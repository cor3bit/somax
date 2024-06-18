"""
Stochastic Quasi-Newton Framework, Algorithm 6.2 from Bottou et al. (2018)
Paper: https://arxiv.org/abs/1606.04838
"""

from typing import Any
from typing import Callable
from typing import NamedTuple, Tuple
from typing import Optional
import dataclasses
from functools import partial

import jax
from jax import tree_map
import jax.lax as lax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxopt._src import base
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_map
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_sum
from jaxopt.tree_util import tree_vdot_real


def select_ith_tree(tree_history, i):
    """Select tree corresponding to ith history entry."""
    return tree_map(lambda a: a[i, ...], tree_history)


def inv_hessian_product(
        pytree: Any,
        s_history: Any,
        y_history: Any,
        rho_history: jnp.ndarray,
        gamma: float = 1.0,
        start: int = 0,
):
    """Product between an approximate Hessian inverse and a pytree.

    Histories are pytrees of the same structure as `pytree` except
    that the leaves are arrays of shape `(history_size, ...)`, where
    `...` means the same shape as `pytree`'s leaves.

    The notation follows the reference below.

    Args:
      pytree: pytree to multiply with.
      s_history: pytree whose leaves contain parameter residuals,
        i.e., `s[k] = x[k+1] - x[k]`
      y_history: pytree whose leaves contain gradient residuals,
        i.e., `y[k] = g[k+1] - g[k]`.
      rho_history: array containing `rho[k] = 1. / vdot(s[k], y[k])`.
      gamma: scalar to use for the initial inverse Hessian approximation,
        i.e., `gamma * I`.
      start: starting index in the circular buffer.

    Returns:
      Product between approximate Hessian inverse and the pytree

    Reference:
      Jorge Nocedal and Stephen Wright.
      Numerical Optimization, second edition.
      Algorithm 7.4 (page 178).
    """

    # Two-loop recursion to compute the product of the inverse Hessian approximation
    history_size = rho_history.shape[0]

    indices = (start + jnp.arange(history_size)) % history_size

    def body_right(r, i):
        si, yi = select_ith_tree(s_history, i), select_ith_tree(y_history, i)
        alpha = rho_history[i] * tree_vdot_real(si, r)
        r = tree_add_scalar_mul(r, -alpha, yi)
        return r, alpha

    r, alpha = jax.lax.scan(body_right, pytree, indices, reverse=True)

    r = tree_scalar_mul(gamma, r)

    def body_left(r, args):
        i, alpha = args
        si, yi = select_ith_tree(s_history, i), select_ith_tree(y_history, i)
        beta = rho_history[i] * tree_vdot_real(yi, r)
        r = tree_add_scalar_mul(r, alpha - beta, si)
        return r, beta

    r, _ = jax.lax.scan(body_left, r, (indices, alpha))

    return r


def init_history(pytree, history_size):
    fun = lambda leaf: jnp.zeros((history_size,) + leaf.shape, dtype=leaf.dtype)
    return tree_map(fun, pytree)


def update_history(history_pytree, new_pytree, last):
    fun = lambda history_array, new_value: history_array.at[last].set(new_value)
    return tree_map(fun, history_pytree, new_pytree)


def compute_gamma(s_history: Any, y_history: Any, last: int):
    """Compute scalar gamma defining the initialization of the approximate Hessian."""
    # Let gamma = vdot(y_history[last], s_history[last]) / sqnorm(y_history[last]).
    # The initial inverse Hessian approximation can be set to gamma * I.
    # See Numerical Optimization, second edition, equation (7.20).
    # Note that unlike BFGS, the initialization can change on every iteration.

    num = tree_sum(tree_map(
        lambda x, y: tree_vdot_real(x[last], y[last]),
        s_history,
        y_history,
    ))

    denom = tree_sum(tree_map(
        lambda x: tree_vdot_real(x[last], x[last]),
        y_history,
    ))

    return jnp.where(denom > 0, num / denom, 1.0)


class SQNState(NamedTuple):
    """Named tuple containing state information."""
    iter_num: int
    stepsize: float
    s_history: Any
    y_history: Any
    rho_history: jnp.ndarray
    gamma: jnp.ndarray
    loss: float


@dataclasses.dataclass(eq=False)
class SQN(base.StochasticSolver):
    loss_fun: Callable

    warmup: int = 0

    history_size: int = 10
    use_gamma: bool = True

    # Either fixed alpha if line_search=False or max_alpha if line_search=True
    learning_rate: Optional[float] = None  # alpha_0

    # Line Search parameters
    line_search: bool = False

    aggressiveness: float = 0.9  # default value recommended by Vaswani et al.
    decrease_factor: float = 0.8  # default value recommended by Vaswani et al.
    increase_factor: float = 1.5  # default value recommended by Vaswani et al.
    reset_option: str = 'increase'  # ['increase', 'goldstein', 'conservative']

    max_stepsize: float = 1.0
    maxls: int = 15

    pre_update: Optional[Callable] = None

    verbose: int = 0

    jit: bool = True
    unroll: base.AutoOrBoolean = "auto"

    def __post_init__(self):
        super().__post_init__()

        self.reference_signature = self.loss_fun

        self.value_and_grad_fun = jax.value_and_grad(self.loss_fun)

    def update(
            self,
            params: Any,
            state: SQNState,
            *args,
            **kwargs,
    ) -> base.OptStep:
        """Performs one iteration of the solver.

        Args:
          params: pytree containing the parameters.
          state: named tuple containing the solver state.
          *args: additional positional arguments to be passed to ``fun``.
          **kwargs: additional keyword arguments to be passed to ``fun``.
        Returns:
          (params, state)
        """
        start = state.iter_num % self.history_size

        # ---------- STEP 0: resolve gradient and curvature batches ---------- #
        # TODO analyze *args and **kwargs
        # split (x,y) pair into (x,) and (y,)
        # if 'targets' in kwargs:
        #     targets = kwargs['targets']
        #     nn_args = args
        # else:
        #     targets = args[-1]
        #     nn_args = args[:-1]

        # ---------- STEP 1: calculate direction with the "gradient batch" ---------- #
        # TODO currently "online L-BFGS": assumes S_k == S_k^H
        # TODO implement "SQN" for S_k != S_k^H

        # value, grad = (state.value, state.grad)
        # descent_direction = tree_scalar_mul(-1.0, tree_conj(grad))

        # TODO: consider adding warmup with k ADAM steps before starting L-BFGS updates

        loss, grad_tree = self.value_and_grad_fun(params, *args, **kwargs)

        hvp_tree = inv_hessian_product(
            grad_tree,
            s_history=state.s_history,
            y_history=state.y_history,
            rho_history=state.rho_history,
            gamma=state.gamma,
            start=start,
        )

        # direction_tree = tree_scalar_mul(-1, hvp_tree)

        next_params = tree_add_scalar_mul(params, -state.stepsize, hvp_tree)

        # ---------- STEP 2: update L-BFGS history with the "curvature batch" ---------- #

        # TODO s,y from curvature batch
        s = tree_sub(next_params, params)

        next_loss, next_grad_tree = self.value_and_grad_fun(next_params, *args, **kwargs)
        y = tree_sub(next_grad_tree, grad_tree)

        vdot_sy = tree_vdot_real(s, y)
        rho = jnp.where(vdot_sy == 0, 0, 1. / vdot_sy)

        s_history = update_history(state.s_history, s, start)
        y_history = update_history(state.y_history, y, start)
        rho_history = update_history(state.rho_history, rho, start)

        if self.use_gamma:
            gamma = compute_gamma(s_history, y_history, start)
        else:
            gamma = jnp.array(1.0)  # , dtype=realdtype

        # ---------- STEP 3: line search for alpha ---------- #
        # f_cur = None
        # f_next = None
        # params_flat, unflatten_fn = ravel_pytree(params)
        # if not self.line_search:
        #     # constant learning rate
        #     stepsize = state.stepsize
        #     next_params = unflatten_fn(params_flat + stepsize * direction)
        # else:
        #     stepsize = self.reset_stepsize(state.stepsize)
        #
        #     goldstein = self.reset_option == 'goldstein'
        #
        #     f_cur = self.loss_fun(params, *nn_args, targets)
        #
        #     # the directional derivative used for Armijo's line search
        #     direct_deriv = grad_loss.T @ direction
        #
        #     direction_tree = unflatten_fn(direction)
        #
        #     stepsize, next_params, f_next = self._armijo_line_search(
        #         goldstein, self.maxls, params, f_cur, stepsize,
        #         direction_tree, direct_deriv, self._coef,
        #         self.decrease_factor, self.increase_factor,
        #         self.max_stepsize, nn_args, targets,
        #     )

        # ---------- bookkeeping ---------- #

        # construct next state
        next_state = SQNState(
            iter_num=state.iter_num + 1,  # Next Iteration
            stepsize=state.stepsize,  # Current alpha
            s_history=s_history,
            y_history=y_history,
            rho_history=rho_history,
            gamma=gamma,
            loss=loss,
        )

        return base.OptStep(params=next_params, state=next_state)

    def init_state(self,
                   init_params: Any,
                   *args,
                   **kwargs) -> SQNState:

        return SQNState(
            iter_num=0,
            stepsize=self.learning_rate,
            s_history=init_history(init_params, self.history_size),
            y_history=init_history(init_params, self.history_size),
            rho_history=jnp.zeros(self.history_size),
            gamma=jnp.array(1.0),
            loss=jnp.inf,
        )

    def optimality_fun(self, params, *args, **kwargs):
        """Optimality function mapping compatible with ``@custom_root``."""
        # return self._grad_fun(params, *args, **kwargs)[0]
        raise NotImplementedError

    def __hash__(self):
        # We assume that the attribute values completely determine the solver.
        return hash(self.attribute_values())
