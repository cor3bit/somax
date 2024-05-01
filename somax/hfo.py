"""
Hessian-Free Optimization (HFO):
- approximate solution to the linear system via the CG method
- adaptive regularization (lambda)
- line search
"""

from typing import Any
from typing import Callable
from typing import NamedTuple, Tuple
from typing import Optional
import dataclasses
from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg
from jaxopt.tree_util import tree_add_scalar_mul, tree_scalar_mul
from jaxopt._src import base
from jaxopt._src import loop


def wolfe_cond_violated(stepsize, coef, f_cur, f_next, direct_deriv):
    eps = jnp.finfo(f_next.dtype).eps
    loss_decrease = f_cur - f_next + eps
    prescribed_decrease = -stepsize * coef * direct_deriv
    return prescribed_decrease > loss_decrease


def curvature_cond_violated(stepsize, coef, f_cur, f_next, direct_deriv):
    loss_decrease = f_cur - f_next
    prescribed_decrease = -stepsize * (1. - coef) * direct_deriv
    return loss_decrease > prescribed_decrease


def armijo_line_search(loss_fun, unroll, jit,
                       goldstein, maxls,
                       params, f_cur, stepsize,
                       direction, direct_deriv,
                       coef, decrease_factor, increase_factor, max_stepsize,
                       args, targets):
    # given direction calculate next params
    next_params = tree_add_scalar_mul(params, stepsize, direction)

    # calculate loss at next params
    f_next = loss_fun(next_params, *args, targets)

    # grad_sqnorm = tree_l2_norm(grad, squared=True)

    def update_stepsize(t):
        """Multiply stepsize per factor, return new params and new value."""
        stepsize, factor = t
        stepsize = stepsize * factor
        stepsize = jnp.minimum(stepsize, max_stepsize)

        next_params = tree_add_scalar_mul(params, stepsize, direction)

        f_next = loss_fun(next_params, *args, targets)

        return stepsize, next_params, f_next

    def body_fun(t):
        stepsize, next_params, f_next, _ = t

        violated = wolfe_cond_violated(stepsize, coef, f_cur, f_next, direct_deriv)

        stepsize, next_params, f_next = lax.cond(
            violated,
            update_stepsize,
            lambda _: (stepsize, next_params, f_next),
            operand=(stepsize, decrease_factor),
        )

        if goldstein:
            goldstein_violated = curvature_cond_violated(
                stepsize, coef, f_cur, f_next, direct_deriv)

            stepsize, next_params, f_next = lax.cond(
                goldstein_violated, update_stepsize,
                lambda _: (stepsize, next_params, f_next),
                operand=(stepsize, increase_factor),
            )

            violated = jnp.logical_or(violated, goldstein_violated)

        return stepsize, next_params, f_next, violated

    init_val = stepsize, next_params, f_next, jnp.array(True)

    ret = loop.while_loop(cond_fun=lambda t: t[-1],  # check boolean violated
                          body_fun=body_fun,
                          init_val=init_val, maxiter=maxls,
                          unroll=unroll, jit=jit)

    return ret[:-1]  # remove boolean


class HFOState(NamedTuple):
    """Named tuple containing state information."""
    iter_num: int
    stepsize: float
    regularizer: float
    cg_guess: Optional[Any]


@dataclasses.dataclass(eq=False)
class HFO(base.StochasticSolver):
    # Jacobian of the residual function
    loss_fun: Callable
    maxcg: int = 3

    # Either fixed alpha if line_search=False or max_alpha if line_search=True
    learning_rate: float = 1.0
    decay_coef = 0.1

    batch_size: Optional[int] = None

    n_classes: Optional[int] = None

    # Adaptive Regularization parameters
    adaptive_lambda: bool = True
    regularizer: float = 1.0
    lambda_decrease_factor: float = 0.99  # default value recommended by Kiros
    lambda_increase_factor: float = 1.01  # default value recommended by Kiros

    # Line Search parameters
    line_search: bool = True

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
        self.grad_fun = jax.grad(self.loss_fun)

        # set up line search
        if self.line_search:
            # !! learning rate is the maximum step size in case of line search
            self.max_stepsize = self.learning_rate

            options = ['increase', 'goldstein', 'conservative']

            if self.reset_option not in options:
                raise ValueError(f"'reset_option' should be one of {options}")
            if self.aggressiveness <= 0. or self.aggressiveness >= 1.:
                raise ValueError(f"'aggressiveness' must belong to open interval (0,1)")

            self._coef = 1 - self.aggressiveness

            unroll = self._get_unroll_option()

            armijo_with_fun = partial(armijo_line_search, self.loss_fun, unroll, self.jit)
            if self.jit:
                jitted_armijo = jax.jit(armijo_with_fun, static_argnums=(0, 1))
                self._armijo_line_search = jitted_armijo
            else:
                self._armijo_line_search = armijo_with_fun

    def update(
            self,
            params: Any,
            state: HFOState,
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

        # ---------- STEP 1: calculate direction with DG ---------- #
        targets = kwargs['targets']
        direction_tree, grad_loss_tree = self.calculate_direction(params, state, targets, *args)

        # ---------- STEP 2: line search for alpha ---------- #
        f_cur = None
        f_next = None
        direct_deriv = None
        if not self.line_search:
            # constant learning rate
            stepsize = state.stepsize
            next_params = tree_add_scalar_mul(params, state.stepsize, direction_tree)
        else:
            stepsize = self.reset_stepsize(state.stepsize)

            goldstein = self.reset_option == 'goldstein'

            f_cur = self.loss_fun(params, *args, targets)

            # the directional derivative used for Armijo's line search
            direction, _ = ravel_pytree(direction_tree)
            grad_loss, _ = ravel_pytree(grad_loss_tree)
            direct_deriv = grad_loss.T @ direction

            stepsize, next_params, f_next = self._armijo_line_search(
                goldstein, self.maxls, params, f_cur, stepsize,
                direction_tree, direct_deriv, self._coef,
                self.decrease_factor, self.increase_factor,
                self.max_stepsize, args, targets,
            )

        # ---------- STEP 3: update (next step) lambda ---------- #
        if not self.adaptive_lambda:
            # constant regularization
            regularizer_next = state.regularizer
        else:
            # numerator: in a good scenario, should be large and negative
            if f_cur is None:
                f_cur = self.loss_fun(params, *args, targets)
            if f_next is None:
                f_next = self.loss_fun(next_params, *args, targets)

            num = f_next - f_cur

            # denominator
            Hv_tree = self.hvp(params, direction_tree, targets, *args)

            # flattening stage
            Hv, _ = ravel_pytree(Hv_tree)

            if direct_deriv is None:
                direction, _ = ravel_pytree(direction_tree)
                grad_loss, _ = ravel_pytree(grad_loss_tree)
                direct_deriv = grad_loss.T @ direction

            denom = 0.5 * jnp.vdot(direction, Hv) + direct_deriv

            # negative denominator means that the direction is a descent direction
            rho = num / denom

            regularizer_next = lax.cond(
                rho < 0.25,
                lambda _: self.lambda_increase_factor * state.regularizer,
                lambda _: lax.cond(
                    rho > 0.75,
                    lambda _: self.lambda_decrease_factor * state.regularizer,
                    lambda _: state.regularizer,
                    None,
                ),
                None,
            )

        # construct the next state
        next_state = HFOState(
            iter_num=state.iter_num + 1,  # Next Iteration
            stepsize=stepsize,  # Current alpha
            regularizer=regularizer_next,  # Next lambda
            cg_guess=tree_scalar_mul(self.decay_coef, direction_tree),  # Next CG guess
        )

        return base.OptStep(params=next_params, state=next_state)

    def init_state(self,
                   init_params: Any,
                   *args,
                   **kwargs) -> HFOState:
        return HFOState(
            iter_num=0,
            stepsize=self.learning_rate,
            regularizer=self.regularizer,
            cg_guess=None,
        )

    def optimality_fun(self, params, *args, **kwargs):
        """Optimality function mapping compatible with ``@custom_root``."""
        # return self._grad_fun(params, *args, **kwargs)[0]
        raise NotImplementedError

    def reset_stepsize(self, stepsize):
        """Return new step size for current step, according to reset_option."""
        if self.reset_option == 'goldstein':
            return stepsize
        if self.reset_option == 'conservative':
            return stepsize
        stepsize = stepsize * self.increase_factor
        return jnp.minimum(stepsize, self.max_stepsize)

    def hvp(self, params, vec, targets, *args):
        # prep grad function to accept only "params"
        def inner_grad_fun(params):
            return self.grad_fun(params, *args, targets)

        return jax.jvp(inner_grad_fun, (params,), (vec,))[1]

    def calculate_direction(self, params, state, targets, *args):
        def mvp(vec):
            # Hv
            hv = self.hvp(params, vec, targets, *args)
            # add regularization, works since (H + lambda*I) v = Hv + lambda*v
            return tree_add_scalar_mul(hv, state.regularizer, vec)

        # --------- Start Here --------- #
        # calculate grad
        grad_tree = self.grad_fun(params, *args, targets)

        # CG iterations
        direction, _ = cg(
            A=mvp,
            b=tree_scalar_mul(-1, grad_tree),
            maxiter=self.maxcg,
            x0=state.cg_guess,  # initial guess
            # M=None,  # preconditioner (see wiesler2013: preconditioner is not important)
        )

        return direction, grad_tree

    def __hash__(self):
        # We assume that the attribute values completely determine the solver.
        return hash(self.attribute_values())
