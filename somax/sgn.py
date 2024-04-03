"""
Stochastic Gauss-Newton (SGN):
- approximate solution to the linear system via the CG method
- adaptive learning rate (line search)
- adaptive regularization (lambda)
- momentum acceleration
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


class SGNState(NamedTuple):
    """Named tuple containing state information."""
    iter_num: int
    # error: float
    # value: float
    stepsize: float
    regularizer: float
    # direction_inf_norm: float
    velocity: Optional[Any]


@dataclasses.dataclass(eq=False)
class SGN(base.StochasticSolver):
    # Jacobian of the residual function
    predict_fun: Callable
    loss_fun: Optional[Callable] = None
    loss_type: str = 'mse'
    maxcg: int = 100

    # Either fixed alpha if line_search=False or max_alpha if line_search=True
    learning_rate: Optional[float] = None

    batch_size: Optional[int] = None

    n_classes: Optional[int] = None

    # Line Search parameters
    line_search: bool = False

    aggressiveness: float = 0.9  # default value recommended by Vaswani et al.
    decrease_factor: float = 0.8  # default value recommended by Vaswani et al.
    increase_factor: float = 1.5  # default value recommended by Vaswani et al.
    reset_option: str = 'increase'  # ['increase', 'goldstein', 'conservative']

    max_stepsize: float = 1.0
    maxls: int = 15

    # Adaptive Regularization parameters
    adaptive_lambda: bool = False
    regularizer: float = 1.0
    # regularizer_eps: float = 1e-5
    lambda_decrease_factor: float = 0.99  # default value recommended by Kiros
    lambda_increase_factor: float = 1.01  # default value recommended by Kiros

    # Momentum parameters
    momentum: float = 0.0

    pre_update: Optional[Callable] = None

    verbose: int = 0

    jit: bool = True
    unroll: base.AutoOrBoolean = "auto"

    def __post_init__(self):
        super().__post_init__()

        self.reference_signature = self.loss_fun

        # self.hvp_fun = jax.jit(jax.hessian_vector_product(self.mse))

        # Regression (MSE)
        if self.loss_type == 'mse':
            self.grad_fun = jax.grad(self.mse)
            self.loss_fun = self.mse_from_predictions
        # Classification (Cross-Entropy)
        elif self.loss_type == 'ce' or self.loss_type == 'xe':
            self.grad_fun = jax.grad(self.ce)
            self.loss_fun = self.ce_from_logits

        # Regression (MSE)
        # if self.loss_type == 'mse':
        #     # TODO
        #     self.jac_fun = jax.vmap(jax.value_and_grad(self.predict_fun), in_axes=self.jac_axis)
        #     self.calculate_direction = self.calculate_direction_mse
        #     self.regularizer_array = self.batch_size * self.regularizer * jnp.eye(self.batch_size)
        # # Classification (Cross-Entropy)
        # elif self.loss_type == 'ce' or self.loss_type == 'xe':
        #     self.loss_fun = self.ce
        #     self.calculate_direction = self.calculate_direction_ce
        #     if self.n_classes == 2:
        #         # TODO check n_classes == 2 does not interfere with the internal functions
        #         self.jac_fun = jax.vmap(jax.grad(self.predict_with_aux, has_aux=True), in_axes=(None, 0))
        #     else:
        #         self.jac_fun = jax.vmap(jax.jacrev(self.predict_with_aux, has_aux=True), in_axes=(None, 0))
        #
        #     self.regularizer_array = self.batch_size * self.regularizer * jnp.eye(self.n_classes * self.batch_size)
        #     self.block_diag_template = jnp.eye(self.batch_size).reshape(self.batch_size, 1, self.batch_size, 1)
        # else:
        #     raise ValueError(f"Loss type \'{self.loss_type}\' not supported.")

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
            state: SGNState,
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

        # convert pytree to JAX array (w)
        # params_flat, unflatten_fn = ravel_pytree(params)

        # ---------- STEP 1: calculate direction with DG ---------- #
        targets = kwargs['targets']
        direction_tree, grad_loss_tree = self.calculate_direction(params, state, targets, *args)

        # TODO restore LS, AT, momentum
        # # ---------- STEP 2: line search for alpha ---------- #
        # f_cur = None
        # f_next = None
        # next_params = None
        # if self.line_search:
        #     stepsize = self.reset_stepsize(state.stepsize)
        #
        #     goldstein = self.reset_option == 'goldstein'
        #
        #     f_cur = self.loss_fun(params, *args, targets)
        #
        #     # the directional derivative used for Armijo's line search
        #     grad_loss = ravel_pytree(grad_loss_tree)[0]
        #     direct_deriv = grad_loss.T @ direction
        #
        #     direction_packed = unflatten_fn(direction)
        #
        #     stepsize, next_params, f_next = self._armijo_line_search(
        #         goldstein, self.maxls, params, f_cur, stepsize, direction_packed, direct_deriv, self._coef,
        #         self.decrease_factor, self.increase_factor, self.max_stepsize, args, targets, )
        # else:
        #     stepsize = state.stepsize
        #
        # # ---------- STEP 3: momentum acceleration ---------- #
        # if self.momentum == 0:
        #     next_velocity = None
        # else:
        #     # next_params = params + stepsize*direction + momentum*(params - previous_params)
        #
        #     if next_params is None:
        #         next_params_flat = params_flat + stepsize * direction
        #     else:
        #         next_params_flat, _ = ravel_pytree(next_params)
        #
        #     next_params_flat = next_params_flat + self.momentum * state.velocity
        #     next_velocity = next_params_flat - params_flat
        #
        #     next_params = unflatten_fn(next_params_flat)
        #
        # # ! params should be "packed" in a pytree before sending to the OptStep
        # if next_params is None:
        #     # the only case is when LS=False, Momentum=0
        #     next_params_flat = params_flat + stepsize * direction
        #     next_params = unflatten_fn(next_params_flat)
        #
        # # ---------- STEP 4: update (next step) lambda ---------- #
        # if self.adaptive_lambda:
        #     # f_cur can be already computed if line search is used
        #     f_cur = self.loss_fun(params, *args, targets) if f_cur is None else f_cur
        #
        #     # if momentum is used, we need to calculate f_next again to take into account the momentum term
        #     f_next = self.loss_fun(next_params, *args, targets) if f_next is None or self.momentum > 0 else f_next
        #
        #     # in a good scenario, should be large and negative
        #     num = f_next - f_cur
        #
        #     if self.momentum > 0:
        #         delta_w = next_velocity
        #     else:
        #         delta_w = stepsize * direction
        #
        #     b = targets.shape[0]
        #
        #     # dimensions: (b x d) @ (d x 1) = (b x 1)
        #     J = None
        #     Q = None
        #     # TODO mvp through jvp
        #
        #     mvp = J @ delta_w
        #     if Q is None:
        #         denom = grad_loss.T @ delta_w + 0.5 * mvp.T @ mvp / b
        #     else:
        #         denom = grad_loss.T @ delta_w + 0.5 * mvp.T @ Q @ mvp / b
        #
        #     # negative denominator means that the direction is a descent direction
        #
        #     rho = num / denom
        #
        #     regularizer_next = lax.cond(
        #         rho < 0.25,
        #         lambda _: self.lambda_increase_factor * state.regularizer,
        #         lambda _: lax.cond(
        #             rho > 0.75,
        #             lambda _: self.lambda_decrease_factor * state.regularizer,
        #             lambda _: state.regularizer,
        #             None,
        #         ),
        #         None,
        #     )
        # else:
        #     regularizer_next = state.regularizer

        next_params = tree_add_scalar_mul(params, state.stepsize, direction_tree)

        # construct the next state
        next_state = SGNState(
            iter_num=state.iter_num + 1,  # Next Iteration
            stepsize=state.stepsize,  # Current alpha
            regularizer=state.regularizer,  # Next lambda
            velocity=None,  # Next velocity
        )

        return base.OptStep(params=next_params, state=next_state)

    def init_state(self,
                   init_params: Any,
                   *args,
                   **kwargs) -> SGNState:
        if self.momentum == 0:
            velocity = None
        else:
            velocity = jnp.zeros_like(ravel_pytree(init_params)[0])

        return SGNState(
            iter_num=0,
            stepsize=self.learning_rate,
            regularizer=self.regularizer,
            velocity=velocity,
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

    def gnhvp(self, params, vec, targets, *args):
        def inner_predict_fun(params):
            return self.predict_fun(params, *args)

        def hvp(network_outputs, vec, targets):
            # prep grad function to accept only "network_outputs"
            def inner_grad_fun(network_outputs):
                return jax.grad(self.loss_fun)(network_outputs, targets)

            return jax.jvp(inner_grad_fun, (network_outputs,), (vec,))[1]

        # H_GN ~ J^T Q J, s.t. hvp is J^T Q J v
        network_outputs, Jv = jax.jvp(inner_predict_fun, (params,), (vec,))

        if self.loss_type == 'mse':
            _, JTJv_fun = jax.vjp(inner_predict_fun, params)
            JTJv = JTJv_fun(Jv)[0]
            return tree_scalar_mul(1 / self.batch_size, JTJv)
        else:
            QJv = hvp(network_outputs, Jv, targets)
            _, JTQJv_fun = jax.vjp(inner_predict_fun, params)
            JTQJv = JTQJv_fun(QJv)[0]
            return JTQJv

    def calculate_direction(self, params, state, targets, *args):
        def mvp(vec):
            # H_{GN}v
            # hv = jax.jvp(jax.grad(stand_alone_loss_fn), (params,), (vec,))[1]
            hv = self.gnhvp(params, vec, targets, *args)
            # add regularization, works since (H + lambda*I) v = Hv + lambda*v
            return tree_add_scalar_mul(hv, state.regularizer, vec)

        # --------- Start Here --------- #
        # calculate grad
        grad_tree = self.grad_fun(params, *args, targets)

        # CG iterations
        # TODO initial guess and preconditioner (kiros13)
        # TODO verify n_iter (kiros13)
        direction, _ = cg(
            A=mvp,
            b=tree_scalar_mul(-1, grad_tree),
            maxiter=self.maxcg,
            # x0=None,  # initial guess
            # M=None,  # preconditioner
        )

        return direction, grad_tree

    def mse(self, params, x, y):
        return self.mse_from_predictions(self.predict_fun(params, x), y)

    def mse_from_predictions(self, preds, y):
        return 0.5 * jnp.mean(jnp.square(y - preds))

    def ce(self, params, x, y):
        logits = self.predict_fun(params, x)
        return self.ce_from_logits(logits, y)

    def ce_from_logits(self, logits, y):
        # b x C
        # jax.nn.log_softmax combines exp() and log() in a numerically stable way.
        log_probs = jax.nn.log_softmax(logits)

        # b x 1
        # if y is one-hot encoded, this operation picks the log probability of the correct class
        residuals = jnp.sum(y * log_probs, axis=-1)

        # 1,
        # average over the batch
        return -jnp.mean(residuals)

    def __hash__(self):
        # We assume that the attribute values completely determine the solver.
        return hash(self.attribute_values())
