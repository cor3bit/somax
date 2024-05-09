"""
! Experimental !
Exact Gauss-Newton (EGN) Solver with the softmax that is part of the model
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
from optax import sigmoid_binary_cross_entropy
from jaxopt.tree_util import tree_add_scalar_mul
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


def flatten_2d_jacobian(jac_tree):
    return jax.vmap(lambda _: ravel_pytree(_)[0], in_axes=(0,))(jac_tree)


def flatten_3d_jacobian(jac_tree):
    flattened_jacobians = jax.vmap(flatten_2d_jacobian)(jac_tree)
    # b, c, m = flattened_jacobians.shape
    # J = flattened_jacobians.reshape(-1, m)
    return flattened_jacobians.reshape(-1, flattened_jacobians.shape[-1])


class EGNProbState(NamedTuple):
    """Named tuple containing state information."""
    iter_num: int
    # error: float
    # value: float
    stepsize: float
    regularizer: float
    # direction_inf_norm: float
    velocity_m: Optional[Any]
    # velocity_v: Optional[Any]


@dataclasses.dataclass(eq=False)
class EGNProb(base.StochasticSolver):
    # Jacobian of the residual function
    predict_fun: Callable
    jac_fun: Optional[Callable] = None
    loss_fun: Optional[Callable] = None

    # vectorization axis for Jacobian, None - no vectorization, 0 - batch axis of the tensor
    # default value (None, 0,) matches predict_fun(params, X)
    jac_axis: Tuple[Optional[int], ...] = (None, 0,)

    # Loss function parameters
    loss_type: str = 'mse'  # ['mse', 'ce']

    # Either fixed alpha if line_search=False or max_alpha if line_search=True
    learning_rate: Optional[float] = None  # alpha_0

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
    regularizer: float = 1.0  # lambda_0
    regularizer_eps: float = 1e-1  # lambda_T
    lambda_decrease_factor: float = 0.99  # default value recommended by Kiros
    lambda_increase_factor: float = 1.01  # default value recommended by Kiros
    total_iterations: Optional[int] = None

    # Momentum parameters
    momentum: float = 0.0
    # beta2: float = 0.0

    pre_update: Optional[Callable] = None

    verbose: int = 0

    jit: bool = True
    unroll: base.AutoOrBoolean = "auto"

    def __post_init__(self):
        super().__post_init__()

        self.reference_signature = self.predict_fun

        # Regression (MSE)
        if self.loss_type == 'mse':
            raise NotImplementedError("MSE loss is not supported yet.")

            if self.loss_fun is None:
                self.loss_fun = self.mse

            self.jac_fun = jax.vmap(jax.value_and_grad(self.predict_fun), in_axes=self.jac_axis)
            self.calculate_direction = self.calculate_direction_mse
            self.regularizer_array = self.batch_size * self.regularizer * jnp.eye(self.batch_size)
        # Classification (Cross-Entropy)
        elif self.loss_type == 'ce' or self.loss_type == 'xe':
            if self.n_classes == 1:  # binary classification
                raise NotImplementedError("Binary classification is not supported yet.")

                if self.loss_fun is None:
                    self.loss_fun = self.ce_binary

                self.jac_fun = jax.vmap(jax.value_and_grad(self.predict_with_true_class), in_axes=(None, 0, 0,))
                self.calculate_direction = self.calculate_direction_ce_binary
            else:
                assert self.loss_fun is not None

                self.jac_fun = jax.vmap(jax.value_and_grad(self.predict_with_true_class), in_axes=(None, 0, 0))
                self.calculate_direction = self.calculate_direction_ce

            self.regularizer_array = self.batch_size * self.regularizer * jnp.eye(self.batch_size)
            # self.block_diag_template = jnp.eye(self.batch_size).reshape(self.batch_size, 1, self.batch_size, 1)
        else:
            raise ValueError(f"Loss type \'{self.loss_type}\' not supported.")

        # set up momentum
        if self.momentum < 0. or self.momentum > 1.:
            raise ValueError(f"'momentum' must belong to closed interval [0,1]")
        # if self.beta2 < 0. or self.beta2 > 1.:
        #     raise ValueError(f"'beta2' must belong to closed interval [0,1]")

        # set up adaptive regularization
        if self.adaptive_lambda:
            raise NotImplementedError("Adaptive regularization is not supported yet.")

        # if self.adaptive_lambda:
        #     if self.total_iterations is None:
        #         raise ValueError(f"'total_iterations' must be provided for adaptive regularization")
        #
        #     # self.lambda_schedule_fn = linear_schedule(
        #     #     self.regularizer, self.regularizer_eps, self.total_iterations, )
        #
        #     self.lambda_schedule_fn = warmup_exponential_decay_schedule(
        #         self.regularizer,
        #         peak_value=self.regularizer,
        #         warmup_steps=int(self.total_iterations * 0.3),
        #         transition_steps=int(self.total_iterations * 0.7),
        #         decay_rate=1e-3,
        #         end_value=self.regularizer_eps, )
        #
        # else:
        #     self.lambda_schedule_fn = constant_schedule(self.regularizer)

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
            state: EGNProbState,
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
        direction, grad_loss, J, Q = self.calculate_direction(params, state, targets, *args)

        # ---------- STEP 2: momentum acceleration ---------- #
        if self.momentum > 0:
            # direction with bias-corrected momentum
            # d = (m * v + (1 - m) * d) / (1 - m^t)
            direction_m = self.momentum * state.velocity_m + (1 - self.momentum) * direction
            bias_corr_m = 1 - self.momentum ** (state.iter_num + 1)
            direction = direction_m / bias_corr_m

            # if self.beta2 > 0:
            #     v_eps = 1e-7
            #     direction_v = self.beta2 * state.velocity_v + (1 - self.beta2) * direction * direction
            #     bias_corr_v = 1 - self.beta2 ** (state.iter_num + 1)
            #     bias_corrected_direction_v = jnp.sqrt(direction_v / bias_corr_v) + v_eps
            #     direction = direction_m / bias_corr_m / bias_corrected_direction_v
            # else:
            #     direction = direction_m / bias_corr_m
            #     direction_v = None
        else:
            direction_m = None
            # direction_v = None

        # ---------- STEP 3: line search for alpha ---------- #
        f_cur = None
        f_next = None
        params_flat, unflatten_fn = ravel_pytree(params)
        if not self.line_search:
            # constant learning rate
            stepsize = state.stepsize
            next_params = unflatten_fn(params_flat + stepsize * direction)
        else:
            stepsize = self.reset_stepsize(state.stepsize)

            goldstein = self.reset_option == 'goldstein'

            f_cur = self.loss_fun(params, *args, targets)

            # the directional derivative used for Armijo's line search
            direct_deriv = grad_loss.T @ direction

            direction_tree = unflatten_fn(direction)

            stepsize, next_params, f_next = self._armijo_line_search(
                goldstein, self.maxls, params, f_cur, stepsize,
                direction_tree, direct_deriv, self._coef,
                self.decrease_factor, self.increase_factor,
                self.max_stepsize, args, targets,
            )

        # ---------- STEP 4: update (next step) lambda ---------- #
        # TODO: adaptive lambda is currently disabled
        regularizer_next = state.regularizer

        # switch to lambda(t) schedule
        # regularizer_next = self.lambda_schedule_fn(state.iter_num + 1)

        # if not self.adaptive_lambda:
        #     # constant lambda
        #     regularizer_next = state.regularizer
        # else:
        #     # f_cur can be already computed if line search is used
        #     f_cur = self.loss_fun(params, *args, targets) if f_cur is None else f_cur
        #
        #     # f_next can be already computed if line search is used
        #
        #     # next_params_no_alpha = unflatten_fn(params_flat + direction)
        #
        #     f_next = self.loss_fun(next_params, *args, targets) if f_next is None else f_next
        #
        #     # in a good scenario, should be large and negative
        #     num = f_next - f_cur
        #
        #     delta_w = stepsize * direction
        #
        #     b = targets.shape[0]
        #
        #     # dimensions: (b x d) @ (d x 1) = (b x 1)
        #     mvp = J @ delta_w
        #     if Q is None:
        #         denom = grad_loss.T @ delta_w + 0.5 * mvp.T @ mvp / b
        #     else:
        #         denom = grad_loss.T @ delta_w + 0.5 * mvp.T @ Q @ mvp / b
        #
        #     # negative denominator means that the direction is a descent direction
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

        # construct next state
        next_state = EGNProbState(
            iter_num=state.iter_num + 1,  # Next Iteration
            stepsize=stepsize,  # Current alpha
            regularizer=regularizer_next,  # Next lambda
            velocity_m=direction_m,  # First moment accumulator
            # velocity_v=direction_v,  # Second moment accumulator
        )

        return base.OptStep(params=next_params, state=next_state)

    def init_state(self,
                   init_params: Any,
                   *args,
                   **kwargs) -> EGNProbState:

        velocity_m = jnp.zeros_like(ravel_pytree(init_params)[0]) if self.momentum > 0 else None
        # velocity_v = jnp.zeros_like(ravel_pytree(init_params)[0]) if self.beta2 > 0 else None

        return EGNProbState(
            iter_num=0,
            stepsize=self.learning_rate,
            regularizer=self.regularizer,
            velocity_m=velocity_m,
            # velocity_v=velocity_v,
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

    def calculate_direction_mse(self, params, state, targets, *args):
        # 1st most time-consuming part - calculate the Jacobian of the Neural Net
        batch_preds, jac_tree = self.jac_fun(params, *args)

        # convert pytree to JAX array (here, J_f)
        J = flatten_2d_jacobian(jac_tree)

        residuals = targets - jnp.squeeze(batch_preds)

        if self.line_search or self.adaptive_lambda:
            # !! we need a minus sign here because J is a Jacobian of f(w) not J of r(w)
            # since r(w)=y-f(w), J_r = -J_f
            grad_loss = -J.T @ residuals / self.batch_size
        else:
            grad_loss = None

        # 2nd most time-consuming part - solve the linear system of dimension (batch_size x batch_size)
        # regularizer_t = state.regularizer + self.regularizer_eps
        # regularizer_t = state.regularizer  # from the schedule
        temp = jnp.linalg.solve(self.regularizer_array + J @ J.T, residuals)

        direction = J.T @ temp

        return direction, grad_loss, J, None

    def calculate_direction_ce(self, params, state, targets, *args):
        # ------ Start of the function ------ #
        # 1st most time-consuming part - calculate the Jacobian by jax.jacrev()
        batch_probs_of_true_class, jac_tree = self.jac_fun(params, targets, *args)

        # convert a 3D pytree (b, c, m) to a 2D array of stacked Jacobians
        J = flatten_2d_jacobian(jac_tree)

        # build block diagonal H from probs
        Q = jnp.diag(1 / batch_probs_of_true_class)

        # calculate (pseudo)residuals
        r = jnp.ones_like(batch_probs_of_true_class)

        # VERIFY: grad_loss is correct
        # grad_loss = -J.T @ r / self.batch_size
        # grad_loss_tree_true = jax.grad(self.loss_fun)(params, *args, targets)
        # grad_loss_true = ravel_pytree(grad_loss_tree_true)[0]
        # diff_inf_norm = jnp.max(jnp.abs(grad_loss - grad_loss_true))

        if self.line_search or self.adaptive_lambda:
            grad_loss = -J.T @ (r / batch_probs_of_true_class) / self.batch_size
        else:
            grad_loss = None

        # calculate the direction
        # regularizer_t = state.regularizer
        temp = jax.scipy.linalg.solve(
            self.regularizer_array * batch_probs_of_true_class + Q @ (J @ J.T), r,
            assume_a='sym',
        )

        direction = J.T @ temp

        return direction, grad_loss, J, Q

    def calculate_direction_ce_binary(self, params, state, targets, *args):
        # 1st most time-consuming part - calculate the Jacobian by jax.jacrev()
        batch_logits, jac_tree = self.jac_fun(params, *args)

        # convert a 3D pytree (b, c, m) to a 2D array of stacked Jacobians
        J = flatten_2d_jacobian(jac_tree)

        # build block diagonal H from logits
        probs = jax.nn.sigmoid(batch_logits)

        Q = jnp.diag(probs * (1 - probs))

        # calculate (pseudo)residuals
        r = targets.reshape(-1, ) - probs

        if self.line_search or self.adaptive_lambda:
            grad_loss = -J.T @ r / self.batch_size
        else:
            grad_loss = None

        # calculate the direction
        # regularizer_t = state.regularizer
        temp = jnp.linalg.solve(self.regularizer_array + Q @ (J @ J.T), r)

        direction = J.T @ temp

        return direction, grad_loss, J, Q

    def mse(self, params, x, y):
        # b x 1
        residuals = y - self.predict_fun(params, x)

        # 1,
        # average over the batch
        return 0.5 * jnp.mean(jnp.square(residuals))

    # def ce(self, params, x, y):
    #     # b x C
    #     logits = self.predict_fun(params, x)
    #
    #     # b x C
    #     # jax.nn.log_softmax combines exp() and log() in a numerically stable way.
    #     log_probs = jax.nn.log_softmax(logits)
    #
    #     # b x 1
    #     # if y is one-hot encoded, this operation picks the log probability of the correct class
    #     residuals = jnp.sum(y * log_probs, axis=-1)
    #
    #     # 1,
    #     # average over the batch
    #     return -jnp.mean(residuals)
    #
    # def ce_binary(self, params, x, y):
    #     # b x 1
    #     logits = self.predict_fun(params, x)
    #
    #     # b x 1
    #     loss = sigmoid_binary_cross_entropy(logits.ravel(), y)
    #
    #     # 1,
    #     # average over the batch
    #     return jnp.mean(loss)

    def predict_with_aux(self, params, *args):
        preds = self.predict_fun(params, *args)
        return preds, preds

    def predict_ravel(self, params, *args):
        preds = self.predict_fun(params, *args)
        return preds[0]

    def predict_with_true_class(self, params, targets, *args):
        preds = self.predict_fun(params, *args)
        pred = jnp.vdot(preds, targets)
        return pred

    def __hash__(self):
        # We assume that the attribute values completely determine the solver.
        return hash(self.attribute_values())
