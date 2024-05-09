"""
GNB Solver with sampling
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


class GNBState(NamedTuple):
    """Named tuple containing state information."""
    iter_num: int
    # error: float
    # value: float
    stepsize: float
    regularizer: float
    # direction_inf_norm: float
    velocity: Optional[Any]


@dataclasses.dataclass(eq=False)
class GNB(base.StochasticSolver):
    # Jacobian of the residual function
    predict_fun: Callable
    jac_fun: Optional[Callable] = None

    # vectorization axis for Jacobian, None - no vectorization, 0 - batch axis of the tensor
    # default value (None, 0,) matches predict_fun(params, X)
    jac_axis: Tuple[Optional[int], ...] = (None, 0,)

    # Loss function parameters
    loss_type: str = 'mse'  # ['mse', 'ce']

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

        self.reference_signature = self.predict_fun

        # Regression (MSE)
        if self.loss_type == 'mse':
            self.loss_fun = self.mse
            self.jac_fun = jax.vmap(jax.value_and_grad(self.predict_fun), in_axes=self.jac_axis)
            self.calculate_direction = self.calculate_direction_mse
            self.regularizer_array = self.batch_size * self.regularizer * jnp.eye(self.batch_size)
        # Classification (Cross-Entropy)
        elif self.loss_type == 'ce' or self.loss_type == 'xe':
            self.loss_fun = self.ce
            self.jac_axis = (None, 0, 0)  # default for classification
            self.jac_fun = jax.vmap(jax.grad(self.ce), in_axes=self.jac_axis)
            self.grad_fun = jax.grad(self.ce_with_aux, has_aux=True)
            self.calculate_direction = self.calculate_direction_ce
            self.regularizer_array = self.batch_size * self.regularizer * jnp.eye(self.batch_size)
            self.block_diag_template = jnp.eye(self.batch_size).reshape(self.batch_size, 1, self.batch_size, 1)
        else:
            raise ValueError(f"Loss type \'{self.loss_type}\' not supported.")

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
            state: GNBState,
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
        params_flat, unflatten_fn = ravel_pytree(params)

        # ---------- STEP 1: calculate direction with DG ---------- #
        targets = kwargs['targets']
        direction, grad_loss, J, Q = self.calculate_direction(params, state, targets, *args)

        # ---------- STEP 2: line search for alpha ---------- #
        f_cur = None
        f_next = None
        next_params = None
        if self.line_search:
            stepsize = self.reset_stepsize(state.stepsize)

            goldstein = self.reset_option == 'goldstein'

            f_cur = self.loss_fun(params, *args, targets)

            # the directional derivative used for Armijo's line search
            direct_deriv = grad_loss.T @ direction

            direction_packed = unflatten_fn(direction)

            stepsize, next_params, f_next = self._armijo_line_search(
                goldstein, self.maxls, params, f_cur, stepsize, direction_packed, direct_deriv, self._coef,
                self.decrease_factor, self.increase_factor, self.max_stepsize, args, targets, )
        else:
            stepsize = state.stepsize

        # ---------- STEP 3: momentum acceleration ---------- #
        if self.momentum == 0:
            next_velocity = None
        else:
            # next_params = params + stepsize*direction + momentum*(params - previous_params)

            if next_params is None:
                next_params_flat = params_flat + stepsize * direction
            else:
                next_params_flat, _ = ravel_pytree(next_params)

            next_params_flat = next_params_flat + self.momentum * state.velocity
            next_velocity = next_params_flat - params_flat

            next_params = unflatten_fn(next_params_flat)

        # ! params should be "packed" in a pytree before sending to the OptStep
        if next_params is None:
            # the only case is when LS=False, Momentum=0
            next_params_flat = params_flat + stepsize * direction
            next_params = unflatten_fn(next_params_flat)

        # ---------- STEP 4: update (next step) lambda ---------- #
        if self.adaptive_lambda:
            # f_cur can be already computed if line search is used
            f_cur = self.loss_fun(params, *args, targets) if f_cur is None else f_cur

            # if momentum is used, we need to calculate f_next again to take into account the momentum term
            f_next = self.loss_fun(next_params, *args, targets) if f_next is None or self.momentum > 0 else f_next

            # in a good scenario, should be large and negative
            num = f_next - f_cur

            if self.momentum > 0:
                delta_w = next_velocity
            else:
                delta_w = stepsize * direction

            b = targets.shape[0]

            # dimensions: (b x d) @ (d x 1) = (b x 1)
            mvp = J @ delta_w
            if Q is None:
                denom = grad_loss.T @ delta_w + 0.5 * mvp.T @ mvp / b
            else:
                denom = grad_loss.T @ delta_w + 0.5 * mvp.T @ Q @ mvp / b

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
        else:
            regularizer_next = state.regularizer

        # construct the next state
        next_state = GNBState(
            iter_num=state.iter_num + 1,  # Next Iteration
            stepsize=stepsize,  # Current alpha
            regularizer=regularizer_next,  # Next lambda
            velocity=next_velocity,  # Next velocity
        )

        return base.OptStep(params=next_params, state=next_state)

    def init_state(self,
                   init_params: Any,
                   *args,
                   **kwargs) -> GNBState:
        if self.momentum == 0:
            velocity = None
        else:
            velocity = jnp.zeros_like(ravel_pytree(init_params)[0])

        return GNBState(
            iter_num=jnp.asarray(0),
            stepsize=jnp.asarray(self.learning_rate),
            regularizer=jnp.asarray(self.regularizer),
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

        # 2st most time-consuming part - solve the linear system of dimension (batch_size x batch_size)
        temp = jax.scipy.linalg.solve(self.regularizer_array + J @ J.T, residuals, assume_a='sym')

        direction = J.T @ temp

        return direction, grad_loss, J, None

    def calculate_direction_ce(self, params, state, targets, *args):
        # prepare keys for vmap() sampling
        # keys = jax.random.split(jax.random.PRNGKey(state.iter_num), num=self.batch_size)

        # OPTION 1
        # 1st most time-consuming part - calculate the Jacobian by jax.jacrev()
        # (selected_logits, (selected_probs, mask)), jac_tree = self.jac_fun(params, keys, *args)

        # jac_tree = self.jac_fun(params, *args, targets)
        #
        # # convert a pytree (b, m) to a 2D array of stacked Jacobians
        # J = flatten_2d_jacobian(jac_tree)
        #
        # # build a (b x b) diagonal matrix
        # # Q = jnp.diag(selected_probs * (1 - selected_probs))
        #
        # # calculate (pseudo-)residuals
        # # selected_targets = jnp.sum(targets * mask, axis=-1)
        # # r = selected_targets - selected_probs #).reshape(-1, )
        # r = jnp.ones((self.batch_size,))
        #
        # if self.line_search or self.adaptive_lambda:
        #     grad_loss = J.T @ r / self.batch_size
        # else:
        #     grad_loss = None
        #
        # # calculate the direction
        # temp = jnp.linalg.solve(self.regularizer_array + J @ J.T, -r)
        #
        # direction = J.T @ temp

        # OPTION 2
        # !! notice that the real targets (y) are used
        grad_loss_tree, logits = self.grad_fun(params, *args, targets)
        grad_loss = ravel_pytree(grad_loss_tree)[0]

        # sample labels (y_hat)
        # Note: sampling is done separately
        # logits = predict_fn(params, x)

        # n_samples = 10
        #
        # keys = jax.random.split(jax.random.PRNGKey(state.iter_num), num=n_samples)
        #
        # L = jnp.zeros((self.batch_size, grad_loss.shape[0]))
        #
        # for key in keys:
        #
        #     samples = jax.random.categorical(key, logits=logits)
        #     targets_sampled = jax.nn.one_hot(samples, num_classes=self.n_classes)
        #
        #     # !! notice that the sampled targets (y_hat) are used
        #     # batch_loss_tree = self.jac_fun(params, *args, targets)
        #     batch_loss_tree = self.jac_fun(params, *args, targets_sampled)
        #     Ls = flatten_2d_jacobian(batch_loss_tree)
        #
        #     L = L + Ls
        #
        # L = L / n_samples

        # no sampling, just largest logits
        # samples = jnp.argmax(logits, axis=-1)
        # targets_sampled = jax.nn.one_hot(samples, num_classes=self.n_classes)

        # sampling
        key = jax.random.PRNGKey(state.iter_num)
        samples = jax.random.categorical(key, logits=logits)
        targets_sampled = jax.nn.one_hot(samples, num_classes=self.n_classes)


        batch_loss_tree = self.jac_fun(params, *args, targets_sampled)
        L = flatten_2d_jacobian(batch_loss_tree)


        # original
        # H_lm = L.T @ L + jnp.eye(L.shape[1]) * self.regularizer * self.batch_size
        # H_lm_inv = jnp.linalg.inv(H_lm)

        # calculate with Woodbury formula
        # aa = jnp.linalg.inv(self.regularizer_array + L @ L.T)
        # bb = jnp.eye(L.shape[1]) - L.T @ aa @ L
        # H_lm_inv2 = bb / self.regularizer / self.batch_size
        #
        # direction = - H_lm_inv2 @ grad_loss * self.batch_size

        # h_gn = L.T @ L / self.batch_size
        #
        # new_grad_fn = jax.grad(self.ce)
        # new_grad_loss_tree = new_grad_fn(params, *args, targets_sampled)
        # new_grad_loss = ravel_pytree(new_grad_loss_tree)[0]
        #
        # h_gn2 = jnp.outer(new_grad_loss, new_grad_loss)

        temp = jax.scipy.linalg.solve(L @ L.T + self.regularizer_array, L @ grad_loss, assume_a='sym')
        direction = (L.T @ temp - grad_loss) / self.regularizer

        # max_norm1 = jnp.max(jnp.abs(grad_loss))
        # max_norm2 = jnp.max(jnp.abs(L))
        # max_norm3 = jnp.max(jnp.abs(temp))
        # max_norm4 = jnp.max(jnp.abs(direction))

        # diff = jnp.linalg.norm(direction - direction2)

        # calculate the direction
        # temp = jnp.linalg.solve(self.regularizer_array + L @ L.T, L @ grad_loss)
        # direction = self.batch_size * L.T @ temp

        return direction, grad_loss, L, None

    def mse(self, params, x, y):
        # b x 1
        residuals = y - self.predict_fun(params, x)

        # 1,
        # average over the batch
        return 0.5 * jnp.mean(jnp.square(residuals))

    def ce(self, params, x, y):
        # b x C
        logits = self.predict_fun(params, x)

        # b x C
        # jax.nn.log_softmax combines exp() and log() in a numerically stable way.
        log_probs = jax.nn.log_softmax(logits)

        # b x 1
        # if y is one-hot encoded, this operation picks the log probability of the correct class
        residuals = jnp.sum(y * log_probs, axis=-1)

        # 1,
        # average over the batch
        return -jnp.mean(residuals)

    def ce_with_aux(self, params, x, y):
        # b x C
        logits = self.predict_fun(params, x)

        # b x C
        # jax.nn.log_softmax combines exp() and log() in a numerically stable way.
        log_probs = jax.nn.log_softmax(logits)

        # b x 1
        # if y is one-hot encoded, this operation picks the log probability of the correct class
        residuals = jnp.sum(y * log_probs, axis=-1)

        # 1,
        # average over the batch
        return -jnp.mean(residuals), logits

    def predict_with_aux(self, params, *args):
        preds = self.predict_fun(params, *args)
        return preds, preds

    def predict_on_sampled_labels(self, params, key, *args):
        # raw prediction scores (logits) from the model: b x c
        logits = self.predict_fun(params, *args)

        # stop gradient flow through samples
        logits_stop = jax.lax.stop_gradient(logits)

        # sample labels from the logits (1 from c classes): b x 1
        samples = jax.random.categorical(key, logits=logits_stop)

        mask = jax.nn.one_hot(samples, self.n_classes)
        mask_stop = jax.lax.stop_gradient(mask)

        # based on the sampled labels, select the logits and probabilities
        selected_logits = jnp.sum(logits * mask_stop, axis=-1)
        selected_probs = jnp.sum(jax.nn.softmax(logits) * mask_stop, axis=-1)

        return selected_logits, (selected_probs, mask_stop)

    def __hash__(self):
        # We assume that the attribute values completely determine the solver.
        return hash(self.attribute_values())
