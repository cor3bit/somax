"""
Incremental Gauss-Newton Descent (IGND) Solver:
"""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional

import dataclasses
from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import chex

from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_scalar_mul, tree_zeros_like
from jaxopt.tree_util import tree_add, tree_sub
from jaxopt._src import base
from jaxopt._src import loop


class IGNDState(NamedTuple):
    """Named tuple containing state information."""
    iter_num: int
    xi: Optional[Any]
    velocity_m: Optional[Any]
    velocity_v: Optional[Any]


@dataclasses.dataclass(eq=False)
class IGND(base.StochasticSolver):
    # Jacobian of the residual function
    predict_fun: Callable
    jac_fun: Optional[Callable] = None

    # Experimental: only for classification problems
    loss_fun: Optional[Callable] = None
    loss_grad_fun: Optional[Callable] = None
    loss_hessian_fun: Optional[Callable] = None

    # Loss function parameters
    # loss_fun: Optional[Callable] = None
    loss_type: str = 'mse'  # ['mse', 'ce']

    learning_rate: float = 0.1

    batch_size: Optional[int] = None

    n_classes: Optional[int] = None

    pre_update: Optional[Callable] = None

    regularizer: float = 1.0

    # Momentum parameters
    momentum: float = 0.0  # 0.9
    beta2: float = 0.0  # 0.999

    verbose: int = 0

    jit: bool = True
    unroll: base.AutoOrBoolean = "auto"

    def __post_init__(self):
        super().__post_init__()

        self.reference_signature = self.predict_fun

        # Regression (MSE)
        if self.loss_type == 'mse':
            self.jac_fun = jax.vmap(jax.value_and_grad(self.predict_fun), in_axes=(None, 0))
            self.calculate_direction = self.calculate_direction_mse
        # Classification (Cross-Entropy)
        elif self.loss_type == 'ce':
            self.jac_fun = jax.vmap(jax.value_and_grad(self.predict_with_targets), in_axes=(None, 0, 0))
            self.calculate_direction = self.calculate_direction_ce
            assert self.loss_grad_fun is not None
            assert self.loss_hessian_fun is not None
        else:
            raise ValueError(f"'loss_type' {self.loss_type} not supported")

        # set up momentum
        if self.momentum < 0. or self.momentum > 1.:
            raise ValueError(f"'momentum' must belong to closed interval [0,1]")
        if self.beta2 < 0. or self.beta2 > 1.:
            raise ValueError(f"'beta2' must belong to closed interval [0,1]")

    def update(
            self,
            params: Any,
            state: IGNDState,
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
        # ---------- STEP 1: calculate direction with QR ---------- #
        targets = kwargs['targets']
        direction, batch_avg_xi = self.calculate_direction(params, state, targets, *args)

        # ---------- STEP 2: momentum acceleration ---------- #
        if self.momentum > 0:
            # direction with bias-corrected momentum
            # d = (m * v + (1 - m) * d) / (1 - m^t)
            direction_m = self.momentum * state.velocity_m + (1 - self.momentum) * direction
            bias_corr_m = 1 - self.momentum ** (state.iter_num + 1)

            if self.beta2 > 0:
                v_eps = 1e-7
                direction_v = self.beta2 * state.velocity_v + (1 - self.beta2) * direction * direction
                bias_corr_v = 1 - self.beta2 ** (state.iter_num + 1)
                bias_corrected_direction_v = jnp.sqrt(direction_v / bias_corr_v) + v_eps
                direction = direction_m / bias_corr_m / bias_corrected_direction_v
            else:
                direction = direction_m / bias_corr_m
                direction_v = None
        else:
            direction_m = None
            direction_v = None

        # ---------- STEP 3: update parameters ---------- #
        params_flat, unflatten_fn = ravel_pytree(params)
        next_params_flat = params_flat + self.learning_rate * direction
        next_params = unflatten_fn(next_params_flat)

        next_state = IGNDState(
            iter_num=state.iter_num + 1,
            xi=batch_avg_xi,
            velocity_m=direction_m,  # First moment accumulator
            velocity_v=direction_v,  # First moment accumulator
        )

        return base.OptStep(params=next_params, state=next_state)

    def init_state(self,
                   init_params: Any,
                   *args,
                   **kwargs) -> IGNDState:

        velocity_m = jnp.zeros_like(ravel_pytree(init_params)[0]) if self.momentum > 0 else None
        velocity_v = jnp.zeros_like(ravel_pytree(init_params)[0]) if self.beta2 > 0 else None

        return IGNDState(
            iter_num=jnp.asarray(0),
            xi=None,
            velocity_m=velocity_m,
            velocity_v=velocity_v,
        )

    def optimality_fun(self, params, *args, **kwargs):
        """Optimality function mapping compatible with ``@custom_root``."""
        # return self._grad_fun(params, *args, **kwargs)[0]
        raise NotImplementedError

    def flatten_jacobian(self, jac_tree):
        return jax.vmap(lambda _: ravel_pytree(_)[0], in_axes=(0,))(jac_tree)

    def calculate_direction_mse(self, params, state, targets, *args):
        # compute the Jacobian (fwd+bwd mode), heaviest part
        batch_preds, jac_tree = self.jac_fun(params, *args)

        # convert pytree to JAX array (here, Jacobian of the DNN, J_f)
        J = self.flatten_jacobian(jac_tree)

        # batch-level xi
        inv_xi = jnp.einsum('ij,ij->i', J, J)

        # residuals
        r = targets - jnp.squeeze(batch_preds)

        # equivalent to -grad scaled by xi
        direction = J.T @ (r / inv_xi) / self.batch_size

        return direction, jnp.mean(1 / inv_xi)

    def calculate_direction_ce(self, params, state, targets, *args):
        # compute the Jacobian (fwd+bwd mode), heaviest part
        batch_probs, jac_tree = self.jac_fun(params, targets, *args)

        # convert pytree to JAX array (here, Jacobian of the DNN, J_f)
        J = self.flatten_jacobian(jac_tree)

        # equivalent to -grad scaled by xi
        dldp = self.loss_grad_fun(batch_probs)

        # VERIFICATION
        # dldw = J.T @ dldp / self.batch_size
        # grads_tree = jax.grad(self.loss_fun)(params, *args, targets=targets)
        # grads = ravel_pytree(grads_tree)[0]
        # chex.assert_trees_all_close(dldw, grads, atol=1e-6, )

        # SGD direction
        # direction = - dldw

        # calculate xi for each sample: (b, )
        # for each sample, calculate the g^T g
        jtj = jnp.einsum('ij,ij->i', J, J)
        q = self.loss_hessian_fun(batch_probs)
        inv_xi = jtj * q

        # equivalent to -grad scaled by xi
        direction = - J.T @ (dldp / inv_xi) / self.batch_size

        return direction, jnp.mean(1 / inv_xi)

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
        residuals = jnp.sum(y * log_probs, axis=1)

        # 1,
        # average over the batch
        return -jnp.mean(residuals)

    def predict_with_aux(self, params, *args):
        preds = self.predict_fun(params, *args)
        return preds, preds

    def predict_with_targets(self, params, targets, *args):
        # preds should be probabilities from the model
        # targets should be one-hot encoded labels
        preds = self.predict_fun(params, *args)
        pred = jnp.vdot(preds, targets)
        return pred

    def __hash__(self):
        # We assume that the attribute values completely determine the solver.
        return hash(self.attribute_values())
