"""
AdaHessian Optimizer
Paper: https://arxiv.org/abs/2006.00719
"""

from typing import Any
from typing import Callable
from typing import NamedTuple, Tuple
from typing import Optional
import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxopt._src import base


class AdaHessianState(NamedTuple):
    """Named tuple containing state information."""
    iter_num: int
    stepsize: float
    velocity_m: Any
    velocity_v_tree: Any
    hess_approx_rng: Any


@dataclasses.dataclass(eq=False)
class AdaHessian(base.StochasticSolver):
    loss_fun: Callable

    learning_rate: float = 1e-3

    spatial_averaging: bool = True

    # Momentum parameters
    beta1: float = 0.9  # the exponential decay rate for the first moment estimates
    beta2: float = 0.999  # the exponential decay rate for the squared hessian estimates

    # Regularization parameters
    weight_decay: float = 0.0  # L2 regularization coefficient

    # Hessian parameters
    hessian_power: float = 1.0  # the power of the hessian approximation, i.e. H^{-hessian_power}
    eps: float = 1e-8  # term added to the denominator to improve numerical stability

    hess_approx_seed: int = 1337

    pre_update: Optional[Callable] = None

    verbose: int = 0

    jit: bool = True
    unroll: base.AutoOrBoolean = "auto"

    def __post_init__(self):
        super().__post_init__()

        self.reference_signature = self.loss_fun

        self.grad_fun = jax.grad(self.loss_fun)

        assert 0 <= self.hessian_power <= 1, "Hessian power must be in [0, 1]"
        assert 0 <= self.weight_decay < 1, "Weight decay must be in [0, 1)"

    def update(
            self,
            params: Any,
            state: AdaHessianState,
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
        # ------- Step 1 -------
        # compute the gradient and the Diagonal Hessian using Hutchinson's method
        next_rng_key, rng_key = jax.random.split(state.hess_approx_rng)
        params_flat, pack_fn = ravel_pytree(params)
        p_shape = params_flat.shape
        grads_tree, diag_hess_tree = self.grad_and_hess(params, rng_key, p_shape, pack_fn, *args, **kwargs)

        # ------- Step 2 -------
        if self.spatial_averaging:
            diag_hess_tree = self.average_layers(diag_hess_tree)

        # ------- Step 3 -------
        # apply temporal averaging (momentum) to the first and second moments
        i = state.iter_num + 1

        # m_t
        grads = ravel_pytree(grads_tree)[0]
        velocity_m = self.beta1 * state.velocity_m + (1 - self.beta1) * grads
        bias_corr_m = velocity_m / (1 - self.beta1 ** i)

        # v_t
        velocity_v_tree = jax.tree_map(lambda v, h: self.beta2 * v + (1 - self.beta2) * h,
                                       state.velocity_v_tree, diag_hess_tree)

        velocity_v = ravel_pytree(velocity_v_tree)[0]

        bias_corr_v = jnp.pow(jnp.sqrt(velocity_v / (1 - self.beta2 ** i)), self.hessian_power)

        # m_t / (v_t + eps)
        direction = -bias_corr_m / (bias_corr_v + self.eps)

        # update parameters
        next_params_flat = params_flat + self.learning_rate * direction

        # ------- Step 4 -------
        # AdamW-style weight decay
        if self.weight_decay > 0:
            next_params_flat -= self.learning_rate * self.weight_decay * params_flat

        # bookkeeping
        next_params = pack_fn(next_params_flat)

        next_state = AdaHessianState(
            iter_num=state.iter_num + 1,
            stepsize=state.stepsize,
            velocity_m=velocity_m,
            velocity_v_tree=velocity_v_tree,
            hess_approx_rng=next_rng_key,
        )

        return base.OptStep(params=next_params, state=next_state)

    def init_state(self,
                   init_params: Any,
                   *args,
                   **kwargs) -> AdaHessianState:
        params_flat, pack_fn = ravel_pytree(init_params)
        velocity_m = jnp.zeros_like(params_flat)
        velocity_v_tree = pack_fn(jnp.zeros_like(params_flat))

        return AdaHessianState(
            iter_num=0,
            stepsize=self.learning_rate,
            velocity_m=velocity_m,
            velocity_v_tree=velocity_v_tree,
            hess_approx_rng=jax.random.PRNGKey(self.hess_approx_seed),
        )

    def optimality_fun(self, params, *args, **kwargs):
        """Optimality function mapping compatible with ``@custom_root``."""
        # return self._grad_fun(params, *args, **kwargs)[0]
        raise NotImplementedError

    def __hash__(self):
        # We assume that the attribute values completely determine the solver.
        return hash(self.attribute_values())

    def grad_and_hess(self, params, rng_key, p_shape, pack_fn, *args, **kwargs):
        # prep grad function to accept only "params"
        def inner_grad_fun(params):
            return self.grad_fun(params, *args, **kwargs)

        # draw from Radamacher distribution
        z = jax.random.rademacher(rng_key, shape=p_shape, dtype=jnp.float32)
        z_tree = pack_fn(z)

        grads_tree, hz_tree = jax.jvp(inner_grad_fun, (params,), (z_tree,))

        diag_hess_tree = jax.tree_map(lambda x, y: x * y, grads_tree, hz_tree)

        return grads_tree, diag_hess_tree

    @staticmethod
    def _average_layer(leaf_node):
        #  !! no abs() in the original implementation
        # see also https://github.com/nestordemeure/AdaHessianJax/blob/main/adahessianJax/hessian_computation.py
        abs_leaf_node = jnp.abs(leaf_node)

        # !! no block size, just average over the layer
        if leaf_node.ndim == 1:
            return abs_leaf_node
        elif leaf_node.ndim in [2, 3]:  # Convolution layers
            return jnp.mean(abs_leaf_node, axis=-1, keepdims=True)
        elif leaf_node.ndim == 4:  # Attention layers
            return jnp.mean(abs_leaf_node, axis=[-2, -1], keepdims=True)
        else:
            raise ValueError("Invalid leaf node dimension")

    def average_layers(self, diag_hess_tree):
        return jax.tree_map(self._average_layer, diag_hess_tree)
