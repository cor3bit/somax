"""
Sophia-H Optimizer
Paper: https://arxiv.org/abs/2305.14342
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
from jaxopt._src import base


class SophiaHState(NamedTuple):
    """Named tuple containing state information."""
    iter_num: int
    stepsize: float
    velocity_m: Any
    velocity_v: Any
    hess_approx_rng: Any


@dataclasses.dataclass(eq=False)
class SophiaH(base.StochasticSolver):
    loss_fun: Callable

    learning_rate: float = 0.85e-3  # proposed as default by levanter

    # Lazy Hessian parameters
    eval_hess_every_k: int = 10

    # Momentum parameters
    beta1: float = 0.965  # proposed as default by levanter
    beta2: float = 0.99  # proposed as default by levanter

    # Regularization parameters
    weight_decay: float = 0.0  # L2 regularization coefficient

    gamma: float = 0.01  # clipping parameter
    clip_th: float = 1.  # clipping threshold
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

        assert 0 <= self.weight_decay < 1, "Weight decay must be in [0, 1)"

    def update(
            self,
            params: Any,
            state: SophiaHState,
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
        # params_flat, pack_fn = ravel_pytree(params)
        # p_shape = params_flat.shape

        # ------- Step 1 -------
        # compute the exact gradient and the Diagonal Hessian estimate
        # incorporate the lazy Hessian evaluation, i.e. re-compute H every k-th iteration
        # apply temporal averaging (momentum) to the first and second moments
        # !! original Sophia paper has no bias correction: 1/(1 - self.beta1 ** i) !!
        next_rng_key, rng_key = jax.random.split(state.hess_approx_rng)
        inputs = (params, state, rng_key, args, kwargs)
        m_t, v_t = lax.cond(
            state.iter_num % self.eval_hess_every_k == 0,
            inputs, self.grad_and_hess_hutchinson,
            inputs, self.grad_and_hess_reuse,
        )

        # TODO apply bias correction for m_t?

        # ------- Step 2 -------
        # clip direction
        direction = jnp.clip(m_t / jnp.maximum(self.gamma * v_t, self.eps), a_min=-self.clip_th, a_max=self.clip_th)

        # update parameters
        params_flat, pack_fn = ravel_pytree(params)
        next_params_flat = params_flat - self.learning_rate * direction

        # ------- Step 4 -------
        # AdamW-style weight decay
        if self.weight_decay > 0:
            next_params_flat -= self.learning_rate * self.weight_decay * params_flat

        # bookkeeping
        next_params = pack_fn(next_params_flat)

        next_state = SophiaHState(
            iter_num=state.iter_num + 1,
            stepsize=state.stepsize,
            velocity_m=m_t,
            velocity_v=v_t,
            hess_approx_rng=next_rng_key,
        )

        return base.OptStep(params=next_params, state=next_state)

    def init_state(self,
                   init_params: Any,
                   *args,
                   **kwargs) -> SophiaHState:
        params_flat, pack_fn = ravel_pytree(init_params)
        velocity_m = jnp.zeros_like(params_flat)
        velocity_v = jnp.zeros_like(params_flat)

        return SophiaHState(
            iter_num=0,
            stepsize=self.learning_rate,
            velocity_m=velocity_m,
            velocity_v=velocity_v,
            hess_approx_rng=jax.random.PRNGKey(self.hess_approx_seed),
        )

    def optimality_fun(self, params, *args, **kwargs):
        """Optimality function mapping compatible with ``@custom_root``."""
        # return self._grad_fun(params, *args, **kwargs)[0]
        raise NotImplementedError

    def __hash__(self):
        # We assume that the attribute values completely determine the solver.
        return hash(self.attribute_values())

    def grad_and_hess_hutchinson(self, inputs):
        # prep grad function to accept only "params"
        def inner_grad_fun(params):
            return self.grad_fun(params, *args, **kwargs)

        params, state, rng_key, args, kwargs = inputs

        # calculate z * Hz
        params_flat, pack_fn = ravel_pytree(params)
        p_shape = params_flat.shape
        z = jax.random.rademacher(rng_key, shape=p_shape, dtype=jnp.float32)
        z_tree = pack_fn(z)
        grads_tree, hz_tree = jax.jvp(inner_grad_fun, (params,), (z_tree,))
        diag_hess_tree = jax.tree_map(lambda x, y: x * y, grads_tree, hz_tree)

        # apply EMA to the first and second moments
        # m_t
        grads = ravel_pytree(grads_tree)[0]
        m_t = self.beta1 * state.velocity_m + (1 - self.beta1) * grads

        # v_t
        v_t = ravel_pytree(diag_hess_tree)[0]
        v_t = self.beta2 * state.velocity_v + (1 - self.beta2) * v_t

        return m_t, v_t

    def grad_and_hess_reuse(self, inputs):
        params, state, rng_key, args, kwargs = inputs

        grads_tree = self.grad_fun(params, *args, **kwargs)

        # m_t
        grads = ravel_pytree(grads_tree)[0]
        m_t = self.beta1 * state.velocity_m + (1 - self.beta1) * grads

        # v_t: reuse the previous estimate
        v_t = state.velocity_v

        return m_t, v_t
