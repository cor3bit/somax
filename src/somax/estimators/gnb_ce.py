from typing import Tuple

import jax, jax.numpy as jnp

from .base import EstimatorPolicy
from ..types import MatVec, Array, PyTree, Params, PRNGKey, Scalar
from ..curvature.base import CurvatureState


class GaussNewtonBartlett(EstimatorPolicy):
    """Diagonal of GGN-CE via Bartlett-corrected empirical Fisher: B * E[(grad_theta l_i)^2]."""

    def __init__(self, n_samples: int = 1):
        self.n_samples = int(n_samples)

    def diagonal(
            self,
            params: Params,
            state: CurvatureState,
            mvp: MatVec,
            rng: PRNGKey,
    ) -> PyTree:
        c = state.cache
        logits = c.logits  # (B, C)
        probs = c.probs  # (B, C)
        JT = c.vjp
        alpha = c.alpha
        B, C = logits.shape
        B_scalar = jnp.asarray(B, logits.dtype)

        S = self.n_samples
        keys = jax.random.split(rng, S)

        # init accumulator (sum of g_samp^2 over samples)
        acc0 = jax.tree_util.tree_map(jnp.zeros_like, params)

        def body(acc, key):
            # Sample labels for this Monte Carlo draw
            y = jax.random.categorical(key, logits=logits, axis=-1)  # (B,)

            # Residual for sampled labels
            r = probs - jax.nn.one_hot(y, C, dtype=logits.dtype)  # (B, C)

            # VJP: mean gradient sample
            (g_samp,) = JT(alpha * r)

            # Accumulate squared gradients
            contrib = jax.tree_util.tree_map(lambda gi: gi * gi, g_samp)
            acc = jax.tree_util.tree_map(lambda a, c: a + c, acc, contrib)
            return acc, None

        accN, _ = jax.lax.scan(body, acc0, keys)

        # Monte Carlo average then Bartlett factor
        mean_G2 = jax.tree_util.tree_map(lambda a: a / S, accN)
        return jax.tree_util.tree_map(lambda a: B_scalar * a, mean_G2)

    def trace(self, params: Params, state: CurvatureState, mvp: MatVec, rng: PRNGKey) -> Scalar:
        raise NotImplementedError

    def spectrum(self, params: Params, state: CurvatureState, mvp: MatVec, rng: PRNGKey, k: int = 16) -> Tuple[
        Array, Array]:
        raise NotImplementedError

    def low_rank(self, params: Params, state: CurvatureState, mvp: MatVec, rng: PRNGKey, k: int = 16
                 ) -> Tuple[Array, Array]:
        raise NotImplementedError
