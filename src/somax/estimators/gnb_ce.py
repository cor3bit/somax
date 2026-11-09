from typing import Tuple

import jax, jax.numpy as jnp

from .base import EstimatorPolicy
from ..types import MatVec, Array, PyTree, Params, PRNGKey, Scalar
from ..curvature.base import CurvatureState


class GaussNewtonBartlett(EstimatorPolicy):
    """Estimate the GGN-CE diagonal as B * E[g_sample**2].

    g_sample is the batch-mean gradient with labels sampled from the model.
    """

    def __init__(self, n_samples: int = 1):
        if int(n_samples) < 1:
            raise ValueError("n_samples must be positive.")
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

        acc0 = jax.tree_util.tree_map(jnp.zeros_like, params)

        def body(acc, key):
            y = jax.random.categorical(key, logits=logits, axis=-1)  # (B,)

            r = probs - jax.nn.one_hot(y, C, dtype=logits.dtype)  # (B, C)

            # Mean gradient for one sampled batch.
            (g_samp,) = JT(alpha * r)

            contrib = jax.tree_util.tree_map(lambda gi: gi * gi, g_samp)
            acc = jax.tree_util.tree_map(lambda a, c: a + c, acc, contrib)
            return acc, None

        accN, _ = jax.lax.scan(body, acc0, keys)

        # Bartlett scaling for the batch-mean gradient.
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
