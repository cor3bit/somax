from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .base import EstimatorPolicy
from ..types import MatVec, Array, PyTree, Params, PRNGKey, Scalar
from ..curvature.base import CurvatureState
from ..utils import tree_vdot


class Hutchinson(EstimatorPolicy):
    """
    Hutchinson estimator for diagonal and trace of a symmetric linear operator C.

        diag(C) ~= E[(C z) .* z], z ~ Rademacher
        tr(C)   ~= E[z^T C z]

    `n_probes` controls the number of Monte Carlo samples.
    """

    def __init__(
            self,
            n_probes: int = 1,
            use_abs: bool = False,
            dtype: Optional[jnp.dtype] = None,
    ):
        self.n_probes = int(n_probes)
        self.use_abs = bool(use_abs)
        self._dtype = dtype  # if None, inferred from params at call time

    # ---- internals ----

    def _make_probe_sampler(self, params: Params):
        """
        Build a closure `make_probe(key) -> PyTree` that samples z ~ Rademacher
        with the same structure as `params`.
        """
        flat, pack = ravel_pytree(params)
        d = int(flat.size)
        dtype = self._dtype or flat.dtype

        def make_probe(key: PRNGKey) -> PyTree:
            z_flat = jax.random.rademacher(key, (d,), dtype=dtype)
            return pack(z_flat)

        return make_probe

    # ---- API ----

    def diagonal(
            self,
            params: Params,
            state: CurvatureState,
            mvp: MatVec,
            rng: PRNGKey,
    ) -> PyTree:
        """
        Estimate diag(C) with Hutchinson: E[(C z)  z], z ~ Rademacher.
        """
        make_probe = self._make_probe_sampler(params)
        keys = jax.random.split(rng, self.n_probes)

        def one(key: PRNGKey) -> PyTree:
            z = make_probe(key)
            Cz = mvp(z)
            prod = jax.tree_util.tree_map(lambda a, b: a * b, Cz, z)
            if self.use_abs:
                prod = jax.tree_util.tree_map(jnp.abs, prod)
            return prod

        acc0 = jax.tree_util.tree_map(jnp.zeros_like, params)

        def scan_body(acc, key):
            contrib = one(key)
            acc = jax.tree_util.tree_map(lambda a, e: a + e, acc, contrib)
            return acc, None

        accN, _ = jax.lax.scan(scan_body, acc0, keys)
        return jax.tree_util.tree_map(lambda a: a / self.n_probes, accN)

    def trace(
            self,
            params: Params,
            state: CurvatureState,
            mvp: MatVec,
            rng: PRNGKey,
    ) -> Scalar:
        """
        Estimate tr(C) with Hutchinson: E[z^T C z], z ~ Rademacher.
        """
        make_probe = self._make_probe_sampler(params)
        keys = jax.random.split(rng, self.n_probes)

        def one(key: PRNGKey) -> Scalar:
            z = make_probe(key)
            return tree_vdot(z, mvp(z))

        vals = jax.vmap(one)(keys)
        return jnp.mean(vals)

    def spectrum(
            self,
            params: Params,
            state: CurvatureState,
            mvp: MatVec,
            rng: PRNGKey,
            k: int = 16,
    ) -> Tuple[Array, Array]:
        raise NotImplementedError

    def low_rank(
            self,
            params: Params,
            state: CurvatureState,
            mvp: MatVec,
            rng: PRNGKey,
            k: int = 16,
    ) -> Tuple[Array, Array]:
        raise NotImplementedError
