from typing import Tuple, Callable
import importlib
import itertools

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .base import EstimatorPolicy
from .hutchinson import Hutchinson
from ..types import Params, PRNGKey, PyTree, Scalar, MatVec, Array
from ..curvature.base import CurvatureState


def _import_decomp():
    """Robustly import matfree.decomp across versions or lazy import styles."""
    try:
        from matfree import decomp  # fast path if re-exported
        return decomp
    except Exception:
        return importlib.import_module("matfree.decomp")  # fallback


def _make_flatten_fns(
        params: Params,
) -> tuple[Callable[[PyTree], Array], Callable[[Array], PyTree], int, jnp.dtype]:
    """
    Build tree-aware pack/unpack for params.

    pack_tree: PyTree -> flat (shape (n,))
    unpack_flat: flat (shape (n,)) -> PyTree with same structure/shapes as params
    n: total number of scalars
    dtype: dtype inferred from params
    """
    flat0, _ = ravel_pytree(params)
    dtype = flat0.dtype

    leaves0, treedef = jax.tree_util.tree_flatten(params)
    sizes = [leaf.size for leaf in leaves0]
    offsets = [0] + list(itertools.accumulate(sizes))
    n = int(offsets[-1])

    def pack_tree(tree: PyTree) -> Array:
        leaves, _ = jax.tree_util.tree_flatten(tree)
        flats = [jnp.reshape(leaf, (-1,)) for leaf in leaves]
        return jnp.concatenate(flats, axis=0)

    def unpack_flat(xf: Array) -> PyTree:
        leaves = []
        for i, leaf0 in enumerate(leaves0):
            start, end = offsets[i], offsets[i + 1]
            seg = xf[start:end]
            leaves.append(seg.reshape(leaf0.shape).astype(leaf0.dtype))
        return jax.tree_util.tree_unflatten(treedef, leaves)

    return pack_tree, unpack_flat, n, dtype


def _mv_flat(
        mvp: MatVec,
        pack_tree: Callable[[PyTree], Array],
        unpack_flat: Callable[[Array], PyTree],
) -> Callable[[Array], Array]:
    """Lift a PyTree matvec mvp(vtree) to a flat matvec mv(x_flat)."""

    def mv(x_flat: Array) -> Array:
        v = unpack_flat(x_flat)
        Av = mvp(v)
        return pack_tree(Av)

    return mv


class DiagPP(EstimatorPolicy):
    """
    Diagonal estimator: low-rank Ritz part from Lanczos (k) plus Hutchinson on residual.

    For a symmetric positive semidefinite operator A:
      1. Approximate A with Q @ T @ Q.T using k Lanczos vectors.
      2. Compute diag_lr = diag(Q @ T @ Q.T).
      3. Estimate diag(R) with Hutchinson, where R = A - Q @ T @ Q.T.

    Final estimate:
        diag(A) ~= diag_lr + Hutchinson(R).

    Notes
    -----
    - This estimator is intentionally heavy:
        * O(k) matvecs with A for tridiagonalisation,
        * O(n k^2) dense work to form Q, T, and their eigendecomposition,
        * plus n_probes_res extra residual matvecs.
      It is meant for diagnostic or occasional use, not every training step.
    """

    def __init__(
            self,
            k: int = 8,
            n_probes_res: int = 8,
            clip_min: float | None = 0.0,
            reortho: str = "none",  # "none" or "full"
    ):
        self.k = int(k)
        self.n_probes_res = int(n_probes_res)
        if self.n_probes_res < 1:
            raise ValueError("n_probes_res must be positive.")
        self.clip_min = clip_min
        self.reortho = str(reortho)
        # Residual estimator (PyTree Hutchinson)
        self._hutch_res = Hutchinson(n_probes=self.n_probes_res)

    def _lowrank_diag(
            self,
            params: Params,
            mvp: MatVec,
            rng: PRNGKey,
    ) -> tuple[
        Array,
        Callable[[Array], Array],
        Callable[[Array], PyTree],
        Callable[[PyTree], Array],
        jnp.dtype,
    ]:
        """
        Compute low-rank diagonal from Lanczos and residual matvec in flat space.

        Returns
        -------
        diag_lr_flat : Array, shape (n,)
            Diagonal of the low-rank approximation Q T Q^T in flat form.
        r_mv_flat : Callable[[Array], Array]
            Flat residual matvec R x = (A - Q T Q^T) x.
        unpack_flat : Callable[[Array], PyTree]
            Flat vector -> PyTree with params structure.
        pack_tree : Callable[[PyTree], Array]
            PyTree -> flat vector compatible with diag_lr_flat.
        dtype : jnp.dtype
            Scalar dtype used internally.
        """
        decomp = _import_decomp()
        pack_tree, unpack_flat, n, dtype = _make_flatten_fns(params)

        Av_flat = _mv_flat(mvp, pack_tree, unpack_flat)
        v0 = jax.random.normal(rng, (n,), dtype=dtype)

        # Lanczos approximation: A ~= Q @ T @ Q.T (Q is n x k).
        tridiag = decomp.tridiag_sym(
            self.k,
            materialize=True,
            reortho=self.reortho,
        )
        Q, T, *_ = tridiag(Av_flat, v0)  # T symmetric (k x k)

        # Ensure Q has shape (n, k)
        if Q.shape[0] != n and Q.shape[1] == n:
            Q = Q.T

        # Eigendecompose T (k x k)
        evals, U = jnp.linalg.eigh(T.astype(dtype))  # evals: (k,), U: (k, k)

        # Ritz vectors in R^n: V = Q U -> (n, k)
        V = Q @ U

        # diag(Q @ T @ Q.T) = sum_j evals[j] * V[:, j]**2.
        diag_lr_flat = (V * V) @ evals  # (n,)
        diag_lr_flat = diag_lr_flat.astype(dtype)

        # Low-rank matvec in flat coordinates: L x = Q T Q^T x
        def lr_mv_flat(xf: Array) -> Array:
            return Q @ (T @ (Q.T @ xf))

        # Residual matvec in flat coordinates: R x = A x - L x
        def r_mv_flat(xf: Array) -> Array:
            return Av_flat(xf) - lr_mv_flat(xf)

        return diag_lr_flat, r_mv_flat, unpack_flat, pack_tree, dtype

    def diagonal(
            self,
            params: Params,
            state: CurvatureState,
            mvp: MatVec,
            rng: PRNGKey,
    ) -> PyTree:
        """
        Estimate diag(A) via low-rank Lanczos + Hutchinson residual.
        """
        # Independent randomness for Lanczos and residual probes.
        k_lz, k_res = jax.random.split(rng, 2)
        diag_lr_flat, r_mv_flat, unpack_flat, pack_tree, dtype = self._lowrank_diag(
            params,
            mvp,
            k_lz,
        )

        def r_mvp(vtree: PyTree) -> PyTree:
            vf = pack_tree(vtree)
            rf = r_mv_flat(vf)
            return unpack_flat(rf)

        diag_res = self._hutch_res.diagonal(params, state, r_mvp, k_res)
        diag_res_flat, _ = ravel_pytree(diag_res)

        diag_flat = diag_lr_flat + diag_res_flat.astype(dtype)

        if self.clip_min is not None:
            diag_flat = jnp.maximum(diag_flat, dtype.type(self.clip_min))

        diag_tree = unpack_flat(diag_flat)
        return jax.tree_util.tree_map(lambda a, p: a.astype(p.dtype), diag_tree, params)

    def trace(
            self,
            params: Params,
            state: CurvatureState,
            mvp: MatVec,
            rng: PRNGKey,
    ) -> Scalar:
        """
        Estimate trace(A) consistent with the diagonal estimator.

        trace(A) ~= sum(diag(Q @ T @ Q.T)) + trace(residual),
        where trace(residual) is approximated with Hutchinson.
        """
        k_lz, k_res = jax.random.split(rng, 2)
        diag_lr_flat, r_mv_flat, unpack_flat, pack_tree, dtype = self._lowrank_diag(
            params,
            mvp,
            k_lz,
        )
        tr_lr = jnp.sum(diag_lr_flat).astype(dtype)

        def r_mvp(vtree: PyTree) -> PyTree:
            vf = pack_tree(vtree)
            rf = r_mv_flat(vf)
            return unpack_flat(rf)

        tr_res = self._hutch_res.trace(params, state, r_mvp, k_res)
        return (tr_lr + tr_res).astype(dtype)

    def spectrum(self, *a, **k):
        raise NotImplementedError("DiagPP does not provide spectrum; use MFLanczos instead.")

    def low_rank(
            self,
            params: Params,
            state: CurvatureState,
            mvp: MatVec,
            rng: PRNGKey,
            k: int = 16,
    ) -> Tuple[Array, Array]:
        raise NotImplementedError("DiagPP does not expose low_rank.")
