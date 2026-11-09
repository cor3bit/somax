import jax
import jax.numpy as jnp

from .base import EstimatorPolicy
from ..types import Params, PRNGKey, MatVec, Scalar
from ..curvature.base import CurvatureState


class MFLanczos(EstimatorPolicy):
    """Lanczos-based spectral extrema estimator (min/max eigenvalue estimates).

    Returns
    -------
    (eig_min, eig_max): Scalar, Scalar
        Estimate extremal eigenvalues of the operator behind `mvp`.

    Notes
    -----
    - Intended for symmetric operators (SPD in the common case).
    - Uses a PyTree-valued starting vector sampled like params.
    """

    def __init__(
            self,
            k: int = 16,
            reortho: str = "none",  # "none", "full"
            custom_vjp: bool = False,
    ):
        self.k = int(k)
        self.reortho = str(reortho)
        self.custom_vjp = bool(custom_vjp)
        self._eigh = None

    def _ensure_eigh(self):
        if self._eigh is None:
            try:
                from matfree import decomp as mf_decomp
                from matfree import eig as mf_eig
            except ImportError as exc:
                raise ImportError(
                    "MFLanczos requires the optional 'matfree' package. "
                    "Install it with: pip install python-somax[matfree]"
                ) from exc

            tri = mf_decomp.tridiag_sym(
                self.k,
                materialize=True,
                reortho=self.reortho,
                custom_vjp=self.custom_vjp,
            )
            self._eigh = mf_eig.eigh_partial(tri)

    def diagonal(self, params: Params, state: CurvatureState, mvp: MatVec, rng: PRNGKey):
        raise NotImplementedError("MFLanczos does not implement diagonal; use Hutchinson/DiagPP.")

    def trace(self, params: Params, state: CurvatureState, mvp: MatVec, rng: PRNGKey):
        raise NotImplementedError("MFLanczos does not implement trace; use Hutchinson/DiagPP.")

    def spectrum(
            self,
            params: Params,
            state: CurvatureState,
            mvp: MatVec,
            rng: PRNGKey,
            k: int = 16,
    ) -> tuple[Scalar, Scalar]:
        del state, k
        self._ensure_eigh()

        # build PyTree-valued v0 with same structure as params
        leaves = jax.tree_util.tree_leaves(params)
        keys = jax.random.split(rng, len(leaves))

        def sample_like(p, key):
            return jax.random.normal(key, p.shape, dtype=p.dtype)

        v0 = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(params),
            [sample_like(p, key) for p, key in zip(leaves, keys)],
        )

        evals, _ = self._eigh(lambda v: mvp(v), v0)  # sorted ascending for SPD
        evals = jnp.asarray(evals)
        return evals[0], evals[-1]

    def low_rank(
            self,
            params: Params,
            state: CurvatureState,
            mvp: MatVec,
            rng: PRNGKey,
            k: int = 16,
    ):
        raise NotImplementedError("MFLanczos does not implement low_rank.")
