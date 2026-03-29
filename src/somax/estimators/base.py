from typing import Protocol, Tuple

from ..curvature.base import CurvatureState
from ..types import MatVec, Array, PyTree, Params, PRNGKey, Scalar


class EstimatorPolicy(Protocol):
    """
    Estimator interface over a matrix-free operator.

    Contract
    --------
    - Stateless-by-contract; may lazily cache param-shape info on first call.
    - All methods must be JIT-safe; RNG comes from the caller.
    - `state` is passed through for estimators that can reuse curvature caches.
    """

    def diagonal(self, params: Params, state: CurvatureState, mvp: MatVec, rng: PRNGKey) -> PyTree:
        """Return a PyTree matching params: estimate of diag(C)."""
        ...

    def trace(self, params: Params, state: CurvatureState, mvp: MatVec, rng: PRNGKey) -> Scalar:
        """Return a scalar estimate of tr(C)."""
        ...

    def spectrum(
            self,
            params: Params,
            state: CurvatureState,
            mvp: MatVec,
            rng: PRNGKey,
            k: int = 16,
    ) -> Tuple[Scalar, Scalar]:
        """Return (eig_min, eig_max) estimates.

        Notes
        -----
        - For SPD operators, these are lower/upper spectral bounds.
        - For non-SPD operators, estimators may return approximate extrema of
          a symmetrized operator; callers must interpret accordingly.
        """
        ...

    def low_rank(
            self,
            params: Params,
            state: CurvatureState,
            mvp: MatVec,
            rng: PRNGKey,
            k: int = 16,
    ) -> Tuple[Array, Array]:
        """Return (evals, vecs_flat) low-rank approximation of curvature."""
        ...
