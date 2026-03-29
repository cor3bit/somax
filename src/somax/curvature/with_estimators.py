from typing import Any, Tuple

from .base import CurvatureOperator, CurvatureState, RowOperator
from ..types import Params, Batch, PRNGKey, PyTree, MatVec, Scalar, Array
from ..estimators.base import EstimatorPolicy


class CurvatureOpWithEstimators:

    def __init__(self, base: CurvatureOperator, estimator: EstimatorPolicy):
        self._base = base
        self._est = estimator

    # ---- delegate CurvatureOperator protocol ----
    def init(self, params: Params, batch: Batch, *, with_grad: bool = False) -> tuple[CurvatureState, Any]:
        return self._base.init(params, batch, with_grad=with_grad)

    def matvec(self, params: Params, state: CurvatureState, v: PyTree) -> PyTree:
        return self._base.matvec(params, state, v)

    def loss(self, params: Params, state: CurvatureState, batch: Batch) -> Scalar:
        return self._base.loss(params, state, batch)

    def loss_only(self, params: Params, batch: Batch) -> Scalar:
        return self._base.loss_only(params, batch)

    def row_op(self, params: Params, state: CurvatureState, batch: Batch) -> RowOperator:
        return self._base.row_op(params, state, batch)

    # ---- estimator facade ----
    def diagonal(self, params: Params, state: CurvatureState, rng: PRNGKey) -> PyTree:
        mvp: MatVec = lambda v: self._base.matvec(params, state, v)
        return self._est.diagonal(params, state, mvp, rng)

    def trace(self, params: Params, state: CurvatureState, rng: PRNGKey) -> Scalar:
        mvp: MatVec = lambda v: self._base.matvec(params, state, v)
        return self._est.trace(params, state, mvp, rng)

    def spectrum(self, params: Params, state: CurvatureState, rng: PRNGKey, k: int = 16) -> Tuple[Scalar, Scalar]:
        mvp: MatVec = lambda v: self._base.matvec(params, state, v)
        return self._est.spectrum(params, state, mvp, rng, k)

    def low_rank(self, params: Params, state: CurvatureState, rng: PRNGKey, k: int = 16) -> Tuple[Array, Array]:
        mvp: MatVec = lambda v: self._base.matvec(params, state, v)
        return self._est.low_rank(params, state, mvp, rng, k)
