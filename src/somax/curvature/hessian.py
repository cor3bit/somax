from typing import Optional, Tuple, Any, Callable

import jax
from flax import struct

from ..types import Params, Updates, Batch, PRNGKey, Scalar, LossFn
from .base import CurvatureOperator, CurvatureState, RowOperator


@struct.dataclass
class _Cache:
    # Primal and linearization artifacts
    loss_val: Scalar
    hvp: Callable[[Updates], Updates]  # v -> H v


class ExactHessian(CurvatureOperator):
    """Exact Hessian operator via Pearlmutter (HVP)."""

    def __init__(
            self,
            *,
            loss_fn: LossFn,
            x_key: str = "x",
            y_key: str = "y",
            reduction: str = "mean",
    ):
        if reduction != "mean":
            raise ValueError("ExactHessian operator supports only reduction='mean'.")

        self.loss_fn = loss_fn
        self.x_key = x_key
        self.y_key = y_key
        self.reduction = reduction

    def init(self, params: Params, batch: Batch, with_grad: bool = False) -> tuple[CurvatureState, Any]:
        def f(p):
            return self.loss_fn(p, batch)

        # Linearize (loss, grad) together so we keep the primal loss and grad,
        # and get an HVP via the JVP of grad.
        (loss_val, g), jvp_vg = jax.linearize(lambda p: jax.value_and_grad(f)(p), params)

        def hvp(v: Updates) -> Updates:
            # jvp_vg(v) = (d_loss, d_grad); we want d_grad = H v
            return jvp_vg(v)[1]

        if with_grad:
            return CurvatureState(cache=_Cache(loss_val=loss_val, hvp=hvp)), g
        else:
            return CurvatureState(cache=_Cache(loss_val=loss_val, hvp=hvp)), ()

    def matvec(self, params: Params, state: CurvatureState, v: Updates) -> Updates:
        return state.cache.hvp(v)

    def loss(self, params: Params, state: CurvatureState, batch: Batch) -> Scalar:
        return state.cache.loss_val

    def loss_only(self, params: Params, batch: Batch) -> Scalar:
        return self.loss_fn(params, batch)

    def row_op(self, params: Params, state: CurvatureState, batch: Batch) -> RowOperator:
        raise NotImplementedError("ExactHessian does not provide a row operator.")
