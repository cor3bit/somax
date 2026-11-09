from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from ..types import Batch, PredictFnLike, Scalar, Updates, Params
from .base import (
    CurvatureOperator,
    CurvatureState,
    RowOperator,
    linearize_predict,
    predict_only,
)


@struct.dataclass
class _Cache:
    yhat: jnp.ndarray  # (B,)
    jvp: Callable[[Updates], jnp.ndarray] = struct.field(pytree_node=False)  # Updates -> (B,)
    vjp: Callable[[jnp.ndarray], Tuple[Updates]] = struct.field(pytree_node=False)  # (B,) -> (Updates,)

    r: jnp.ndarray  # (B,)
    b: int = struct.field(pytree_node=False)
    alpha: jnp.ndarray  # scalar array 1/B


def _squeeze_scalar_output(yhat: jnp.ndarray) -> jnp.ndarray:
    if yhat.ndim == 2 and yhat.shape[-1] == 1:
        return yhat.squeeze(-1)
    return yhat


class GGNMSE(CurvatureOperator):
    """Generalized Gauss-Newton for 0.5 * mean (yhat - y)^2 (mean reduction)."""

    def __init__(
            self,
            *,
            predict_fn: PredictFnLike,
            x_key: str = "x",
            y_key: str = "y",
            reduction: str = "mean",
            has_aux: bool = False,
    ):
        if reduction != "mean":
            raise ValueError("GGNMSE supports only reduction='mean'.")
        self.predict_fn = predict_fn
        self.x_key = x_key
        self.y_key = y_key
        self.reduction = reduction
        self.has_aux = bool(has_aux)

    def init(
            self,
            params: Params,
            batch: Batch,
            *,
            model_state: Any = (),
            with_grad: bool = False,
    ) -> tuple[CurvatureState, Any]:
        x = batch[self.x_key]

        yhat, Jv, model_state_new = linearize_predict(
            predict_fn=self.predict_fn,
            params=params,
            x=x,
            model_state=model_state,
            has_aux=self.has_aux,
            output_transform=_squeeze_scalar_output,
        )
        JT = jax.linear_transpose(Jv, params)

        y = batch[self.y_key]
        if y.ndim == 2 and y.shape[-1] == 1:
            y = y.squeeze(-1)

        if yhat.ndim != 1 or y.shape != yhat.shape:
            raise ValueError("GGNMSE requires predictions and targets of shape (B,) or (B, 1).")

        r = (yhat - y).astype(yhat.dtype)
        b = int(r.shape[0])
        alpha = jnp.asarray(1.0 / b, dtype=yhat.dtype)

        if with_grad:
            grad = JT(alpha * r)[0]
        else:
            grad = ()

        cache = _Cache(yhat=yhat, jvp=Jv, vjp=JT, r=r, b=b, alpha=alpha)
        return CurvatureState(
            cache=cache,
            model_state_new=model_state_new,
        ), grad

    def matvec(self, params: Params, state: CurvatureState, v: Updates) -> Updates:
        c: _Cache = state.cache
        u = c.jvp(v)
        (Av,) = c.vjp(c.alpha * u)
        return Av

    def loss(self, params: Params, state: CurvatureState, batch: Batch) -> Scalar:
        c: _Cache = state.cache
        return 0.5 * jnp.mean(jnp.square(c.r))

    def loss_only(
            self,
            params: Params,
            batch: Batch,
            *,
            model_state: Any = (),
    ) -> Scalar:
        x = batch[self.x_key]
        y = batch[self.y_key]
        yhat = predict_only(
            predict_fn=self.predict_fn,
            params=params,
            x=x,
            model_state=model_state,
            has_aux=self.has_aux,
            output_transform=_squeeze_scalar_output,
        )
        if y.ndim == 2 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        if yhat.ndim != 1 or y.shape != yhat.shape:
            raise ValueError("GGNMSE requires predictions and targets of shape (B,) or (B, 1).")
        r = (yhat - y).astype(yhat.dtype)
        return 0.5 * jnp.mean(jnp.square(r))

    def row_op(self, params: Params, state: CurvatureState, batch: Batch) -> RowOperator:
        c: _Cache = state.cache
        b = int(c.b)

        def vjp_row(u: jnp.ndarray) -> Updates:
            (out,) = c.vjp(u)
            return out

        rhs = c.r.reshape((b,))

        return RowOperator(
            rhs=rhs,
            b=b,
            jvp=c.jvp,
            vjp=vjp_row,
        )
