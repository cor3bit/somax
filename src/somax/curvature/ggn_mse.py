from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from ..types import Batch, PredictFn, Scalar, Updates, Params
from .base import CurvatureOperator, CurvatureState, RowOperator


@struct.dataclass
class _Cache:
    yhat: jnp.ndarray  # (B,)
    jvp: Callable[[Updates], jnp.ndarray] = struct.field(pytree_node=False)  # Updates -> (B,)
    vjp: Callable[[jnp.ndarray], Tuple[Updates]] = struct.field(pytree_node=False)  # (B,) -> (Updates,)

    r: jnp.ndarray  # (B,)
    b: int = struct.field(pytree_node=False)
    alpha: jnp.ndarray  # scalar array 1/B


class GGNMSE(CurvatureOperator):
    """Generalized Gauss-Newton for 0.5 * mean (yhat - y)^2 (mean reduction)."""

    def __init__(
            self,
            *,
            predict_fn: PredictFn,
            x_key: str = "x",
            y_key: str = "y",
            reduction: str = "mean",
    ):
        if reduction != "mean":
            raise ValueError("GGNMSE supports only reduction='mean'.")
        self.predict_fn = predict_fn
        self.x_key = x_key
        self.y_key = y_key
        self.reduction = reduction

    def init(self, params: Params, batch: Batch, *, with_grad: bool = False) -> tuple[CurvatureState, Any]:
        x = batch[self.x_key]

        def f(p):
            yhat = self.predict_fn(p, x)
            if yhat.ndim == 2 and yhat.shape[-1] == 1:
                yhat = yhat.squeeze(-1)
            return yhat

        yhat, Jv = jax.linearize(f, params)
        JT = jax.linear_transpose(Jv, params)

        y = batch[self.y_key]
        if y.ndim == 2 and y.shape[-1] == 1:
            y = y.squeeze(-1)

        r = (yhat - y).astype(yhat.dtype)
        b = int(r.shape[0])
        alpha = jnp.asarray(1.0 / b, dtype=yhat.dtype)

        if with_grad:
            grad = JT(alpha * r)[0]
        else:
            grad = ()

        cache = _Cache(yhat=yhat, jvp=Jv, vjp=JT, r=r, b=b, alpha=alpha)
        return CurvatureState(cache=cache), grad

    def matvec(self, params: Params, state: CurvatureState, v: Updates) -> Updates:
        c: _Cache = state.cache
        u = c.jvp(v)
        (Av,) = c.vjp(c.alpha * u)
        return Av

    def loss(self, params: Params, state: CurvatureState, batch: Batch) -> Scalar:
        c: _Cache = state.cache
        return 0.5 * jnp.mean(jnp.square(c.r))

    def loss_only(self, params: Params, batch: Batch) -> Scalar:
        x = batch[self.x_key]
        y = batch[self.y_key]
        yhat = self.predict_fn(params, x)
        if yhat.ndim == 2 and yhat.shape[-1] == 1:
            yhat = yhat.squeeze(-1)
        if y.ndim == 2 and y.shape[-1] == 1:
            y = y.squeeze(-1)
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
