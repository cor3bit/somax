from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from optax import softmax_cross_entropy, softmax_cross_entropy_with_integer_labels

from ..types import Batch, PredictFn, Scalar, Updates, Params
from .base import CurvatureOperator, CurvatureState, RowOperator


@struct.dataclass
class _Cache:
    logits: jnp.ndarray  # (B, C)
    jvp: Callable[[Updates], jnp.ndarray] = struct.field(pytree_node=False)  # Updates -> (B, C)
    vjp: Callable[[jnp.ndarray], Tuple[Updates]] = struct.field(pytree_node=False)  # (B, C) -> (Updates,)

    probs: jnp.ndarray  # (B, C)
    r: jnp.ndarray  # (B, C) = probs - one_hot(y)
    b: int = struct.field(pytree_node=False)
    alpha: jnp.ndarray  # scalar array 1/B


class GGNCE(CurvatureOperator):
    """Generalized Gauss-Newton for softmax cross-entropy (mean reduction)."""

    def __init__(
            self,
            *,
            predict_fn: PredictFn,
            x_key: str = "x",
            y_key: str = "y",
            reduction: str = "mean",
    ):
        if reduction != "mean":
            raise ValueError("GGNCE supports only reduction='mean'.")
        self.predict_fn = predict_fn
        self.x_key = x_key
        self.y_key = y_key
        self.reduction = reduction

    def init(self, params: Params, batch: Batch, *, with_grad: bool = False) -> tuple[CurvatureState, Any]:
        x = batch[self.x_key]
        y = batch[self.y_key]

        def f(p):
            return self.predict_fn(p, x)

        logits, Jv = jax.linearize(f, params)
        JT = jax.linear_transpose(Jv, params)

        probs = jax.lax.stop_gradient(jax.nn.softmax(logits, axis=-1))
        b, n_classes = logits.shape
        alpha = jnp.asarray(1.0 / b, dtype=logits.dtype)

        if y.ndim == 2 and y.shape[-1] == 1:
            y_int = y.squeeze(-1).astype(jnp.int32)
        else:
            y_int = y.astype(jnp.int32)

        r = probs - jax.nn.one_hot(y_int, n_classes, dtype=probs.dtype)

        if with_grad:
            grad = JT(alpha * r)[0]
        else:
            grad = ()

        cache = _Cache(
            logits=logits,
            jvp=Jv,
            vjp=JT,
            probs=probs,
            r=r,
            b=int(b),
            alpha=alpha,
        )
        return CurvatureState(cache=cache), grad

    def matvec(self, params: Params, state: CurvatureState, v: Updates) -> Updates:
        c: _Cache = state.cache
        Jv = c.jvp(v)
        tmp = Jv * c.probs
        dot = jnp.sum(tmp, axis=-1, keepdims=True)
        QJv = tmp - c.probs * dot
        (Av,) = c.vjp(c.alpha * QJv)
        return Av

    def loss(self, params: Params, state: CurvatureState, batch: Batch) -> Scalar:
        c: _Cache = state.cache
        y = batch[self.y_key]

        if jnp.issubdtype(y.dtype, jnp.integer) or (y.ndim <= 2 and y.shape[-1] == 1):
            y_int = y.squeeze(-1) if (y.ndim == 2 and y.shape[-1] == 1) else y
            y_int = y_int.astype(jnp.int32)
            loss_vals = softmax_cross_entropy_with_integer_labels(logits=c.logits, labels=y_int)
        else:
            y_oh = y.astype(c.logits.dtype)
            loss_vals = softmax_cross_entropy(logits=c.logits, labels=y_oh)

        return jnp.mean(loss_vals)

    def loss_only(self, params: Params, batch: Batch) -> Scalar:
        x = batch[self.x_key]
        y = batch[self.y_key]
        logits = self.predict_fn(params, x)

        if jnp.issubdtype(y.dtype, jnp.integer) or (y.ndim <= 2 and y.shape[-1] == 1):
            y_int = y.squeeze(-1) if (y.ndim == 2 and y.shape[-1] == 1) else y
            y_int = y_int.astype(jnp.int32)
            loss_vals = softmax_cross_entropy_with_integer_labels(logits=logits, labels=y_int)
        else:
            y_oh = y.astype(logits.dtype)
            loss_vals = softmax_cross_entropy(logits=logits, labels=y_oh)

        return jnp.mean(loss_vals)

    def row_op(self, params: Params, state: CurvatureState, batch: Batch) -> RowOperator:
        """Row operator for CE with an SPD row-space system.

        Parameter-space GGN:
          H = (1/B) J^T Q J
        where Q is the per-example Fisher on logits (PSD, singular).

        Make it Cholesky-safe by factoring:
          Q_eps = Q + eps I = L L^T
        and define:
          J_tilde = Q_eps^{1/2} J  (use L as the square-root)

        Then the row-space normal matrix:
          A = J_tilde J_tilde^T + mu I
        is SPD for mu > 0.

        Notes:
        - rhs is Q_eps^{-T} r so that J_tilde^T rhs = J^T r.
        - Any reduction scaling (e.g., 1/B) should be applied outside curvature code.
        """
        c: _Cache = state.cache
        B, C = c.probs.shape
        n_rows = B * C

        logits = c.logits
        probs = c.probs

        eps = jnp.asarray(1e-6, dtype=logits.dtype)
        I = jnp.eye(C, dtype=logits.dtype)
        Q = probs[..., None] * I - probs[:, :, None] * probs[:, None, :]

        L = jnp.linalg.cholesky(Q + eps * I)  # (B, C, C), lower

        def _Qsqrt_apply(Z: jnp.ndarray) -> jnp.ndarray:
            return jnp.einsum("bij,bj->bi", L, Z)

        def _Qsqrt_T_apply(U: jnp.ndarray) -> jnp.ndarray:
            return jnp.einsum("bji,bj->bi", L, U)

        def _Qsqrt_inv_T_apply(R: jnp.ndarray) -> jnp.ndarray:
            return jax.lax.linalg.triangular_solve(
                L,
                R[..., None],
                left_side=True,
                lower=True,
                transpose_a=True,
            )[..., 0]

        def jvp_flat(v: Updates) -> jnp.ndarray:
            Z = c.jvp(v)  # (B, C)
            Zt = _Qsqrt_apply(Z)
            return Zt.reshape((n_rows,))

        def vjp_flat(u_flat: jnp.ndarray) -> Updates:
            U = u_flat.reshape((B, C))
            Ut = _Qsqrt_T_apply(U)
            (out,) = c.vjp(Ut)
            return out

        rhs = _Qsqrt_inv_T_apply(c.r).reshape((n_rows,))

        return RowOperator(
            rhs=rhs,
            b=int(B),
            jvp=jvp_flat,
            vjp=vjp_flat,
        )
