"""FGNHA curvature operator for mean softmax cross-entropy.

This is the conservative/base implementation, extended with an optional
direct row-Gram fast path for frozen-feature linear-head experiments.

Base representation
-------------------
Linearize logits once, construct a stable frozen row coefficient, and expose
both parameter-space and row-space views of the same whitened factor.

For a batch of logits ``z`` and hard integer labels ``y``, define frozen
probabilities ``p = softmax(z)`` and

    s_i = sqrt(clip(p_{i,y_i}) / (1 - clip(p_{i,y_i}))).

The whitened row factor is

    L_i dz_i = s_i * (sum_c p_{i,c} dz_{i,c} - dz_{i,y_i}).

Then under mean reduction:

    grad = (1 / B) L^T rhs,      rhs_i = 1 / s_i
    H    = (1 / B) L^T L

and the row-space solver binds

    (L L^T + B * lam * I) u = rhs,
    step = L^T u.

Optional direct row Gram
------------------------
Set ``gram_mode="linear_head"`` only when:
  - ``batch[x_key]`` already contains frozen features with shape ``(B, H)``, and
  - the trainable model is exactly a linear classifier head

        logits = x @ W + b        if ``linear_head_has_bias=True``
        logits = x @ W            otherwise.

In that mode we additionally cache the exact row Gram

    G = (x x^T + 1_bias) * (coeff coeff^T),

where

    coeff_i = s_i * (p_i - e_{y_i}).

Then ``row_op.gram_mv(u) = G u`` can bypass the generic
``row_op.jvp(row_op.vjp(u))`` composition inside row-space CG.

Do not enable ``gram_mode="linear_head"`` for a generic MLP/CNN body; it would
omit body parameters from the row Gram.
"""

from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import struct
from optax import softmax_cross_entropy_with_integer_labels

from ..types import Batch, JvpFn, Params, PredictFnLike, Scalar, Updates, VjpFn
from .base import (
    CurvatureOperator,
    CurvatureState,
    RowOperator,
    linearize_predict,
    predict_only,
)

GramMode = Literal["none", "linear_head"]


@struct.dataclass
class _Cache:
    # Linearization of logits z(params, x).
    jvp_logits: JvpFn = struct.field(pytree_node=False)
    vjp_logits: VjpFn = struct.field(pytree_node=False)

    # Frozen output-side row coefficient a_i = s_i * (p_i - one_hot(y_i))
    # and row RHS 1 / s_i; subsequent matvecs reuse these cached values.
    rhs_factor: jnp.ndarray  # (B,), 1 / s
    row_coeff: jnp.ndarray  # (B, C), s[:, None] * (p - one_hot(y))
    loss0: Scalar  # stable cached CE loss at the linearization point

    # Optional direct row Gram. When disabled, gram has shape (0, 0).
    gram: jnp.ndarray
    has_gram: bool = struct.field(pytree_node=False, default=False)

    # Metadata.
    b: int = struct.field(pytree_node=False, default=0)
    alpha: Scalar = 1.0  # 1 / B


class FGNHA(CurvatureOperator):
    """Fast Gauss-Newton Hessian Approximation for cross-entropy.

    Reuses one logits linearization and optionally caches a linear-head row Gram.
    """

    def __init__(
            self,
            *,
            predict_fn: PredictFnLike,
            x_key: str = "x",
            y_key: str = "y",
            reduction: str = "mean",
            eps: float = 1e-6,
            gram_mode: GramMode = "none",
            linear_head_has_bias: bool = True,
            gram_dtype: Any | None = None,
            has_aux: bool = False,
    ):
        if reduction != "mean":
            raise ValueError("FGNHA supports only reduction='mean'.")
        if gram_mode not in ("none", "linear_head"):
            raise ValueError("gram_mode must be 'none' or 'linear_head'.")

        self.predict_fn = predict_fn
        self.x_key = x_key
        self.y_key = y_key
        self.reduction = reduction
        self.eps = float(eps)

        self.gram_mode = gram_mode
        self.linear_head_has_bias = bool(linear_head_has_bias)
        self.gram_dtype = gram_dtype
        self.has_aux = bool(has_aux)

    @staticmethod
    def _integer_labels(y: jnp.ndarray) -> jnp.ndarray:
        """Validate and return hard integer labels.

        FGNHA has one true-versus-rest row per example and is derived for hard
        class labels.  Silently applying ``argmax`` to dense targets would make
        the gradient/curvature inconsistent with the reported dense-target loss.
        """
        if not jnp.issubdtype(y.dtype, jnp.integer):
            raise ValueError(
                "FGNHA supports hard integer labels only; dense/soft targets "
                "(label smoothing, mixup, CutMix) are not supported."
            )
        if y.ndim == 1:
            return y.astype(jnp.int32)
        if y.ndim == 2 and y.shape[-1] == 1:
            return y.squeeze(-1).astype(jnp.int32)
        raise ValueError(
            "FGNHA labels must have shape (B,) or (B, 1)."
        )

    @staticmethod
    def _cast_if_needed(x: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
        return x if x.dtype == dtype else x.astype(dtype)

    def _maybe_linear_head_gram(
            self,
            x: jnp.ndarray,
            row_coeff: jnp.ndarray,
            dtype: jnp.dtype,
    ) -> tuple[jnp.ndarray, bool]:
        """Return a cached direct row Gram if enabled.

        This fast path is exact only when x already are frozen features and the
        trainable params are exactly a linear classifier head.
        """
        if self.gram_mode == "none":
            return jnp.zeros((0, 0), dtype=dtype), False

        if x.ndim != 2:
            raise ValueError("gram_mode='linear_head' expects x/features with shape (B, H).")

        h = jax.lax.stop_gradient(self._cast_if_needed(x, dtype))  # (B, H)
        coeff = jax.lax.stop_gradient(self._cast_if_needed(row_coeff, dtype))  # (B, C)

        k_h = h @ h.T
        k_c = coeff @ coeff.T

        gram = k_h * k_c
        if self.linear_head_has_bias:
            gram = gram + k_c

        return jax.lax.stop_gradient(gram), True

    def init(
            self,
            params: Params,
            batch: Batch,
            with_grad: bool = False,
            *,
            model_state: Any = (),
    ) -> tuple[CurvatureState, Any]:
        x = batch[self.x_key]
        y_int = self._integer_labels(batch[self.y_key])
        eps = self.eps

        logits, Jv_logits, model_state_new = linearize_predict(
            predict_fn=self.predict_fn,
            params=params,
            x=x,
            model_state=model_state,
            has_aux=self.has_aux,
        )
        if logits.ndim != 2 or y_int.shape[0] != logits.shape[0]:
            raise ValueError("FGNHA requires logits (B, C) and integer labels (B,).")
        JT_logits = jax.linear_transpose(Jv_logits, params)

        probs = jax.lax.stop_gradient(jax.nn.softmax(logits, axis=-1))
        B = probs.shape[0]
        b_idx = jnp.arange(B, dtype=jnp.int32)

        p_star = probs[b_idx, y_int]
        # Ensure the clipping interval remains strictly inside (0, 1) even in
        # low-precision dtypes where ``1 - eps`` may round back to one.
        zero = jnp.asarray(0.0, probs.dtype)
        one = jnp.asarray(1.0, probs.dtype)
        eps_arr = jnp.asarray(eps, probs.dtype)
        lower = jnp.maximum(eps_arr, jnp.nextafter(zero, one))
        upper = jnp.minimum(one - eps_arr, jnp.nextafter(one, zero))
        p_star_safe = jnp.clip(p_star, lower, upper)
        p_dagger_safe = one - p_star_safe

        row_scale = jnp.sqrt(p_star_safe / p_dagger_safe)
        rhs_factor = 1.0 / row_scale
        alpha = jnp.asarray(1.0 / B, dtype=probs.dtype)
        loss0 = jax.lax.stop_gradient(jnp.mean(
            softmax_cross_entropy_with_integer_labels(logits=logits, labels=y_int)
        ))

        # Exact CE output gradient at the linearization point.  Build the true
        # class component as minus the sum of the competitor probabilities.
        # This is algebraically p_y - 1, but remains informative when p_y has
        # rounded to exactly one while tiny competitor probabilities survive.
        grad_logits = probs.at[b_idx, y_int].set(0.0)
        grad_logits = grad_logits.at[b_idx, y_int].set(
            -jnp.sum(grad_logits, axis=-1)
        )

        # Cache the whitened row coefficient explicitly instead of repeatedly
        # evaluating s * (E_p[z] - z_y).  The latter suffers cancellation for
        # confident predictions and, in parameter-space matvecs, creates an
        # avoidable s**2 intermediate.
        row_coeff = jax.lax.stop_gradient(row_scale[:, None] * grad_logits)

        if with_grad:
            grad = JT_logits(alpha * grad_logits)[0]
        else:
            grad = ()

        # Optional direct row Gram for frozen-feature linear-head experiments.
        if self.gram_mode == "none":
            gram = jnp.zeros((0, 0), dtype=probs.dtype)
            has_gram = False
        else:
            gram_dtype = probs.dtype if self.gram_dtype is None else self.gram_dtype
            gram, has_gram = self._maybe_linear_head_gram(x, row_coeff, gram_dtype)

        cache = _Cache(
            jvp_logits=Jv_logits,
            vjp_logits=JT_logits,
            rhs_factor=jax.lax.stop_gradient(rhs_factor),
            row_coeff=row_coeff,
            loss0=loss0,
            gram=gram,
            has_gram=has_gram,
            b=int(B),
            alpha=alpha,
        )
        return CurvatureState(
            cache=cache,
            model_state_new=model_state_new,
        ), grad

    def matvec(self, params: Params, state: CurvatureState, v: Updates) -> Updates:
        """Parameter-space matvec ``(1 / B) L^T L v``."""
        del params
        c = state.cache

        Z = c.jvp_logits(v)  # (B, C)
        LZ = jnp.einsum("bc,bc->b", c.row_coeff, Z)
        U = LZ[:, None] * c.row_coeff
        return c.vjp_logits(c.alpha * U)[0]

    def loss(self, params: Params, state: CurvatureState, batch: Batch) -> Scalar:
        """Cached mean CE loss at the linearization point."""
        del params, batch
        return state.cache.loss0

    def loss_only(
            self,
            params: Params,
            batch: Batch,
            *,
            model_state: Any = (),
    ) -> Scalar:
        """Forward-only mean cross-entropy; no linearization."""
        x = batch[self.x_key]
        y = batch[self.y_key]
        logits = predict_only(
            predict_fn=self.predict_fn,
            params=params,
            x=x,
            model_state=model_state,
            has_aux=self.has_aux,
        )

        y_int = self._integer_labels(y)
        if logits.ndim != 2 or y_int.shape[0] != logits.shape[0]:
            raise ValueError("FGNHA requires logits (B, C) and integer labels (B,).")
        loss_vals = softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=y_int
        )
        return jnp.mean(loss_vals)

    def row_op(self, params: Params, state: CurvatureState, batch: Batch) -> RowOperator:
        """Expose row primitives and optional direct row Gram to row solvers."""
        del params, batch
        c = state.cache

        def jvp_row(w: Updates) -> jnp.ndarray:
            Z = c.jvp_logits(w)  # (B, C)
            return jnp.einsum("bc,bc->b", c.row_coeff, Z)

        def vjp_row(u: jnp.ndarray) -> Updates:
            return c.vjp_logits(u[:, None] * c.row_coeff)[0]

        kwargs = {}
        if c.has_gram:
            def gram_mv(u: jnp.ndarray) -> jnp.ndarray:
                return (c.gram @ u).astype(u.dtype)

            kwargs["gram_mv"] = gram_mv

        return RowOperator(
            rhs=c.rhs_factor,
            b=c.b,
            jvp=jvp_row,
            vjp=vjp_row,
            **kwargs,
        )
