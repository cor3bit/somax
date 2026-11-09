"""Exact generalized Gauss-Newton operator for mean softmax cross-entropy."""

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from optax import softmax_cross_entropy, softmax_cross_entropy_with_integer_labels

from ..types import Batch, Params, PredictFnLike, Scalar, Updates
from .base import (
    CurvatureOperator,
    CurvatureState,
    RowOperator,
    linearize_predict,
    predict_only,
)


def _has_integer_labels(y: jnp.ndarray) -> bool:
    return jnp.issubdtype(y.dtype, jnp.integer) and (
            y.ndim == 1 or (y.ndim == 2 and y.shape[-1] == 1)
    )


def _integer_labels(y: jnp.ndarray) -> jnp.ndarray:
    if not _has_integer_labels(y):
        raise ValueError("Integer labels must have shape (B,) or (B, 1).")
    if y.ndim == 2:
        y = y.squeeze(-1)
    return y.astype(jnp.int32)


def _output_residual(probs: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Return the exact CE logits gradient for hard or normalized dense targets."""
    batch_size = probs.shape[0]

    if _has_integer_labels(targets):
        labels = _integer_labels(targets)
        if labels.shape[0] != batch_size:
            raise ValueError("GGNCE integer labels must match the logits batch size.")
        batch_idx = jnp.arange(batch_size, dtype=jnp.int32)

        # Algebraically this is probs - one_hot(labels).  Constructing the true
        # class entry as minus the competitor mass remains informative when the
        # true-class probability has rounded to exactly one in finite precision.
        residual = probs.at[batch_idx, labels].set(0.0)
        return residual.at[batch_idx, labels].set(
            -jnp.sum(residual, axis=-1)
        )

    if targets.shape == probs.shape:
        # Contract: each dense target row is a probability distribution.  This
        # covers one-hot targets, label smoothing, mixup, and CutMix.
        return probs - targets.astype(probs.dtype)

    raise ValueError(
        "GGNCE labels must be integer class indices with shape (B,) or (B, 1), "
        "or normalized dense targets with the same shape as logits."
    )


def _loss_values(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    if _has_integer_labels(targets):
        labels = _integer_labels(targets)
        if labels.shape[0] != logits.shape[0]:
            raise ValueError("GGNCE integer labels must match the logits batch size.")
        return softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=labels,
        )
    if targets.shape == logits.shape:
        return softmax_cross_entropy(
            logits=logits,
            labels=targets.astype(logits.dtype),
        )
    raise ValueError(
        "GGNCE labels must be integer class indices with shape (B,) or (B, 1), "
        "or normalized dense targets with the same shape as logits."
    )


def _softmax_fisher_mv(probs: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """Apply Q z for Q = diag(p) - p p^T without materializing Q."""
    pz = probs * z
    return pz - probs * jnp.sum(pz, axis=-1, keepdims=True)


def _fisher_factor_mv(sqrt_probs: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """Apply S z for S = (I - q q^T) diag(q), q = sqrt(p)."""
    dz = sqrt_probs * z
    return dz - sqrt_probs * jnp.sum(
        sqrt_probs * dz,
        axis=-1,
        keepdims=True,
    )


def _fisher_factor_transpose_mv(
        sqrt_probs: jnp.ndarray,
        u: jnp.ndarray,
) -> jnp.ndarray:
    """Apply S^T u for S = (I - q q^T) diag(q), q = sqrt(p)."""
    projected = u - sqrt_probs * jnp.sum(
        sqrt_probs * u,
        axis=-1,
        keepdims=True,
    )
    return sqrt_probs * projected


@struct.dataclass
class _Cache:
    logits: jnp.ndarray  # (B, C)
    jvp: Callable[[Updates], jnp.ndarray] = struct.field(pytree_node=False)
    vjp: Callable[[jnp.ndarray], Tuple[Updates]] = struct.field(pytree_node=False)

    probs: jnp.ndarray  # (B, C)
    residual: jnp.ndarray  # (B, C), exact CE gradient with respect to logits
    b: int = struct.field(pytree_node=False)
    alpha: jnp.ndarray  # scalar array 1 / B


class GGNCE(CurvatureOperator):
    """Generalized Gauss-Newton for mean softmax cross-entropy.

    Integer labels and normalized dense targets are supported.  The critical
    parameter-space path used by ``sgn_ce`` applies

        (1 / B) J^T [diag(p) - p p^T] J

    with one JVP and one VJP per matrix-vector product.  ``row_op`` exposes an
    exact, optional factorization for generic row-space solvers; ``sgn_ce`` does
    not use that path.
    """

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
            raise ValueError("GGNCE supports only reduction='mean'.")
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
        targets = batch[self.y_key]

        logits, jvp, model_state_new = linearize_predict(
            predict_fn=self.predict_fn,
            params=params,
            x=x,
            model_state=model_state,
            has_aux=self.has_aux,
        )
        if logits.ndim != 2:
            raise ValueError("GGNCE requires logits with shape (B, C).")
        vjp = jax.linear_transpose(jvp, params)

        probs = jax.lax.stop_gradient(jax.nn.softmax(logits, axis=-1))
        batch_size = logits.shape[0]
        alpha = jnp.asarray(1.0 / batch_size, dtype=logits.dtype)
        residual = _output_residual(probs, targets)

        grad = vjp(alpha * residual)[0] if with_grad else ()

        cache = _Cache(
            logits=logits,
            jvp=jvp,
            vjp=vjp,
            probs=probs,
            residual=residual,
            b=int(batch_size),
            alpha=alpha,
        )
        return CurvatureState(
            cache=cache,
            model_state_new=model_state_new,
        ), grad

    def matvec(self, params: Params, state: CurvatureState, v: Updates) -> Updates:
        del params
        c: _Cache = state.cache
        logits_tangent = c.jvp(v)
        fisher_tangent = _softmax_fisher_mv(c.probs, logits_tangent)
        return c.vjp(c.alpha * fisher_tangent)[0]

    def loss(self, params: Params, state: CurvatureState, batch: Batch) -> Scalar:
        del params
        return jnp.mean(_loss_values(state.cache.logits, batch[self.y_key]))

    def loss_only(
            self,
            params: Params,
            batch: Batch,
            *,
            model_state: Any = (),
    ) -> Scalar:
        logits = predict_only(
            predict_fn=self.predict_fn,
            params=params,
            x=batch[self.x_key],
            model_state=model_state,
            has_aux=self.has_aux,
        )
        if logits.ndim != 2:
            raise ValueError("GGNCE requires logits with shape (B, C).")
        return jnp.mean(_loss_values(logits, batch[self.y_key]))

    def row_op(self, params: Params, state: CurvatureState, batch: Batch) -> RowOperator:
        """Expose an exact matrix-free row factor of the softmax GGN.

        For each example, let ``q = sqrt(p)``, ``P = I - q q^T``, and

            S = P diag(q).

        Since ``P`` is an orthogonal projector,

            S^T S = diag(p) - p p^T = Q.

        Hence ``L = S J`` satisfies ``L^T L = J^T Q J``.  For cross-entropy
        residual ``r`` with ``sum(r) = 0``, ``rhs = r / sqrt(p)`` satisfies
        ``S^T rhs = r``.  No dense ``C x C`` matrix or Cholesky factor is formed.

        This optional row representation has ``B * C`` rows and is not used by
        the standard parameter-space ``sgn_ce`` method.
        """
        del params, batch
        c: _Cache = state.cache
        batch_size, n_classes = c.probs.shape
        n_rows = batch_size * n_classes
        sqrt_probs = jnp.sqrt(c.probs)
        # Zero probability with nonzero CE residual cannot be represented by
        # this Fisher row factor. Surface the invalid solve under jit instead
        # of silently dropping the corresponding gradient component.
        invalid_factorization = jnp.any((sqrt_probs == 0) & (c.residual != 0))

        def jvp_flat(v: Updates) -> jnp.ndarray:
            logits_tangent = c.jvp(v)
            result = _fisher_factor_mv(sqrt_probs, logits_tangent).reshape((n_rows,))
            return jnp.where(invalid_factorization, jnp.nan, result)

        def vjp_flat(u_flat: jnp.ndarray) -> Updates:
            u = u_flat.reshape((batch_size, n_classes))
            cotangent = _fisher_factor_transpose_mv(sqrt_probs, u)
            result = c.vjp(cotangent)[0]
            return jax.tree_util.tree_map(
                lambda leaf: jnp.where(invalid_factorization, jnp.nan, leaf), result
            )

        # Underflowed probabilities with zero residual are harmless. Nonzero
        # residuals are flagged as invalid by the row operators above.
        safe_sqrt_probs = jnp.where(sqrt_probs > 0, sqrt_probs, 1.0)
        rhs = jnp.where(
            sqrt_probs > 0,
            c.residual / safe_sqrt_probs,
            0.0,
        ).reshape((n_rows,))

        return RowOperator(
            rhs=rhs,
            b=int(batch_size),
            jvp=jvp_flat,
            vjp=vjp_flat,
        )
