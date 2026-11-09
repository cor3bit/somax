from typing import Any, Callable, Protocol, Tuple, TypeAlias

import jax
from flax import struct

from ..types import Array, Batch, JvpFn, Params, PredictFnLike, Scalar, Updates, VjpFn

Empty: TypeAlias = tuple[()]
MaybeGrads: TypeAlias = Updates | Empty
RowGramFn: TypeAlias = Callable[[Array], Array]


def _stop_gradient_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(jax.lax.stop_gradient, tree)


def linearize_predict(
        *,
        predict_fn: PredictFnLike,
        params: Params,
        x: Any,
        model_state: Any = (),
        has_aux: bool = False,
        output_transform: Callable[[Any], Any] | None = None,
) -> tuple[Any, Callable[[Updates], Any], Any]:
    """Linearize a stateless or model-state-aware prediction function once.

    Contracts:
    - ``has_aux=False``: ``predict_fn(params, x) -> predictions``.
    - ``has_aux=True``:
      ``predict_fn(params, x, model_state) -> (predictions, model_state_new)``.

    With ``has_aux=True``, JAX differentiates only ``predictions``. The model
    state update is produced by the same primal evaluation, stopped from higher
    order differentiation, and is not recomputed by calls to the returned JVP.
    """
    transform = (lambda y: y) if output_transform is None else output_transform
    model_state = _stop_gradient_tree(model_state)

    if has_aux:
        def f(p: Params):
            predictions, model_state_new = predict_fn(p, x, model_state)
            return transform(predictions), model_state_new

        predictions, jvp, model_state_new = jax.linearize(
            f, params, has_aux=True
        )
        model_state_new = _stop_gradient_tree(model_state_new)
        return predictions, jvp, model_state_new

    def f(p: Params):
        return transform(predict_fn(p, x))

    predictions, jvp = jax.linearize(f, params)
    return predictions, jvp, model_state


def predict_only(
        *,
        predict_fn: PredictFnLike,
        params: Params,
        x: Any,
        model_state: Any = (),
        has_aux: bool = False,
        output_transform: Callable[[Any], Any] | None = None,
) -> Any:
    """Evaluate predictions without creating a linearization closure."""
    transform = (lambda y: y) if output_transform is None else output_transform
    model_state = _stop_gradient_tree(model_state)
    if has_aux:
        predictions, _ = predict_fn(params, x, model_state)
        return transform(predictions)
    return transform(predict_fn(params, x))


@struct.dataclass
class RowOperator:
    """Row-space primitives for row lane.

    Invariants:
    - Numeric fields are JAX arrays (PyTree leaves).
    - Callables are marked static (pytree_node=False).
    - No Optional/None leaves in traced pytrees.
    """

    rhs: Array  # (m,)
    b: int = struct.field(pytree_node=False)
    jvp: JvpFn = struct.field(pytree_node=False)  # Updates -> (m,)
    vjp: VjpFn = struct.field(pytree_node=False)  # (m,) -> Updates
    gram_mv: RowGramFn | None = struct.field(pytree_node=False, default=None)


@struct.dataclass
class CurvatureState:
    """Ephemeral per-step state/caches for a curvature operator.

    ``cache`` and ``model_state_new`` must contain JIT-safe PyTree content.
    ``model_state_new`` is committed once by the executor after the solve; CG
    matvecs use only the fixed linearization stored in ``cache``.
    """
    cache: Any = ()
    model_state_new: Any = ()


class CurvatureOperator(Protocol):
    """Matrix-free curvature operator for ``(C + lam I) s = g``.

    Design:
    - init() performs the single allowed linearization per outer step and caches needed artifacts.
    - loss() must be derived from cached artifacts (no extra linearization).
    - loss_only() is forward-only and must not create linearization closures.
    """

    def init(
            self,
            params: Params,
            batch: Batch,
            *,
            model_state: Any = (),
            with_grad: bool = False,
    ) -> Tuple[CurvatureState, MaybeGrads]:
        ...

    def matvec(self, params: Params, state: CurvatureState, v: Updates) -> Updates:
        ...

    def loss(self, params: Params, state: CurvatureState, batch: Batch) -> Scalar:
        ...

    def loss_only(
            self,
            params: Params,
            batch: Batch,
            *,
            model_state: Any = (),
    ) -> Scalar:
        ...

    def row_op(self, params: Params, state: CurvatureState, batch: Batch) -> RowOperator:
        ...
