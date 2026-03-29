from typing import Any, Protocol, Tuple, TypeAlias

from flax import struct

from ..types import Array, Batch, JvpFn, Params, Scalar, Updates, VjpFn

Empty: TypeAlias = tuple[()]
MaybeGrads: TypeAlias = Updates | Empty


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


@struct.dataclass
class CurvatureState:
    """Ephemeral per-step state/caches for a curvature operator.
    """
    cache: Any = ()


class CurvatureOperator(Protocol):
    """Matrix-free curvature operator for (C + lam I) s = g.

    Design:
    - init() performs the single allowed linearization per outer step and caches needed artifacts.
    - loss() must be derived from cached artifacts (no extra linearization).
    - loss_only() is forward-only and must not create linearization closures.
    """

    def init(self, params: Params, batch: Batch, *, with_grad: bool = False) -> Tuple[CurvatureState, MaybeGrads]:
        ...

    def matvec(self, params: Params, state: CurvatureState, v: Updates) -> Updates:
        ...

    def loss(self, params: Params, state: CurvatureState, batch: Batch) -> Scalar:
        ...

    def loss_only(self, params: Params, batch: Batch) -> Scalar:
        ...

    def row_op(self, params: Params, state: CurvatureState, batch: Batch) -> RowOperator:
        ...
