from typing import Any, Protocol

from flax import struct

from ..types import Scalar, StepInfo


@struct.dataclass
class DampingState:
    """Opaque state for damping policies (PyTree).

    Invariant:
      - Stable PyTree structure (no None leaves).
      - `lam` is a scalar array (dtype typically float32).
    """

    lam: Scalar
    aux: Any = ()


class DampingPolicy(Protocol):
    """Damping policy interface.

    Damping is a pure controller:
      - reads StepInfo (static schema; missing keys are default-filled upstream)
      - updates only `DampingState`
    """

    def init(self) -> DampingState: ...

    def update(self, state: DampingState, info: StepInfo) -> DampingState: ...

    def needed_metrics(self) -> frozenset[str]: ...
