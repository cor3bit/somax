from typing import Protocol, Optional, Tuple, Any, FrozenSet

from flax import struct

from ..types import Params, PRNGKey, Scalar, PrecondFn
from ..curvature.base import CurvatureState, CurvatureOperator


@struct.dataclass
class PrecondState:
    cache: Any


class PreconditionerPolicy(Protocol):
    """Preconditioner policy interface.

    Contract:
      - build(...) returns (pre_fn, state2)
      - pre_fn is either:
          * a callable implementing approximate M^{-1} v, or
          * None, meaning "no preconditioning" (identity)
      - state2 must have the same pytree structure as state
    """
    is_identity: bool = False

    def init(self, params: Params) -> PrecondState: ...

    def build(
            self,
            params: Params,
            op: CurvatureOperator,
            cstate: CurvatureState,
            *,
            rng: Optional[PRNGKey],
            lam: Optional[Scalar],
            state: PrecondState,
    ) -> Tuple[Optional[PrecondFn], PrecondState]:
        ...

    def needed_metrics(self) -> FrozenSet[str]:
        """Planner-time metric requirements (default: empty)."""
        return frozenset()
