from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..types import MatVec, Updates, PrecondFn
from .base import NullSolverState


@dataclass(frozen=True)
class IdentitySolve:
    """Diag-lane solver: solution is exactly preconditioned RHS.

    Contract:
      - This solver is only meaningful if a preconditioner is provided.
      - If `precond` is None: raise ValueError (explicit contract).
    """

    space: str = "diag"

    def init(self, params: Updates) -> NullSolverState:
        return NullSolverState()

    def solve(
            self,
            A_mv: MatVec,
            rhs: Updates,
            *,
            state: Any,
            precond: Optional[PrecondFn] = None,
            **kwargs: Any,
    ) -> tuple[Updates, Dict[str, Any], None]:
        if precond is None:
            raise ValueError("IdentitySolve requires a non-None precond.")
        # Must not call A_mv in diag lane.
        sol = precond(rhs)
        info = {"mode": "direct_precond"}
        return sol, info, state
