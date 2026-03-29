"""Solver protocols.

A solver is a matrix-free linear solver for a linear system A x = b, expressed
by a matvec callable.

Lane selection convention (static, compile-time):
  - solver.space == "diag"  -> diagonal lane (preconditioner-only)
  - solver.space == "param" -> parameter-space solve (PyTree rhs)
  - solver.space == "row"   -> row-space solve (1D rhs)

Row-space solvers solve for the row variable u and return u (an Array). The
executor is responsible for mapping u -> parameter-space direction s via the
curvature RowOperator.backproject callable.
"""

from typing import Any, Dict, Protocol, Tuple

from flax import struct

from ..types import Array, MatVec, PrecondFn, RowPrecondFn, Updates

PrecondLike = PrecondFn | RowPrecondFn


@struct.dataclass
class NullSolverState:
    """Stateless placeholder state."""
    pass


class LinearSolver(Protocol):
    """Matrix-free linear solver interface."""

    space: str  # "diag" | "param" | "row"

    def init(self, params: Updates) -> Any:
        """Return an initial solver state (PyTree)."""
        ...

    def solve(
            self,
            A_mv: MatVec,
            rhs: Updates,
            *,
            state: Any,
            precond: PrecondLike | None = None,
    ) -> Tuple[Updates, Dict[str, Array], Any]:
        """Solve A x = rhs and return (x, info, new_state)."""
        ...
