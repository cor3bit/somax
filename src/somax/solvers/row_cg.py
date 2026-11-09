from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp

from ..types import Array, MatVec, Updates
from .base import LinearSolver, PrecondLike, NullSolverState
from .cg import ConjugateGradient


class RowConjugateGradient(LinearSolver):
    space = "row"

    def __init__(
            self,
            *,
            maxiter: int = 10,
            tol: float = 1e-4,
            warm_start: bool = False,
            stabilise_every: int = 0,
            preconditioned: bool = False,
            solve_dtype: Optional[jnp.dtype] = None,
            backend: str = "pcg",
            telemetry_residual: bool = False,
    ):
        if preconditioned:
            raise NotImplementedError("RowConjugateGradient: preconditioning not implemented.")
        if warm_start:
            raise NotImplementedError(
                "RowConjugateGradient: row-space warm start is intentionally disabled. "
                "This avoids row-dimension-dependent solver state and first-step retrace churn."
            )

        self._cg = ConjugateGradient(
            maxiter=maxiter,
            tol=tol,
            warm_start=False,
            preconditioned=False,
            stabilise_every=stabilise_every,
            solve_dtype=solve_dtype,
            backend=backend,
            telemetry_residual=telemetry_residual,
        )

    def init(self, params: Updates) -> Any:
        del params
        return NullSolverState()

    def solve(
            self,
            A_mv: MatVec,
            rhs: Updates,
            *,
            state: Any,
            precond: PrecondLike | None = None,
    ) -> Tuple[Updates, Dict[str, Any], Any]:
        del state

        if precond is not None:
            raise NotImplementedError("RowConjugateGradient: preconditioning not implemented.")

        r: Array = jnp.asarray(rhs)
        if r.ndim != 1:
            raise ValueError("RowConjugateGradient.solve expects rhs with shape (m,).")

        return self._cg.solve(A_mv, r, state=NullSolverState(), precond=None)
