from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp

from ..types import Array, MatVec, Updates
from .base import LinearSolver, PrecondLike, NullSolverState
from .cg import ConjugateGradient, CGState


class RowConjugateGradient(LinearSolver):
    """Row-space Conjugate Gradient solver.

    Solves (in row space):
        A u = rhs
    where A is typically (J J^T + mu * D).

    Contract:
      - rhs is a 1D Array with shape (m,)
      - A_mv: Array(m,) -> Array(m,)
      - warm_start stores the previous solution u in CGState.last_x
    """

    space = "row"

    def __init__(
            self,
            *,
            maxiter: int = 50,
            tol: float = 1e-4,
            warm_start: bool = True,
            stabilise_every: int = 10,
            preconditioned: bool = False,
            assume_spd: bool = True,
            solve_dtype: Optional[jnp.dtype] = None,
            backend: str = "pcg",
            telemetry_residual: bool = False,
    ):
        if preconditioned:
            raise NotImplementedError("RowConjugateGradient: preconditioning not implemented.")

        self._warm_start = bool(warm_start)

        # Preconditioning is not implemented for row CG yet.
        self._cg = ConjugateGradient(
            maxiter=maxiter,
            tol=tol,
            warm_start=self._warm_start,
            preconditioned=False,
            stabilise_every=stabilise_every,
            assume_spd=assume_spd,
            solve_dtype=solve_dtype,
            backend=backend,
            telemetry_residual=telemetry_residual,
        )

    def init(self, params: Updates) -> Any:
        del params
        return CGState(last_x=jnp.zeros((1,))) if self._warm_start else NullSolverState()

    def solve(
            self,
            A_mv: MatVec,
            rhs: Updates,
            *,
            state: Any,
            precond: PrecondLike | None = None,
    ) -> Tuple[Updates, Dict[str, Any], Any]:
        if precond is not None:
            raise NotImplementedError("RowConjugateGradient: preconditioning not implemented.")

        r: Array = jnp.asarray(rhs)
        if r.ndim != 1:
            raise ValueError("RowConjugateGradient.solve expects rhs with shape (m,).")

        if not self._warm_start:
            # Ignore any provided state to keep behavior deterministic.
            return self._cg.solve(A_mv, r, state=NullSolverState(), precond=None)

        # Ensure state provides a valid x0 with the right structure.
        if not isinstance(state, CGState):
            x0 = jnp.zeros_like(r)
        else:
            x0 = jnp.asarray(state.last_x)
            if x0.ndim != 1 or x0.shape != r.shape:
                x0 = jnp.zeros_like(r)
            elif x0.dtype != r.dtype:
                x0 = x0.astype(r.dtype)

        return self._cg.solve(A_mv, r, state=CGState(last_x=x0), precond=None)
