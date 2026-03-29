from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from ..types import MatVec, Updates
from .base import LinearSolver, NullSolverState, PrecondLike


class RowCholesky(LinearSolver):
    """Direct row-space solve via dense Cholesky.

    For small row dimension m only.

    Solves A u = rhs in row space and returns u (Array).
    Backprojection u -> s is handled by the executor.
    """

    space = "row"

    def __init__(
        self,
        *,
        symmetrize: bool = True,
        jitter: float = 0.0,
        solve_dtype: Optional[jnp.dtype] = None,
    ):
        self.symmetrize = bool(symmetrize)
        self.jitter = float(jitter)
        self.solve_dtype = solve_dtype

    def init(self, params: Updates) -> NullSolverState:
        return NullSolverState()

    def solve(
        self,
        A_mv: MatVec,
        rhs: Updates,
        *,
        state: Any,
        precond: PrecondLike | None = None,
    ) -> Tuple[Updates, Dict[str, Any], Any]:
        del precond

        r = jnp.asarray(rhs)
        if r.ndim != 1:
            raise ValueError("RowCholesky.solve expects rhs with shape (m,).")

        m = r.shape[0]
        I = jnp.eye(m, dtype=r.dtype)
        A = jax.vmap(lambda e: A_mv(e), in_axes=0, out_axes=0)(I)  # rows
        A = jnp.transpose(A, (1, 0))  # columns

        if self.symmetrize:
            A = 0.5 * (A + A.T)

        if self.solve_dtype is not None and A.dtype != self.solve_dtype:
            A_s = A.astype(self.solve_dtype)
            r_s = r.astype(self.solve_dtype)
        else:
            A_s, r_s = A, r

        if self.jitter > 0.0:
            A_s = A_s + self.jitter * jnp.eye(m, dtype=A_s.dtype)

        L = jnp.linalg.cholesky(A_s)
        y = lax.linalg.triangular_solve(L, r_s, left_side=True, lower=True, transpose_a=False)
        u = lax.linalg.triangular_solve(L, y, left_side=True, lower=True, transpose_a=True)

        if self.solve_dtype is not None:
            u = u.astype(r.dtype)

        info: Dict[str, Any] = {
            "mode": "row_cholesky",
            "row_dim": jnp.asarray(m, dtype=jnp.int32),
        }
        return u, info, state
