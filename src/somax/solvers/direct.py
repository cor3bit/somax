from typing import Optional, Tuple, Dict, Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from ..types import Updates, MatVec, PrecondFn
from .base import LinearSolver, NullSolverState


class DirectSPD(LinearSolver):
    """Tiny dense direct solve (for tests/sanity).

    Builds a dense matrix A by applying A_mv to basis vectors in the flattened
    parameter space, then solves A x = b with jax.scipy.linalg.solve.

    Complexity is O(n^2) matvecs and O(n^3) dense solve -- ONLY use for small n.
    """

    space = "param"

    def __init__(self):
        pass

    def init(self, params: Updates) -> NullSolverState:
        return NullSolverState()

    def solve(
            self,
            A_mv: MatVec,
            rhs: Updates,
            *,
            state: Any,
            precond: PrecondFn = None,
    ) -> Tuple[Updates, Dict[str, Any], Any]:
        # Flatten RHS to define packing
        rhs_flat, pack = ravel_pytree(rhs)
        n = rhs_flat.size

        # Standard basis in R^n
        eye = jnp.eye(n, dtype=rhs_flat.dtype)  # (n, n)

        def col_apply(e_col: jnp.ndarray) -> jnp.ndarray:
            v = pack(e_col)  # PyTree shaped like b
            Av = A_mv(v)  # PyTree result
            Av_flat, _ = ravel_pytree(Av)
            return Av_flat  # (n,)

        # Build dense A matrix with vmap over columns
        A = jax.vmap(col_apply, in_axes=1, out_axes=1)(eye)  # (n, n)

        # Solve A x = b
        x_flat = jax.scipy.linalg.solve(A, rhs_flat, assume_a="pos")
        x = pack(x_flat)
        return x, {"mode": "dense_direct", "n": int(n)}, state
