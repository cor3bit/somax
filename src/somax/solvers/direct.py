from typing import Any, Dict, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.flatten_util import ravel_pytree

from ..types import MatVec, Updates
from .base import LinearSolver, NullSolverState, PrecondLike

VectorSpace = Literal["pytree", "array"]
SolveMethod = Literal["chol", "scipy"]
Lane = Literal["param", "row"]


class DirectSPD(LinearSolver):
    """Dense direct SPD solve from a matrix-free operator.

    This class is a small wrapper that can operate in either:
      - parameter space: rhs / solution are PyTrees
      - row space:       rhs / solution are explicit vectors

    It materializes a dense matrix A by applying A_mv to basis vectors, then
    solves A x = rhs with either:
      - explicit Cholesky
      - jax.scipy.linalg.solve(..., assume_a="pos")

    Complexity:
      - O(n) matvecs and O(n^2) storage to materialize A
      - O(n^3) dense solve

    Intended for small systems, tests, and sanity checks.
    """

    space: str

    def __init__(
            self,
            *,
            space: Lane = "param",
            vector_space: Optional[VectorSpace] = None,
            solve_method: SolveMethod = "scipy",
            symmetrize: bool = False,
            jitter: float = 0.0,
            solve_dtype: Optional[jnp.dtype] = None,
            info_mode: Optional[str] = None,
    ):
        if space not in ("param", "row"):
            raise ValueError(f"DirectSPD.space must be 'param' or 'row', got {space!r}.")

        if vector_space is None:
            vector_space = "array" if space == "row" else "pytree"

        if vector_space not in ("pytree", "array"):
            raise ValueError(
                f"DirectSPD.vector_space must be 'pytree' or 'array', got {vector_space!r}."
            )

        if solve_method not in ("chol", "scipy"):
            raise ValueError(
                f"DirectSPD.solve_method must be 'chol' or 'scipy', got {solve_method!r}."
            )

        if space == "row" and vector_space != "array":
            raise ValueError(
                "DirectSPD with space='row' must use vector_space='array'."
            )

        self.space = space
        self.vector_space = vector_space
        self.solve_method = solve_method
        self.symmetrize = bool(symmetrize)
        self.jitter = float(jitter)
        self.solve_dtype = solve_dtype
        self.info_mode = info_mode

    def init(self, params: Updates) -> NullSolverState:
        del params
        return NullSolverState()

    def _flatten_rhs(self, rhs: Updates) -> Tuple[jnp.ndarray, Any]:
        if self.vector_space == "pytree":
            rhs_flat, pack = ravel_pytree(rhs)
            if rhs_flat.ndim != 1:
                raise ValueError(
                    f"DirectSPD(vector_space='pytree') expects flattened rhs to be 1D, "
                    f"got shape {rhs_flat.shape}."
                )
            return rhs_flat, pack

        rhs_arr = jnp.asarray(rhs)
        if rhs_arr.ndim != 1:
            raise ValueError(
                f"DirectSPD(vector_space='array') expects rhs with shape (n,), "
                f"got shape {rhs_arr.shape}."
            )
        return rhs_arr, (lambda x: x)

    def _flatten_out_like_rhs(self, out: Updates, *, n: int) -> jnp.ndarray:
        if self.vector_space == "pytree":
            out_flat, _ = ravel_pytree(out)
        else:
            out_flat = jnp.asarray(out)

        if out_flat.ndim != 1:
            raise ValueError(
                f"DirectSPD expects A_mv output to flatten to shape (n,), "
                f"got shape {out_flat.shape}."
            )
        if out_flat.shape[0] != n:
            raise ValueError(
                f"DirectSPD expects A_mv output size {n}, got {out_flat.shape[0]}."
            )
        return out_flat

    def _build_dense(self, A_mv: MatVec, *, pack: Any, n: int, dtype: jnp.dtype) -> jnp.ndarray:
        eye = jnp.eye(n, dtype=dtype)

        def col_apply(e_col: jnp.ndarray) -> jnp.ndarray:
            v = pack(e_col)
            Av = A_mv(v)
            return self._flatten_out_like_rhs(Av, n=n)

        # Columns of A in the chosen basis
        A = jax.vmap(col_apply, in_axes=1, out_axes=1)(eye)
        return A

    def _solve_dense(self, A: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
        if self.solve_method == "chol":
            L = jnp.linalg.cholesky(A)
            y = lax.linalg.triangular_solve(
                L,
                rhs[:, None],
                left_side=True,
                lower=True,
                transpose_a=False,
            )
            x = lax.linalg.triangular_solve(
                L,
                y,
                left_side=True,
                lower=True,
                transpose_a=True,
            )
            return jnp.squeeze(x, axis=-1)

        return jax.scipy.linalg.solve(A, rhs, assume_a="pos")

    def solve(
            self,
            A_mv: MatVec,
            rhs: Updates,
            *,
            state: Any,
            precond: PrecondLike | None = None,
    ) -> Tuple[Updates, Dict[str, Any], Any]:
        del precond

        rhs_flat, pack = self._flatten_rhs(rhs)
        n = int(rhs_flat.shape[0])

        A = self._build_dense(A_mv, pack=pack, n=n, dtype=rhs_flat.dtype)

        if self.symmetrize:
            A = 0.5 * (A + A.T)

        if self.solve_dtype is not None:
            A_s = A.astype(self.solve_dtype)
            rhs_s = rhs_flat.astype(self.solve_dtype)
        else:
            A_s = A
            rhs_s = rhs_flat

        if self.jitter > 0.0:
            A_s = A_s + jnp.asarray(self.jitter, dtype=A_s.dtype) * jnp.eye(n, dtype=A_s.dtype)

        x_flat = self._solve_dense(A_s, rhs_s)

        if self.solve_dtype is not None:
            x_flat = x_flat.astype(rhs_flat.dtype)

        x = pack(x_flat)

        if self.info_mode is not None:
            mode = self.info_mode
        elif self.space == "row" and self.solve_method == "chol":
            mode = "row_cholesky"
        else:
            mode = "dense_direct"

        info: Dict[str, Any] = {
            "mode": mode,
            "dim": jnp.asarray(n, dtype=jnp.int32),
            "space": self.space,
            "vector_space": self.vector_space,
            "solve_method": self.solve_method,
        }

        if self.space == "row":
            info["row_dim"] = jnp.asarray(n, dtype=jnp.int32)
        else:
            info["n"] = jnp.asarray(n, dtype=jnp.int32)

        return x, info, state
