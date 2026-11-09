from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from .. import metrics as mx
from ..curvature.base import RowOperator
from ..types import Scalar, Updates
from ..utils import tree_add_scaled, tree_norm2, tree_vdot, tree_zeros_like
from .base import NullSolverState, PrecondLike
from .row_cg import RowConjugateGradient
from .row_common import Reduction, mu_from_lambda


class LiftedRowConjugateGradient(RowConjugateGradient):
    """Unpreconditioned, unstabilized row CG with a lifted parameter direction.

    The structured entry point ``solve_row_direction`` solves the row system

        (L L.T + mu I) u = rhs,         mu = B * lam  (mean reduction),

    while accumulating ``d = L.T @ u`` in parameter space. At iteration k,
    the matvec already computes ``q = L.T @ p``; reusing that vector in
    ``d += alpha * q`` removes the final backprojection VJP.

    This implementation is intentionally limited to cold-start CG with the
    internal ``pcg`` backend (unpreconditioned) and no stabilization. All
    unsupported algorithmic settings are rejected at construction time.
    The structured entry point rejects explicit Gram shortcuts and unsupported
    dtypes instead of silently executing ordinary row CG.

    The inherited ``solve(A_mv, rhs, ...)`` still returns the ordinary row
    solution, for compatibility with the existing row-solver interface.
    Only ``solve_row_direction`` performs lifting. The output direction is
    unsigned and has no additional ``1/B`` scaling.

    Assumes a fixed adjoint pair (L, L.T) and an SPD row system. Equality to
    ordinary row CG at each finite iteration holds in exact arithmetic; the
    parameter-space accumulation order differs in floating point.
    """

    def __init__(
            self,
            *,
            maxiter: int = 10,
            tol: float = 1e-4,
            warm_start: bool = False,
            stabilise_every: int = 0,
            preconditioned: bool = False,
            solve_dtype: jnp.dtype | None = None,
            backend: str = "pcg",
            telemetry_residual: bool = False,
    ):
        # Preserve RowConjugateGradient's constructor signature while making
        # the implemented lifted-CG variant explicit.
        if preconditioned:
            raise NotImplementedError(
                "LiftedRowConjugateGradient: preconditioning not implemented."
            )
        if warm_start:
            raise NotImplementedError(
                "LiftedRowConjugateGradient: row-space warm start is not implemented."
            )
        if int(stabilise_every) != 0:
            raise NotImplementedError(
                "LiftedRowConjugateGradient: residual stabilization is not implemented; "
                "set stabilise_every=0 or use RowConjugateGradient."
            )
        if backend != "pcg":
            raise NotImplementedError(
                "LiftedRowConjugateGradient: only backend='pcg' (internal CG) "
                "is implemented."
            )
        if solve_dtype is not None:
            raise NotImplementedError(
                "LiftedRowConjugateGradient: solve_dtype casting is not implemented."
            )

        super().__init__(
            maxiter=maxiter,
            tol=tol,
            warm_start=False,
            stabilise_every=0,
            preconditioned=False,
            solve_dtype=None,
            backend="pcg",
            telemetry_residual=telemetry_residual,
        )
        self._lift_maxiter = int(maxiter)
        self._lift_tol = float(tol)

    def solve_row_direction(
            self,
            row_op: RowOperator,
            lam: Scalar,
            *,
            params: Updates,
            state: Any,
            reduction: Reduction = "mean",
            precond: PrecondLike | None = None,
    ) -> tuple[Updates, dict[str, Any], Any]:
        """Return ``(unsigned_parameter_direction, CG_info, solver_state)``.

        ``params`` is used only as a PyTree shape/dtype template, not as a
        gradient or initialization guess. For mean reduction, ``mu = B*lam``;
        for sum reduction, ``mu = lam``. There is no extra output scaling.
        """
        del state
        if precond is not None:
            raise NotImplementedError(
                "LiftedRowConjugateGradient: preconditioning is not implemented."
            )
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'.")
        if getattr(row_op, "gram_mv", None) is not None:
            raise NotImplementedError(
                "LiftedRowConjugateGradient: explicit Gram shortcuts are not "
                "supported. Use RowConjugateGradient for Gram-based solves."
            )

        rhs = jnp.asarray(row_op.rhs)
        if rhs.ndim != 1:
            raise ValueError("solve_row_direction expects rhs with shape (m,).")

        lam = jnp.asarray(lam)
        if lam.ndim != 0:
            raise ValueError("lam must be a scalar.")

        mu = mu_from_lambda(lam, b=int(row_op.b), reduction=reduction)

        if mu.dtype != rhs.dtype:
            raise TypeError("Row damping and RHS must have the same dtype.")

        direction, info = _lifted_cg(
            row_op, rhs, mu, params, tol=self._lift_tol, maxiter=self._lift_maxiter
        )
        return direction, info, NullSolverState()


def _lifted_cg(
        row_op: RowOperator,
        rhs: jax.Array,
        mu: Scalar,
        params: Updates,
        *,
        tol: float,
        maxiter: int,
) -> tuple[Updates, dict[str, jax.Array]]:
    """Cold-start, unstabilized CG; return L.T @ u without a final VJP."""
    tiny = jnp.finfo(rhs.dtype).tiny
    b2 = jnp.maximum(tree_norm2(rhs), tiny)
    tol2 = (tol * tol) * b2
    d0 = tree_zeros_like(params)
    r0 = rhs
    rTr0 = tree_vdot(r0, r0)
    k0 = jnp.asarray(0, jnp.int32)

    def cond(carry):
        _d, _r, _p, rTr, k = carry
        return jnp.logical_and(rTr > tol2, k < maxiter)

    def body(carry):
        d, r, p, rTr, k = carry
        q = row_op.vjp(p)
        Ap = row_op.jvp(q) + mu * p
        alpha = rTr / tree_vdot(p, Ap)

        d_new = tree_add_scaled(d, alpha, q)
        r_new = tree_add_scaled(r, -alpha, Ap)
        rTr_new = tree_vdot(r_new, r_new)
        beta = rTr_new / rTr
        p_new = tree_add_scaled(r_new, beta, p)
        return d_new, r_new, p_new, rTr_new, k + 1

    d, _r, _p, rTr, k = lax.while_loop(cond, body, (d0, r0, r0, rTr0, k0))
    info = {
        mx.CG_ITERS: k,
        mx.CG_MAXITER: jnp.asarray(maxiter, k.dtype),
        mx.CG_RESID: jnp.sqrt(rTr / b2),
        mx.CG_CONVERGED: rTr <= tol2,
    }
    return d, info
