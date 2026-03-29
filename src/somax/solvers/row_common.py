"""Row-space linear system helpers.

Given a RowOperator and damping lambda, bind the row-space system:

  (J J^T + mu I) u = rhs

where:
  - rhs is a row-space vector of shape (m,)
  - J is the implicit row Jacobian defined by RowOperator.jvp/vjp
  - mu is derived from lambda and the reduction convention:
      reduction="mean": mu = B * lam
      reduction="sum":  mu = lam

The solver returns u. The binder also returns a backprojection function
that maps row-space u to a parameter-space direction:

  d = backproject(u)

Convention:
- This binder constructs backproject from row_op.vjp only.
- Any extra scaling (e.g., 1/B for mean reduction) and the choice of sign
  (whether d is a descent direction or a solve result) is handled outside
  this module (executor / optimizer).
"""

from typing import Callable, Literal, Tuple

import jax.numpy as jnp

from ..curvature.base import RowOperator
from ..types import Array, Scalar, Updates

Reduction = Literal["mean", "sum"]


def mu_from_lambda(lam: Scalar, *, b: int, reduction: Reduction) -> Scalar:
    """Convert parameter-space damping lambda to row-space mu.

    Conventions:
      - mean reduction (1/B): mu = B * lam
      - sum reduction:        mu = lam
    """
    if reduction == "mean":
        return lam * jnp.asarray(b, dtype=lam.dtype)
    return lam


def bind_row_system(
        row_op: RowOperator,
        lam: Scalar,
        *,
        reduction: Reduction = "mean",
) -> Tuple[Callable[[Array], Array], Array, Callable[[Array], Updates], Scalar]:
    """Bind row-space matvec, rhs, backproject, and mu.

    Returns:
      A_mv: u -> (J J^T + mu I) u
      rhs: row_op.rhs
      backproject: u -> J^T u   (unscaled; old convention)
      mu: row damping used
    """
    b = int(row_op.b)
    mu = mu_from_lambda(lam, b=b, reduction=reduction)

    def A_mv(u: Array) -> Array:
        # (J J^T) u = J (J^T u)
        p = row_op.vjp(u)  # Updates (param pytree)
        ju = row_op.jvp(p)  # (m,) row vector
        return ju + mu * u

    return A_mv, row_op.rhs, row_op.vjp, mu
