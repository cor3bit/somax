from typing import Optional, Tuple, Any

import jax
import jax.numpy as jnp

from .base import PreconditionerPolicy, PrecondState
from ..types import Params, Updates, PRNGKey, Scalar, PrecondFn
from ..curvature.base import CurvatureOperator, CurvatureState

_DIAG_SENTINEL: Any = object()


class DiagDirectPrecond(PreconditionerPolicy):
    """
    Diagonal preconditioner from an instantaneous diagonal estimate.

        M^{-1} r = r / (diag [+ lambda] + eps)

    Sources
    -------
    - Preferred: caller provides `diag=...` to build() (decouples from CurvatureOperator).
    - Fallback: use op.diagonal(params, cstate, rng) if available.

    Args
    ----
    eps:
        Numerical stabilizer added to the diagonal (always).
    add_lambda:
        If True, add incoming damping `lam` (step damping) to the diagonal.
    clip_nonneg:
        If True, clip diagonal to be nonnegative before adding eps / lambda.
        Safety for noisy estimators that may produce small negatives.
    """

    def __init__(
            self,
            *,
            eps: float = 1e-8,
            add_lambda: bool = True,
            clip_nonneg: bool = True,
    ):
        self.eps = float(eps)
        self.add_lambda = bool(add_lambda)
        self.clip_nonneg = bool(clip_nonneg)

    def init(self, params: Params) -> PrecondState:
        return PrecondState(cache=())

    def build(
            self,
            params: Params,
            op: CurvatureOperator,
            cstate: CurvatureState,
            *,
            rng: Optional[PRNGKey],
            lam: Optional[Scalar],
            state: PrecondState,
            diag: Any = _DIAG_SENTINEL,
    ) -> Tuple[Optional[PrecondFn], PrecondState]:
        if diag is _DIAG_SENTINEL:
            if not hasattr(op, "diagonal"):
                raise ValueError(
                    "DiagDirectPrecond requires either `diag=...` or "
                    "CurvatureOperator.diagonal(params, cstate, rng)."
                )
            diag = op.diagonal(params, cstate, rng)

        eps = self.eps
        add_lambda = self.add_lambda
        clip_nonneg = self.clip_nonneg
        lam_val = lam  # may be None (Python-level static branch)

        def _shift_leaf(d):
            if clip_nonneg:
                d = jnp.maximum(d, jnp.asarray(0.0, d.dtype))
            if add_lambda and (lam_val is not None):
                d = d + jnp.asarray(lam_val, d.dtype)
            d = d + jnp.asarray(eps, d.dtype)
            return d

        denom = jax.tree_util.tree_map(_shift_leaf, diag)

        def M_inv(r: Updates) -> Updates:
            return jax.tree_util.tree_map(lambda ri, di: ri / di, r, denom)

        return M_inv, state
