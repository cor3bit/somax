from typing import Optional, Tuple

from ..types import Params, PRNGKey, Scalar, PrecondFn
from ..curvature.base import CurvatureState, CurvatureOperator
from .base import PrecondState, PreconditionerPolicy


class IdentityPrecond(PreconditionerPolicy):
    """A no-op preconditioner.

    Contract:
      - build() returns (None, state), meaning "no preconditioning"
      - state is stable and does not depend on params/op/cstate
      - safe to use in any lane that supports identity/no-preconditioning
    """

    is_identity: bool = True

    def init(self, params: Params) -> PrecondState:
        # params ignored by design; keep a pytree-stable state.
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
    ) -> Tuple[Optional[PrecondFn], PrecondState]:
        # All inputs ignored: identity preconditioner is constant.
        # Return the same state object to avoid churn in the outer state pytree.
        return None, state
