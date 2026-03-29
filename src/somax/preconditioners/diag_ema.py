from typing import Optional, Tuple, Any, Callable

import jax
import jax.numpy as jnp
from flax import struct

from .base import PreconditionerPolicy, PrecondState, PrecondFn
from ..types import Params, Updates, PRNGKey, Scalar
from ..curvature.base import CurvatureState, CurvatureOperator


@struct.dataclass
class DiagEMAState(PrecondState):
    step: jnp.ndarray
    t_ema: jnp.ndarray  # number of EMA refreshes
    v: Updates  # EMA accumulator (raw or squared)


class DiagEMAPrecond(PreconditionerPolicy):
    """EMA diagonal preconditioner.

    Modes
    -----
    - AdaHessian-style (gamma is None):
        contrib = (diag_est)^2, denom = (v_hat)^(alpha/2) + eps
    - Sophia-H-style (gamma is not None):
        contrib = diag_est, denom = max(gamma * v_hat, eps)

    Diagonal source
    --------------
    By default, uses op.diagonal(params, cstate, rng) if available.
    You may supply a custom `diag_fn` to decouple diagonal estimation from the
    curvature operator (e.g. grad2-based or shared diag pipelines).

    Notes
    -----
    This is curvature/diag scaling, not a Jacobi preconditioner for (C + lam I)
    unless your diag source matches that intent.
    """

    def __init__(
            self,
            *,
            beta1: float | None = None,
            beta2: float = 0.999,
            hessian_power: float = 1.0,  # AdaHessian only (alpha)
            eps: float = 1e-8,
            spatial_averaging: bool = False,
            eval_every_k: int = 1,
            gamma: float | None = None,  # if set -> Sophia-H denom: max{gamma*h_hat, eps}
            add_lambda: bool = False,  # if True, add damping lambda to denom
            diag_fn: Optional[Callable[[Params, CurvatureOperator, CurvatureState, PRNGKey], Updates]] = None,
    ):
        if beta1 is not None:
            assert 0.0 <= beta1 <= 1.0
        assert 0.0 <= beta2 <= 1.0

        self.eval_every_k = int(eval_every_k)
        assert self.eval_every_k >= 1

        self.beta1 = None if beta1 is None else float(beta1)
        self.beta2 = jnp.asarray(beta2, jnp.float32)
        self.alpha = float(hessian_power)
        self.eps = float(eps)
        self.spatial_averaging = bool(spatial_averaging)

        self.gamma = None if gamma is None else jnp.asarray(gamma, jnp.float32)
        self.square_estimate = self.gamma is None
        self.add_lambda = bool(add_lambda)

        self._diag_fn = diag_fn

    def init(self, params: Params) -> DiagEMAState:
        zeros_like = jax.tree_util.tree_map(jnp.zeros_like, params)
        return DiagEMAState(
            step=jnp.array(0, jnp.int32),
            t_ema=jnp.array(0, jnp.int32),
            v=zeros_like,
            cache=(),
        )

    def _diag_est(
            self,
            params: Params,
            op: CurvatureOperator,
            cstate: CurvatureState,
            rng: PRNGKey,
    ) -> Updates:
        if self._diag_fn is not None:
            return self._diag_fn(params, op, cstate, rng)
        if not hasattr(op, "diagonal"):
            raise ValueError(
                "DiagEMAPrecond requires either `diag_fn` or CurvatureOperator.diagonal(...)."
            )
        return op.diagonal(params, cstate, rng)

    def build(
            self,
            params: Params,
            op: CurvatureOperator,
            cstate: CurvatureState,
            *,
            rng: Optional[PRNGKey],
            lam: Optional[Scalar],
            state: DiagEMAState,
    ) -> Tuple[Optional[PrecondFn], PrecondState]:
        if rng is None:
            raise ValueError("DiagEMAPrecond.build requires rng (used for probe/diag estimation).")

        step = state.step
        do_refresh = (step % jnp.asarray(self.eval_every_k, step.dtype)) == 0

        def _refresh(_):
            k = jax.random.fold_in(rng, step)
            diag_est = self._diag_est(params, op, cstate, k)

            if self.spatial_averaging:
                diag_est = jax.tree_util.tree_map(_spatial_average_leaf, diag_est)

            if self.square_estimate:
                contrib = jax.tree_util.tree_map(lambda d: d * d, diag_est)
            else:
                contrib = diag_est

            v_new = jax.tree_util.tree_map(
                lambda v_old, c: self.beta2 * v_old + (1.0 - self.beta2) * c,
                state.v,
                contrib,
            )
            return v_new, state.t_ema + jnp.array(1, state.t_ema.dtype)

        v_new, t_ema_next = jax.lax.cond(
            do_refresh,
            _refresh,
            lambda _: (state.v, state.t_ema),
            operand=None,
        )

        # Bias correction by number of refreshes
        b2c = jnp.where(
            t_ema_next > 0,
            1.0 - jnp.power(self.beta2, t_ema_next.astype(jnp.float32)),
            1.0,
        )
        v_hat = jax.tree_util.tree_map(lambda vv: vv / b2c, v_new)

        eps = self.eps

        if self.gamma is not None:
            denom = jax.tree_util.tree_map(
                lambda h: jnp.maximum(self.gamma * h, jnp.asarray(eps, h.dtype)),
                v_hat,
            )
        else:
            half_alpha = 0.5 * self.alpha
            denom = jax.tree_util.tree_map(
                lambda vv: jnp.power(jnp.clip(vv, min=0.0), half_alpha) + jnp.asarray(eps, vv.dtype),
                v_hat,
            )

        if self.add_lambda and lam is not None:
            denom = jax.tree_util.tree_map(lambda d: d + jnp.asarray(lam, d.dtype), denom)

        def M_inv(g: Updates) -> Updates:
            return jax.tree_util.tree_map(lambda gg, dd: gg / dd, g, denom)

        new_state = DiagEMAState(
            step=step + jnp.array(1, step.dtype),
            t_ema=t_ema_next,
            v=v_new,
            cache=(),
        )
        return M_inv, new_state


def _spatial_average_leaf(leaf: jnp.ndarray) -> jnp.ndarray:
    if leaf.ndim == 1:
        return leaf
    if leaf.ndim in (2, 3):
        return jnp.mean(leaf, axis=-1, keepdims=True)
    if leaf.ndim == 4:
        return jnp.mean(leaf, axis=(-2, -1), keepdims=True)
    return leaf
