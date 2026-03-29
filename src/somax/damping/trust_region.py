from typing import Any, FrozenSet

import jax
import jax.numpy as jnp
from flax import struct

from .. import metrics as mx
from ..types import StepInfo
from ..utils import cadence
from .base import DampingPolicy, DampingState


@struct.dataclass
class TrustRegionAux:
    """Diagnostics for TR damping (stored in StepInfo by executor when requested)."""

    last_rho: jnp.ndarray
    last_action: jnp.ndarray  # int32: -1 dec, 0 keep, +1 inc
    lam_clipped: jnp.ndarray  # bool


@struct.dataclass
class TrustRegionDamping(DampingPolicy):
    """LM-style trust-region damping driven by gain ratio rho.

    Update rule:
      - if rho < lower: increase lambda by factor `inc`
      - if rho > upper: decrease lambda by factor `dec`
      - else: keep lambda

    Guards:
      - update only on cadence (every_k)
      - skip if rho is not finite or rho_valid is False
      - clip rho to [-rho_clip, rho_clip] before comparing
      - clamp lambda to [min_lam, max_lam]
    """

    lam0: float = 1.0
    lower: float = 0.25
    upper: float = 0.75
    inc: float = 1.5
    dec: float = 0.5
    every_k: int = 1
    min_lam: float = 1e-12
    max_lam: float = 1e6
    rho_clip: float = 5.0
    dtype: Any = jnp.float32

    def __post_init__(self):
        if self.every_k < 1:
            raise ValueError("every_k must be >= 1")
        if not (self.min_lam > 0.0):
            raise ValueError("min_lam must be > 0")
        if not (self.max_lam >= self.min_lam):
            raise ValueError("max_lam must be >= min_lam")

    def needed_metrics(self) -> FrozenSet[str]:
        return frozenset({mx.STEP, mx.RHO, mx.RHO_VALID})

    def init(self) -> DampingState:
        lam0 = jnp.asarray(self.lam0, self.dtype)
        lam0 = jnp.clip(
            lam0,
            jnp.asarray(self.min_lam, lam0.dtype),
            jnp.asarray(self.max_lam, lam0.dtype),
        )
        aux = TrustRegionAux(
            last_rho=jnp.asarray(jnp.nan, lam0.dtype),
            last_action=jnp.asarray(0, jnp.int32),
            lam_clipped=jnp.asarray(False, jnp.bool_),
        )
        return DampingState(lam=lam0, aux=aux)

    def update(self, state: DampingState, info: StepInfo) -> DampingState:
        step = info[mx.STEP]
        rho = jnp.asarray(info[mx.RHO], dtype=state.lam.dtype)
        rho_valid = jnp.asarray(info[mx.RHO_VALID], dtype=jnp.bool_)

        do = cadence(step, self.every_k)

        rho = jnp.clip(
            rho,
            -jnp.asarray(self.rho_clip, state.lam.dtype),
            jnp.asarray(self.rho_clip, state.lam.dtype),
        )
        finite = jnp.isfinite(rho)
        ok = jnp.logical_and(do, jnp.logical_and(finite, rho_valid))

        lower = jnp.asarray(self.lower, state.lam.dtype)
        upper = jnp.asarray(self.upper, state.lam.dtype)
        inc = jnp.asarray(self.inc, state.lam.dtype)
        dec = jnp.asarray(self.dec, state.lam.dtype)
        min_lam = jnp.asarray(self.min_lam, state.lam.dtype)
        max_lam = jnp.asarray(self.max_lam, state.lam.dtype)

        def _apply(_):
            act_inc = rho < lower
            act_dec = rho > upper

            action = jnp.where(
                act_inc,
                jnp.asarray(1, jnp.int32),
                jnp.where(act_dec, jnp.asarray(-1, jnp.int32), jnp.asarray(0, jnp.int32)),
            )

            lam1 = jnp.where(act_inc, state.lam * inc, state.lam)
            lam2 = jnp.where(act_dec, lam1 * dec, lam1)

            lam3 = jnp.clip(lam2, min_lam, max_lam)
            clipped = jnp.logical_or(lam3 == min_lam, lam3 == max_lam)

            aux2 = TrustRegionAux(last_rho=rho, last_action=action, lam_clipped=clipped)
            return DampingState(lam=lam3, aux=aux2)

        return jax.lax.cond(ok, _apply, lambda _: state, operand=None)
