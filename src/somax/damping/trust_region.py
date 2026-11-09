"""Reactive gain-ratio damping.

The module name and ``TrustRegionDamping`` alias are retained for backwards
compatibility.  The implemented policy does not reject or retry steps.
"""

import math
from typing import Any, FrozenSet

import jax
import jax.numpy as jnp
from flax import struct

from .. import metrics as mx
from ..types import StepInfo
from ..utils import cadence
from .base import DampingPolicy, DampingState


@struct.dataclass
class GainRatioDampingAux:
    """Per-step diagnostics for gain-ratio damping."""

    action: jnp.ndarray       # int32: -1 decrease, 0 keep/skip, +1 increase
    lam_clipped: jnp.ndarray  # bool: proposed lambda was clipped to its bounds


@struct.dataclass
class GainRatioDamping(DampingPolicy):
    """Choose the *next step's* damping from the current gain ratio.

    This is a reactive damping controller, not a transactional trust-region or
    classical Levenberg-Marquardt method.  At outer step ``t`` the executor:

    1. solves with ``lambda_t``;
    2. always commits the candidate parameter, Optax, and model-state updates;
    3. measures the same-batch gain ratio ``rho_t``;
    4. calls this controller to produce ``lambda_{t+1}``.

    Update rule:

    - ``rho_t < lower``: increase damping by ``inc``;
    - ``rho_t > upper``: decrease damping by ``dec``;
    - otherwise: keep damping unchanged.

    Invalid/nonfinite ratios are ignored.  The ratio is clipped before threshold
    comparisons, and lambda is clamped to ``[min_lam, max_lam]``.
    """

    lam0: float = 1.0
    lower: float = 0.25
    upper: float = 0.75
    inc: float = 1.5
    dec: float = 0.5
    every_k: int = 1
    min_lam: float = 1e-12
    max_lam: float = 1e6
    gain_ratio_clip: float = 5.0
    dtype: Any = jnp.float32

    def __post_init__(self):
        if self.every_k < 1:
            raise ValueError("every_k must be >= 1")
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
            raise ValueError("lower and upper must be finite")
        if not self.lower < self.upper:
            raise ValueError("lower must be strictly smaller than upper")
        if not (math.isfinite(self.inc) and self.inc > 1.0):
            raise ValueError("inc must be finite and > 1")
        if not (math.isfinite(self.dec) and 0.0 < self.dec < 1.0):
            raise ValueError("dec must be finite and lie in (0, 1)")
        if not (math.isfinite(self.min_lam) and self.min_lam > 0.0):
            raise ValueError("min_lam must be finite and > 0")
        if not (math.isfinite(self.max_lam) and self.max_lam >= self.min_lam):
            raise ValueError("max_lam must be finite and >= min_lam")
        if not (math.isfinite(self.gain_ratio_clip) and self.gain_ratio_clip > 0.0):
            raise ValueError("gain_ratio_clip must be finite and > 0")
        if not math.isfinite(self.lam0):
            raise ValueError("lam0 must be finite")

    def needed_metrics(self) -> FrozenSet[str]:
        return frozenset({mx.STEP, mx.GAIN_RATIO, mx.GAIN_RATIO_VALID})

    def init(self) -> DampingState:
        lam0 = jnp.asarray(self.lam0, self.dtype)
        lam0 = jnp.clip(
            lam0,
            jnp.asarray(self.min_lam, lam0.dtype),
            jnp.asarray(self.max_lam, lam0.dtype),
        )
        aux = GainRatioDampingAux(
            action=jnp.asarray(0, jnp.int32),
            lam_clipped=jnp.asarray(False, jnp.bool_),
        )
        return DampingState(lam=lam0, aux=aux)

    def update(self, state: DampingState, info: StepInfo) -> DampingState:
        step = info[mx.STEP]
        gain_ratio_raw = jnp.asarray(
            info[mx.GAIN_RATIO], dtype=state.lam.dtype
        )
        gain_ratio_valid = jnp.asarray(info[mx.GAIN_RATIO_VALID], dtype=jnp.bool_)

        do_update = cadence(step, self.every_k)
        finite = jnp.isfinite(gain_ratio_raw)
        gain_ratio = jnp.clip(
            gain_ratio_raw,
            -jnp.asarray(self.gain_ratio_clip, state.lam.dtype),
            jnp.asarray(self.gain_ratio_clip, state.lam.dtype),
        )
        valid = jnp.logical_and(
            do_update,
            jnp.logical_and(finite, gain_ratio_valid),
        )

        lower = jnp.asarray(self.lower, state.lam.dtype)
        upper = jnp.asarray(self.upper, state.lam.dtype)
        inc = jnp.asarray(self.inc, state.lam.dtype)
        dec = jnp.asarray(self.dec, state.lam.dtype)
        min_lam = jnp.asarray(self.min_lam, state.lam.dtype)
        max_lam = jnp.asarray(self.max_lam, state.lam.dtype)

        def _apply(_):
            increase = gain_ratio < lower
            decrease = gain_ratio > upper

            action = jnp.where(
                increase,
                jnp.asarray(1, jnp.int32),
                jnp.where(
                    decrease,
                    jnp.asarray(-1, jnp.int32),
                    jnp.asarray(0, jnp.int32),
                ),
            )

            proposed = jnp.where(increase, state.lam * inc, state.lam)
            proposed = jnp.where(decrease, proposed * dec, proposed)
            lam_next = jnp.clip(proposed, min_lam, max_lam)
            clipped = lam_next != proposed

            return DampingState(
                lam=lam_next,
                aux=GainRatioDampingAux(action=action, lam_clipped=clipped),
            )

        def _skip(_):
            return DampingState(
                lam=state.lam,
                aux=GainRatioDampingAux(
                    action=jnp.asarray(0, jnp.int32),
                    lam_clipped=jnp.asarray(False, jnp.bool_),
                ),
            )

        return jax.lax.cond(valid, _apply, _skip, operand=None)


# Backwards-compatible names.  New code should use ``GainRatioDamping`` and
# ``lam_policy="gain_ratio"`` to avoid implying step rejection.
TrustRegionDamping = GainRatioDamping
TrustRegionAux = GainRatioDampingAux
