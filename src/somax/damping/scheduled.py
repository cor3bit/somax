"""Deterministic schedule-based damping."""

import math
from typing import Any, FrozenSet

import jax
import jax.numpy as jnp
import optax
from flax import struct

from .. import metrics as mx
from ..types import StepInfo
from .base import DampingPolicy, DampingState


@struct.dataclass
class ScheduledDamping(DampingPolicy):
    """Scale a reference damping value with a step-indexed schedule.

    At outer step ``t`` the linear system uses

    ``lambda_t = lam0 * factor_schedule(t)``.

    The schedule follows the same callable convention as an Optax learning-rate
    schedule: it receives a scalar integer step and returns a scalar factor.
    Consequently, standard Optax schedules can be passed directly.

    The executor calls :meth:`update` after completing outer step ``t``.  The
    returned state therefore stores ``lambda_{t+1}``, ready for the next solve.

    ``factor_schedule`` must return finite, non-negative scalar values.  Somax
    deliberately does not clip or otherwise modify the schedule output: the
    configured schedule is the damping policy.
    """

    factor_schedule: optax.Schedule = struct.field(pytree_node=False)
    lam0: float = 1.0
    dtype: Any = jnp.float32

    def __post_init__(self) -> None:
        if not callable(self.factor_schedule):
            raise TypeError("factor_schedule must be callable")
        if not (math.isfinite(self.lam0) and self.lam0 >= 0.0):
            raise ValueError("lam0 must be finite and non-negative")

    def needed_metrics(self) -> FrozenSet[str]:
        return frozenset({mx.STEP})

    def _lam_at(self, step: jax.Array, *, dtype: Any | None = None) -> jax.Array:
        out_dtype = self.dtype if dtype is None else dtype
        factor = jnp.asarray(self.factor_schedule(step), dtype=out_dtype)
        lam0 = jnp.asarray(self.lam0, dtype=out_dtype)
        return lam0 * factor

    def init(self) -> DampingState:
        step0 = jnp.asarray(0, jnp.int32)
        return DampingState(lam=self._lam_at(step0), aux=())

    def update(self, state: DampingState, info: StepInfo) -> DampingState:
        step = jnp.asarray(info[mx.STEP], jnp.int32)
        next_step = step + jnp.asarray(1, jnp.int32)
        lam_next = self._lam_at(next_step, dtype=state.lam.dtype)
        return DampingState(lam=lam_next, aux=())
