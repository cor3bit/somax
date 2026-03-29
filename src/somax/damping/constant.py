from typing import Any, FrozenSet

import jax.numpy as jnp
from flax import struct

from ..types import StepInfo
from .base import DampingPolicy, DampingState


@struct.dataclass
class ConstantDamping(DampingPolicy):
    """Constant Levenberg-Marquardt damping.

    Keeps lambda fixed at initialization value.
    """

    lam0: float = 1.0
    dtype: Any = jnp.float32

    def needed_metrics(self) -> FrozenSet[str]:
        return frozenset()

    def init(self) -> DampingState:
        lam0 = jnp.asarray(self.lam0, self.dtype)
        return DampingState(lam=lam0, aux=())

    def update(self, state: DampingState, info: StepInfo) -> DampingState:
        return state
