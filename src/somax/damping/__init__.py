"""Damping policies for Somax."""

from .base import DampingPolicy, DampingState
from .constant import ConstantDamping
from .scheduled import ScheduledDamping
from .trust_region import GainRatioDamping, TrustRegionDamping

__all__ = [
    "DampingPolicy",
    "DampingState",
    "ConstantDamping",
    "ScheduledDamping",
    "GainRatioDamping",
    "TrustRegionDamping",
]
