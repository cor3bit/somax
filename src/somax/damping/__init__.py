"""Damping policies for Somax."""

from .base import DampingPolicy, DampingState
from .constant import ConstantDamping
from .trust_region import TrustRegionDamping

__all__ = [
    "DampingPolicy",
    "DampingState",
    "ConstantDamping",
    "TrustRegionDamping",
]
