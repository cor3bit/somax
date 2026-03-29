"""Stateful preconditioners that produce M^{-1} for (C + lambda*I)s = g."""

from .base import PreconditionerPolicy, PrecondState, PrecondFn
from .diag_ema import DiagEMAPrecond
from .diag_direct import DiagDirectPrecond
from .identity import IdentityPrecond

__all__ = [
    "PreconditionerPolicy",
    "PrecondState",

    "IdentityPrecond",
    "DiagEMAPrecond",
    "DiagDirectPrecond",
]
