from .base import EstimatorPolicy
from .hutchinson import Hutchinson
from .gnb_ce import GaussNewtonBartlett
from .diagpp import DiagPP
from .lanczos import MFLanczos


def make_estimator(name: str, **kwargs) -> EstimatorPolicy:
    n = name.lower()
    if n in ("hutch", "hutchinson"): return Hutchinson(**kwargs)
    if n in ("gnb", "gnb_ce", "gaussnewtonbartlett"): return GaussNewtonBartlett(**kwargs)
    if n in ("diagpp", "hutchpp", "diag++", "hutch++"): return DiagPP(**kwargs)
    if n in ("lanczos", "mflanczos"): return MFLanczos(**kwargs)
    raise ValueError(f"Unknown estimator: {name}")


__all__ = [
    "make_estimator",
    "EstimatorPolicy",
    "Hutchinson",
    "GaussNewtonBartlett",
    "MFLanczos",
    "DiagPP",
]
