from .base import EstimatorPolicy
from .hutchinson import Hutchinson
from .gnb_ce import GaussNewtonBartlett


def make_estimator(name: str, **kwargs) -> EstimatorPolicy:
    n = name.lower()
    if n in ("hutch", "hutchinson"): return Hutchinson(**kwargs)
    if n in ("gnb", "gnb_ce", "gaussnewtonbartlett"): return GaussNewtonBartlett(**kwargs)
    raise ValueError(f"Unknown estimator: {name}")


__all__ = [
    "make_estimator",
    "EstimatorPolicy",
    "Hutchinson",
    "GaussNewtonBartlett",
]
