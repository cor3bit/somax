from .assembler import assemble, SecondOrderMethod, SecondOrderState
from .presets import make, list_methods, describe

# Import methods once so @register runs and explicit methods exist.
from . import methods as _methods  # noqa: F401

from importlib.metadata import PackageNotFoundError, version as _version

for _dist_name in ("python-somax", "somax"):
    try:
        __version__ = _version(_dist_name)
        break
    except PackageNotFoundError:
        continue
else:
    __version__ = "0+unknown"

# Re-export explicit methods at the top level.
from .methods import (
    adahessian,
    ggn_direct_mse,
    ggn_direct_ce,
    newton_direct,
    egn_ce,
    egn_mse,
    newton_cg,
    sgn_mse,
    sgn_ce,
    sophia_g,
    sophia_h,
)

__all__ = [
    "__version__",

    # Low-level construction
    "assemble",
    "SecondOrderMethod",
    "SecondOrderState",

    # Mid-level interface for CLI/config
    "make",
    "list_methods",
    "describe",

    # High-level standard methods for direct import
    "adahessian",
    "ggn_direct_mse",
    "ggn_direct_ce",
    "newton_direct",
    "egn_ce",
    "egn_mse",
    "newton_cg",
    "sgn_mse",
    "sgn_ce",
    "sophia_g",
    "sophia_h",
]
