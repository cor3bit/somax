# Import modules so @register runs (side effects are intentional).
from . import egn  # noqa: F401

# Re-export explicit callables, e.g., for "from somax.methods import egn_ce".
from .adahessian import adahessian
from .direct_methods import ggn_direct_ce, ggn_direct_mse, newton_direct
from .egn import egn_ce, egn_mse
from .newton_cg import newton_cg
from .sgn import sgn_mse, sgn_ce
from .sophia_g import sophia_g
from .sophia_h import sophia_h

__all__ = [
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
