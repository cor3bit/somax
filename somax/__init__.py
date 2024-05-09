# Diagonal Scaling
# TODO

# Hessian Free Optimization
from somax.hf.newton_cg import NewtonCG

# Quasi-Newton Methods
# TODO

# Gauss-Newton Methods
from somax.gn.egn import EGN
from somax.gn.fast_egn import FastEGN
from somax.gn.egn_probs import EGNProb
from somax.gn.gnb import GNB
from somax.gn.ignd import IGND
from somax.gn.lb import LB
from somax.gn.sgn import SGN

# Natural Gradient Methods
from somax.ng.swm_ng import SWMNG

__version__ = '0.0.1.dev'
