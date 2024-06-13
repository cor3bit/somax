# Diagonal Scaling
from somax.diagonal.adahessian import AdaHessian
from somax.diagonal.sophia_g import SophiaG
from somax.diagonal.sophia_h import SophiaH

# Hessian Free Optimization
from somax.hf.newton_cg import NewtonCG

# Quasi-Newton Methods
from somax.qn.sqn import SQN

# Gauss-Newton Methods
from somax.gn.egn import EGN
from somax.gn.sgn import SGN

# Natural Gradient Methods
from somax.ng.swm_ng import SWMNG

__version__ = '0.0.1'
