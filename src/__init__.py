# Diagonal Scaling
from src.somax.diagonal.adahessian import AdaHessian
from src.somax.diagonal.sophia_g import SophiaG
from src.somax.diagonal.sophia_h import SophiaH

# Hessian Free Optimization
from src.somax.hf.newton_cg import NewtonCG

# Quasi-Newton Methods
from src.somax.qn.sqn import SQN

# Gauss-Newton Methods
from src.somax.gn.egn import EGN
from src.somax.gn.sgn import SGN

# Natural Gradient Methods
from src.somax.ng.swm_ng import SWMNG

__version__ = '0.0.1'
