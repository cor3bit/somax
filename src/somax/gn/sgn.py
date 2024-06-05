"""
Stochastic Gauss-Newton (SGN):
- approximate solution to the linear system via the CG method
- adaptive regularization (lambda)
Paper: https://arxiv.org/abs/2006.02409
"""

from typing import Any
from typing import Callable
from typing import NamedTuple, Tuple
from typing import Optional
import dataclasses
from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg
from optax import sigmoid_binary_cross_entropy
from jaxopt.tree_util import tree_add_scalar_mul, tree_scalar_mul
from jaxopt._src import base
from jaxopt._src import loop


class SGNState(NamedTuple):
    """Named tuple containing state information."""
    iter_num: int
    # error: float
    # value: float
    stepsize: float
    regularizer: float
    # direction_inf_norm: float


@dataclasses.dataclass(eq=False)
class SGN(base.StochasticSolver):
    predict_fun: Callable

    loss_fun: Optional[Callable] = None
    loss_type: str = 'mse'
    maxcg: int = 10

    learning_rate: float = 1.0  # default value recommended by Gargiani et al.

    batch_size: Optional[int] = None

    n_classes: Optional[int] = None

    # Adaptive Regularization parameters
    adaptive_lambda: bool = False
    regularizer: float = 1e-3  # default value recommended by Gargiani et al.
    lambda_decrease_factor: float = 0.99  # default value recommended by Kiros
    lambda_increase_factor: float = 1.01  # default value recommended by Kiros

    pre_update: Optional[Callable] = None

    verbose: int = 0

    jit: bool = True
    unroll: base.AutoOrBoolean = "auto"

    def __post_init__(self):
        super().__post_init__()

        self.reference_signature = self.loss_fun

        # Regression (MSE)
        if self.loss_type == 'mse':
            if self.loss_fun is None:  # mostly the case for supervised learning
                self.loss_fun = self.mse

            self.loss_fun_from_preds = self.mse_from_predictions

        # Classification (Cross-Entropy)
        elif self.loss_type == 'ce' or self.loss_type == 'xe':
            if self.n_classes == 1:  # binary classification
                if self.loss_fun is None:  # mostly the case for supervised learning
                    self.loss_fun = self.ce_binary

                self.loss_fun_from_preds = self.ce_binary_from_logits

            else:
                if self.loss_fun is None:  # mostly the case for supervised learning
                    self.loss_fun = self.ce

                self.loss_fun_from_preds = self.ce_from_logits

        self.grad_fun = jax.grad(self.loss_fun)

    def update(
            self,
            params: Any,
            state: SGNState,
            *args,
            **kwargs,
    ) -> base.OptStep:
        """Performs one iteration of the solver.

        Args:
          params: pytree containing the parameters.
          state: named tuple containing the solver state.
          *args: additional positional arguments to be passed to ``fun``.
          **kwargs: additional keyword arguments to be passed to ``fun``.
        Returns:
          (params, state)
        """

        # ---------- STEP 1: calculate direction with DG ---------- #
        # TODO analyze *args and **kwargs
        # split (x,y) pair into (x,) and (y,)
        if 'targets' in kwargs:
            targets = kwargs['targets']
            nn_args = args
        else:
            targets = args[-1]
            nn_args = args[:-1]

        direction_tree, grad_loss_tree = self.calculate_direction(params, state, targets, *nn_args)

        # # ---------- STEP 2: update (next step) lambda ---------- #
        # TODO

        next_params = tree_add_scalar_mul(params, state.stepsize, direction_tree)

        # construct the next state
        next_state = SGNState(
            iter_num=state.iter_num + 1,  # Next Iteration
            stepsize=state.stepsize,  # Current alpha
            regularizer=state.regularizer,  # Next lambda
        )

        return base.OptStep(params=next_params, state=next_state)

    def init_state(self,
                   init_params: Any,
                   *args,
                   **kwargs) -> SGNState:
        return SGNState(
            iter_num=0,
            stepsize=self.learning_rate,
            regularizer=self.regularizer,
        )

    def optimality_fun(self, params, *args, **kwargs):
        """Optimality function mapping compatible with ``@custom_root``."""
        # return self._grad_fun(params, *args, **kwargs)[0]
        raise NotImplementedError

    def gnhvp(self, params, vec, targets, *args):
        def inner_predict_fun(params):
            return self.predict_fun(params, *args)

        def hvp(network_outputs, vec, targets):
            # prep grad function to accept only "network_outputs"
            def inner_grad_fun(network_outputs):
                return jax.grad(self.loss_fun_from_preds)(network_outputs, targets)

            return jax.jvp(inner_grad_fun, (network_outputs,), (vec,))[1]

        # H_GN ~ J^T Q J, s.t. hvp is J^T Q J v
        network_outputs, Jv = jax.jvp(inner_predict_fun, (params,), (vec,))

        if self.loss_type == 'mse':
            _, JTJv_fun = jax.vjp(inner_predict_fun, params)
            JTJv = JTJv_fun(Jv)[0]
            return tree_scalar_mul(1 / self.batch_size, JTJv)
        else:
            QJv = hvp(network_outputs, Jv, targets)
            _, JTQJv_fun = jax.vjp(inner_predict_fun, params)
            JTQJv = JTQJv_fun(QJv)[0]
            return JTQJv

    def calculate_direction(self, params, state, targets, *args):
        def mvp(vec):
            # H_{GN}v
            # hv = jax.jvp(jax.grad(stand_alone_loss_fn), (params,), (vec,))[1]
            hv = self.gnhvp(params, vec, targets, *args)
            # add regularization, works since (H + lambda*I) v = Hv + lambda*v
            return tree_add_scalar_mul(hv, state.regularizer, vec)

        # --------- Start Here --------- #
        # calculate grad
        grad_tree = self.grad_fun(params, *args, targets)

        # CG iterations
        # TODO initial guess and preconditioner
        direction, _ = cg(
            A=mvp,
            b=tree_scalar_mul(-1, grad_tree),
            maxiter=self.maxcg,
            # x0=None,  # initial guess
            # M=None,  # preconditioner
        )

        return direction, grad_tree

    def mse(self, params, x, y):
        return self.mse_from_predictions(self.predict_fun(params, x), y)

    @staticmethod
    def mse_from_predictions(preds, y):
        return 0.5 * jnp.mean(jnp.square(y - preds))

    def ce(self, params, x, y):
        logits = self.predict_fun(params, x)
        return self.ce_from_logits(logits, y)

    @staticmethod
    def ce_from_logits(logits, y):
        # b x C
        # jax.nn.log_softmax combines exp() and log() in a numerically stable way.
        log_probs = jax.nn.log_softmax(logits)

        # b x 1
        # if y is one-hot encoded, this operation picks the log probability of the correct class
        residuals = jnp.sum(y * log_probs, axis=-1)

        # 1,
        # average over the batch
        return -jnp.mean(residuals)

    def ce_binary(self, params, x, y):
        logits = self.predict_fun(params, x)
        return self.ce_binary_from_logits(logits, y)

    def ce_binary_from_logits(self, logits, y):
        # b x 1
        loss = sigmoid_binary_cross_entropy(logits.ravel(), y)

        # 1,
        # average over the batch
        return jnp.mean(loss)

    def __hash__(self):
        # We assume that the attribute values completely determine the solver.
        return hash(self.attribute_values())
