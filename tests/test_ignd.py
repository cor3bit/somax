import pytest

import numpy as np

import jax
import jax.numpy as jnp

from somax import IGND
from utils import load_iris, load_california, MLPRegressorMini, MLPClassifierMini


def test_ignd_mse():
    @jax.jit
    def mse(params, x, y):
        residuals = y - predict_fn(params, x)
        return 0.5 * jnp.mean(jnp.square(residuals))

    # --------------- jax setup ---------------
    # jax.config.update("jax_disable_jit", True)

    # --------------- dataset & models ---------------
    seed = 1337
    rng = jax.random.PRNGKey(seed)
    b = 32

    X_train, X_test, Y_train, Y_test = load_california()

    # model
    model = MLPRegressorMini()
    params = model.init(rng, X_train[0])
    predict_fn = jax.jit(model.apply)

    # --------------- init solvers ---------------

    solver = IGND(
        predict_fun=predict_fn,
        loss_type='mse',
        learning_rate=1.0,
        regularizer=0.1,
        batch_size=b,
    )

    opt_state = solver.init_state(params)

    loss_t0 = mse(params, X_test, Y_test)

    # update
    test_set_loss = [loss_t0]
    for i in range(5):
        batch_X = X_train[i * b:(i + 1) * b, :]
        batch_y = Y_train[i * b:(i + 1) * b]

        params, opt_state = solver.update(params, opt_state, batch_X, targets=batch_y)

        test_set_loss.append(mse(params, X_test, Y_test))

    realized_losses = jnp.array(test_set_loss)

    actual_losses = jnp.array(np.array([
        3.1070921, 0.9547686, 0.56324023, 0.5226982, 0.48368528, 0.4937724,
    ]))

    assert jnp.allclose(realized_losses, actual_losses, atol=1e-6), "Step size mismatch"


@pytest.mark.skip(reason="Not implemented")
def test_ignd_ce():
    raise NotImplementedError
