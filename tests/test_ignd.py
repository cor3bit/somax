import random
import pytest

import numpy as np

import jax
import jax.numpy as jnp

from egn.ignd import IGND
from egn import load_data
from egn import model_zoo as zoo


def test_ignd_mse():
    @jax.jit
    def mse(params, x, y):
        residuals = y - predict_fn(params, x)
        return 0.5 * jnp.mean(jnp.square(residuals))

    # --------------- jax setup ---------------
    # jax.config.update("jax_disable_jit", True)

    # --------------- dataset & models ---------------
    seed = 1337

    dataset_id = 'california_housing'
    (X_train, X_test, Y_train, Y_test), is_clf, n_classes = load_data(dataset_id, test_size=0.1, seed=seed)

    # model
    model = zoo.MLPRegressorMedium()
    params = model.init(jax.random.PRNGKey(seed), X_train[0])
    predict_fn = jax.jit(model.apply)

    # --------------- init solvers ---------------
    b = 32

    solver = IGND(
        predict_fun=predict_fn,
        loss_type='mse',
        learning_rate=1.0,
        regularizer=0.0,
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
        2.5230525,
        0.8790639,
        0.56046385,
        0.5382391,
        0.56748986,
        0.528028,
    ]))

    assert jnp.allclose(realized_losses, actual_losses, atol=1e-6), "Step size mismatch"


@pytest.mark.skip(reason="Not implemented")
def test_ignd_ce():
    raise NotImplementedError
