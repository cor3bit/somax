import pytest

import numpy as np

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from somax import SWMNG
from utils import load_california, load_iris, MLPRegressorMini, MLPClassifierMini


# @pytest.mark.skip(reason="Disable test for debugging purposes")
def test_swm_ng_mse():
    @jax.jit
    def mse(params, x, y):
        residuals = y - predict_fn(params, x)
        return 0.5 * jnp.mean(jnp.square(residuals))

    # --------------- jax setup ---------------
    # jax.config.update("jax_disable_jit", True)

    # --------------- initialization ---------------
    seed = 1337
    rng = jax.random.PRNGKey(seed)
    b = 32
    regularizer = 1.0

    # dataset
    X_train, X_test, Y_train, Y_test = load_california()

    # model
    model = MLPRegressorMini()

    predict_fn = jax.jit(model.apply)

    true_loss = np.array([3.1070921, 2.4282007, 1.8896264, 1.4194626, 1.0841045, 0.8781257, ])

    params = model.init(rng, X_train[0])
    params_flat, pack_fn = ravel_pytree(params)
    d = params_flat.shape[0]

    # solver
    solver = SWMNG(
        loss_fun=mse,
        loss_type='mse',
        learning_rate=1.0,
        regularizer=regularizer,
        batch_size=b,
    )

    # init sover state and params
    opt_state = solver.init_state(params)

    # --------------- verify calculation ---------------

    loss_t0 = mse(params, X_test, Y_test)

    # update
    test_set_loss = [loss_t0]
    for i in range(5):
        batch_X = X_train[i * b:(i + 1) * b, :]
        batch_y = Y_train[i * b:(i + 1) * b]

        params, opt_state = solver.update(params, opt_state, batch_X, targets=batch_y)

        test_set_loss.append(mse(params, X_test, Y_test))

    realized_losses = jnp.array(test_set_loss)

    actual_losses = jnp.array(true_loss)

    assert jnp.allclose(realized_losses, actual_losses, atol=1e-4), "Realized Loss mismatch"


# @pytest.mark.skip(reason="Disable test for debugging purposes")
def test_swm_ng_ce():
    @jax.jit
    def ce(params, inputs, labels_ohe):
        logits = predict_fn(params, inputs)
        log_probs = jax.nn.log_softmax(logits)
        residuals = jnp.sum(labels_ohe * log_probs, axis=-1)
        return -jnp.mean(residuals)

    # --------------- jax setup ---------------
    # jax.config.update("jax_disable_jit", True)

    # --------------- initialization ---------------
    seed = 1337
    rng = jax.random.PRNGKey(seed)
    b = 16
    regularizer = 1.0

    # dataset
    (X_train, X_test, Y_train, Y_test), is_clf, c = load_iris()

    # model
    model = MLPClassifierMini(c)
    predict_fn = jax.jit(model.apply)

    true_loss = np.array([1.1646403, 1.1412463, 1.0816532, 1.0446402, 1.004406, 0.77636904, ])

    # solver
    solver = SWMNG(
        loss_fun=ce,
        loss_type='ce',
        learning_rate=1.0,
        regularizer=regularizer,
        batch_size=b,
        n_classes=c,
    )

    # init sover state and params
    params = model.init(rng, X_train[0])
    opt_state = solver.init_state(params)

    params_flat, pack_fn = ravel_pytree(params)
    d = params_flat.shape[0]

    # --------------- verify calculation ---------------

    loss_t0 = ce(params, X_test, Y_test)

    # update
    test_set_loss = [loss_t0]
    for i in range(5):
        batch_X = X_train[i * b:(i + 1) * b, :]
        batch_y = Y_train[i * b:(i + 1) * b]

        params, opt_state = solver.update(params, opt_state, batch_X, targets=batch_y)

        test_set_loss.append(ce(params, X_test, Y_test))

    realized_losses = jnp.array(test_set_loss)

    actual_losses = jnp.array(true_loss)

    assert jnp.allclose(realized_losses, actual_losses, atol=1e-4), "Realized Loss mismatch"
