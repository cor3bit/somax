import pytest

import numpy as np

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from src import SophiaH
from utils import load_california, load_iris, MLPRegressorMini, MLPClassifierMini


def flatten_2d_jacobian(jac_tree):
    return jax.vmap(lambda _: ravel_pytree(_)[0], in_axes=(0,))(jac_tree)


# @pytest.mark.skip(reason="Disable test for debugging purposes")
def test_sophia_h_mse():
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

    # dataset
    X_train, X_test, Y_train, Y_test = load_california()

    # model
    model = MLPRegressorMini()

    predict_fn = jax.jit(model.apply)

    true_loss = np.array([3.1070921, 2.3950484, 1.7493383, 1.1818779, 1.0049131, 1.5003884, ])

    params = model.init(rng, X_train[0])
    params_flat, pack_fn = ravel_pytree(params)
    d = params_flat.shape[0]

    # solver
    solver = SophiaH(
        loss_fun=mse,
        learning_rate=0.1,
        weight_decay=1e-4,
        eval_hess_every_k=1,
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

        params, opt_state = solver.update(params, opt_state, batch_X, batch_y)

        test_set_loss.append(mse(params, X_test, Y_test))

    realized_losses = jnp.array(test_set_loss)

    actual_losses = jnp.array(true_loss)

    assert jnp.allclose(realized_losses, actual_losses, atol=1e-4), "Realized Loss mismatch"


# @pytest.mark.skip(reason="Disable test for debugging purposes")
def test_sophia_h_ce():
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

    # dataset
    (X_train, X_test, Y_train, Y_test), is_clf, c = load_iris()

    # model
    model = MLPClassifierMini(c)
    predict_fn = jax.jit(model.apply)

    true_loss = np.array([1.1646403, 1.1408378, 1.0884053, 1.0289913, 1.0301952, 0.8387284, ])

    # solver
    solver = SophiaH(
        loss_fun=ce,
        learning_rate=0.1,
        weight_decay=1e-4,
        eval_hess_every_k=1,
    )

    # init sover state and params
    params = model.init(rng, X_train[0])
    opt_state = solver.init_state(params)

    # --------------- verify calculation ---------------

    loss_t0 = ce(params, X_test, Y_test)

    # update
    test_set_loss = [loss_t0]
    for i in range(5):
        batch_X = X_train[i * b:(i + 1) * b, :]
        batch_y = Y_train[i * b:(i + 1) * b]

        params, opt_state = solver.update(params, opt_state, batch_X, batch_y)

        test_set_loss.append(ce(params, X_test, Y_test))

    realized_losses = jnp.array(test_set_loss)

    actual_losses = jnp.array(true_loss)

    assert jnp.allclose(realized_losses, actual_losses, atol=1e-4), "Realized Loss mismatch"
