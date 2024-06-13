import pytest

import numpy as np

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from somax import SophiaG
from utils import load_california, load_iris, MLPRegressorMini, MLPClassifierMini


# @pytest.mark.skip(reason="Disable test for debugging purposes")
def test_sophia_g_ce():
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

    Y_train_labels = jnp.argmax(Y_train, axis=-1)

    # model
    model = MLPClassifierMini(c)
    predict_fn = jax.jit(model.apply)

    true_loss = np.array([1.1646403, 1.1408378, 1.0884051, 1.0287585, 1.028964, .83854806, ])

    # solver
    solver = SophiaG(
        predict_fun=predict_fn,

        learning_rate=0.1,
        weight_decay=1e-4,
        eval_hess_every_k=2,
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
        batch_y = Y_train_labels[i * b:(i + 1) * b]

        params, opt_state = solver.update(params, opt_state, batch_X, batch_y)

        test_set_loss.append(ce(params, X_test, Y_test))

    realized_losses = jnp.array(test_set_loss)

    actual_losses = jnp.array(true_loss)

    assert jnp.allclose(realized_losses, actual_losses, atol=1e-4), "Realized Loss mismatch"
