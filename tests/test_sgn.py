import pytest

import numpy as np

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from somax import SGN
from utils import load_california, load_iris, MLPRegressorMini, MLPClassifierMini


def flatten_2d_jacobian(jac_tree):
    return jax.vmap(lambda _: ravel_pytree(_)[0], in_axes=(0,))(jac_tree)


def flatten_3d_jacobian(jac_tree):
    flattened_jacobians = jax.vmap(flatten_2d_jacobian)(jac_tree)
    # b, c, m = flattened_jacobians.shape
    # J = flattened_jacobians.reshape(-1, m)
    return flattened_jacobians.reshape(-1, flattened_jacobians.shape[-1])


def calculate_gn_hessian_ce(predict_fn, params, batch_X, batch_y, b, c):
    def calculate_hess_pieces(probs):
        return jax.vmap(lambda p: jnp.diag(p) - jnp.outer(p, p))(probs)

    def build_block_diag(hess_pieces):
        n_dims = b * c
        block_diag_template = jnp.eye(b).reshape(b, 1, b, 1)
        block_diag_q = block_diag_template * hess_pieces.reshape(b, c, 1, c)
        return block_diag_q.reshape(n_dims, n_dims)

    jac_fn = jax.jacrev(predict_fn)

    jac_tree = jac_fn(params, batch_X)
    J = flatten_3d_jacobian(jac_tree)

    # build block diagonal H from logits
    batch_logits = predict_fn(params, batch_X)
    probs = jax.nn.softmax(batch_logits)
    hess_logits = calculate_hess_pieces(probs)
    Q = build_block_diag(hess_logits)

    H_gn = J.T @ Q @ J / b

    return H_gn


def calculate_gn_hessian_mse(predict_fn, params, batch_X, batch_y, b):
    jac_fn = jax.jacrev(predict_fn)
    jac_tree = jac_fn(params, batch_X)
    J = flatten_2d_jacobian(jac_tree)
    H_gn = J.T @ J / b

    return H_gn


# @pytest.mark.skip(reason="Disable test for debugging purposes")
def test_sgn_mse():
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

    max_diffs = {
        2: 0.01,
        10: 1e-5,
        50: 1e-5,

    }

    true_losses = {
        2: np.array([3.1070921, 0.98605144, 0.4565691, 0.4050824, 0.37491602, 0.33616212, ]),
        10: np.array([3.1070921, 0.98780066, 0.45470968, 0.4062884, 0.3772855, 0.317104, ]),
        50: np.array([3.1070921, 0.98780066, 0.45470968, 0.4062884, 0.3772855, 0.317104, ]),
    }

    for maxcg_iter in [2, 10, 50]:
        # print(f"maxcg_iter: {maxcg_iter}")

        params = model.init(rng, X_train[0])
        params_flat, pack_fn = ravel_pytree(params)
        d = params_flat.shape[0]

        # solver
        solver = SGN(
            predict_fun=predict_fn,
            loss_type='mse',
            maxcg=maxcg_iter,
            learning_rate=1.0,
            regularizer=regularizer,
            batch_size=b,
        )

        # init sover state and params
        opt_state = solver.init_state(params)

        # --------------- verify hvp and mvp ---------------
        batch_X = X_train[:b, :]
        batch_y = Y_train[:b]

        vec = jax.random.normal(rng, (d,))
        vec_tree = pack_fn(vec)

        gnhvp_sgn_tree = solver.gnhvp(params, vec_tree, batch_y, batch_X)
        gnhvp_sgn = ravel_pytree(gnhvp_sgn_tree)[0]

        # manual
        true_gn_hess = calculate_gn_hessian_mse(predict_fn, params, batch_X, batch_y, b)
        gnhvp_true = true_gn_hess @ vec

        assert jnp.allclose(gnhvp_sgn, gnhvp_true, atol=1e-5), "GN HVP mismatch"

        # --------------- verify CG solver ---------------
        dir_tree, _ = solver.calculate_direction(params, opt_state, batch_y, batch_X)
        dir_sgn = ravel_pytree(dir_tree)[0]

        # manual, Hd=-g
        grad_loss_tree = jax.grad(mse)(params, batch_X, batch_y)
        grad_loss = ravel_pytree(grad_loss_tree)[0]
        reg_gn_hess = true_gn_hess + jnp.eye(d) * regularizer
        dir_true = jnp.linalg.solve(reg_gn_hess, -grad_loss)

        dir_diff = jnp.max(jnp.abs(dir_sgn - dir_true))

        assert dir_diff < max_diffs[maxcg_iter], "Direction mismatch"

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

        actual_losses = jnp.array(true_losses[maxcg_iter])

        assert jnp.allclose(realized_losses, actual_losses, atol=1e-4), "Realized Loss mismatch"


# @pytest.mark.skip(reason="Disable test for debugging purposes")
def test_sgn_ce():
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
    b = 32
    regularizer = 1.0

    # dataset
    (X_train, X_test, Y_train, Y_test), is_clf, c = load_iris()

    # model
    model = MLPClassifierMini(c)
    predict_fn = jax.jit(model.apply)

    max_diffs = {
        2: 0.001,
        10: 1e-6,
        50: 1e-6,
    }

    true_losses = {
        2: np.array([1.1646403, 1.1155982, 1.0912483, 0.9720879, 0.8604481, 0.7592329, ]),
        10: np.array([1.1646403, 1.1158458, 1.09132, 0.97225153, 0.86185616, 0.7586591, ]),
        50: np.array([1.1646403, 1.1158458, 1.09132, 0.97225153, 0.86185616, 0.7586591, ]),
    }

    for maxcg_iter in [2, 10, 50, ]:
        print(f"maxcg_iter: {maxcg_iter}")

        # solver
        solver = SGN(
            predict_fun=predict_fn,
            loss_type='ce',
            maxcg=maxcg_iter,
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

        # --------------- verify hvp and mvp ---------------
        batch_X = X_train[:b, :]
        batch_y = Y_train[:b]

        vec = jax.random.normal(rng, (d,))
        vec_tree = pack_fn(vec)

        gnhvp_sgn_tree = solver.gnhvp(params, vec_tree, batch_y, batch_X)
        gnhvp_sgn = ravel_pytree(gnhvp_sgn_tree)[0]

        # manual
        true_gn_hess = calculate_gn_hessian_ce(predict_fn, params, batch_X, batch_y, b, c)
        gnhvp_true = true_gn_hess @ vec

        assert jnp.allclose(gnhvp_sgn, gnhvp_true, atol=1e-5), "GN HVP mismatch"

        # --------------- verify CG solver ---------------
        dir_tree, _ = solver.calculate_direction(params, opt_state, batch_y, batch_X)
        dir_sgn = ravel_pytree(dir_tree)[0]

        # manual, Hd=-g
        grad_loss_tree = jax.grad(ce)(params, batch_X, batch_y)
        grad_loss = ravel_pytree(grad_loss_tree)[0]
        reg_gn_hess = true_gn_hess + jnp.eye(d) * regularizer
        dir_true = jnp.linalg.solve(reg_gn_hess, -grad_loss)

        dir_diff = jnp.max(jnp.abs(dir_sgn - dir_true))

        assert dir_diff < max_diffs[maxcg_iter], "Direction mismatch"

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

        actual_losses = jnp.array(true_losses[maxcg_iter])

        assert jnp.allclose(realized_losses, actual_losses, atol=1e-4), "Realized Loss mismatch"
