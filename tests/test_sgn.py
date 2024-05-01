import pytest

import numpy as np

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from benchmarks.utils.data_loader import load_data
from benchmarks.utils import model_zoo as zoo
from somax import SGN


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


# @pytest.mark.skip(reason="For debugging only")
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
    maxcg_iter = 10

    # model
    model = zoo.MLPRegressorMini()
    predict_fn = jax.jit(model.apply)

    # solver
    solver = SGN(
        predict_fun=predict_fn,
        loss_type='mse',
        maxcg=maxcg_iter,
        learning_rate=1.0,
        regularizer=regularizer,
        batch_size=b,
    )

    # dataset
    dataset_id = 'california_housing'
    (X_train, X_test, Y_train, Y_test), is_clf, n_classes = load_data(
        dataset_id, test_size=0.1, seed=seed)

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

    if maxcg_iter == 2:
        assert dir_diff < 0.01, "Direction mismatch"
    elif maxcg_iter == 10:
        assert dir_diff < 1e-5, "Direction mismatch"
    elif maxcg_iter == 50:
        assert dir_diff < 1e-5, "Direction mismatch"
    else:
        raise ValueError(f"Unknown maxcg_iter: {maxcg_iter}")

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

    actual_losses = jnp.array(np.array(
        [3.1070921, 0.96741265, 0.49240723, 0.39509234, 0.37976882, 0.34163678, ]))

    assert jnp.allclose(realized_losses, actual_losses, atol=1e-4), "Realized Loss mismatch"


# @pytest.mark.skip(reason="For debugging only")
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
    c = 3
    regularizer = 1.0
    maxcg_iter = 10

    # model
    model = zoo.MLPClassifierMini(c)
    predict_fn = jax.jit(model.apply)

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

    # dataset
    dataset_id = 'iris'
    (X_train, X_test, Y_train, Y_test), is_clf, n_classes = load_data(
        dataset_id, test_size=0.1, seed=seed)
    Y_test = jax.nn.one_hot(Y_test, c)

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

    if maxcg_iter == 2:
        assert dir_diff < 0.001, "Direction mismatch"
    elif maxcg_iter == 10:
        assert dir_diff < 1e-6, "Direction mismatch"
    elif maxcg_iter == 100:
        assert dir_diff < 1e-6, "Direction mismatch"
    else:
        raise ValueError(f"Unknown maxcg_iter: {maxcg_iter}")

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

    actual_losses = jnp.array(np.array(
        [1.1646405, 1.1309346, 1.0883108, 1.021174, 0.7846394, 0.62611365]))

    assert jnp.allclose(realized_losses, actual_losses, atol=1e-4), "Realized Loss mismatch"
