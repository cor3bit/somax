import pytest

import numpy as np

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from benchmarks.utils.data_loader import load_data
from benchmarks.utils import model_zoo as zoo
from somax import HFO


def flatten_hessian(hessian_dict):
    flattened = []

    for layer_key, layer_dict in hessian_dict['params'].items():
        for key1, inner_dict in layer_dict.items():
            if key1 == 'bias':
                hess_rows = jax.vmap(lambda _: ravel_pytree(_)[0], in_axes=(0,))(inner_dict)
            elif key1 == 'kernel':
                collapsed_tree = jax.tree_map(lambda t: t.reshape(t.shape[0] * t.shape[1], *t.shape[2:]), inner_dict)

                hess_rows = jax.vmap(lambda _: ravel_pytree(_)[0], in_axes=(0,))(collapsed_tree)
            else:
                raise ValueError(f"Unknown key: {key1}")

            flattened.append(hess_rows)

    flat_hessian_2d = jnp.concatenate(flattened)

    return flat_hessian_2d


# @pytest.mark.skip(reason="For debugging only")
def test_hfo_mse():
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
    alpha = 0.5
    regularizer = 1.0
    maxcg_iter = 3

    # solver
    solver = HFO(
        loss_fun=mse,
        maxcg=maxcg_iter,
        learning_rate=alpha,
        regularizer=regularizer,
        batch_size=b,
    )

    # model
    model = zoo.MLPRegressorMini()
    predict_fn = jax.jit(model.apply)

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
    hvp_hfo_tree = solver.hvp(params, vec_tree, batch_y, batch_X)
    hvp_hfo = ravel_pytree(hvp_hfo_tree)[0]

    # manual
    true_hess_tree = jax.hessian(mse)(params, batch_X, batch_y)
    true_hess = flatten_hessian(true_hess_tree)
    hvp_true = true_hess @ vec

    assert jnp.allclose(hvp_hfo, hvp_true, atol=1e-5), "HVP mismatch"

    # --------------- verify CG solver ---------------
    dir_tree, _ = solver.calculate_direction(params, opt_state, batch_y, batch_X)
    dir_hfo = ravel_pytree(dir_tree)[0]

    # manual, Hd=-g
    grad_loss_tree = jax.grad(mse)(params, batch_X, batch_y)
    grad_loss = ravel_pytree(grad_loss_tree)[0]
    reg_hess = true_hess + jnp.eye(d) * regularizer
    dir_true = jnp.linalg.solve(reg_hess, -grad_loss)

    dir_diff = jnp.max(jnp.abs(dir_hfo - dir_true))

    if maxcg_iter == 3:
        # maxcg=10, dir_diff~0.06
        assert dir_diff < 1.5, "Direction mismatch"
    elif maxcg_iter == 10:
        # maxcg=10, dir_diff~0.06
        assert dir_diff < 0.07, "Direction mismatch"
    elif maxcg_iter == 20:
        # maxcg=20, dir_diff~0.0023
        assert dir_diff < 0.003, "Direction mismatch"
    elif maxcg_iter == 50:
        # maxcg=50, dir_diff~1e-5
        assert dir_diff < 5e-4, "Direction mismatch"
    else:
        raise ValueError(f"Unknown maxcg_iter: {maxcg_iter}")

    # --------------- verify calculation ---------------

    loss_t0 = mse(params, X_test, Y_test)

    # update
    test_set_loss = [loss_t0]

    lambdas = [regularizer, ]

    for i in range(5):
        batch_X = X_train[i * b:(i + 1) * b, :]
        batch_y = Y_train[i * b:(i + 1) * b]

        params, opt_state = solver.update(params, opt_state, batch_X, targets=batch_y)

        test_set_loss.append(mse(params, X_test, Y_test))
        lambdas.append(opt_state.regularizer)

    realized_losses = jnp.array(test_set_loss)

    actual_losses = jnp.array(np.array(
        [3.1070921, 2.0268407, 1.3703537, 0.79599005, 0.6253868, 0.5462274, ]))

    assert jnp.allclose(realized_losses, actual_losses, atol=1e-3), "Realized Loss mismatch"


# @pytest.mark.skip(reason="For debugging only")
def test_hfo_ce():
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

    # solver
    solver = HFO(
        loss_fun=ce,
        maxcg=maxcg_iter,
        learning_rate=1.0,
        regularizer=regularizer,
        batch_size=b,
        n_classes=c,
    )

    # model
    model = zoo.MLPClassifierMini(c)
    predict_fn = jax.jit(model.apply)

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
    hvp_hfo_tree = solver.hvp(params, vec_tree, batch_y, batch_X)
    hvp_hfo = ravel_pytree(hvp_hfo_tree)[0]

    # manual
    true_hess_tree = jax.hessian(ce)(params, batch_X, batch_y)
    true_hess = flatten_hessian(true_hess_tree)
    hvp_true = true_hess @ vec

    assert jnp.allclose(hvp_hfo, hvp_true, atol=1e-5), "HVP mismatch"

    # --------------- verify CG solver ---------------
    dir_tree, _ = solver.calculate_direction(params, opt_state, batch_y, batch_X)
    dir_hfo = ravel_pytree(dir_tree)[0]

    # manual, Hd=-g
    grad_loss_tree = jax.grad(ce)(params, batch_X, batch_y)
    grad_loss = ravel_pytree(grad_loss_tree)[0]
    reg_hess = true_hess + jnp.eye(d) * regularizer
    dir_true = jnp.linalg.solve(reg_hess, -grad_loss)

    dir_diff = jnp.max(jnp.abs(dir_hfo - dir_true))

    if maxcg_iter == 10:
        # maxcg=10, dir_diff~1e-6
        assert dir_diff < 1e-5, "Direction mismatch"
    else:
        raise ValueError(f"Unknown maxcg_iter: {maxcg_iter}")

    # --------------- verify calculation ---------------

    loss_t0 = ce(params, X_test, Y_test)

    # update
    test_set_loss = [loss_t0]
    lambdas = [regularizer, ]
    for i in range(5):
        batch_X = X_train[i * b:(i + 1) * b, :]
        batch_y = Y_train[i * b:(i + 1) * b]

        params, opt_state = solver.update(params, opt_state, batch_X, targets=batch_y)

        test_set_loss.append(ce(params, X_test, Y_test))
        lambdas.append(opt_state.regularizer)

    realized_losses = jnp.array(test_set_loss)

    actual_losses = jnp.array(np.array([1.1646405, 1.134731, 1.0789027, 0.9477376, 0.45464972, 0.41376147]))

    assert jnp.allclose(realized_losses, actual_losses, atol=1e-4), "Realized Loss mismatch"
