import pytest

import numpy as np

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from somax import NewtonCG
from utils import load_iris, load_california, MLPRegressorMini, MLPClassifierMini


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


# @pytest.mark.skip(reason="Disable test for debugging purposes")
def test_newton_cg_mse():
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

    # dataset
    X_train, X_test, Y_train, Y_test = load_california()

    max_diffs = {
        3: 0.6,
        10: 0.07,
        50: 1e-5,
    }

    true_losses = {
        3: np.array([3.1070921, 2.0306454, 1.3865501, 0.6482527, 0.49732292, 0.37662107, ]),
        10: np.array([3.1070921, 1.8493116, 0.9016294, 0.7295554, 0.5904904, 0.5468303, ]),
        50: np.array([3.1070921, 1.8460542, 1.1549302, 0.80039966, 0.98126984, 0.84186745, ]),
    }

    for maxcg_iter in [3, 10, 50]:
        # print(f"maxcg_iter: {maxcg_iter}")

        # solver
        solver = NewtonCG(
            loss_fun=mse,
            maxcg=maxcg_iter,
            learning_rate=alpha,
            regularizer=regularizer,
            batch_size=b,
        )

        # model
        model = MLPRegressorMini()
        predict_fn = jax.jit(model.apply)

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

        assert dir_diff < max_diffs[maxcg_iter], "Direction mismatch"

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

        actual_losses = jnp.array(true_losses[maxcg_iter])

        # TODO stable only for float64
        assert jnp.allclose(realized_losses, actual_losses, atol=1e-1), "Realized Loss mismatch"


# @pytest.mark.skip(reason="Disable test for debugging purposes")
def test_newton_cg_ce():
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

    # model
    model = MLPClassifierMini(c)
    predict_fn = jax.jit(model.apply)

    # dataset
    (X_train, X_test, Y_train, Y_test), is_clf, n_classes = load_iris()

    max_diffs = {
        3: 0.0005,
        10: 1e-6,
        50: 1e-6,
    }

    true_losses = {
        3: np.array([1.1646403, 1.1184257, 1.0817341, 0.79667425, 0.5941576, 0.50947213, ]),
        10: np.array([1.1646403, 1.1182823, 1.0832202, 0.81386983, 0.5965343, 0.5097831, ]),
        50: np.array([1.1646403, 1.1182823, 1.0832202, 0.81386983, 0.5965343, 0.5097831, ]),
    }

    for maxcg_iter in [3, 10, 50]:
        print(f"maxcg_iter: {maxcg_iter}")

        # solver
        solver = NewtonCG(
            loss_fun=ce,
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

        assert dir_diff < max_diffs[maxcg_iter], "Direction mismatch"

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

        actual_losses = jnp.array(true_losses[maxcg_iter])

        # TODO stable only for float64
        assert jnp.allclose(realized_losses, actual_losses, atol=1e-1), "Realized Loss mismatch"
