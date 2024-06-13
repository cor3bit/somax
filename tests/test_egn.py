import pytest

import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, hessian
from jax.flatten_util import ravel_pytree

from somax import EGN


def predict_fn_mse(params, x):
    # Ax + b
    outputs = jnp.dot(x, params['w']) + params['b']
    outputs = jnp.squeeze(outputs)
    return outputs


def mse_loss(params, x, y):
    residuals = y - predict_fn_mse(params, x)
    return 0.5 * jnp.mean(jnp.square(residuals))


def flatten_2d_jacobian(jac_tree):
    return jax.vmap(lambda _: ravel_pytree(_)[0], in_axes=(0,))(jac_tree)


def predict_fn_ce(params, x):
    # Ax + b
    logits = jnp.dot(x, params['w']) + params['b']
    return logits


def predict_with_aux(params, x):
    logits = predict_fn_ce(params, x)
    return logits, logits


def ce_loss(params, x, y):
    logits = predict_fn_ce(params, x)
    log_probs = jax.nn.log_softmax(logits)
    residuals = jnp.sum(y * log_probs, axis=1)
    ce_loss = -jnp.mean(residuals)
    return ce_loss


def flatten_hessian(hessian_dict):
    flattened = []
    for key1, inner_dict in hessian_dict.items():
        if key1 == 'b':
            hess_rows = jax.vmap(lambda _: ravel_pytree(_)[0], in_axes=(0,))(inner_dict)
        elif key1 == 'w':
            collapsed_tree = jax.tree_map(lambda t: t.reshape(t.shape[0] * t.shape[1], *t.shape[2:]), inner_dict)
            hess_rows = jax.vmap(lambda _: ravel_pytree(_)[0], in_axes=(0,))(collapsed_tree)
        else:
            raise ValueError(f"Unknown key: {key1}")

        flattened.append(hess_rows)

    flat_hessian_2d = jnp.concatenate(flattened)

    return flat_hessian_2d


def flatten_3d_jacobian(jac_tree):
    flattened_jacobians = jax.vmap(flatten_2d_jacobian)(jac_tree)
    # b, c, m = flattened_jacobians.shape
    # J = flattened_jacobians.reshape(-1, m)
    return flattened_jacobians.reshape(-1, flattened_jacobians.shape[-1])


def block_diag(batch):
    b, n, m = batch.shape
    # Create an identity matrix of shape (b, b)
    eye = jnp.eye(b).reshape(b, 1, b, 1)
    # Reshape the batch for broadcasting
    batch = batch.reshape(b, n, 1, m)
    # Use outer product to create block diagonal structure
    result = eye * batch
    # Reshape to final block diagonal matrix
    return result.reshape(b * n, b * m)


def test_egn_mse():
    # jax.config.update("jax_disable_jit", True)

    # seeding
    np.random.seed(1337)

    # HPs
    params = {
        'w': jnp.array([[0.1, 0.2, 0.3, 0.1, 0.1], ]).T,
        'b': jnp.array([0.1]),
    }

    x = jnp.array(
        [[0.23612788, 0.09510499, 0.80695184, 0.31310285, 0.41774893],
         [0.57449531, 0.19120059, 0.52991965, 0.58265076, 0.80095254],
         [0.14088843, 0.73286944, 0.1400755, 0.42825239, 0.30697055],
         [0.33390875, 0.28142191, 0.78089622, 0.06147314, 0.40186186]],
    )
    y = jnp.array([5.3, 4.1, 1.2, 3.3])

    # number of parameters
    m = params['w'].shape[0] * params['w'].shape[1] + 1

    b = x.shape[0]  # batch size

    # EXACT GRAD AND HESSIAN
    outputs = predict_fn_mse(params, x)

    loss_true = mse_loss(params, x, y)
    grad_loss_true = grad(mse_loss, argnums=0)(params, x, y)
    hess_loss_true = hessian(mse_loss, argnums=0)(params, x, y)

    # ----------- Step 1. Check the gradient -----------

    jac_fn = jax.vmap(jax.value_and_grad(predict_fn_mse), in_axes=(None, 0))

    preds, jac_tree = jac_fn(params, x)
    assert jnp.allclose(preds, outputs)

    residuals = y - jnp.squeeze(preds)

    loss = 0.5 * jnp.mean(residuals * residuals)
    assert jnp.allclose(loss_true, loss), "Loss mismatch"

    J = flatten_2d_jacobian(jac_tree)

    assert J.shape == (b, m)

    grad_loss = -J.T @ residuals / b

    true_grads_flat, _ = ravel_pytree(grad_loss_true)
    grad_max_diff = jnp.max(jnp.abs(true_grads_flat - grad_loss))
    assert grad_max_diff < 1e-6, "Gradient mismatch"

    # ----------- Step 2. Check the hessian -----------

    hess_loss_gn = J.T @ J / b

    hess_true_flat = flatten_hessian(hess_loss_true)

    hess_max_diff = jnp.max(jnp.abs(hess_loss_gn - hess_true_flat))
    assert hess_max_diff < 1e-6, "Hessian mismatch"

    # ----------- Step 3. Check the direction -----------
    # regularizer = 0.
    regularizer = 1.
    temp = jnp.linalg.solve(b * regularizer * jnp.eye(b) + J @ J.T, residuals)
    direction = J.T @ temp

    # Hd=-g
    # so that (H_gn + lambda*I)d + g = 0
    scaled_direction = (hess_loss_gn + regularizer * jnp.eye(m)) @ direction
    lin_solve_diff = jnp.max(jnp.abs(scaled_direction + grad_loss))
    assert lin_solve_diff < 1e-6, "Linear system solver error"

    # ----------- Step 4. Verify with EGN update() -----------

    solver = EGN(
        predict_fun=predict_fn_mse,  # linreg.apply,
        loss_type='mse',
        learning_rate=1.0,
        regularizer=1.0,
        batch_size=b,
    )

    opt_state = solver.init_state(params, x)

    # update
    new_params, new_opt_state = solver.update(params, opt_state, x, targets=y)
    new_params_flat = ravel_pytree(new_params)[0]

    # check if the update is correct
    params_flat = ravel_pytree(params)[0]
    manual_params = params_flat + direction

    update_diff = jnp.max(jnp.abs(new_params_flat - manual_params))
    assert update_diff < 1e-6, "Update() error"


def test_egn_ce():
    # jax.config.update("jax_disable_jit", True)

    # seeding
    np.random.seed(1337)

    # HPs
    params = {
        'w': jnp.array([
            [0.1, 0.2, 0.3, 0.1, 0.1],
            [0.2, 0.1, 0.1, 0.3, 0.2],
            [0.1, 0.1, 0.2, 0.2, 0.3]],
        ).T,
        'b': jnp.array([0.1, 0.1, 0.1]),
    }

    x = jnp.array(
        [[0.23612788, 0.09510499, 0.80695184, 0.31310285, 0.41774893],
         [0.57449531, 0.19120059, 0.52991965, 0.58265076, 0.80095254],
         [0.14088843, 0.73286944, 0.1400755, 0.42825239, 0.30697055],
         [0.33390875, 0.28142191, 0.78089622, 0.06147314, 0.40186186]],
    )
    y_labels = np.array([0, 1, 2, 1])

    c = 3  # number of classes (output dimension)
    y_ohe = jax.nn.one_hot(y_labels, c)

    # number of parameters
    m = params['w'].shape[0] * params['w'].shape[1] + params['b'].shape[0]

    b = x.shape[0]  # batch size

    # EXACT GRAD AND HESSIAN
    logits_true = predict_fn_ce(params, x)

    loss_true = ce_loss(params, x, y_ohe)

    grad_loss_true = grad(ce_loss, argnums=0)(params, x, y_ohe)

    hess_loss_true = hessian(ce_loss, argnums=0)(params, x, y_ohe)
    hess_true_flat = flatten_hessian(hess_loss_true)

    # ----------- Step 1. Check the gradient -----------

    jac_fn = jax.vmap(jax.jacrev(predict_with_aux, has_aux=True), in_axes=(None, 0))

    # jac_fn = jax.jacrev(model_with_aux, has_aux=True)

    jac_tree, logits = jac_fn(params, x)
    assert jnp.allclose(logits, logits_true)

    # from jac_tree to 2-D array of stacked Jacobians
    J = flatten_3d_jacobian(jac_tree)
    assert J.shape == (c * b, m)

    probs = jax.nn.softmax(logits)

    gr = probs - y_ohe
    r = gr.reshape(-1, )
    grad_loss = J.T @ r / b

    # np_J = np.array(J)
    # np_gr = np.array(gr)
    # np_gr_col = np.array(gr_col)

    true_grads_flat, _ = ravel_pytree(grad_loss_true)
    grad_max_diff = jnp.max(jnp.abs(true_grads_flat - grad_loss))
    assert grad_max_diff < 1e-6, "Gradient mismatch"

    # ----------- Step 2. Check the hessian -----------

    # build diag H from logits
    new_fn = lambda p: jnp.diag(p) - jnp.outer(p, p)
    hess_logits = jax.vmap(new_fn)(probs)
    Q = block_diag(hess_logits)

    # Hessian of the loss function
    hess_loss_gn = J.T @ Q @ J / b

    # visualize in PyCharm data view
    # np_hess_true_flat = np.array(hess_true_flat)
    # np_hess_loss_gn = np.array(hess_loss_gn)
    # np_H_ = np.array(H_)
    # np_J = np.array(J)

    hess_max_diff = jnp.max(jnp.abs(hess_loss_gn - hess_true_flat))
    assert hess_max_diff < 1e-6, "Hessian mismatch"

    # ----------- Step 3. Check the direction -----------
    # regularizer = 0.
    regularizer = 1.
    temp = jnp.linalg.solve(b * regularizer * jnp.eye(Q.shape[0]) + Q @ J @ J.T, r)
    direction = -J.T @ temp

    # Hd=-g
    # so that (H_gn + lambda*I)d + g = 0
    scaled_direction = (hess_loss_gn + regularizer * jnp.eye(m)) @ direction
    lin_solve_diff = jnp.max(jnp.abs(scaled_direction + grad_loss))
    assert lin_solve_diff < 3e-6, "Linear system solver error"

    # ----------- Step 4. Verify with EGN update() -----------

    solver = EGN(
        predict_fun=predict_fn_ce,
        loss_type='ce',
        learning_rate=1.0,
        regularizer=1.0,
        batch_size=b,
        n_classes=c,
    )

    opt_state = solver.init_state(params, x)

    # update
    new_params, new_opt_state = solver.update(params, opt_state, x, targets=y_ohe)
    new_params_flat = ravel_pytree(new_params)[0]

    # check if the update is correct
    params_flat = ravel_pytree(params)[0]
    manual_params = params_flat + direction

    update_diff = jnp.max(jnp.abs(new_params_flat - manual_params))
    assert update_diff < 1e-6, "Update() error"
