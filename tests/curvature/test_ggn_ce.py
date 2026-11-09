import jax
import jax.numpy as jnp
import chex
import numpy as np
import optax
import pytest
from jax.flatten_util import ravel_pytree

from somax.curvature.ggn_ce import GGNCE
from somax import utils

_ADJOINT_RTOL = 4e-3
_ADJOINT_ATOL = 5e-4
_FISHER_RTOL = 5e-3
_FISHER_ATOL = 2e-4


def _random_like_tree(key, tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(
        treedef,
        [jax.random.normal(k, a.shape, a.dtype) for k, a in zip(keys, leaves)],
    )


def test_ggnce_rejects_single_label_broadcast_to_batch():
    params = jnp.zeros((2,), dtype=jnp.float32)
    batch = {"x": jnp.zeros((3, 2)), "y": jnp.array([0], dtype=jnp.int32)}
    op = GGNCE(predict_fn=lambda p, x: jnp.broadcast_to(p, (x.shape[0], 2)))
    with pytest.raises(ValueError, match="batch size"):
        op.init(params, batch)
    with pytest.raises(ValueError, match="batch size"):
        op.loss_only(params, batch)


def test_ggnce_underflow_cannot_silently_drop_nonzero_row_gradient():
    params = jnp.array([-1000.0, 0.0], dtype=jnp.float32)
    batch = {"x": jnp.zeros((1, 1)), "y": jnp.array([0], dtype=jnp.int32)}
    op = GGNCE(predict_fn=lambda p, x: jnp.broadcast_to(p, (x.shape[0], 2)))
    state, grad = op.init(params, batch, with_grad=True)
    row = op.row_op(params, state, batch)
    assert jnp.any(grad != 0)
    assert jnp.any(~jnp.isfinite(jax.jit(row.vjp)(row.rhs)))


def _make_problem(key, shapes):
    B, D, H, C = shapes["B"], shapes["D"], shapes["H"], shapes["C"]
    kx, ky, kW1, kb1, kW2, kb2 = jax.random.split(key, 6)

    x = jax.random.normal(kx, (B, D), jnp.float32)
    y = jax.random.randint(ky, (B,), 0, C, dtype=jnp.int32)

    params = {
        "W1": jax.random.normal(kW1, (D, H), jnp.float32) * 0.2,
        "b1": jax.random.normal(kb1, (H,), jnp.float32) * 0.05,
        "W2": jax.random.normal(kW2, (H, C), jnp.float32) * 0.2,
        "b2": jax.random.normal(kb2, (C,), jnp.float32) * 0.05,
    }
    return params, {"x": x, "y": y}


def _dense_logits_fisher_matvec(params, x, probs, v, predict_fn):
    B, C = probs.shape

    def logit_bc(p, xi):
        return predict_fn(p, xi)

    J_tree = jax.vmap(jax.jacrev(logit_bc), in_axes=(None, 0))(params, x)
    J = utils.flatten_3d_jacobian(J_tree)
    v_flat, _ = ravel_pytree(v)
    Z = (J @ v_flat).reshape((B, C))
    pZ = probs * Z
    QZ = pZ - probs * jnp.sum(pZ, axis=-1, keepdims=True)
    return (J.T @ QZ.reshape((B * C,))) / jnp.asarray(B, probs.dtype)


def test_ggnce_loss_matches_loss_only(key, small_shapes, mlp_logits, tol):
    params, batch = _make_problem(key, small_shapes)

    op = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)

    l_cached = op.loss(params, state, batch)
    l_only = op.loss_only(params, batch)
    l_ext = optax.softmax_cross_entropy_with_integer_labels(
        mlp_logits(params, batch["x"]), batch["y"]
    ).mean()

    assert jnp.allclose(l_cached, l_only, **tol)
    assert jnp.allclose(l_cached, l_ext, **tol)


def test_ggnce_grad_matches_autodiff(key, small_shapes, mlp_logits, tol):
    params, batch = _make_problem(key, small_shapes)

    op = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    _, grad = op.init(params, batch, with_grad=True)

    grad_ref = jax.grad(lambda p: op.loss_only(p, batch))(params)
    chex.assert_trees_all_close(grad, grad_ref, **tol)


def test_ggnce_integer_and_one_hot_targets_are_equivalent(
        key, small_shapes, mlp_logits, tol
):
    params, batch = _make_problem(key, small_shapes)
    one_hot_batch = {
        "x": batch["x"],
        "y": jax.nn.one_hot(batch["y"], small_shapes["C"]),
    }
    op = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")

    state_int, grad_int = op.init(params, batch, with_grad=True)
    state_dense, grad_dense = op.init(params, one_hot_batch, with_grad=True)
    v = _random_like_tree(jax.random.fold_in(key, 17), params)

    chex.assert_trees_all_close(grad_int, grad_dense, **tol)
    chex.assert_trees_all_close(
        op.matvec(params, state_int, v),
        op.matvec(params, state_dense, v),
        **tol,
    )
    np.testing.assert_allclose(
        op.loss(params, state_int, batch),
        op.loss(params, state_dense, one_hot_batch),
        **tol,
    )


def test_ggnce_label_smoothed_gradient_matches_autodiff(
        key, small_shapes, mlp_logits, tol
):
    params, batch = _make_problem(key, small_shapes)
    C = small_shapes["C"]
    smoothing = 0.1
    one_hot = jax.nn.one_hot(batch["y"], C)
    targets = (1.0 - smoothing) * one_hot + smoothing / C
    dense_batch = {"x": batch["x"], "y": targets}

    op = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    _, grad = op.init(params, dense_batch, with_grad=True)
    grad_ref = jax.grad(lambda p: op.loss_only(p, dense_batch))(params)

    chex.assert_trees_all_close(grad, grad_ref, **tol)


def test_ggnce_rejects_invalid_target_shape(key, small_shapes, mlp_logits):
    params, batch = _make_problem(key, small_shapes)
    bad_batch = {"x": batch["x"], "y": jnp.zeros((small_shapes["B"], 2), jnp.float32)}
    op = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")

    with pytest.raises(ValueError, match="same shape as logits"):
        op.init(params, bad_batch, with_grad=True)
    with pytest.raises(ValueError, match="same shape as logits"):
        op.loss_only(params, bad_batch)


def test_ggnce_matvec_matches_dense_logits_fisher(key, small_shapes, mlp_logits):
    params, batch = _make_problem(key, small_shapes)
    x = batch["x"]

    op = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)
    v = _random_like_tree(jax.random.fold_in(key, 1), params)

    probs = jax.nn.softmax(mlp_logits(params, x), axis=-1)
    Av_ref = _dense_logits_fisher_matvec(params, x, probs, v, mlp_logits)
    Av_flat, _ = ravel_pytree(op.matvec(params, state, v))

    chex.assert_trees_all_close(
        Av_flat,
        Av_ref,
        rtol=_FISHER_RTOL,
        atol=_FISHER_ATOL,
    )


def test_ggnce_matvec_is_target_independent(key, small_shapes, mlp_logits, tol):
    params, batch = _make_problem(key, small_shapes)
    C = small_shapes["C"]
    v = _random_like_tree(jax.random.fold_in(key, 31), params)

    hard_op = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    hard_state, _ = hard_op.init(params, batch, with_grad=False)

    dense_targets = jnp.full((small_shapes["B"], C), 1.0 / C, jnp.float32)
    dense_batch = {"x": batch["x"], "y": dense_targets}
    dense_state, _ = hard_op.init(params, dense_batch, with_grad=False)

    chex.assert_trees_all_close(
        hard_op.matvec(params, hard_state, v),
        hard_op.matvec(params, dense_state, v),
        **tol,
    )


def test_ggnce_row_op_adjointness_and_shapes(key, small_shapes, mlp_logits):
    params, batch = _make_problem(key, small_shapes)
    B, C = small_shapes["B"], small_shapes["C"]

    op = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)
    row = op.row_op(params, state, batch)

    assert row.b == B
    assert row.rhs.shape == (B * C,)

    u = jax.random.normal(jax.random.fold_in(key, 2), (B * C,), jnp.float32)
    v = _random_like_tree(jax.random.fold_in(key, 3), params)

    JTu = row.vjp(u)
    lhs = jnp.dot(u, row.jvp(v))
    rhs = jnp.dot(ravel_pytree(JTu)[0], ravel_pytree(v)[0])

    np.testing.assert_allclose(lhs, rhs, rtol=_ADJOINT_RTOL, atol=_ADJOINT_ATOL)


@pytest.mark.parametrize("dense_targets", [False, True])
def test_ggnce_row_rhs_backprojects_to_gradient_numerator(
        key, small_shapes, mlp_logits, dense_targets
):
    params, batch = _make_problem(key, small_shapes)
    B, C = small_shapes["B"], small_shapes["C"]
    if dense_targets:
        one_hot = jax.nn.one_hot(batch["y"], C)
        targets = 0.9 * one_hot + 0.1 / C
        batch = {"x": batch["x"], "y": targets}

    op = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    state, grad = op.init(params, batch, with_grad=True)
    row = op.row_op(params, state, batch)

    lhs = row.vjp(row.rhs)
    rhs = jax.tree_util.tree_map(lambda g: B * g, grad)

    chex.assert_trees_all_close(
        lhs,
        rhs,
        rtol=_FISHER_RTOL,
        atol=_FISHER_ATOL,
    )


def test_ggnce_row_operator_matches_exact_parameter_space_fisher(
        key, small_shapes, mlp_logits
):
    """Regression: row_op must factor Q exactly, with no epsilon*J^T J term."""
    B = small_shapes["B"]
    params, batch = _make_problem(key, small_shapes)

    op = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)
    row = op.row_op(params, state, batch)
    v = _random_like_tree(jax.random.fold_in(key, 4), params)

    Av_param = op.matvec(params, state, v)
    Av_row = row.vjp(jnp.asarray(1.0 / B, jnp.float32) * row.jvp(v))

    chex.assert_trees_all_close(
        Av_row,
        Av_param,
        rtol=_FISHER_RTOL,
        atol=_FISHER_ATOL,
    )


def test_ggnce_matvec_is_symmetric_and_psd(
        key,
        small_shapes,
        mlp_logits,
):
    params, batch = _make_problem(key, small_shapes)
    op = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)

    u = _random_like_tree(jax.random.fold_in(key, 41), params)
    v = _random_like_tree(jax.random.fold_in(key, 42), params)
    Au = op.matvec(params, state, u)
    Av = op.matvec(params, state, v)

    u_flat, _ = ravel_pytree(u)
    v_flat, _ = ravel_pytree(v)
    Au_flat, _ = ravel_pytree(Au)
    Av_flat, _ = ravel_pytree(Av)

    lhs = jnp.vdot(u_flat, Av_flat)
    rhs = jnp.vdot(Au_flat, v_flat)
    np.testing.assert_allclose(lhs, rhs, rtol=_FISHER_RTOL, atol=_FISHER_ATOL)

    row = op.row_op(params, state, batch)
    row_v = row.jvp(v)
    q_form = jnp.vdot(v_flat, Av_flat)
    q_factor = jnp.asarray(1.0 / row.b, row_v.dtype) * jnp.vdot(row_v, row_v)
    np.testing.assert_allclose(q_form, q_factor, rtol=_FISHER_RTOL, atol=_FISHER_ATOL)
    assert float(q_factor) >= 0.0


@pytest.mark.parametrize("dense_targets", [False, True])
def test_ggnce_mean_reduction_is_invariant_to_batch_duplication(
        key,
        small_shapes,
        mlp_logits,
        tol,
        dense_targets,
):
    params, batch = _make_problem(key, small_shapes)
    if dense_targets:
        C = small_shapes["C"]
        one_hot = jax.nn.one_hot(batch["y"], C)
        batch = {"x": batch["x"], "y": 0.9 * one_hot + 0.1 / C}

    batch_twice = {
        "x": jnp.concatenate([batch["x"], batch["x"]], axis=0),
        "y": jnp.concatenate([batch["y"], batch["y"]], axis=0),
    }
    op = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")

    state, grad = op.init(params, batch, with_grad=True)
    state_twice, grad_twice = op.init(params, batch_twice, with_grad=True)
    v = _random_like_tree(jax.random.fold_in(key, 43), params)

    np.testing.assert_allclose(
        op.loss(params, state, batch),
        op.loss(params, state_twice, batch_twice),
        **tol,
    )

    # Duplicating batches changes float32 reduction order.
    dup_tol = {"rtol": 1e-3, "atol": 1e-4}

    chex.assert_trees_all_close(grad, grad_twice, **dup_tol)
    chex.assert_trees_all_close(
        op.matvec(params, state, v),
        op.matvec(params, state_twice, v),
        **dup_tol,
    )
