import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.flatten_util import ravel_pytree

from somax.curvature.fgnha_ce import FGNHA
from somax.curvature.ggn_ce import GGNCE
from somax.solvers.row_common import bind_row_system
from somax.solvers.row_cg import RowConjugateGradient


def test_fgnha_rejects_single_label_broadcast_to_batch():
    params = jnp.zeros((2,), dtype=jnp.float32)
    batch = {"x": jnp.zeros((3, 1)), "y": jnp.array([0], dtype=jnp.int32)}
    op = FGNHA(predict_fn=lambda p, x: jnp.broadcast_to(p, (x.shape[0], 2)))
    with pytest.raises(ValueError, match="integer labels"):
        op.init(params, batch)
    with pytest.raises(ValueError, match="integer labels"):
        op.loss_only(params, batch)

# Float32 scalar identities compose several reductions (JVP/VJP + class
# reductions + flatten-dot), so use dedicated tolerances.
ADJOINT_RTOL = 4e-3
ADJOINT_ATOL = 5e-4
GRAM_RTOL = 2e-3
GRAM_ATOL = 5e-4


def random_like_tree(key, tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(
        treedef,
        [jax.random.normal(k, a.shape, a.dtype) for k, a in zip(keys, leaves)],
    )


def make_problem(key, shapes):
    B, D, H, C = shapes["B"], shapes["D"], shapes["H"], shapes["C"]
    kx, ky, kW1, kb1, kW2, kb2 = jax.random.split(key, 6)
    x = jax.random.normal(kx, (B, D), jnp.float32)
    y = jax.random.randint(ky, (B,), 0, C, dtype=jnp.int32)
    params = {
        "W1": jax.random.normal(kW1, (D, H), jnp.float32) * 0.3,
        "b1": jax.random.normal(kb1, (H,), jnp.float32) * 0.05,
        "W2": jax.random.normal(kW2, (H, C), jnp.float32) * 0.3,
        "b2": jax.random.normal(kb2, (C,), jnp.float32) * 0.05,
    }
    return params, {"x": x, "y": y}


def make_linear_head_problem(key, *, B: int, D: int, C: int):
    kx, ky, kW, kb = jax.random.split(key, 4)
    x = jax.random.normal(kx, (B, D), jnp.float32)
    y = jax.random.randint(ky, (B,), 0, C, dtype=jnp.int32)
    params = {
        "W": jax.random.normal(kW, (D, C), jnp.float32) * 0.2,
        "b": jax.random.normal(kb, (C,), jnp.float32) * 0.01,
    }
    return params, {"x": x, "y": y}


def _stable_row_coeff(logits, labels, eps=1e-6):
    probs = jax.nn.softmax(logits, axis=-1)
    batch_idx = jnp.arange(logits.shape[0], dtype=jnp.int32)

    p_star = probs[batch_idx, labels]
    zero = jnp.asarray(0.0, probs.dtype)
    one = jnp.asarray(1.0, probs.dtype)
    eps_arr = jnp.asarray(eps, probs.dtype)
    lower = jnp.maximum(eps_arr, jnp.nextafter(zero, one))
    upper = jnp.minimum(one - eps_arr, jnp.nextafter(one, zero))
    p_star_safe = jnp.clip(p_star, lower, upper)
    row_scale = jnp.sqrt(p_star_safe / (one - p_star_safe))

    residual = probs.at[batch_idx, labels].set(0.0)
    residual = residual.at[batch_idx, labels].set(-jnp.sum(residual, axis=-1))
    return row_scale[:, None] * residual


def _matvec_reference_from_definition(predict_fn, params, batch, v, eps=1e-6):
    x = batch["x"]
    y = batch["y"]
    logits = predict_fn(params, x)
    row_coeff = _stable_row_coeff(logits, y, eps=eps)

    _, Z = jax.jvp(lambda p: predict_fn(p, x), (params,), (v,))
    _, pullback = jax.vjp(lambda p: predict_fn(p, x), params)

    LZ = jnp.einsum("bc,bc->b", row_coeff, Z)
    U = LZ[:, None] * row_coeff
    return pullback(U / jnp.asarray(logits.shape[0], logits.dtype))[0]


def test_fgnha_loss_matches_loss_only(key, small_shapes, mlp_logits, tol):
    params, batch = make_problem(key, small_shapes)
    op = FGNHA(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)

    l_cached = op.loss(params, state, batch)
    l_only = op.loss_only(params, batch)
    l_ext = optax.softmax_cross_entropy_with_integer_labels(
        mlp_logits(params, batch["x"]), batch["y"]
    ).mean()

    assert jnp.allclose(l_cached, l_only, **tol)
    assert jnp.allclose(l_cached, l_ext, **tol)


def test_fgnha_rejects_dense_targets_in_init_and_loss_only(key, small_shapes, mlp_logits):
    params, batch = make_problem(key, small_shapes)
    dense_targets = jax.nn.one_hot(batch["y"], small_shapes["C"])
    dense_batch = {"x": batch["x"], "y": dense_targets}
    op = FGNHA(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")

    match = "hard integer labels only"
    with pytest.raises(ValueError, match=match):
        op.init(params, dense_batch, with_grad=True)
    with pytest.raises(ValueError, match=match):
        op.loss_only(params, dense_batch)


def test_fgnha_accepts_integer_column_labels(key, small_shapes, mlp_logits, tol, assert_allclose_tree):
    params, batch = make_problem(key, small_shapes)
    column_batch = {"x": batch["x"], "y": batch["y"][:, None]}
    op = FGNHA(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")

    state_flat, grad_flat = op.init(params, batch, with_grad=True)
    state_col, grad_col = op.init(params, column_batch, with_grad=True)

    assert jnp.allclose(op.loss(params, state_flat, batch), op.loss(params, state_col, column_batch), **tol)
    assert_allclose_tree(grad_flat, grad_col, **tol)


def test_fgnha_cache_keeps_only_required_row_bindings(key, small_shapes, mlp_logits):
    params, batch = make_problem(key, small_shapes)
    op = FGNHA(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)
    cache = state.cache

    for name in (
            "jvp_logits",
            "vjp_logits",
            "rhs_factor",
            "row_coeff",
            "loss0",
            "gram",
            "has_gram",
            "b",
            "alpha",
    ):
        assert hasattr(cache, name)

    for removed_name in ("probs", "labels", "row_scale", "b_idx", "eps"):
        assert not hasattr(cache, removed_name)

    assert cache.row_coeff.shape == (small_shapes["B"], small_shapes["C"])
    assert cache.rhs_factor.shape == (small_shapes["B"],)


def test_fgnha_grad_matches_autodiff(key, small_shapes, mlp_logits, tol, assert_allclose_tree):
    params, batch = make_problem(key, small_shapes)
    op = FGNHA(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    _, grad = op.init(params, batch, with_grad=True)

    grad_ref = jax.grad(lambda p: op.loss_only(p, batch))(params)
    assert_allclose_tree(grad, grad_ref, **tol)


def test_fgnha_matvec_matches_operator_definition(key, small_shapes, mlp_logits, tol, assert_allclose_tree):
    params, batch = make_problem(key, small_shapes)
    op = FGNHA(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)

    v = random_like_tree(jax.random.fold_in(key, 1), params)
    Av = op.matvec(params, state, v)
    Av_ref = _matvec_reference_from_definition(mlp_logits, params, batch, v, eps=op.eps)
    assert_allclose_tree(Av, Av_ref, **tol)


def test_fgnha_saturated_logits_keep_nonzero_competitor_row_signal():
    B, C = 2, 5
    x = jnp.zeros((B, 1), jnp.float32)
    y = jnp.zeros((B,), jnp.int32)
    params = jnp.asarray([18.0, 0.0, 0.0, 0.0, 0.0], jnp.float32)

    def repeated_logits(p, inputs):
        return jnp.broadcast_to(p, (inputs.shape[0], p.shape[0]))

    op = FGNHA(predict_fn=repeated_logits, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, {"x": x, "y": y}, with_grad=False)
    row = op.row_op(params, state, {"x": x, "y": y})

    # Perturb only the true-class logit. The old E_p[Z] - Z_y evaluation can
    # round this row action to zero when p_y rounds to one.
    tangent = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0], jnp.float32)
    row_tangent = row.jvp(tangent)

    assert jnp.all(jnp.isfinite(row_tangent))
    assert jnp.all(jnp.abs(row_tangent) > 0.0)


def test_fgnha_row_op_adjointness_and_matvec(key, small_shapes, mlp_logits, tol, assert_allclose_tree):
    params, batch = make_problem(key, small_shapes)
    B = small_shapes["B"]
    op = FGNHA(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)
    row = op.row_op(params, state, batch)

    assert row.b == B
    assert row.rhs.shape == (B,)
    assert row.gram_mv is None

    u = jax.random.normal(jax.random.fold_in(key, 2), (B,), jnp.float32)
    v = random_like_tree(jax.random.fold_in(key, 3), params)

    JTu = row.vjp(u)
    lhs = jnp.dot(u, row.jvp(v))
    rhs = jnp.dot(ravel_pytree(JTu)[0], ravel_pytree(v)[0])
    np.testing.assert_allclose(lhs, rhs, rtol=ADJOINT_RTOL, atol=ADJOINT_ATOL)

    Av = op.matvec(params, state, v)
    Av_row = row.vjp(state.cache.alpha * row.jvp(v))
    assert_allclose_tree(Av, Av_row, **tol)


def test_fgnha_linear_head_gram_mv_matches_generic_path(
        key,
        linear_logits,
        tol,
        assert_allclose_tree,
):
    B, D, C = 7, 5, 11
    params, batch = make_linear_head_problem(key, B=B, D=D, C=C)

    op = FGNHA(
        predict_fn=linear_logits,
        x_key="x",
        y_key="y",
        reduction="mean",
        gram_mode="linear_head",
        linear_head_has_bias=True,
    )
    state, _ = op.init(params, batch, with_grad=False)
    row = op.row_op(params, state, batch)

    assert row.b == B
    assert row.rhs.shape == (B,)
    assert row.gram_mv is not None

    u = jax.random.normal(jax.random.fold_in(key, 11), (B,), jnp.float32)
    generic = row.jvp(row.vjp(u))
    direct = row.gram_mv(u)
    np.testing.assert_allclose(direct, generic, rtol=GRAM_RTOL, atol=GRAM_ATOL)

    lam = jnp.asarray(0.17, jnp.float32)
    A_mv, rhs, backproject, mu = bind_row_system(row, lam, reduction="mean")

    np.testing.assert_allclose(rhs, row.rhs, **tol)
    np.testing.assert_allclose(A_mv(u), direct + mu * u, rtol=GRAM_RTOL, atol=GRAM_ATOL)
    assert_allclose_tree(backproject(u), row.vjp(u), **tol)


def test_fgnha_float64_gram_preserves_float32_row_cg_dtype():
    with jax.enable_x64():
        params = {
            "W": jnp.ones((3, 2), jnp.float32) * 0.1,
            "b": jnp.zeros((2,), jnp.float32),
        }
        batch = {
            "x": jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10,
            "y": jnp.asarray([0, 1, 0, 1], jnp.int32),
        }
        op = FGNHA(
            predict_fn=lambda p, x: x @ p["W"] + p["b"],
            gram_mode="linear_head",
            gram_dtype=jnp.float64,
        )
        state, _ = op.init(params, batch, with_grad=False)
        row = op.row_op(params, state, batch)
        assert row.gram_mv(row.rhs).dtype == row.rhs.dtype == jnp.float32

        A_mv, rhs, _, _ = bind_row_system(row, jnp.asarray(0.1, jnp.float32), reduction="mean")
        solver = RowConjugateGradient(maxiter=2, tol=1e-5)
        solution, _, _ = solver.solve(A_mv, rhs, state=solver.init(params))
        assert jnp.all(jnp.isfinite(solution))


def test_fgnha_matvec_is_symmetric_and_psd(
        key,
        small_shapes,
        mlp_logits,
):
    params, batch = make_problem(key, small_shapes)
    op = FGNHA(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)

    u = random_like_tree(jax.random.fold_in(key, 21), params)
    v = random_like_tree(jax.random.fold_in(key, 22), params)
    Au = op.matvec(params, state, u)
    Av = op.matvec(params, state, v)

    u_flat, _ = ravel_pytree(u)
    v_flat, _ = ravel_pytree(v)
    Au_flat, _ = ravel_pytree(Au)
    Av_flat, _ = ravel_pytree(Av)

    lhs = jnp.vdot(u_flat, Av_flat)
    rhs = jnp.vdot(Au_flat, v_flat)
    np.testing.assert_allclose(lhs, rhs, rtol=GRAM_RTOL, atol=GRAM_ATOL)

    row = op.row_op(params, state, batch)
    row_v = row.jvp(v)
    q_form = jnp.vdot(v_flat, Av_flat)
    q_factor = jnp.asarray(1.0 / row.b, row_v.dtype) * jnp.vdot(
        row_v,
        row_v,
    )
    np.testing.assert_allclose(q_form, q_factor, rtol=GRAM_RTOL, atol=GRAM_ATOL)
    assert float(q_factor) >= 0.0


def test_fgnha_mean_reduction_is_invariant_to_batch_duplication(
        key,
        small_shapes,
        mlp_logits,
        tol,
        assert_allclose_tree,
):
    params, batch = make_problem(key, small_shapes)
    batch_twice = {
        "x": jnp.concatenate([batch["x"], batch["x"]], axis=0),
        "y": jnp.concatenate([batch["y"], batch["y"]], axis=0),
    }
    op = FGNHA(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")

    state, grad = op.init(params, batch, with_grad=True)
    state_twice, grad_twice = op.init(params, batch_twice, with_grad=True)
    v = random_like_tree(jax.random.fold_in(key, 23), params)


    np.testing.assert_allclose(
        op.loss(params, state, batch),
        op.loss(params, state_twice, batch_twice),
        **tol,
    )

    # Duplicating batches changes float32 reduction order.
    dup_tol = {"rtol": 1e-3, "atol": 1e-4}

    assert_allclose_tree(grad, grad_twice, **dup_tol)
    assert_allclose_tree(
        op.matvec(params, state, v),
        op.matvec(params, state_twice, v),
        **dup_tol,
    )


def test_fgnha_output_curvature_is_rank_one_psd_lower_bound_of_softmax_fisher(
        key,
        small_shapes,
):
    batch_size = small_shapes["B"]
    n_classes = small_shapes["C"]
    logits = 0.5 * jax.random.normal(
        jax.random.fold_in(key, 51),
        (batch_size, n_classes),
        jnp.float32,
    )
    labels = jax.random.randint(
        jax.random.fold_in(key, 52),
        (batch_size,),
        0,
        n_classes,
        dtype=jnp.int32,
    )
    batch = {
        "x": jnp.zeros((batch_size, 1), jnp.float32),
        "y": labels,
    }

    def identity_logits(params, _x):
        return params

    op = FGNHA(
        predict_fn=identity_logits,
        x_key="x",
        y_key="y",
        reduction="mean",
    )
    state, _ = op.init(logits, batch, with_grad=False)

    probs = jax.nn.softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, n_classes, dtype=probs.dtype)
    residual = probs - one_hot
    p_true = jnp.take_along_axis(probs, labels[:, None], axis=-1)[:, 0]
    scale_sq = p_true / (1.0 - p_true)
    rank_one = scale_sq[:, None, None] * jnp.einsum(
        "bi,bj->bij",
        residual,
        residual,
    )
    fisher = jax.vmap(jnp.diag)(probs) - jnp.einsum(
        "bi,bj->bij",
        probs,
        probs,
    )

    tangent = jax.random.normal(
        jax.random.fold_in(key, 53),
        logits.shape,
        logits.dtype,
    )
    expected = jnp.einsum("bij,bj->bi", rank_one, tangent) / batch_size
    actual = op.matvec(logits, state, tangent)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    remainder = 0.5 * (
        fisher - rank_one + jnp.swapaxes(fisher - rank_one, -1, -2)
    )
    remainder_eigs = jnp.linalg.eigvalsh(remainder)
    assert float(jnp.min(remainder_eigs)) >= -2e-6


def test_fgnha_is_psd_lower_bound_of_ggnce_in_parameter_space(
        key,
        small_shapes,
        mlp_logits,
):
    params, batch = make_problem(key, small_shapes)
    fgn = FGNHA(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    ggn = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    fgn_state, _ = fgn.init(params, batch, with_grad=False)
    ggn_state, _ = ggn.init(params, batch, with_grad=False)

    for index in range(3):
        v = random_like_tree(jax.random.fold_in(key, 60 + index), params)
        v_flat, _ = ravel_pytree(v)
        fgn_v_flat, _ = ravel_pytree(fgn.matvec(params, fgn_state, v))
        ggn_v_flat, _ = ravel_pytree(ggn.matvec(params, ggn_state, v))

        q_fgn = jnp.vdot(v_flat, fgn_v_flat)
        q_remainder = jnp.vdot(v_flat, ggn_v_flat - fgn_v_flat)
        scale = jnp.maximum(jnp.abs(jnp.vdot(v_flat, ggn_v_flat)), 1.0)
        assert float(q_fgn) >= -2e-6 * float(scale)
        assert float(q_remainder) >= -3e-6 * float(scale)
