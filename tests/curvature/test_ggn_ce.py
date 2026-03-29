import jax
import jax.numpy as jnp
import chex
from jax.flatten_util import ravel_pytree
import optax

from somax.curvature.ggn_ce import GGNCE
from somax import utils


def test_ggnce_loss_matches_loss_only(key, small_shapes, mlp_logits, tol):
    B, D, H, C = small_shapes["B"], small_shapes["D"], small_shapes["H"], small_shapes["C"]
    kx, ky, kW1, kb1, kW2, kb2 = jax.random.split(key, 6)

    x = jax.random.normal(kx, (B, D), jnp.float32)
    y = jax.random.randint(ky, (B,), 0, C, dtype=jnp.int32)

    params = {
        "W1": jax.random.normal(kW1, (D, H), jnp.float32) * 0.2,
        "b1": jax.random.normal(kb1, (H,), jnp.float32) * 0.05,
        "W2": jax.random.normal(kW2, (H, C), jnp.float32) * 0.2,
        "b2": jax.random.normal(kb2, (C,), jnp.float32) * 0.05,
    }
    batch = {"x": x, "y": y}

    op = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)

    l_cached = op.loss(params, state, batch)
    l_only = op.loss_only(params, batch)
    assert jnp.allclose(l_cached, l_only, **tol)

    l_ext = optax.softmax_cross_entropy_with_integer_labels(
        mlp_logits(params, x), y
    ).mean()
    assert jnp.allclose(l_cached, l_ext, **tol)


def test_ggnce_grad_matches_autodiff(key, small_shapes, mlp_logits, tol):
    B, D, H, C = small_shapes["B"], small_shapes["D"], small_shapes["H"], small_shapes["C"]
    kx, ky, kW1, kb1, kW2, kb2 = jax.random.split(key, 6)

    x = jax.random.normal(kx, (B, D), jnp.float32)
    y = jax.random.randint(ky, (B,), 0, C, dtype=jnp.int32)
    params = {
        "W1": jax.random.normal(kW1, (D, H), jnp.float32) * 0.2,
        "b1": jax.random.normal(kb1, (H,), jnp.float32) * 0.05,
        "W2": jax.random.normal(kW2, (H, C), jnp.float32) * 0.2,
        "b2": jax.random.normal(kb2, (C,), jnp.float32) * 0.05,
    }
    batch = {"x": x, "y": y}

    op = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    state, g = op.init(params, batch, with_grad=True)

    g_ref = jax.grad(lambda p: op.loss_only(p, batch))(params)
    chex.assert_trees_all_close(g, g_ref, **tol)


def test_ggnce_matvec_matches_dense_logits_fisher(key, small_shapes, mlp_logits, tol):
    # Reference: H v = (1/B) J^T Q J v where Q per-example Fisher on logits
    B, D, H, C = small_shapes["B"], small_shapes["D"], small_shapes["H"], small_shapes["C"]
    kx, ky, kW1, kb1, kW2, kb2, kv = jax.random.split(key, 7)

    x = jax.random.normal(kx, (B, D), jnp.float32)
    y = jax.random.randint(ky, (B,), 0, C, dtype=jnp.int32)
    params = {
        "W1": jax.random.normal(kW1, (D, H), jnp.float32) * 0.2,
        "b1": jax.random.normal(kb1, (H,), jnp.float32) * 0.05,
        "W2": jax.random.normal(kW2, (H, C), jnp.float32) * 0.2,
        "b2": jax.random.normal(kb2, (C,), jnp.float32) * 0.05,
    }
    batch = {"x": x, "y": y}

    op = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)

    v = jax.tree_map(lambda a: jax.random.normal(kv, a.shape, jnp.float32), params)

    # Build dense J_logits (B*C, P) via per-logit grads
    def logit_bc(p, xi):
        return mlp_logits(p, xi)  # (C,)

    J_tree = jax.vmap(jax.jacrev(logit_bc), in_axes=(None, 0))(params, x)  # leaves (B, C, ...)
    J = utils.flatten_3d_jacobian(J_tree)  # (B*C, P)

    logits = mlp_logits(params, x)  # (B, C)
    probs = jax.nn.softmax(logits, axis=-1)  # (B, C)

    # Q acts per-example: Qz = diag(p) z - p (p^T z)
    # As a block matrix on (B*C,), it is block-diagonal with blocks Q_i.
    # Implement Q via reshaping.
    def apply_Q(flat):
        Z = flat.reshape((B, C))
        p = probs
        pZ = p * Z
        dot = jnp.sum(pZ, axis=-1, keepdims=True)
        QZ = pZ - p * dot
        return QZ.reshape((B * C,))

    v_flat, _ = ravel_pytree(v)
    Jv = J @ v_flat  # (B*C,)
    QJv = apply_Q(Jv)
    Av_ref = (J.T @ QJv) / jnp.asarray(B, jnp.float32)

    Av = op.matvec(params, state, v)
    Av_flat, _ = ravel_pytree(Av)
    chex.assert_trees_all_close(Av_flat, Av_ref, **tol)


def test_ggnce_row_op_shapes_and_backproject_scaling(key, small_shapes, mlp_logits, tol):
    B, D, H, C = small_shapes["B"], small_shapes["D"], small_shapes["H"], small_shapes["C"]
    kx, ky, kW1, kb1, kW2, kb2, ku = jax.random.split(key, 7)

    x = jax.random.normal(kx, (B, D), jnp.float32)
    y = jax.random.randint(ky, (B,), 0, C, dtype=jnp.int32)
    params = {
        "W1": jax.random.normal(kW1, (D, H), jnp.float32) * 0.2,
        "b1": jax.random.normal(kb1, (H,), jnp.float32) * 0.05,
        "W2": jax.random.normal(kW2, (H, C), jnp.float32) * 0.2,
        "b2": jax.random.normal(kb2, (C,), jnp.float32) * 0.05,
    }
    batch = {"x": x, "y": y}

    op = GGNCE(predict_fn=mlp_logits, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)
    row = op.row_op(params, state, batch)

    assert row.b == B
    assert row.rhs.shape == (B * C,)

    u = jax.random.normal(ku, (B * C,), jnp.float32)
    JTu = row.vjp(u)

    JTu_flat, _ = ravel_pytree(JTu)

    kv = jax.random.split(ku, 2)[0]
    v = jax.tree_map(lambda a: jax.random.normal(kv, a.shape, jnp.float32), params)
    lhs = jnp.dot(u, row.jvp(v))
    rhs = jnp.dot(JTu_flat, ravel_pytree(v)[0])
    assert jnp.allclose(lhs, rhs, **tol)
