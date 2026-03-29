import jax
import jax.numpy as jnp
import chex
import optax
from jax.flatten_util import ravel_pytree

from somax.curvature.ggn_mse import GGNMSE
from somax import utils


def test_ggnmse_loss_matches_loss_only(key, small_shapes, linear_predict, tol, mse_loss):
    B, D = small_shapes["B"], small_shapes["D"]
    kx, ky, kp = jax.random.split(key, 3)

    x = jax.random.normal(kx, (B, D), jnp.float32)
    y = jax.random.normal(ky, (B,), jnp.float32)

    params = {
        "W": jax.random.normal(kp, (D, 1), jnp.float32) * 0.1,
        "b": jnp.zeros((), jnp.float32),
    }
    batch = {"x": x, "y": y}

    op = GGNMSE(predict_fn=linear_predict, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)

    l_cached = op.loss(params, state, batch)
    l_only = op.loss_only(params, batch)

    assert jnp.allclose(l_cached, l_only, **tol)

    l_ext = optax.l2_loss(linear_predict(params, x), y).mean()
    assert jnp.allclose(l_cached, l_ext, **tol)


def test_ggnmse_grad_matches_autodiff(key, small_shapes, linear_predict, tol):
    B, D = small_shapes["B"], small_shapes["D"]
    kx, ky, kp = jax.random.split(key, 3)

    x = jax.random.normal(kx, (B, D), jnp.float32)
    y = jax.random.normal(ky, (B,), jnp.float32)

    params = {
        "W": jax.random.normal(kp, (D, 1), jnp.float32) * 0.2,
        "b": jnp.zeros((), jnp.float32),
    }
    batch = {"x": x, "y": y}

    op = GGNMSE(predict_fn=linear_predict, x_key="x", y_key="y", reduction="mean")
    state, g = op.init(params, batch, with_grad=True)

    g_ref = jax.grad(lambda p: op.loss_only(p, batch))(params)
    chex.assert_trees_all_close(g, g_ref, **tol)


def test_ggnmse_matvec_matches_jtj(key, small_shapes, linear_predict, tol):
    # For MSE: matvec = (1/B) J^T J v
    B, D = small_shapes["B"], small_shapes["D"]
    kx, ky, kp, kv = jax.random.split(key, 4)

    x = jax.random.normal(kx, (B, D), jnp.float32)
    y = jax.random.normal(ky, (B,), jnp.float32)

    params = {
        "W": jax.random.normal(kp, (D, 1), jnp.float32) * 0.2,
        "b": jnp.zeros((), jnp.float32),
    }
    batch = {"x": x, "y": y}

    op = GGNMSE(predict_fn=linear_predict, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)

    v = {
        "W": jax.random.normal(kv, (D, 1), jnp.float32),
        "b": jax.random.normal(kv, (), jnp.float32),
    }

    # Reference via explicit per-example grads of yhat (Jacobian)
    def yhat_i(p, xi):
        return linear_predict(p, xi[None, :])[0]

    J_tree = jax.vmap(jax.grad(yhat_i), in_axes=(None, 0))(params, x)  # leaves (B, ...)
    J = utils.flatten_2d_jacobian(J_tree)  # (B, P)

    v_flat, _ = ravel_pytree(v)
    Av_ref = (J.T @ (J @ v_flat)) / jnp.asarray(B, jnp.float32)

    Av = op.matvec(params, state, v)
    Av_flat, _ = ravel_pytree(Av)

    chex.assert_trees_all_close(Av_flat, Av_ref, **tol)


def test_ggnmse_row_op_adjointness_and_backproject(key, small_shapes, linear_predict, tol):
    B, D = small_shapes["B"], small_shapes["D"]
    kx, ky, kp, ku, kv = jax.random.split(key, 5)

    x = jax.random.normal(kx, (B, D), jnp.float32)
    y = jax.random.normal(ky, (B,), jnp.float32)
    params = {
        "W": jax.random.normal(kp, (D, 1), jnp.float32) * 0.2,
        "b": jnp.zeros((), jnp.float32),
    }
    batch = {"x": x, "y": y}

    op = GGNMSE(predict_fn=linear_predict, x_key="x", y_key="y", reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)
    row = op.row_op(params, state, batch)

    assert row.b == B
    assert row.rhs.shape == (B,)

    v = {"W": jax.random.normal(kv, (D, 1), jnp.float32), "b": jax.random.normal(kv, (), jnp.float32)}
    u = jax.random.normal(ku, (B,), jnp.float32)

    assert_row_op_adjointness(row, u, v, tol)


def assert_row_op_adjointness(row, u, v, tol):
    from jax.flatten_util import ravel_pytree
    lhs = jnp.vdot(u, row.jvp(v))
    JTu = row.vjp(u)
    rhs = jnp.vdot(ravel_pytree(JTu)[0], ravel_pytree(v)[0])
    assert jnp.allclose(lhs, rhs, **tol)
