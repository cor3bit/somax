import jax
import jax.numpy as jnp
import chex
from jax.flatten_util import ravel_pytree
import optax

from somax.curvature.hessian import ExactHessian


def test_exact_hessian_loss_matches_loss_only(key, small_shapes, mse_loss, init_mlp_regression, tol):
    B, D, H = small_shapes["B"], small_shapes["D"], small_shapes["H"]
    kparams, kdata = jax.random.split(key, 2)

    params = init_mlp_regression(kparams, D, H)
    teacher, batch = (None, {"x": jax.random.normal(kdata, (B, D), jnp.float32),
                             "y": jax.random.normal(kdata, (B,), jnp.float32)})

    loss_fn = lambda p, b: mse_loss(p, b)

    op = ExactHessian(loss_fn=loss_fn, reduction="mean")
    state, _ = op.init(params, batch, with_grad=False)

    l_cached = op.loss(params, state, batch)
    l_only = op.loss_only(params, batch)
    assert jnp.allclose(l_cached, l_only, **tol)

    l_ext = mse_loss(params, batch)
    assert jnp.allclose(l_cached, l_ext, **tol)


def test_exact_hessian_grad_and_hvp_match_autodiff(key, small_shapes, mse_loss, init_mlp_regression, tol):
    B, D, H = small_shapes["B"], small_shapes["D"], small_shapes["H"]
    kparams, kx, ky, kv = jax.random.split(key, 4)

    params = init_mlp_regression(kparams, D, H)
    batch = {
        "x": jax.random.normal(kx, (B, D), jnp.float32),
        "y": jax.random.normal(ky, (B,), jnp.float32),
    }

    loss_fn = lambda p, b: mse_loss(p, b)
    op = ExactHessian(loss_fn=loss_fn, reduction="mean")

    state, g = op.init(params, batch, with_grad=True)
    g_ref = jax.grad(lambda p: op.loss_only(p, batch))(params)
    chex.assert_trees_all_close(g, g_ref, **tol)

    v = jax.tree_util.tree_map(lambda a: jax.random.normal(kv, a.shape, jnp.float32), params)
    Hv = op.matvec(params, state, v)

    # Reference HVP via jax.jvp of grad
    def grad_fn(p):
        return jax.grad(lambda pp: op.loss_only(pp, batch))(p)

    _, Hv_ref = jax.jvp(grad_fn, (params,), (v,))
    chex.assert_trees_all_close(Hv, Hv_ref, **tol)
