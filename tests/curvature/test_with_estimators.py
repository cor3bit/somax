import jax
import jax.numpy as jnp
import chex
from jax.flatten_util import ravel_pytree

from somax.curvature.with_estimators import CurvatureOpWithEstimators
from somax.curvature.ggn_mse import GGNMSE


class _DummyEstimator:
    # Minimal EstimatorPolicy-like object.
    # We only use diagonal/trace in this test.
    def diagonal(self, params, state, mv, rng):
        flat, pack = ravel_pytree(params)
        # Hutchinson-diag estimator with 1 probe (deterministic sign pattern)
        z = jax.random.rademacher(rng, flat.shape, dtype=flat.dtype)
        v = pack(z)
        Hv = mv(v)
        Hv_flat, _ = ravel_pytree(Hv)
        return pack(z * Hv_flat)

    def trace(self, params, state, mv, rng):
        flat, pack = ravel_pytree(params)
        z = jax.random.rademacher(rng, flat.shape, dtype=flat.dtype)
        v = pack(z)
        Hv = mv(v)
        Hv_flat, _ = ravel_pytree(Hv)
        return jnp.dot(z, Hv_flat)

    def spectrum(self, params, state, mv, rng, k):
        raise NotImplementedError

    def low_rank(self, params, state, mv, rng, k):
        raise NotImplementedError


def test_with_estimators_delegates_and_exposes_facade(key, small_shapes, linear_predict, tol):
    B, D = small_shapes["B"], small_shapes["D"]
    kx, ky, kp, krng = jax.random.split(key, 4)

    x = jax.random.normal(kx, (B, D), jnp.float32)
    y = jax.random.normal(ky, (B,), jnp.float32)
    params = {"W": jax.random.normal(kp, (D, 1), jnp.float32) * 0.1, "b": jnp.zeros((), jnp.float32)}
    batch = {"x": x, "y": y}

    base = GGNMSE(predict_fn=linear_predict, x_key="x", y_key="y", reduction="mean")
    state, g = base.init(params, batch, with_grad=True)

    wrapped = CurvatureOpWithEstimators(base, _DummyEstimator())

    # Delegation: init/matvec/loss/loss_only/row_op should behave the same
    state2, g2 = wrapped.init(params, batch, with_grad=True)
    chex.assert_trees_all_close(g2, g, **tol)

    v = {"W": jnp.ones((D, 1), jnp.float32), "b": jnp.ones((), jnp.float32)}
    Av1 = base.matvec(params, state, v)
    Av2 = wrapped.matvec(params, state2, v)
    chex.assert_trees_all_close(Av2, Av1, **tol)

    l1 = base.loss(params, state, batch)
    l2 = wrapped.loss(params, state2, batch)
    assert jnp.allclose(l1, l2, **tol)

    # Facade: diagonal/trace should run and return correct types/shapes
    diag = wrapped.diagonal(params, state2, rng=krng)
    diag_flat, _ = ravel_pytree(diag)
    params_flat, _ = ravel_pytree(params)
    assert diag_flat.shape == params_flat.shape

    tr = wrapped.trace(params, state2, rng=krng)
    assert tr.shape == ()
