import jax
import jax.numpy as jnp
import chex
from jax import lax

from somax.curvature.ggn_ce import GGNCE
from somax.estimators.gnb_ce import GaussNewtonBartlett
from somax import utils


def _manual_bartlett_from_cache(cstate, params, rng, n_samples):
    c = cstate.cache
    logits, probs, JT, alpha = c.logits, c.probs, c.vjp, c.alpha
    B, C = logits.shape
    B_scalar = jnp.asarray(B, logits.dtype)
    S = int(n_samples)

    keys = jax.random.split(rng, S)
    acc0 = utils.tree_zeros_like(params)

    def body(acc, rk):
        y = jax.random.categorical(rk, logits=logits, axis=-1)  # (B,)
        r = probs - jax.nn.one_hot(y, C, dtype=logits.dtype)  # (B,C)
        (g_s,) = JT(alpha * r)  # mean loss scaling
        g2 = jax.tree_util.tree_map(lambda gi: gi * gi, g_s)
        acc = utils.tree_add(acc, g2)
        return acc, None

    accN, _ = lax.scan(body, acc0, keys)
    mean_G2 = utils.tree_scale(accN, jnp.asarray(1.0 / S, dtype=B_scalar.dtype))
    return utils.tree_scale(mean_G2, B_scalar)


def _rel_err_tree(a, b):
    af = utils.flatten_tree(a)
    bf = utils.flatten_tree(b)
    return float(jnp.linalg.norm(af - bf) / (jnp.linalg.norm(bf) + 1e-12))


def test_gnb_ce_matches_manual_for_various_n_samples(linear_logits, keys, dtype, tol):
    B, D, C = 32, 8, 4
    kx, kp, ky = keys(3)

    x = jax.random.normal(kx, (B, D), dtype=dtype)
    y = jax.random.randint(ky, (B,), 0, C, dtype=jnp.int32)

    params = {
        "W": jax.random.normal(kp, (D, C), dtype=dtype) * 0.1,
        "b": jax.random.normal(keys(), (C,), dtype=dtype) * 0.1,
    }

    cop = GGNCE(predict_fn=linear_logits, x_key="x", y_key="y", reduction="mean")
    cst, _ = cop.init(params, {"x": x, "y": y}, with_grad=False)

    for S in [1, 4, 16]:
        rk = keys()
        est = GaussNewtonBartlett(n_samples=S)
        got = est.diagonal(params, cst, mvp=lambda v: cop.matvec(params, cst, v), rng=rk)
        ref = _manual_bartlett_from_cache(cst, params, rk, n_samples=S)
        chex.assert_trees_all_close(got, ref, atol=1e-7, rtol=1e-7)


def test_gnb_ce_multisample_variance_shrinks(linear_logits, keys, dtype):
    B, D, C = 64, 16, 5
    kx, kp, ky = keys(3)

    x = jax.random.normal(kx, (B, D), dtype=dtype)
    y = jax.random.randint(ky, (B,), 0, C, dtype=jnp.int32)

    params = {
        "W": jax.random.normal(kp, (D, C), dtype=dtype) * 0.1,
        "b": jax.random.normal(keys(), (C,), dtype=dtype) * 0.1,
    }

    cop = GGNCE(predict_fn=linear_logits, x_key="x", y_key="y", reduction="mean")
    cst, _ = cop.init(params, {"x": x, "y": y}, with_grad=False)

    est1 = GaussNewtonBartlett(n_samples=1)
    est32 = GaussNewtonBartlett(n_samples=32)

    k1, k32, kref = keys(3)
    got1 = est1.diagonal(params, cst, mvp=lambda v: cop.matvec(params, cst, v), rng=k1)
    got32 = est32.diagonal(params, cst, mvp=lambda v: cop.matvec(params, cst, v), rng=k32)
    ref = _manual_bartlett_from_cache(cst, params, kref, n_samples=256)

    e1 = _rel_err_tree(got1, ref)
    e32 = _rel_err_tree(got32, ref)
    assert e32 < 0.75 * e1, f"e1={e1:.4g}, e32={e32:.4g}"


def test_gnb_ce_jit_safe_and_rng_changes_output(linear_logits, keys, dtype):
    B, D, C = 32, 10, 4
    kx, kp, ky = keys(3)

    x = jax.random.normal(kx, (B, D), dtype=dtype)
    y = jax.random.randint(ky, (B,), 0, C, dtype=jnp.int32)

    params = {
        "W": jax.random.normal(kp, (D, C), dtype=dtype) * 0.1,
        "b": jax.random.normal(keys(), (C,), dtype=dtype) * 0.1,
    }

    cop = GGNCE(predict_fn=linear_logits, x_key="x", y_key="y", reduction="mean")
    cst, _ = cop.init(params, {"x": x, "y": y}, with_grad=False)

    est = GaussNewtonBartlett(n_samples=4)

    @jax.jit
    def run(rk):
        diag = est.diagonal(params, cst, mvp=lambda v: cop.matvec(params, cst, v), rng=rk)
        return jnp.sum(utils.flatten_tree(diag))

    out1 = run(keys())
    out2 = run(keys())
    assert jnp.isfinite(out1) and jnp.isfinite(out2)
    assert not jnp.allclose(out1, out2)


def test_gnb_ce_extreme_logits_still_finite(linear_logits, keys, dtype):
    B, D, C = 32, 6, 3
    scale = 30.0

    kx, kp, ky = keys(3)
    x = jax.random.normal(kx, (B, D), dtype=dtype) * scale
    y = jax.random.randint(ky, (B,), 0, C, dtype=jnp.int32)

    params = {
        "W": jax.random.normal(kp, (D, C), dtype=dtype) * scale,
        "b": jax.random.normal(keys(), (C,), dtype=dtype) * scale,
    }

    cop = GGNCE(predict_fn=linear_logits, x_key="x", y_key="y", reduction="mean")
    cst, _ = cop.init(params, {"x": x, "y": y}, with_grad=False)

    est = GaussNewtonBartlett(n_samples=8)
    got = est.diagonal(params, cst, mvp=lambda v: cop.matvec(params, cst, v), rng=keys())
    assert bool(utils.tree_all_finite(got))
