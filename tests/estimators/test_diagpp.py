import pytest

import jax
import jax.numpy as jnp

matfree = pytest.importorskip("matfree", reason="DiagPP tests require matfree")

from somax.estimators.diagpp import DiagPP
from somax import utils


def test_diagpp_mixed_precision_unpacks_original_tangent_dtypes():
    with jax.enable_x64():
        params = {"a": jnp.ones((4,), dtype=jnp.float32),
                  "b": jnp.ones((4,), dtype=jnp.float64)}
        _, mvp = jax.linearize(
            lambda p: jax.tree_util.tree_map(lambda leaf: 2.0 * leaf, p), params
        )
        estimated = DiagPP(k=2, n_probes_res=4).diagonal(
            params, state=None, mvp=mvp, rng=jax.random.PRNGKey(0)
        )
        assert estimated["a"].dtype == params["a"].dtype
        assert estimated["b"].dtype == params["b"].dtype
        assert jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda leaf: jnp.all(jnp.isfinite(leaf)), estimated)
        )


def _spiky_spectrum_spd(key, n, dtype=jnp.float32):
    q, _ = jnp.linalg.qr(jax.random.normal(key, (n, n), dtype))
    d = jnp.asarray(1.0 / (1.0 + jnp.arange(n, dtype=dtype)) ** 1.5, dtype)
    d = 50.0 * d
    A = (q * d) @ q.T
    A = 0.5 * (A + A.T)
    return A


def test_diagpp_diag_matches_true_diag_and_improves_with_k(make_block_pytree, mv_from_dense, keys):
    template, pack, _ = make_block_pytree(shape_a=(64,), shape_b=(64,))
    n = utils.flatten_tree(template).shape[0]

    A = _spiky_spectrum_spd(keys(), n, dtype=utils.flatten_tree(template).dtype)
    mv = mv_from_dense(A, pack=pack)

    diag_true = pack(jnp.diag(A))

    est_small = DiagPP(k=8, n_probes_res=256, clip_min=0.0, reortho="full")
    est_large = DiagPP(k=32, n_probes_res=256, clip_min=0.0, reortho="full")

    d_small = est_small.diagonal(template, state=None, mvp=mv, rng=keys())
    d_large = est_large.diagonal(template, state=None, mvp=mv, rng=keys())

    def rel_err(diag_est):
        e = utils.flatten_tree(utils.tree_sub(diag_est, diag_true))
        t = utils.flatten_tree(diag_true)
        return float(jnp.linalg.norm(e) / (jnp.linalg.norm(t) + 1e-12))

    err_small = rel_err(d_small)
    err_large = rel_err(d_large)

    assert err_large < 0.85 * err_small, f"err_small={err_small:.4g}, err_large={err_large:.4g}"


def test_diagpp_trace_close_to_true_trace(make_spd, make_block_pytree, mv_from_dense, keys):
    template, pack, _ = make_block_pytree(shape_a=(40,), shape_b=(24,))
    n = utils.flatten_tree(template).shape[0]

    A = make_spd(n, keys(), cond=25.0, eps=1e-6)
    mv = mv_from_dense(A, pack=pack)

    est = DiagPP(k=24, n_probes_res=64, clip_min=0.0, reortho="none")
    tr_hat = est.trace(template, state=None, mvp=mv, rng=keys())
    tr_true = jnp.trace(A)

    assert jnp.allclose(tr_hat, tr_true, rtol=3e-2, atol=3e-2)
