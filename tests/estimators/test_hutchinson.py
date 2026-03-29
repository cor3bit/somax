import jax
import jax.numpy as jnp
import chex

from somax.estimators import Hutchinson
from somax import utils


def _diag_tree_like_params(params, key, *, min_abs=0.5):
    leaves = jax.tree_util.tree_leaves(params)
    ks = jax.random.split(key, len(leaves))

    def make_leaf(x, k):
        x = jnp.asarray(x)
        return jnp.abs(jax.random.normal(k, x.shape, x.dtype)) + x.dtype.type(min_abs)

    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(params),
        [make_leaf(x, k) for x, k in zip(leaves, ks)],
    )


def _signed_tree_like_params(params, key):
    leaves = jax.tree_util.tree_leaves(params)
    ks = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(params),
        [jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, ks)],
    )


def test_hutchinson_diag_exact_on_pointwise_diagonal(key, keys, dtype):
    k_params = keys()
    params = {
        "a": jax.random.normal(k_params, (64,), dtype=dtype),
        "b": jax.random.normal(keys(), (8, 4), dtype=dtype),
    }

    d_tree = _diag_tree_like_params(params, keys())
    est = Hutchinson(n_probes=1, use_abs=False)

    def mv(v):
        return jax.tree_util.tree_map(lambda d, x: d * x, d_tree, v)

    got = est.diagonal(params, state=None, mvp=mv, rng=keys())
    chex.assert_trees_all_close(got, d_tree, atol=0.0, rtol=0.0)


def test_hutchinson_trace_exact_on_pointwise_diagonal(keys, dtype, tol):
    params = {
        "a": jax.random.normal(keys(), (32,), dtype=dtype),
        "b": jax.random.normal(keys(), (6, 5), dtype=dtype),
    }
    d_tree = _diag_tree_like_params(params, keys())

    est = Hutchinson(n_probes=64)

    def mv(v):
        return jax.tree_util.tree_map(lambda d, x: d * x, d_tree, v)

    tr_hat = est.trace(params, state=None, mvp=mv, rng=keys())
    tr_true = jax.tree_util.tree_reduce(jnp.add, jax.tree_util.tree_map(jnp.sum, d_tree), initializer=0.0)
    chex.assert_trees_all_close(tr_hat, tr_true, **tol)


def test_hutchinson_error_shrinks_with_probes_on_dense_spd(make_spd, mv_from_dense, make_block_pytree, keys):
    # Build SPD A in flat space and lift to PyTree via pack.
    template, pack, _ = make_block_pytree(shape_a=(40,), shape_b=(24,))
    n = utils.flatten_tree(template).shape[0]

    A = make_spd(n, keys(), cond=30.0, eps=1e-6)
    mv = mv_from_dense(A, pack=pack)

    # True diagonal for the lifted operator is just diag(A) packed.
    diag_true = pack(jnp.diag(A))

    def rel_err(diag_est):
        e = utils.flatten_tree(utils.tree_sub(diag_est, diag_true))
        t = utils.flatten_tree(diag_true)
        return jnp.linalg.norm(e) / (jnp.linalg.norm(t) + 1e-12)

    est16 = Hutchinson(n_probes=16)
    est256 = Hutchinson(n_probes=256)

    diag16 = est16.diagonal(template, state=None, mvp=mv, rng=keys())
    diag256 = est256.diagonal(template, state=None, mvp=mv, rng=keys())

    e16 = float(rel_err(diag16))
    e256 = float(rel_err(diag256))

    # Expect ~1/sqrt(N) improvement; allow slack for randomness/device.
    assert e256 < 0.75 * e16, f"e16={e16:.4g}, e256={e256:.4g}"


def test_hutchinson_use_abs_matches_abs_diag(keys, dtype):
    params = {
        "a": jax.random.normal(keys(), (20,), dtype=dtype),
        "b": jax.random.normal(keys(), (5, 3), dtype=dtype),
    }
    d_tree = _signed_tree_like_params(params, keys())
    est = Hutchinson(n_probes=1, use_abs=True)

    def mv(v):
        return jax.tree_util.tree_map(lambda d, x: d * x, d_tree, v)

    got = est.diagonal(params, state=None, mvp=mv, rng=keys())
    chex.assert_trees_all_close(got, jax.tree_util.tree_map(jnp.abs, d_tree), atol=0.0, rtol=0.0)


def test_hutchinson_stateless_across_param_structures(keys, dtype):
    est = Hutchinson(n_probes=1)

    p1 = {"w": jax.random.normal(keys(), (17,), dtype=dtype)}
    d1 = _diag_tree_like_params(p1, keys())
    mv1 = lambda v: utils.tree_scale(v, 0.0) if False else jax.tree_util.tree_map(lambda d, x: d * x, d1, v)

    p2 = {"w": jax.random.normal(keys(), (4, 3), dtype=dtype), "b": jax.random.normal(keys(), (3,), dtype=dtype)}
    d2 = _diag_tree_like_params(p2, keys())
    mv2 = lambda v: jax.tree_util.tree_map(lambda d, x: d * x, d2, v)

    got1 = est.diagonal(p1, state=None, mvp=mv1, rng=keys())
    got2 = est.diagonal(p2, state=None, mvp=mv2, rng=keys())

    chex.assert_trees_all_close(got1, d1, atol=0.0, rtol=0.0)
    chex.assert_trees_all_close(got2, d2, atol=0.0, rtol=0.0)
