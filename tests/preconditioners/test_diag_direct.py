import pytest

import jax
import jax.numpy as jnp
import chex
from flax import struct

from somax.preconditioners.diag_direct import DiagDirectPrecond
from somax.curvature.base import CurvatureState
from somax import utils


@struct.dataclass
class _Cache:
    diag: object


class _DiagOp:
    """Toy op that exposes diagonal() only (duck-typed for DiagDirectPrecond)."""

    def __init__(self, diag_tree):
        self._diag = diag_tree

    def diagonal(self, params, cstate, rng):
        # Return diag from cache (ignore rng).
        return cstate.cache.diag


def _rand_params(keys, dtype):
    k1, k2 = keys(2)
    return {
        "w": jax.random.normal(k1, (8,), dtype=dtype),
        "b": jax.random.normal(k2, (3, 5), dtype=dtype),
    }


def _full_like(tree, val):
    return jax.tree_util.tree_map(lambda x: jnp.full_like(x, val), tree)


def test_diag_direct_exact_jacobi_when_diag_provided(keys, dtype):
    params = _rand_params(keys, dtype)
    diag = jax.tree_util.tree_map(lambda p: jnp.abs(p) + p.dtype.type(0.3), params)

    pre = DiagDirectPrecond(eps=0.0, add_lambda=False, clip_nonneg=False)
    pst = pre.init(params)

    # Should be ignored because we pass diag explicitly.
    op = _DiagOp(diag_tree=_full_like(params, 123.0))
    cst = CurvatureState(cache=_Cache(diag=_full_like(params, 999.0)))

    M_inv, pst2 = pre.build(params, op, cst, rng=keys(), lam=None, state=pst, diag=diag)
    assert pst2 is pst

    r = _full_like(params, 1.0)
    out = M_inv(r)
    ref = jax.tree_util.tree_map(lambda ri, di: ri / di, r, diag)
    chex.assert_trees_all_close(out, ref, atol=0.0, rtol=0.0)


def test_diag_direct_fallbacks_to_op_diagonal_when_diag_missing(keys, dtype, tol):
    params = _rand_params(keys, dtype)
    diag = _full_like(params, 2.0)
    op = _DiagOp(diag_tree=diag)
    cst = CurvatureState(cache=_Cache(diag=diag))

    pre = DiagDirectPrecond(eps=1.0, add_lambda=True, clip_nonneg=True)
    pst = pre.init(params)
    lam = jnp.array(3.0, dtype=dtype)

    M_inv, _ = pre.build(params, op, cst, rng=keys(), lam=lam, state=pst)

    r = _full_like(params, 4.0)
    out = M_inv(r)
    denom = 2.0 + 3.0 + 1.0
    ref = _full_like(params, 4.0 / denom)
    chex.assert_trees_all_close(out, ref, **tol)


def test_diag_direct_clip_nonneg_clamps_before_shift(keys, dtype, tol):
    params = _rand_params(keys, dtype)
    diag = _full_like(params, -2.0)
    op = _DiagOp(diag_tree=diag)
    cst = CurvatureState(cache=_Cache(diag=diag))

    pre = DiagDirectPrecond(eps=1.0, add_lambda=True, clip_nonneg=True)
    pst = pre.init(params)
    lam = jnp.array(3.0, dtype=dtype)

    M_inv, _ = pre.build(params, op, cst, rng=keys(), lam=lam, state=pst)

    r = _full_like(params, 4.0)
    out = M_inv(r)

    # clip_nonneg => 0; denom = 0 + lam + eps = 4.
    ref = _full_like(params, 1.0)
    chex.assert_trees_all_close(out, ref, **tol)


def test_diag_direct_no_clip_allows_negative(keys, dtype, tol):
    params = _rand_params(keys, dtype)
    diag = _full_like(params, -2.0)
    op = _DiagOp(diag_tree=diag)
    cst = CurvatureState(cache=_Cache(diag=diag))

    pre = DiagDirectPrecond(eps=1.0, add_lambda=False, clip_nonneg=False)
    pst = pre.init(params)

    M_inv, _ = pre.build(params, op, cst, rng=keys(), lam=None, state=pst)

    r = _full_like(params, 2.0)
    out = M_inv(r)

    # denom = -2 + eps = -1 -> 2 / -1 = -2
    ref = _full_like(params, -2.0)
    chex.assert_trees_all_close(out, ref, **tol)


def test_diag_direct_jit_safe_and_finite(keys, dtype):
    params = _rand_params(keys, dtype)
    diag = jax.tree_util.tree_map(lambda p: jnp.abs(p) + p.dtype.type(0.1), params)
    op = _DiagOp(diag_tree=diag)
    cst = CurvatureState(cache=_Cache(diag=diag))

    pre = DiagDirectPrecond(eps=1e-6, add_lambda=True, clip_nonneg=True)

    @jax.jit
    def run(p, k):
        pst = pre.init(p)
        lam = jnp.array(0.0, dtype=dtype)
        M_inv, _ = pre.build(p, op, cst, rng=k, lam=lam, state=pst, diag=diag)
        out = M_inv(jax.tree_util.tree_map(jnp.ones_like, p))
        return jnp.sum(utils.flatten_tree(out))

    val = run(params, keys())
    assert val.dtype == dtype
    assert jnp.isfinite(val)


def test_diag_direct_requires_diagonal_if_no_diag_passed(keys, dtype):
    class _NoDiagOp:
        pass

    params = _rand_params(keys, dtype)
    op = _NoDiagOp()
    cst = CurvatureState(cache=())

    pre = DiagDirectPrecond()
    pst = pre.init(params)

    with pytest.raises(ValueError):
        pre.build(params, op, cst, rng=keys(), lam=None, state=pst)
