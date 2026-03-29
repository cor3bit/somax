import pytest

import jax
import jax.numpy as jnp
import chex
from flax import struct

from somax.preconditioners.diag_ema import DiagEMAPrecond, DiagEMAState
from somax.curvature.base import CurvatureState
from somax import utils


class _FakeOp:
    """Fake curvature operator exposing diagonal(params, cstate, rng)->PyTree."""

    def __init__(self, diag_fn):
        self._diag_fn = diag_fn

    def diagonal(self, params, cstate, rng):
        return self._diag_fn(params, cstate, rng)


@struct.dataclass
class _Cache:
    any: object = None


def _rand_params(keys, dtype):
    k1, k2, k3, k4 = keys(4)
    return {
        "w1": jax.random.normal(k1, (8, 4), dtype=dtype),
        "b1": jax.random.normal(k2, (4,), dtype=dtype),
        "w2": jax.random.normal(k3, (5, 7, 3), dtype=dtype),
        "conv": jax.random.normal(k4, (3, 3, 2, 4), dtype=dtype),
    }


def _ones_like(tree):
    return jax.tree_util.tree_map(jnp.ones_like, tree)


def _full_like(tree, val):
    return jax.tree_util.tree_map(lambda x: jnp.full_like(x, val), tree)


def test_init_shapes_and_defaults(keys, dtype):
    params = _rand_params(keys, dtype)
    pre = DiagEMAPrecond()
    st = pre.init(params)

    assert isinstance(st, DiagEMAState)
    assert int(st.step) == 0
    assert int(st.t_ema) == 0
    chex.assert_trees_all_equal_shapes(st.v, params)
    chex.assert_trees_all_close(st.v, _full_like(params, 0.0), atol=0.0, rtol=0.0)


def test_adahessian_single_refresh_expected_denom_and_v(keys, dtype, tol):
    params = _rand_params(keys, dtype)
    dval = 2.0

    op = _FakeOp(lambda p, cs, rng: _full_like(p, dval))
    pre = DiagEMAPrecond(beta2=0.9, hessian_power=1.0, eps=1e-8, spatial_averaging=False)

    st = pre.init(params)
    cst = CurvatureState(cache=_Cache())

    M_inv, st2 = pre.build(params, op, cst, rng=keys(), lam=None, state=st)

    assert int(st2.step) == 1
    assert int(st2.t_ema) == 1

    g = _full_like(params, 3.0)
    out = M_inv(g)

    # AdaHessian path: v <- EMA(diag^2), bias-correct -> v_hat = d^2, denom=(v_hat)^(alpha/2)+eps with alpha=1 => |d|+eps.
    ref = _full_like(params, 3.0 / (dval + 1e-8))
    chex.assert_trees_all_close(out, ref, **tol)

    v_expected = _full_like(params, (1.0 - 0.9) * (dval ** 2))
    chex.assert_trees_all_close(st2.v, v_expected, **tol)


def test_adahessian_bias_correction_invariance_multiple_steps(keys, dtype, tol):
    params = _rand_params(keys, dtype)
    dval = 1.7
    op = _FakeOp(lambda p, cs, rng: _full_like(p, dval))

    pre = DiagEMAPrecond(beta2=0.5, hessian_power=1.0, eps=0.0)
    st = pre.init(params)
    cst = CurvatureState(cache=_Cache())

    denoms = []
    for _ in range(3):
        M_inv, st = pre.build(params, op, cst, rng=keys(), lam=None, state=st)
        out = M_inv(_ones_like(params))  # = 1/denom
        denoms.append(jax.tree_util.tree_map(lambda x: 1.0 / x, out))

    for D in denoms:
        chex.assert_trees_all_close(D, _full_like(params, dval), **tol)

    assert int(st.step) == 3
    assert int(st.t_ema) == 3


def test_eval_every_k_gates_refresh(keys, dtype):
    params = _rand_params(keys, dtype)
    dval = 1.2
    op = _FakeOp(lambda p, cs, rng: _full_like(p, dval))

    pre = DiagEMAPrecond(beta2=0.9, eval_every_k=2, eps=0.0)
    st0 = pre.init(params)
    cst = CurvatureState(cache=_Cache())

    # step 0 refresh
    M0, st1 = pre.build(params, op, cst, rng=keys(), lam=None, state=st0)
    assert int(st1.step) == 1 and int(st1.t_ema) == 1

    # step 1 no refresh
    M1, st2 = pre.build(params, op, cst, rng=keys(), lam=None, state=st1)
    assert int(st2.step) == 2 and int(st2.t_ema) == 1

    x = _full_like(params, 1.0)
    chex.assert_trees_all_close(M0(x), M1(x), atol=0.0, rtol=0.0)


def test_sophiah_raw_diag_path_gamma_and_eps(keys, dtype, tol):
    params = _rand_params(keys, dtype)
    raw = 0.2
    gamma = 3.0
    eps = 1e-4

    op = _FakeOp(lambda p, cs, rng: _full_like(p, raw))
    pre = DiagEMAPrecond(beta2=0.95, gamma=gamma, eps=eps)

    st = pre.init(params)
    cst = CurvatureState(cache=_Cache())

    M_inv, st2 = pre.build(params, op, cst, rng=keys(), lam=None, state=st)
    denom = max(gamma * raw, eps)

    out = M_inv(_ones_like(params))
    ref = _full_like(params, 1.0 / denom)
    chex.assert_trees_all_close(out, ref, **tol)
    assert int(st2.t_ema) == 1


def test_spatial_averaging_broadcasts_correctly(keys, tol):
    # Explicit fp32 for pure shape test.
    k = keys()
    params = {
        "vec": jax.random.normal(k, (5,), dtype=jnp.float32),
        "mat": jax.random.normal(keys(), (4, 6), dtype=jnp.float32),
        "ten": jax.random.normal(keys(), (3, 5, 7), dtype=jnp.float32),
        "img": jax.random.normal(keys(), (2, 3, 5, 7), dtype=jnp.float32),
    }

    op = _FakeOp(lambda p, cs, rng: jax.tree_util.tree_map(jnp.abs, p))
    pre = DiagEMAPrecond(spatial_averaging=True, eps=0.0)

    st = pre.init(params)
    cst = CurvatureState(cache=_Cache())

    M_inv, _ = pre.build(params, op, cst, rng=keys(), lam=None, state=st)
    out = M_inv(_ones_like(params))

    def inv_expected(leaf):
        a = jnp.abs(leaf)
        if leaf.ndim == 1:
            return 1.0 / a
        if leaf.ndim in (2, 3):
            m = jnp.mean(a, axis=-1, keepdims=True)
            return jnp.broadcast_to(1.0 / m, leaf.shape)
        if leaf.ndim == 4:
            m = jnp.mean(a, axis=(-2, -1), keepdims=True)
            return jnp.broadcast_to(1.0 / m, leaf.shape)
        return 1.0 / a

    ref = jax.tree_util.tree_map(inv_expected, params)
    chex.assert_trees_all_close(out, ref, **tol)


def test_add_lambda_shifts_denominator(keys, dtype, tol):
    params = _rand_params(keys, dtype)
    dval = 2.0
    lam = jnp.array(3.0, dtype=dtype)

    op = _FakeOp(lambda p, cs, rng: _full_like(p, dval))
    pre = DiagEMAPrecond(beta2=0.9, hessian_power=1.0, eps=1e-8, add_lambda=True)

    st = pre.init(params)
    cst = CurvatureState(cache=_Cache())

    M_inv, _ = pre.build(params, op, cst, rng=keys(), lam=lam, state=st)
    out = M_inv(_ones_like(params))

    ref = _full_like(params, 1.0 / (dval + 1e-8 + 3.0))
    chex.assert_trees_all_close(out, ref, **tol)


def test_build_requires_rng(keys, dtype):
    params = _rand_params(keys, dtype)
    op = _FakeOp(lambda p, cs, rng: _full_like(p, 1.0))
    pre = DiagEMAPrecond()
    st = pre.init(params)
    cst = CurvatureState(cache=_Cache())

    with pytest.raises(ValueError):
        pre.build(params, op, cst, rng=None, lam=None, state=st)


def test_m_inv_finite_and_jittable(keys, dtype):
    params = _rand_params(keys, dtype)
    op = _FakeOp(lambda p, cs, rng: _full_like(p, 2.5))
    pre = DiagEMAPrecond(beta2=0.9, eps=1e-6, hessian_power=1.0)

    st = pre.init(params)
    cst = CurvatureState(cache=_Cache())

    M_inv, _ = pre.build(params, op, cst, rng=keys(), lam=None, state=st)
    x = _full_like(params, 3.0)

    @jax.jit
    def apply(v):
        out = M_inv(v)
        return jnp.sum(utils.flatten_tree(out))

    val = apply(x)
    assert val.dtype == dtype
    assert jnp.isfinite(val)


def test_diag_fn_override_bypasses_op_diagonal(keys, dtype, tol):
    params = _rand_params(keys, dtype)
    dval = 1.25

    class _RaisingOp:
        def diagonal(self, params, cstate, rng):
            raise AssertionError("op.diagonal should not be called when diag_fn is provided")

    called = {"ok": False}

    def diag_fn(p, op, cstate, rng):
        called["ok"] = True
        return _full_like(p, dval)

    op = _RaisingOp()
    pre = DiagEMAPrecond(beta2=0.0, hessian_power=1.0, eps=0.0, diag_fn=diag_fn)

    st = pre.init(params)
    cst = CurvatureState(cache=_Cache())

    M_inv, st2 = pre.build(params, op, cst, rng=keys(), lam=None, state=st)

    assert called["ok"] is True
    assert int(st2.step) == 1
    assert int(st2.t_ema) == 1

    out = M_inv(_ones_like(params))
    ref = _full_like(params, 1.0 / dval)
    chex.assert_trees_all_close(out, ref, **tol)
