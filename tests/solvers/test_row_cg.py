import pytest

import jax
import jax.numpy as jnp

import somax.metrics as mx
from somax.solvers.cg import CGState
from somax.solvers.row_cg import RowConjugateGradient
from somax.solvers.row_cholesky import RowCholesky

from tests.conftest import make_full_rank_J, normal_eq_solution


def _as_vec(x):
    x = jnp.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"expected 1D vector, got shape {x.shape}")
    return x


def _to_flat_pytree(x):
    flat, _ = jax.flatten_util.ravel_pytree(x)
    return flat


def test_row_cg_matches_row_cholesky(keys, make_block_pytree, bind_row_system_from_J, tol):
    key = keys()
    m, n = 40, 20

    _template, pack, _zeros_like = make_block_pytree(shape_a=(n,), shape_b=(0,))
    J = make_full_rank_J(m, n, key, dtype=jnp.float32)
    rhs_row = jax.random.normal(keys(), (m,), dtype=jnp.float32)

    lam = jnp.asarray(1e-3, dtype=jnp.float32)
    bsz = 8

    A_mv, rhs, backproject, mu = bind_row_system_from_J(J, rhs_row, pack, lam=lam, b=bsz, reduction="mean")

    rcg = RowConjugateGradient(
        backend="pcg",
        tol=1e-6,
        maxiter=2000,
        stabilise_every=10,
        warm_start=True,
    )
    st = rcg.init(rhs)
    u_cg, info_cg, _ = rcg.solve(A_mv, rhs, state=st, precond=None)

    chol = RowCholesky(symmetrize=True, jitter=1e-6, solve_dtype=None)
    u_ch, info_ch, _ = chol.solve(A_mv, rhs, state=None, precond=None)

    assert info_ch["mode"] == "row_cholesky"
    assert mx.CG_ITERS in info_cg
    assert mx.CG_RESID in info_cg
    assert mx.CG_CONVERGED in info_cg

    s_cg = backproject(u_cg)
    s_ch = backproject(u_ch)

    # Compare in parameter space.
    s_cg_f = _to_flat_pytree(s_cg)
    s_ch_f = _to_flat_pytree(s_ch)
    assert jnp.allclose(s_cg_f, s_ch_f, rtol=2e-3, atol=2e-3)

    # Compare to explicit normal-equations reference:
    # s = J^T u solves (J^T J + mu I) s = J^T rhs_row
    s_ref = normal_eq_solution(J, rhs_row, float(mu))
    assert jnp.allclose(s_cg_f, s_ref, **tol)


def test_row_cg_rejects_precond(keys, make_block_pytree, bind_row_system_from_J):
    key = keys()
    m, n = 16, 8

    _template, pack, _zeros_like = make_block_pytree(shape_a=(n,), shape_b=(0,))
    J = make_full_rank_J(m, n, key, dtype=jnp.float32)
    rhs_row = jax.random.normal(keys(), (m,), dtype=jnp.float32)

    lam = jnp.asarray(1e-3, dtype=jnp.float32)
    bsz = 8

    A_mv, rhs, _backproject, _mu = bind_row_system_from_J(J, rhs_row, pack, lam=lam, b=bsz, reduction="mean")

    rcg = RowConjugateGradient(backend="pcg", tol=1e-6, maxiter=50, stabilise_every=10, warm_start=True)

    st = rcg.init(rhs)
    with pytest.raises(NotImplementedError):
        rcg.solve(A_mv, rhs, state=st, precond=lambda x: x)


def test_row_cg_warm_start_accepts_non_cg_state(keys, make_block_pytree, bind_row_system_from_J):
    key = keys()
    m, n = 24, 12

    _template, pack, _zeros_like = make_block_pytree(shape_a=(n,), shape_b=(0,))
    J = make_full_rank_J(m, n, key, dtype=jnp.float32)
    rhs_row = jax.random.normal(keys(), (m,), dtype=jnp.float32)

    lam = jnp.asarray(1e-4, dtype=jnp.float32)
    bsz = 8

    A_mv, rhs, _backproject, _mu = bind_row_system_from_J(J, rhs_row, pack, lam=lam, b=bsz, reduction="mean")

    rcg = RowConjugateGradient(backend="pcg", tol=1e-6, maxiter=200, stabilise_every=10, warm_start=True)

    # Pass a dummy state. RowCG should fall back to x0=zeros_like(rhs) internally.
    bogus_state = object()
    u, info, _ = rcg.solve(A_mv, rhs, state=bogus_state, precond=None)

    u = _as_vec(u)
    assert u.shape == (m,)
    assert mx.CG_ITERS in info
    assert mx.CG_RESID in info
    assert jnp.isfinite(info[mx.CG_RESID])


def test_row_cg_warm_start_resets_on_shape_mismatch(keys, make_block_pytree, bind_row_system_from_J):
    key = keys()
    m, n = 32, 16

    _template, pack, _zeros_like = make_block_pytree(shape_a=(n,), shape_b=(0,))
    J = make_full_rank_J(m, n, key, dtype=jnp.float32)
    rhs_row = jax.random.normal(keys(), (m,), dtype=jnp.float32)

    lam = jnp.asarray(1e-4, dtype=jnp.float32)
    bsz = 8

    A_mv, rhs, _bp, _mu = bind_row_system_from_J(J, rhs_row, pack, lam=lam, b=bsz, reduction="mean")

    rcg = RowConjugateGradient(backend="pcg", tol=1e-6, maxiter=100, stabilise_every=10, warm_start=True)

    # State has wrong shape; should be reset to zeros_like(rhs).
    st_bad = CGState(last_x=jnp.zeros((m + 3,), dtype=jnp.float32))
    u, info, st_new = rcg.solve(A_mv, rhs, state=st_bad, precond=None)

    u = _as_vec(u)
    assert u.shape == (m,)
    assert isinstance(st_new, CGState)
    assert st_new.last_x.shape == (m,)
    assert mx.CG_ITERS in info
    assert jnp.isfinite(info[mx.CG_RESID])


def test_row_cg_warm_start_casts_dtype(keys, make_block_pytree, bind_row_system_from_J):
    key = keys()
    m, n = 20, 10

    _template, pack, _zeros_like = make_block_pytree(shape_a=(n,), shape_b=(0,))
    J = make_full_rank_J(m, n, key, dtype=jnp.float32)
    rhs_row = jax.random.normal(keys(), (m,), dtype=jnp.float32)

    lam = jnp.asarray(1e-4, dtype=jnp.float32)
    bsz = 8

    A_mv, rhs, _bp, _mu = bind_row_system_from_J(J, rhs_row, pack, lam=lam, b=bsz, reduction="mean")

    rcg = RowConjugateGradient(backend="pcg", tol=1e-6, maxiter=120, stabilise_every=10, warm_start=True)

    # State last_x is float64; should be cast to rhs dtype.
    st_bad = CGState(last_x=jnp.zeros((m,), dtype=jnp.float64))
    _u, _info, st_new = rcg.solve(A_mv, rhs, state=st_bad, precond=None)

    assert isinstance(st_new, CGState)
    assert st_new.last_x.dtype == jnp.float32


def test_row_cg_warm_start_does_not_blow_up_iters(keys, make_block_pytree, bind_row_system_from_J):
    key = keys()
    m, n = 96, 40

    _template, pack, _zeros_like = make_block_pytree(shape_a=(n,), shape_b=(0,))
    J = make_full_rank_J(m, n, key, dtype=jnp.float32)

    rhs_row1 = jax.random.normal(keys(), (m,), dtype=jnp.float32)
    rhs_row2 = rhs_row1 + 1e-2 * jax.random.normal(keys(), (m,), dtype=jnp.float32)

    lam = jnp.asarray(1e-4, dtype=jnp.float32)
    bsz = 8

    A_mv1, rhs1, _bp1, _mu1 = bind_row_system_from_J(J, rhs_row1, pack, lam=lam, b=bsz, reduction="mean")
    A_mv2, rhs2, _bp2, _mu2 = bind_row_system_from_J(J, rhs_row2, pack, lam=lam, b=bsz, reduction="mean")

    rcg = RowConjugateGradient(backend="pcg", tol=1e-6, maxiter=2000, stabilise_every=10, warm_start=True)

    st = rcg.init(rhs1)
    _, info1, st = rcg.solve(A_mv1, rhs1, state=st, precond=None)
    _, info2, _ = rcg.solve(A_mv2, rhs2, state=st, precond=None)

    it1 = int(info1[mx.CG_ITERS])
    it2 = int(info2[mx.CG_ITERS])

    # Warm-start should not catastrophically increase iterations.
    # (Not requiring strict improvement to avoid flakiness.)
    assert it2 <= max(it1 + 5, int(1.5 * it1) + 1)


def test_row_cg_warm_start_false_ignores_state(keys, make_block_pytree, bind_row_system_from_J):
    key = keys()
    m, n = 24, 12

    _template, pack, _zeros_like = make_block_pytree(shape_a=(n,), shape_b=(0,))
    J = make_full_rank_J(m, n, key, dtype=jnp.float32)
    rhs_row = jax.random.normal(keys(), (m,), dtype=jnp.float32)

    lam = jnp.asarray(1e-4, dtype=jnp.float32)
    bsz = 8

    A_mv, rhs, _bp, _mu = bind_row_system_from_J(J, rhs_row, pack, lam=lam, b=bsz, reduction="mean")

    rcg = RowConjugateGradient(backend="pcg", tol=1e-6, maxiter=200, stabilise_every=10, warm_start=False)

    # When warm_start=False, RowCG delegates and should accept a Null-like state.
    bogus_state = object()
    u, info, st_new = rcg.solve(A_mv, rhs, state=bogus_state, precond=None)

    u = _as_vec(u)
    assert u.shape == (m,)
    assert mx.CG_ITERS in info
    assert mx.CG_RESID in info
    assert jnp.isfinite(info[mx.CG_RESID])
    # State type is solver-dependent; just ensure we got something back.
    assert st_new is not None
