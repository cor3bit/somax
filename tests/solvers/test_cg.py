import pytest

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree

import somax.metrics as mx
from somax.solvers.cg import CGState, ConjugateGradient
from somax.utils import tree_vdot


def _flat(x):
    return ravel_pytree(x)[0]


def _rel_resid(A_mv, x, b):
    r = jtu.tree_map(lambda Ax, bb: Ax - bb, A_mv(x), b)
    r2 = tree_vdot(r, r)
    b2 = tree_vdot(b, b)
    dt = jnp.result_type(*[jnp.asarray(t).dtype for t in jtu.tree_leaves(b)])
    tiny = jnp.asarray(jnp.finfo(dt).tiny, dt)
    return jnp.sqrt(r2 / jnp.maximum(b2, tiny))


def _assert_pcg_info(info):
    assert mx.CG_ITERS in info
    assert mx.CG_RESID in info
    assert mx.CG_CONVERGED in info
    assert mx.CG_MAXITER in info
    assert jnp.isfinite(info[mx.CG_RESID])


def _tree_shape_dtype(tree):
    return jtu.tree_map(lambda x: (jnp.asarray(x).shape, jnp.asarray(x).dtype), tree)


def _available_reference_backends():
    backends = ["scipy"]
    try:
        import lineax as _  # noqa: F401
        backends.append("lineax")
    except Exception:
        pass
    return backends


@pytest.mark.parametrize("backend", _available_reference_backends())
def test_reference_cg_solve_dtype_preserves_linearized_primal_dtype(backend):
    params = jnp.array([1.0, 2.0], dtype=jnp.bfloat16)
    _, linearized = jax.linearize(lambda x: 2.0 * x, params)
    rhs = jnp.array([2.0, 4.0], dtype=jnp.bfloat16)
    solver = ConjugateGradient(backend=backend, solve_dtype=jnp.float32, maxiter=20, tol=1e-4)
    solution, _, _ = solver.solve(linearized, rhs, state=solver.init(rhs))
    assert solution.dtype == rhs.dtype
    assert jnp.allclose(solution.astype(jnp.float32), jnp.array([1.0, 2.0]), atol=0.02)


def test_lineax_cg_reuses_solver_for_different_vector_shapes():
    pytest.importorskip("lineax")
    solver = ConjugateGradient(backend="lineax", maxiter=20, tol=1e-5)
    A_mv = lambda v: 2.0 * v
    for n in (2, 3):
        rhs = jnp.ones((n,), dtype=jnp.float32)
        solution, _, _ = solver.solve(A_mv, rhs, state=solver.init(rhs))
        assert solution.shape == rhs.shape
        assert jnp.allclose(solution, 0.5 * rhs, atol=1e-4)


def test_lineax_cg_uses_supplied_preconditioner():
    pytest.importorskip("lineax")
    rhs = jnp.array([2.0, 3.0], dtype=jnp.float32)
    solver = ConjugateGradient(backend="lineax", preconditioned=True, maxiter=10, tol=1e-5)
    solution, _, _ = solver.solve(
        lambda v: jnp.array([2.0, 3.0]) * v,
        rhs,
        state=solver.init(rhs),
        precond=lambda v: v / jnp.array([2.0, 3.0]),
    )
    assert jnp.allclose(solution, jnp.ones_like(rhs), atol=1e-4)


def _make_vector_spd_problem(keys, make_spd, mv_from_dense, *, n=32, cond=80.0):
    A = make_spd(n=n, key=keys(), cond=cond, eps=1e-6)
    b = jax.random.normal(keys(), (n,), dtype=jnp.float32)
    return A, b, mv_from_dense(A, pack=None)


def _make_pytree_spd_problem(keys, make_spd, mv_from_dense, make_block_pytree, *, n_a=7, n_b=5, cond=80.0):
    template, pack, _ = make_block_pytree(shape_a=(n_a,), shape_b=(n_b,))
    n = int(ravel_pytree(template)[0].shape[0])
    A = make_spd(n=n, key=keys(), cond=cond, eps=1e-6)
    b = pack(jax.random.normal(keys(), (n,), dtype=jnp.float32))
    return A, b, mv_from_dense(A, pack=pack)


# -----------------------------------------------------------------------------
# Production backend: core correctness and invariants
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("use_pytree", [False, True])
def test_pcg_solves_spd_to_true_residual(
        use_pytree, keys, make_spd, mv_from_dense, make_block_pytree
):
    if use_pytree:
        _, b, A_mv = _make_pytree_spd_problem(
            keys, make_spd, mv_from_dense, make_block_pytree, cond=80.0
        )
    else:
        _, b, A_mv = _make_vector_spd_problem(keys, make_spd, mv_from_dense, n=32, cond=80.0)

    cg = ConjugateGradient(
        backend="pcg",
        tol=1e-6,
        maxiter=600,
        stabilise_every=0,
        warm_start=False,
        preconditioned=False,
    )
    x, info, st2 = cg.solve(A_mv, b, state=cg.init(b), precond=None)

    _assert_pcg_info(info)
    assert not isinstance(st2, CGState)
    assert _rel_resid(A_mv, x, b) <= 2.0e-5


@pytest.mark.parametrize("preconditioned", [False, True])
def test_raw_denominator_not_clipped_on_tiny_spd_curvature(preconditioned):
    # Regression for the old jnp.maximum(pAp, eps) denominator clip.
    dtype = jnp.float32
    A = jnp.asarray([[1.0e-10]], dtype=dtype)
    b = jnp.asarray([1.0], dtype=dtype)

    def A_mv(x):
        return A @ x

    cg = ConjugateGradient(
        backend="pcg",
        tol=0.0,
        maxiter=1,
        stabilise_every=0,
        warm_start=False,
        preconditioned=preconditioned,
    )
    precond = (lambda r: r) if preconditioned else None
    x, info, _ = cg.solve(A_mv, b, state=cg.init(b), precond=precond)

    _assert_pcg_info(info)
    assert int(info[mx.CG_ITERS]) == 1
    assert jnp.allclose(x, jnp.asarray([1.0e10], dtype=dtype), rtol=1e-5, atol=0.0)
    assert _rel_resid(A_mv, x, b) <= 1.0e-5


def test_zero_rhs_converges_without_iterations():
    b = jnp.zeros((4,), dtype=jnp.float32)
    cg = ConjugateGradient(
        backend="pcg",
        tol=1e-6,
        maxiter=4,
        stabilise_every=0,
        warm_start=False,
        preconditioned=False,
    )
    x, info, _ = cg.solve(
        lambda v: 3.0 * v,
        b,
        state=cg.init(b),
        precond=None,
    )

    _assert_pcg_info(info)
    assert int(info[mx.CG_ITERS]) == 0
    assert bool(info[mx.CG_CONVERGED])
    assert float(info[mx.CG_RESID]) == 0.0
    assert jnp.all(x == 0.0)


def test_maxiter_zero_returns_warm_start_unchanged(keys, make_spd, mv_from_dense):
    A = make_spd(n=32, key=keys(), cond=40.0, eps=1e-6)
    A_mv = mv_from_dense(A)
    b = jax.random.normal(keys(), (32,), dtype=jnp.float32)

    cg_ref = ConjugateGradient(
        backend="pcg",
        tol=1e-6,
        maxiter=200,
        stabilise_every=0,
        warm_start=True,
        preconditioned=False,
    )
    x_ref, info_ref, st = cg_ref.solve(A_mv, b, state=cg_ref.init(b), precond=None)
    assert int(info_ref[mx.CG_ITERS]) > 0

    cg0 = ConjugateGradient(
        backend="pcg",
        tol=1e-12,
        maxiter=0,
        stabilise_every=0,
        warm_start=True,
        preconditioned=False,
    )
    x0, info0, st2 = cg0.solve(A_mv, b, state=st, precond=None)

    assert int(info0[mx.CG_ITERS]) == 0
    assert jnp.allclose(_flat(x0), _flat(x_ref), rtol=0.0, atol=0.0)
    assert jnp.allclose(_flat(st2.last_x), _flat(x_ref), rtol=0.0, atol=0.0)


def test_warm_start_same_rhs_exits_immediately(keys):
    n = 32
    b = jax.random.normal(keys(), (n,), dtype=jnp.float32)

    def A_mv(x):
        return x

    cg = ConjugateGradient(
        backend="pcg",
        tol=1e-8,
        maxiter=8,
        stabilise_every=0,
        warm_start=True,
        preconditioned=False,
    )
    st = cg.init(b)
    x1, info1, st = cg.solve(A_mv, b, state=st, precond=None)
    x2, info2, st = cg.solve(A_mv, b, state=st, precond=None)

    assert int(info1[mx.CG_ITERS]) == 1
    assert int(info2[mx.CG_ITERS]) == 0
    assert jnp.allclose(x1, b, rtol=0.0, atol=0.0)
    assert jnp.allclose(x2, b, rtol=0.0, atol=0.0)
    assert isinstance(st, CGState)


def test_warm_start_nearby_rhs_reduces_iterations():
    dtype = jnp.float32
    diag = jnp.asarray([1.0, 2.0, 4.0, 8.0] * 4, dtype=dtype)
    b1 = jnp.ones_like(diag)
    b2 = b1.at[0].add(jnp.asarray(1.0e-2, dtype))

    def A_mv(x):
        return diag * x

    warm = ConjugateGradient(
        backend="pcg",
        tol=1e-7,
        maxiter=16,
        stabilise_every=0,
        warm_start=True,
        preconditioned=False,
    )
    cold = ConjugateGradient(
        backend="pcg",
        tol=1e-7,
        maxiter=16,
        stabilise_every=0,
        warm_start=False,
        preconditioned=False,
    )

    # Starting from the exact solution for b1 makes the warm residual for b2
    # lie in a single eigendirection. This avoids a random, flaky comparison.
    warm_state = CGState(last_x=b1 / diag)
    x_warm, info_warm, _ = warm.solve(
        A_mv, b2, state=warm_state, precond=None
    )
    x_cold, info_cold, _ = cold.solve(
        A_mv, b2, state=cold.init(b2), precond=None
    )

    _assert_pcg_info(info_warm)
    _assert_pcg_info(info_cold)
    assert int(info_warm[mx.CG_ITERS]) < int(info_cold[mx.CG_ITERS])
    assert _rel_resid(A_mv, x_warm, b2) <= 1.0e-6
    assert _rel_resid(A_mv, x_cold, b2) <= 1.0e-6


@pytest.mark.parametrize("preconditioned", [False, True])
def test_stabilized_internal_cg_paths_jit_run_twice(
        preconditioned, keys, make_spd, mv_from_dense
):
    A = make_spd(n=32, key=keys(), cond=200.0, eps=1e-6)
    A_mv = mv_from_dense(A)
    b1 = jax.random.normal(keys(), (32,), dtype=jnp.float32)
    b2 = b1 * jnp.asarray(1.001, b1.dtype)
    precond = (lambda r: r) if preconditioned else None
    stabilise_every = 5

    cg = ConjugateGradient(
        backend="pcg",
        tol=1e-6,
        maxiter=300,
        stabilise_every=stabilise_every,
        warm_start=True,
        preconditioned=preconditioned,
    )

    @jax.jit
    def run(rhs, st):
        x, info, st2 = cg.solve(A_mv, rhs, state=st, precond=precond)
        return x, info[mx.CG_RESID], info[mx.CG_ITERS], st2

    st0 = cg.init(b1)
    x1, rel1, iters1, st1 = run(b1, st0)
    x2, rel2, _, st2 = run(b2, st1)

    # The first solve must reach a replacement step; otherwise this test would
    # compile the stabilized core without exercising its replacement branch.
    assert int(iters1) >= stabilise_every
    assert jnp.isfinite(rel1) & jnp.isfinite(rel2)
    assert _rel_resid(A_mv, x1, b1) <= 2.0e-5
    assert _rel_resid(A_mv, x2, b2) <= 2.0e-5
    assert _tree_shape_dtype(st0) == _tree_shape_dtype(st1) == _tree_shape_dtype(st2)


@pytest.mark.parametrize("preconditioned", [False, True])
@pytest.mark.parametrize("stabilise_every", [12, 13])
def test_stabilise_every_ge_maxiter_is_effectively_off(
        preconditioned, stabilise_every, keys, make_spd, mv_from_dense
):
    _, b, A_mv = _make_vector_spd_problem(
        keys, make_spd, mv_from_dense, n=32, cond=80.0
    )
    maxiter = 12
    precond = (lambda r: r) if preconditioned else None

    kwargs = dict(
        backend="pcg",
        tol=0.0,
        maxiter=maxiter,
        warm_start=False,
        preconditioned=preconditioned,
    )
    cg_off = ConjugateGradient(stabilise_every=0, **kwargs)
    cg_ge = ConjugateGradient(stabilise_every=stabilise_every, **kwargs)

    x_off, info_off, _ = cg_off.solve(
        A_mv, b, state=cg_off.init(b), precond=precond
    )
    x_ge, info_ge, _ = cg_ge.solve(
        A_mv, b, state=cg_ge.init(b), precond=precond
    )

    _assert_pcg_info(info_off)
    _assert_pcg_info(info_ge)
    assert int(info_off[mx.CG_ITERS]) == maxiter
    assert int(info_ge[mx.CG_ITERS]) == maxiter
    assert jnp.allclose(x_ge, x_off, rtol=0.0, atol=0.0)
    assert jnp.allclose(
        info_ge[mx.CG_RESID], info_off[mx.CG_RESID], rtol=0.0, atol=0.0
    )


def test_pcg_jit_warm_start_state_shape_stable(keys, make_spd, mv_from_dense):
    A = make_spd(n=32, key=keys(), cond=120.0, eps=1e-6)
    A_mv = mv_from_dense(A)
    b = jax.random.normal(keys(), (32,), dtype=jnp.float32)

    cg = ConjugateGradient(
        backend="pcg",
        tol=1e-6,
        maxiter=300,
        stabilise_every=0,
        warm_start=True,
        preconditioned=False,
    )

    @jax.jit
    def run(rhs, st):
        x, info, st2 = cg.solve(A_mv, rhs, state=st, precond=None)
        return x, info[mx.CG_RESID], st2

    st0 = cg.init(b)
    _, rel1, st1 = run(b, st0)
    _, rel2, st2 = run(b * jnp.asarray(1.001, b.dtype), st1)

    assert jnp.isfinite(rel1) & jnp.isfinite(rel2)
    assert _tree_shape_dtype(st0) == _tree_shape_dtype(st1) == _tree_shape_dtype(st2)


def test_pcg_rejects_explicit_solve_dtype():
    with pytest.raises(ValueError, match="solve_dtype=None"):
        ConjugateGradient(
            backend="pcg",
            solve_dtype=jnp.float32,
            stabilise_every=0,
            warm_start=False,
            preconditioned=False,
        )


# -----------------------------------------------------------------------------
# Preconditioned CG behavior
# -----------------------------------------------------------------------------


def test_preconditioned_flag_enforced(keys, make_spd, mv_from_dense):
    A, b, A_mv = _make_vector_spd_problem(keys, make_spd, mv_from_dense, n=16, cond=50.0)
    invd = 1.0 / jnp.clip(jnp.diag(A), 1e-8, jnp.inf)
    M_inv = lambda v: invd * v

    cg_plain = ConjugateGradient(
        backend="pcg", preconditioned=False, tol=1e-6, maxiter=5, stabilise_every=0
    )
    with pytest.raises(ValueError):
        cg_plain.solve(A_mv, b, state=cg_plain.init(b), precond=M_inv)

    cg_prec = ConjugateGradient(
        backend="pcg", preconditioned=True, tol=1e-6, maxiter=5, stabilise_every=0
    )
    with pytest.raises(ValueError):
        cg_prec.solve(A_mv, b, state=cg_prec.init(b), precond=None)


def test_identity_preconditioner_matches_unpreconditioned_fixed_budget(keys, make_spd, mv_from_dense):
    A, b, A_mv = _make_vector_spd_problem(keys, make_spd, mv_from_dense, n=64, cond=200.0)
    K = 40

    cg = ConjugateGradient(
        backend="pcg", preconditioned=False, tol=0.0, maxiter=K, stabilise_every=0, warm_start=False
    )
    pcg = ConjugateGradient(
        backend="pcg", preconditioned=True, tol=0.0, maxiter=K, stabilise_every=0, warm_start=False
    )

    x_cg, info_cg, _ = cg.solve(A_mv, b, state=cg.init(b), precond=None)
    x_pcg, info_pcg, _ = pcg.solve(A_mv, b, state=pcg.init(b), precond=lambda r: r)

    assert int(info_cg[mx.CG_ITERS]) == K
    assert int(info_pcg[mx.CG_ITERS]) == K
    assert jnp.allclose(x_pcg, x_cg, rtol=2e-6, atol=2e-6)


def _make_diag_plus_lowrank_spd(key, n: int, rank: int, dtype):
    kD, kU, kb = jax.random.split(key, 3)
    del kD
    diag = jnp.exp(jnp.linspace(jnp.asarray(0.0, dtype), jnp.asarray(6.0, dtype), n))
    U = jnp.asarray(1e-2, dtype) * jax.random.normal(kU, (n, rank), dtype=dtype)
    A = jnp.diag(diag) + U @ U.T
    b = jax.random.normal(kb, (n,), dtype=dtype)
    return A, b, diag


@pytest.mark.parametrize("use_pytree", [False, True])
def test_diag_preconditioner_improves_diag_dominated_fixed_budget(
        use_pytree, keys, mv_from_dense, make_block_pytree
):
    dtype = jnp.float32
    A, b_vec, diag_vec = _make_diag_plus_lowrank_spd(keys(), n=64, rank=4, dtype=dtype)

    if use_pytree:
        _, pack, _ = make_block_pytree(shape_a=(40,), shape_b=(24,))
        b = pack(b_vec)
        diag = pack(diag_vec)
        A_mv = mv_from_dense(A, pack=pack)
    else:
        b = b_vec
        diag = diag_vec
        A_mv = mv_from_dense(A, pack=None)

    from somax.preconditioners.diag_direct import DiagDirectPrecond

    class _DummyOp:
        pass

    policy = DiagDirectPrecond(eps=0.0, add_lambda=False, clip_nonneg=True)
    M_inv, _ = policy.build(
        params=b,
        op=_DummyOp(),
        cstate=None,
        rng=None,
        lam=None,
        state=policy.init(b),
        diag=diag,
    )

    K = 12
    cg_plain = ConjugateGradient(
        backend="pcg", preconditioned=False, tol=0.0, maxiter=K, stabilise_every=0, warm_start=False
    )
    cg_prec = ConjugateGradient(
        backend="pcg", preconditioned=True, tol=0.0, maxiter=K, stabilise_every=0, warm_start=False
    )

    x0, _, _ = cg_plain.solve(A_mv, b, state=cg_plain.init(b), precond=None)
    x1, _, _ = cg_prec.solve(A_mv, b, state=cg_prec.init(b), precond=M_inv)

    rr0 = _rel_resid(A_mv, x0, b)
    rr1 = _rel_resid(A_mv, x1, b)
    assert rr1 <= 0.2 * rr0


# -----------------------------------------------------------------------------
# Reference backends smoke tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _available_reference_backends())
@pytest.mark.parametrize("use_pytree", [False, True])
def test_reference_backends_solve_spd_to_low_residual(
        backend, use_pytree, keys, make_spd, mv_from_dense, make_block_pytree
):
    if use_pytree:
        _, b, A_mv = _make_pytree_spd_problem(
            keys, make_spd, mv_from_dense, make_block_pytree, cond=60.0
        )
    else:
        _, b, A_mv = _make_vector_spd_problem(
            keys, make_spd, mv_from_dense, n=24, cond=60.0
        )

    cg = ConjugateGradient(
        backend=backend,
        tol=1e-6,
        maxiter=800,
        stabilise_every=0,
        warm_start=False,
        preconditioned=False,
        telemetry_residual=True,
    )
    x, info, _ = cg.solve(A_mv, b, state=cg.init(b), precond=None)

    assert mx.CG_MAXITER in info
    assert mx.CG_RESID in info
    assert _rel_resid(A_mv, x, b) <= 1.0e-4


def test_reference_backend_telemetry_residual_matches_true(keys, make_spd, mv_from_dense):
    A, b, A_mv = _make_vector_spd_problem(keys, make_spd, mv_from_dense, n=24, cond=50.0)

    cg = ConjugateGradient(
        backend="scipy",
        tol=1e-8,
        maxiter=800,
        stabilise_every=0,
        warm_start=False,
        preconditioned=False,
        telemetry_residual=True,
    )
    x, info, _ = cg.solve(A_mv, b, state=cg.init(b), precond=None)

    rr_true = _rel_resid(A_mv, x, b)
    assert mx.CG_RESID in info
    assert jnp.allclose(info[mx.CG_RESID], rr_true, rtol=1e-6, atol=1e-6)


def test_reference_backends_accept_explicit_solve_dtype(keys, make_spd, mv_from_dense):
    A, b, A_mv = _make_vector_spd_problem(keys, make_spd, mv_from_dense, n=24, cond=50.0)

    cg = ConjugateGradient(
        backend="scipy",
        tol=1e-6,
        maxiter=400,
        stabilise_every=0,
        warm_start=False,
        preconditioned=False,
        solve_dtype=jnp.float32,
        telemetry_residual=True,
    )
    x, info, _ = cg.solve(A_mv, b, state=cg.init(b), precond=None)

    assert mx.CG_RESID in info
    assert _rel_resid(A_mv, x, b) <= 1.0e-4


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        ConjugateGradient(backend="nope")
