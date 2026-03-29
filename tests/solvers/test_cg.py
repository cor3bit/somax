import pytest

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import somax.metrics as mx
from somax.utils import tree_vdot
from somax.solvers.cg import ConjugateGradient


def _rel_resid(A_mv, x, b):
    r = jtu.tree_map(lambda Ax, bb: Ax - bb, A_mv(x), b)
    r2 = tree_vdot(r, r)
    b2 = tree_vdot(b, b)
    dt = jnp.result_type(*[t.dtype for t in jtu.tree_leaves(b)])
    tiny = jnp.asarray(jnp.finfo(dt).tiny, dt)
    return jnp.sqrt(r2 / jnp.maximum(b2, tiny))


def _available_backends():
    backends = ["pcg", "scipy"]
    try:
        import lineax as _  # noqa: F401

        backends.append("lineax")
    except Exception:
        pass
    return backends


@pytest.mark.parametrize("backend", _available_backends())
@pytest.mark.parametrize("use_pytree", [False, True])
def test_all_backends_solve_spd_to_low_residual(
        backend, use_pytree, key, make_spd, mv_from_dense, make_block_pytree
):
    # Keep problem small so reference backends are stable/fast.
    if use_pytree:
        template, pack, _zeros_like = make_block_pytree(shape_a=(9,), shape_b=(7,))
        n = int(jax.flatten_util.ravel_pytree(template)[0].shape[0])
        kA, kb = jax.random.split(key)
        A = make_spd(n=n, key=kA, cond=60.0, eps=1e-6)
        b = pack(jax.random.normal(kb, (n,), dtype=jnp.float32))
        A_mv = mv_from_dense(A, pack=pack)
    else:
        kA, kb = jax.random.split(key)
        n = 32
        A = make_spd(n=n, key=kA, cond=60.0, eps=1e-6)
        b = jax.random.normal(kb, (n,), dtype=jnp.float32)
        A_mv = mv_from_dense(A, pack=None)

    # For backend parity: unpreconditioned solve.
    cg = ConjugateGradient(
        backend=backend,
        tol=1e-6,
        maxiter=800,
        stabilise_every=10,
        warm_start=False,
        preconditioned=False,
        telemetry_residual=True,  # references compute exact residual; pcg already provides one
        assume_spd=True,
    )
    st = cg.init(b)
    x, info, _ = cg.solve(A_mv, b, state=st, precond=None)

    rr = _rel_resid(A_mv, x, b)

    # Core PCG reports a residual; reference backends should also populate it if telemetry_residual=True.
    assert mx.CG_MAXITER in info
    assert jnp.isfinite(rr)

    # Loose-but-meaningful bound for float32 across backends.
    assert rr <= 1.0e-4


# -----------------------
# PCG-specific: correctness
# -----------------------


@pytest.mark.parametrize("use_pytree", [False, True])
def test_pcg_converges_on_spd_float32(
        use_pytree, key, make_spd, mv_from_dense, make_block_pytree
):
    if use_pytree:
        template, pack, _zeros_like = make_block_pytree(shape_a=(7,), shape_b=(5,))
        n = int(jax.flatten_util.ravel_pytree(template)[0].shape[0])
        kA, kb = jax.random.split(key)
        A = make_spd(n=n, key=kA, cond=80.0, eps=1e-6)
        b = pack(jax.random.normal(kb, (n,), dtype=jnp.float32))
        A_mv = mv_from_dense(A, pack=pack)
    else:
        kA, kb = jax.random.split(key)
        A = make_spd(n=32, key=kA, cond=80.0, eps=1e-6)
        b = jax.random.normal(kb, (32,), dtype=jnp.float32)
        A_mv = mv_from_dense(A, pack=None)

    cg = ConjugateGradient(
        backend="pcg",
        tol=1e-6,
        maxiter=600,
        stabilise_every=10,
        warm_start=False,
        preconditioned=False,
        assume_spd=True,
    )
    st = cg.init(b)
    x, info, _ = cg.solve(A_mv, b, state=st, precond=None)

    # Schema: must include these keys for pcg.
    assert mx.CG_ITERS in info
    assert mx.CG_RESID in info
    assert mx.CG_CONVERGED in info
    assert mx.CG_MAXITER in info

    rr_true = _rel_resid(A_mv, x, b)
    assert rr_true <= 7.0e-6


def test_pcg_maxiter_zero_returns_initial_guess(keys, make_spd, mv_from_dense):
    kA = keys()
    n = 32
    A = make_spd(n=n, key=kA, cond=40.0, eps=1e-6)
    A_mv = mv_from_dense(A)

    b = jax.random.normal(keys(), (n,), dtype=jnp.float32)

    # First, get a nontrivial warm-start state from a normal solve.
    cg_ref = ConjugateGradient(
        backend="pcg",
        tol=1e-6,
        maxiter=200,
        stabilise_every=10,
        warm_start=True,
        preconditioned=False,
        assume_spd=True,
    )
    st = cg_ref.init(b)
    x_ref, info_ref, st = cg_ref.solve(A_mv, b, state=st, precond=None)
    assert int(info_ref[mx.CG_ITERS]) > 0

    # Now: maxiter=0 should return x0 exactly (here: the warm-start x_ref).
    cg0 = ConjugateGradient(
        backend="pcg",
        tol=1e-12,  # irrelevant
        maxiter=0,
        stabilise_every=0,
        warm_start=True,
        preconditioned=False,
        assume_spd=True,
    )
    x0, info0, st2 = cg0.solve(A_mv, b, state=st, precond=None)

    x0_f = jax.flatten_util.ravel_pytree(x0)[0]
    x_ref_f = jax.flatten_util.ravel_pytree(x_ref)[0]
    assert jnp.allclose(x0_f, x_ref_f, rtol=0.0, atol=0.0)
    assert int(info0[mx.CG_ITERS]) == 0
    # state updated (still warm_start) but should stay equal to x0
    st2_f = jax.flatten_util.ravel_pytree(st2.last_x)[0]
    assert jnp.allclose(st2_f, x_ref_f, rtol=0.0, atol=0.0)


def test_pcg_warm_start_reduces_iters(keys, make_spd, mv_from_dense):
    kA = keys()
    n = 96
    A = make_spd(n=n, key=kA, cond=500.0, eps=1e-6)
    A_mv = mv_from_dense(A)

    b1 = jax.random.normal(keys(), (n,), dtype=jnp.float32)
    b2 = b1 + 1e-2 * jax.random.normal(keys(), (n,), dtype=jnp.float32)

    cg = ConjugateGradient(
        backend="pcg",
        tol=1e-6,
        maxiter=600,
        stabilise_every=10,
        warm_start=True,
        preconditioned=False,
        assume_spd=True,
    )
    st = cg.init(b1)

    _, info1, st = cg.solve(A_mv, b1, state=st, precond=None)
    _, info2, _ = cg.solve(A_mv, b2, state=st, precond=None)

    assert int(info2[mx.CG_ITERS]) <= int(info1[mx.CG_ITERS])


def test_pcg_jit_runs_twice_no_crash(keys, make_spd, mv_from_dense):
    A = make_spd(n=32, key=keys(), cond=200.0, eps=1e-6)
    A_mv = mv_from_dense(A)

    cg = ConjugateGradient(
        backend="pcg",
        tol=1e-6,
        maxiter=300,
        stabilise_every=5,
        warm_start=True,
        preconditioned=False,
        assume_spd=True,
    )

    @jax.jit
    def run(rhs, st):
        x, info, st2 = cg.solve(A_mv, rhs, state=st, precond=None)
        return x, info[mx.CG_RESID], st2

    b = jax.random.normal(keys(), (32,), dtype=jnp.float32)
    st0 = cg.init(b)

    _, rel1, st1 = run(b, st0)
    _, rel2, _ = run(b * 1.001, st1)

    assert jnp.isfinite(rel1) & jnp.isfinite(rel2)


def test_pcg_stabilise_every_ge_maxiter_is_effectively_off(keys, make_spd, mv_from_dense):
    A = make_spd(n=48, key=keys(), cond=300.0, eps=1e-6)
    A_mv = mv_from_dense(A)
    b = jax.random.normal(keys(), (48,), dtype=jnp.float32)

    cg_off = ConjugateGradient(
        backend="pcg",
        tol=1e-6,
        maxiter=40,
        stabilise_every=0,
        warm_start=False,
        preconditioned=False,
        assume_spd=True,
    )
    cg_ge = ConjugateGradient(
        backend="pcg",
        tol=1e-6,
        maxiter=40,
        stabilise_every=40,  # should never trigger
        warm_start=False,
        preconditioned=False,
        assume_spd=True,
    )

    x0, info0, _ = cg_off.solve(A_mv, b, state=cg_off.init(b), precond=None)
    x1, info1, _ = cg_ge.solve(A_mv, b, state=cg_ge.init(b), precond=None)

    # Not necessarily identical iter-by-iter, but should solve to comparable residual.
    rr0 = _rel_resid(A_mv, x0, b)
    rr1 = _rel_resid(A_mv, x1, b)
    assert rr0 <= 2e-5
    assert rr1 <= 2e-5
    assert jnp.isfinite(info0[mx.CG_RESID])
    assert jnp.isfinite(info1[mx.CG_RESID])


# -----------------------
# PCG-specific: preconditioner behavior + edge cases
# -----------------------


def test_preconditioned_flag_enforced(keys, make_spd, mv_from_dense):
    A = make_spd(n=16, key=keys(), cond=50.0, eps=1e-6)
    A_mv = mv_from_dense(A)
    b = jax.random.normal(keys(), (16,), dtype=jnp.float32)

    d = jnp.clip(jnp.diag(A), 1e-8, jnp.inf)
    invd = 1.0 / d
    M_inv = lambda v: invd * v

    cg_plain = ConjugateGradient(backend="pcg", preconditioned=False, tol=1e-6, maxiter=5)
    st = cg_plain.init(b)
    with pytest.raises(ValueError):
        cg_plain.solve(A_mv, b, state=st, precond=M_inv)

    cg_prec = ConjugateGradient(backend="pcg", preconditioned=True, tol=1e-6, maxiter=5)
    st = cg_prec.init(b)
    with pytest.raises(ValueError):
        cg_prec.solve(A_mv, b, state=st, precond=None)


def test_preconditioning_helps_fixed_budget(keys, make_spd, mv_from_dense):
    # Construct a diagonally scaled SPD to make Jacobi meaningful.
    n = 128
    A0 = make_spd(n=n, key=keys(), cond=1000.0, eps=1e-6)
    scale = jnp.exp(jnp.linspace(-1.5, 1.5, n)).astype(jnp.float32)
    D = jnp.diag(jnp.sqrt(scale))
    A = D @ A0 @ D

    b = jax.random.normal(keys(), (n,), dtype=jnp.float32)

    d = jnp.clip(jnp.diag(A), 1e-8, jnp.inf)
    invd = 1.0 / d
    M_inv = lambda v: invd * v

    A_mv = mv_from_dense(A)

    cg0 = ConjugateGradient(
        backend="pcg",
        preconditioned=False,
        tol=1e-8,
        maxiter=20,
        stabilise_every=10,
        warm_start=False,
        assume_spd=True,
    )
    cg1 = ConjugateGradient(
        backend="pcg",
        preconditioned=True,
        tol=1e-8,
        maxiter=20,
        stabilise_every=10,
        warm_start=False,
        assume_spd=True,
    )

    _, info0, _ = cg0.solve(A_mv, b, state=cg0.init(b), precond=None)
    _, info1, _ = cg1.solve(A_mv, b, state=cg1.init(b), precond=M_inv)

    rr0 = float(info0[mx.CG_RESID])
    rr1 = float(info1[mx.CG_RESID])

    # Hardness sanity: unpreconditioned should not magically hit tol in 20 iters.
    assert rr0 > 5e-2
    # Preconditioning should help, but keep threshold loose to avoid seed/device flakiness.
    assert rr1 < 0.99 * rr0


def test_pcg_precond_identity_matches_unpreconditioned(keys, make_spd, mv_from_dense):
    A = make_spd(n=64, key=keys(), cond=200.0, eps=1e-6)
    A_mv = mv_from_dense(A)
    b = jax.random.normal(keys(), (64,), dtype=jnp.float32)

    I = lambda v: v

    cg0 = ConjugateGradient(
        backend="pcg",
        preconditioned=False,
        tol=1e-7,
        maxiter=300,
        stabilise_every=10,
        warm_start=False,
        assume_spd=True,
    )
    cg1 = ConjugateGradient(
        backend="pcg",
        preconditioned=True,
        tol=1e-7,
        maxiter=300,
        stabilise_every=10,
        warm_start=False,
        assume_spd=True,
    )

    x0, info0, _ = cg0.solve(A_mv, b, state=cg0.init(b), precond=None)
    x1, info1, _ = cg1.solve(A_mv, b, state=cg1.init(b), precond=I)

    rr0 = _rel_resid(A_mv, x0, b)
    rr1 = _rel_resid(A_mv, x1, b)

    # Both should solve well; identity precond shouldn't catastrophically change behavior.
    assert rr0 <= 3e-5
    assert rr1 <= 3e-5
    assert jnp.isfinite(info0[mx.CG_RESID])
    assert jnp.isfinite(info1[mx.CG_RESID])


def test_pcg_with_identity_precond_matches_cg(keys, mv_from_dense, make_spd):
    n = 64
    A = make_spd(n=n, key=keys(), cond=200.0, eps=1e-6)
    b = jax.random.normal(keys(), (n,), dtype=jnp.float32)
    A_mv = mv_from_dense(A, pack=None)

    K = 40

    cg = ConjugateGradient(backend="pcg", preconditioned=False, tol=0.0, maxiter=K, stabilise_every=K, warm_start=False)
    pcg = ConjugateGradient(backend="pcg", preconditioned=True, tol=0.0, maxiter=K, stabilise_every=K, warm_start=False)

    x_cg, _, _ = cg.solve(A_mv, b, state=cg.init(b), precond=None)
    x_pcg, _, _ = pcg.solve(A_mv, b, state=pcg.init(b), precond=lambda r: r)

    rr_cg = _rel_resid(A_mv, x_cg, b)
    rr_pcg = _rel_resid(A_mv, x_pcg, b)

    assert rr_pcg <= 1.05 * rr_cg


def _make_gn_system(key, n: int, bsz: int, mu: float, dtype):
    kG, kb = jax.random.split(key, 2)
    G = jax.random.normal(kG, (bsz, n), dtype=dtype)
    A = (G.T @ G) / jnp.asarray(bsz, dtype=dtype) + jnp.asarray(mu, dtype=dtype) * jnp.eye(n, dtype=dtype)
    b = jax.random.normal(kb, (n,), dtype=dtype)
    diag = jnp.mean(G * G, axis=0) + jnp.asarray(mu, dtype=dtype)
    return A, b, diag


def _make_diag_plus_lowrank_spd(key, n: int, rank: int, dtype):
    """Construct an SPD system A = D + U U^T with a wide diagonal spectrum.

    Purpose: deterministic-ish unit-test instance where Jacobi preconditioning
    (using diag(D)) should provide a clear benefit under a fixed CG budget.

    Args:
      key: PRNGKey
      n: dimension
      rank: low-rank factor rank
      dtype: jnp dtype (e.g., jnp.float32)

    Returns:
      A: (n,n) SPD matrix
      b: (n,) rhs
      diag: (n,) diagonal used for Jacobi (diag(D))
    """
    kD, kU, kb = jax.random.split(key, 3)

    # Diagonal with large dynamic range: exp(linspace(0, 6)) ~ [1, 1e6].
    exponents = jnp.linspace(jnp.asarray(0.0, dtype), jnp.asarray(6.0, dtype), n)
    diag = jnp.exp(exponents)
    D = jnp.diag(diag)

    # Small low-rank perturbation to avoid being purely diagonal.
    U = jnp.asarray(1e-2, dtype) * jax.random.normal(kU, (n, rank), dtype=dtype)
    A = D + (U @ U.T)

    b = jax.random.normal(kb, (n,), dtype=dtype)
    return A, b, diag


@pytest.mark.parametrize("use_pytree", [False, True])
def test_pcg_accepts_diag_direct_batch_grad_squared_diag(
        use_pytree, keys, mv_from_dense, make_block_pytree
):
    dtype = jnp.float32
    mu = 1e-2
    bsz = 32
    n = 64

    A, b_vec, diag_vec = _make_gn_system(keys(), n=n, bsz=bsz, mu=mu, dtype=dtype)

    if use_pytree:
        _template, pack, _zeros_like = make_block_pytree(shape_a=(40,), shape_b=(24,))
        b = pack(b_vec)
        diag = pack(diag_vec)
        A_mv = mv_from_dense(A, pack=pack)
    else:
        b = b_vec
        diag = diag_vec
        A_mv = mv_from_dense(A, pack=None)

    from somax.preconditioners.diag_direct import DiagDirectPrecond

    policy = DiagDirectPrecond(eps=1e-8, add_lambda=False, clip_nonneg=True)

    class _DummyOp:
        pass

    M_inv, _pst = policy.build(
        params=b,
        op=_DummyOp(),
        cstate=None,
        rng=None,
        lam=None,
        state=policy.init(b),
        diag=diag,
    )

    cg_prec = ConjugateGradient(
        backend="pcg",
        preconditioned=True,
        tol=1e-6,
        maxiter=60,
        stabilise_every=60,  # disable stabilization to avoid conflating behaviors
        warm_start=False,
        assume_spd=True,
    )

    x, info, _ = cg_prec.solve(A_mv, b, state=cg_prec.init(b), precond=M_inv)

    assert mx.CG_ITERS in info
    assert mx.CG_RESID in info
    assert mx.CG_CONVERGED in info
    assert jnp.isfinite(info[mx.CG_RESID])

    # Also verify solution is finite and same structure as b.
    xb = jax.flatten_util.ravel_pytree(x)[0]
    assert jnp.all(jnp.isfinite(xb))


@pytest.mark.parametrize("use_pytree", [False, True])
def test_pcg_diag_direct_improves_on_diag_dominated_system(
        use_pytree, keys, mv_from_dense, make_block_pytree
):
    dtype = jnp.float32
    n = 64
    rank = 4

    A, b_vec, diag_vec = _make_diag_plus_lowrank_spd(keys(), n=n, rank=rank, dtype=dtype)

    if use_pytree:
        _template, pack, _ = make_block_pytree(shape_a=(40,), shape_b=(24,))
        b = pack(b_vec)
        diag = pack(diag_vec)
        A_mv = mv_from_dense(A, pack=pack)
    else:
        b = b_vec
        diag = diag_vec
        A_mv = mv_from_dense(A, pack=None)

    from somax.preconditioners.diag_direct import DiagDirectPrecond
    policy = DiagDirectPrecond(eps=0.0, add_lambda=False, clip_nonneg=True)

    class _DummyOp:
        pass

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
    cg_plain = ConjugateGradient(backend="pcg", preconditioned=False, tol=0.0, maxiter=K, stabilise_every=K,
                                 warm_start=False, assume_spd=True)
    cg_prec = ConjugateGradient(backend="pcg", preconditioned=True, tol=0.0, maxiter=K, stabilise_every=K,
                                warm_start=False, assume_spd=True)

    x0, _, _ = cg_plain.solve(A_mv, b, state=cg_plain.init(b), precond=None)
    x1, _, _ = cg_prec.solve(A_mv, b, state=cg_prec.init(b), precond=M_inv)

    rr0 = _rel_resid(A_mv, x0, b)
    rr1 = _rel_resid(A_mv, x1, b)

    assert rr1 <= 0.2 * rr0


# -----------------------
# Reference backend behavior / invariants
# -----------------------


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        ConjugateGradient(backend="nope")


def test_reference_backend_telemetry_residual_matches_true(keys, make_spd, mv_from_dense):
    # Use scipy backend (always available in JAX).
    A = make_spd(n=24, key=keys(), cond=50.0, eps=1e-6)
    A_mv = mv_from_dense(A)
    b = jax.random.normal(keys(), (24,), dtype=jnp.float32)

    cg = ConjugateGradient(
        backend="scipy",
        tol=1e-8,
        maxiter=800,
        warm_start=False,
        preconditioned=False,
        telemetry_residual=True,
        assume_spd=True,
    )
    st = cg.init(b)
    x, info, _ = cg.solve(A_mv, b, state=st, precond=None)

    assert mx.CG_RESID in info
    rr_true = _rel_resid(A_mv, x, b)
    assert jnp.allclose(info[mx.CG_RESID], rr_true, rtol=1e-6, atol=1e-6)
