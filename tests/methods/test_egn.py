import pytest

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import somax
from somax import metrics as mx
from somax.solvers.row_cg import RowConjugateGradient
from somax.solvers.row_cg_lifted import LiftedRowConjugateGradient


def test_egn_mse_smoke(key, medium_shapes, mlp_apply, make_teacher_regression, init_mlp_regression, mse_loss):
    B, D, H = medium_shapes["B"], medium_shapes["D"], medium_shapes["H"]
    _, batch = make_teacher_regression(key, B, D, H)
    params = init_mlp_regression(jax.random.PRNGKey(1), D, H)

    method = somax.make(
        "egn_mse",
        predict_fn=mlp_apply,
        lam0=1e-2,
        learning_rate=0.2,
    )
    st = method.init(params)

    @jax.jit
    def step(p, s, k):
        return method.step(p, batch, s, k)

    l0 = mse_loss(params, batch)
    for i in range(30):
        params, st, info = step(params, st, jax.random.fold_in(key, i))
    l1 = mse_loss(params, batch)

    assert jnp.isfinite(l0) and jnp.isfinite(l1)
    assert l1 < 0.1 * l0
    assert l1 < 1e-4
    assert int(st.step) == 30
    assert int(info["step"]) == 29
    assert method.plan.lane == "row"


def test_egn_ce_smoke(key, medium_shapes, mlp_logits, make_teacher_classification, init_mlp_classification, ce_loss,
                      accuracy):
    B, D, H, C = medium_shapes["B"], medium_shapes["D"], medium_shapes["H"], medium_shapes["C"]
    _, batch = make_teacher_classification(key, B, D, H, C)
    params = init_mlp_classification(jax.random.PRNGKey(2), D, H, C)

    method = somax.make(
        "egn_ce",
        predict_fn=mlp_logits,
        lam0=1e-2,
        learning_rate=0.2,
    )
    st = method.init(params)

    @jax.jit
    def step(p, s, k):
        return method.step(p, batch, s, k)

    l0, a0 = ce_loss(params, batch), accuracy(params, batch)
    for i in range(40):
        params, st, _ = step(params, st, jax.random.fold_in(key, 1000 + i))
    l1, a1 = ce_loss(params, batch), accuracy(params, batch)

    assert jnp.isfinite(l1)
    assert l1 < 0.1 * l0
    assert a1 > a0 + 0.4
    assert a1 > 0.95
    assert method.plan.lane == "row"


def test_egn_mse_cg_smoke_decreases(
        key,
        medium_shapes,
        mlp_apply,
        make_teacher_regression,
        init_mlp_regression,
        mse_loss,
):
    B, D, H = medium_shapes["B"], medium_shapes["D"], medium_shapes["H"]
    _, batch = make_teacher_regression(key, B, D, H)
    params = init_mlp_regression(jax.random.PRNGKey(11), D, H)

    method = somax.make(
        "egn_mse_cg",
        predict_fn=mlp_apply,
        lam0=1e-2,
        tol=1e-6,
        maxiter=20,
        warm_start=False,
        stabilise_every=10,
        record_cg_stats=True,
        learning_rate=0.2,
    )
    st = method.init(params)

    @jax.jit
    def step(p, s, k):
        return method.step(p, batch, s, k)

    l0 = mse_loss(params, batch)
    info_last = None
    for i in range(35):
        params, st, info_last = step(params, st, jax.random.fold_in(key, i))
    l1 = mse_loss(params, batch)

    assert jnp.isfinite(l0) and jnp.isfinite(l1)
    assert l1 < 0.2 * l0
    assert l1 < 1e-4
    assert method.plan.lane == "row"
    assert int(st.step) == 35
    assert int(info_last["step"]) == 34


def test_egn_mse_cg_info_schema_and_optional_cg_stats(
        key,
        medium_shapes,
        mlp_apply,
        make_teacher_regression,
        init_mlp_regression,
):
    # Explicitly requested CG telemetry must be present and finite.
    B, D, H = medium_shapes["B"], medium_shapes["D"], medium_shapes["H"]
    _, batch = make_teacher_regression(key, B, D, H)
    params = init_mlp_regression(jax.random.PRNGKey(12), D, H)

    lam0 = 0.5
    method = somax.make(
        "egn_mse_cg",
        predict_fn=mlp_apply,
        lam0=lam0,
        tol=1e-6,
        maxiter=10,
        warm_start=False,
        stabilise_every=10,
        record_cg_stats=True,
        learning_rate=0.0,  # isolate info/state, do not move params
        record_lam=True,
    )
    st = method.init(params)

    params2, st2, info = method.step(params, batch, st, rng=jax.random.PRNGKey(0))

    assert int(st2.step) == 1
    assert int(info["step"]) == 0
    assert "lam_used" in info
    assert jnp.isfinite(info["lam_used"])
    assert float(info["lam_used"]) > 0.0

    assert 0 <= int(info[mx.CG_ITERS]) <= int(info[mx.CG_MAXITER])
    assert jnp.isfinite(info[mx.CG_RESID])
    assert jnp.asarray(info[mx.CG_CONVERGED]).shape == ()

    # lr=0 should keep params unchanged
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: jnp.all(a == b), params2, params)
    )


@pytest.mark.parametrize("kind", ["mse", "ce"])
@pytest.mark.parametrize("maxiter", [2, 5])
def test_egn_cg_lifted_matches_standard_step(
        kind, maxiter, key, small_shapes,
        mlp_apply, mlp_logits,
        init_mlp_regression, init_mlp_classification,
        make_teacher_regression, make_teacher_classification,
        monkeypatch,
):
    """Lifted and ordinary row CG produce equivalent optimizer updates."""
    B, D, H, C = (small_shapes[k] for k in ("B", "D", "H", "C"))

    if kind == "mse":
        _, batch = make_teacher_regression(key, B, D, H)
        params = init_mlp_regression(jax.random.PRNGKey(21), D, H)
        predict = mlp_apply
    else:
        _, batch = make_teacher_classification(key, B, D, H, C)
        params = init_mlp_classification(jax.random.PRNGKey(21), D, H, C)
        predict = mlp_logits

    options = dict(
        predict_fn=predict,
        lam0=0.3,
        maxiter=maxiter,
        tol=0.0,
        learning_rate=1.0,
        record_cg_stats=True,
    )
    ordinary = somax.make(f"egn_{kind}_cg", lifted=False, **options)
    lifted = somax.make(f"egn_{kind}_cg", lifted=True, **options)

    assert type(ordinary.solver) is RowConjugateGradient
    assert type(lifted.solver) is LiftedRowConjugateGradient
    assert ordinary.plan.lane == lifted.plan.lane == "row"

    def forbidden_solve(*args, **kwargs):
        pytest.fail("Lifted solver called ordinary solve()")

    monkeypatch.setattr(lifted.solver, "solve", forbidden_solve)

    p0, s0, i0 = jax.jit(ordinary.step)(
        params, batch, ordinary.init(params), key
    )
    p1, s1, i1 = jax.jit(lifted.step)(
        params, batch, lifted.init(params), key
    )

    # Compare actual parameter updates, not parameter values.
    flat, _ = ravel_pytree(params)
    flat0, _ = ravel_pytree(p0)
    flat1, _ = ravel_pytree(p1)

    delta0 = flat0 - flat
    delta1 = flat1 - flat

    assert jnp.all(jnp.isfinite(delta0))
    assert jnp.all(jnp.isfinite(delta1))

    rel_error = jnp.linalg.norm(delta1 - delta0) / jnp.maximum(
        jnp.linalg.norm(delta0), 1e-12
    )
    assert float(rel_error) < 2e-4

    # Same CG behavior and optimizer-step progression.
    assert int(s0.step) == int(s1.step) == 1
    assert 0 < int(i0[mx.CG_ITERS]) == int(i1[mx.CG_ITERS]) <= maxiter
    assert bool(i0[mx.CG_CONVERGED]) == bool(i1[mx.CG_CONVERGED])

    assert jnp.isfinite(i0[mx.CG_RESID])
    assert jnp.isfinite(i1[mx.CG_RESID])
    assert jnp.allclose(
        i0[mx.CG_RESID], i1[mx.CG_RESID],
        rtol=1e-5, atol=1e-7,
    )
