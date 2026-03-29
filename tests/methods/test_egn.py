import jax
import jax.numpy as jnp

import somax


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
        warm_start=True,
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
    # Telemetry is allowed to be gating-driven. This test enforces:
    # - step and lam_used are present and sane
    # - if CG stats are emitted, they are finite and nonnegative
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
        warm_start=True,
        stabilise_every=10,
        record_cg_stats=True,
        learning_rate=0.0,  # isolate info/state, do not move params
        record_lam=True,
    )
    st = method.init(params)

    params2, st2, info = method.step(params, batch, st, rng=jax.random.PRNGKey(0))

    assert int(st2.step) == 1
    assert int(info["step"]) == 0
    # lam_used is already commonly emitted by damping policies; keep this invariant.
    assert "lam_used" in info
    assert jnp.isfinite(info["lam_used"])
    assert float(info["lam_used"]) > 0.0

    # Optional CG stats (names may differ by backend). Validate whichever exist.
    maybe_int_keys = ("cg_iters", "cg_num_iters", "iters", "num_iters")
    for k in maybe_int_keys:
        if k in info:
            assert int(info[k]) >= 0

    maybe_float_keys = ("cg_resid", "cg_r_norm", "resid", "r_norm")
    for k in maybe_float_keys:
        if k in info:
            assert jnp.isfinite(info[k])

    # lr=0 should keep params unchanged
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: jnp.all(a == b), params2, params)
    )


def test_egn_mse_cg_warm_start_flag_does_not_change_numerics_on_first_step(
        key,
        medium_shapes,
        mlp_apply,
        make_teacher_regression,
        init_mlp_regression,
        mse_loss,
):
    # First step should be warm-start independent (no previous solution exists).
    B, D, H = medium_shapes["B"], medium_shapes["D"], medium_shapes["H"]
    _, batch = make_teacher_regression(key, B, D, H)
    params0 = init_mlp_regression(jax.random.PRNGKey(13), D, H)

    def run_once(warm_start: bool):
        method = somax.make(
            "egn_mse_cg",
            predict_fn=mlp_apply,
            lam0=1e-2,
            tol=1e-6,
            maxiter=10,
            warm_start=warm_start,
            stabilise_every=10,
            record_cg_stats=False,
            learning_rate=0.2,
        )
        st = method.init(params0)
        p1, s1, info1 = method.step(params0, batch, st, rng=jax.random.PRNGKey(0))
        return p1, s1, info1, method

    p_ws, _, _, m_ws = run_once(True)
    p_cold, _, _, m_cold = run_once(False)

    assert m_ws.plan.lane == "row"
    assert m_cold.plan.lane == "row"
    # Allow tiny float32 jitter; first step should match closely.
    flat_ws, _ = jax.flatten_util.ravel_pytree(p_ws)
    flat_cold, _ = jax.flatten_util.ravel_pytree(p_cold)
    assert jnp.allclose(flat_ws, flat_cold, atol=1e-6, rtol=1e-6)
    assert jnp.isfinite(mse_loss(p_ws, batch))
