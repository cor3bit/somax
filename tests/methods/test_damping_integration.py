import jax
import jax.numpy as jnp
import pytest

import somax
from somax import metrics as mx


def _run_steps(method, params, batch, key, n_steps: int):
    st = method.init(params)

    @jax.jit
    def step(p, s, k):
        return method.step(p, batch, s, k)

    info_last = None
    for i in range(n_steps):
        params, st, info_last = step(params, st, jax.random.fold_in(key, i))
    return params, st, info_last


# -------------------------
# EGN (param lane) damping
# -------------------------


def test_egn_mse_tr_damping_decreases_loss(
        key, medium_shapes, mlp_apply, make_teacher_regression, init_mlp_regression, mse_loss
):
    B, D, H = medium_shapes["B"], medium_shapes["D"], medium_shapes["H"]
    _, batch = make_teacher_regression(key, B, D, H)
    params = init_mlp_regression(jax.random.PRNGKey(3), D, H)

    method = somax.make(
        "egn_mse",
        predict_fn=mlp_apply,
        lam_policy="tr",
        lam0=1e-2,
        lam_kwargs={
            "every_k": 1,
            "lower": 0.25,
            "upper": 0.75,
            "inc": 1.5,
            "dec": 0.5,
            "gain_ratio_clip": 5.0,
        },
        learning_rate=0.2,
    )

    l0 = mse_loss(params, batch)
    params2, st2, _ = _run_steps(method, params, batch, key, n_steps=20)
    l1 = mse_loss(params2, batch)

    assert jnp.isfinite(l1)
    assert l1 < l0


# -------------------------
# SGN (param lane) damping
# -------------------------


def test_sgn_mse_tr_damping_decreases_loss(
        key, medium_shapes, mlp_apply, make_teacher_regression, init_mlp_regression, mse_loss
):
    B, D, H = medium_shapes["B"], medium_shapes["D"], medium_shapes["H"]
    _, batch = make_teacher_regression(key, B, D, H)
    params = init_mlp_regression(jax.random.PRNGKey(4), D, H)

    method = somax.make(
        "sgn_mse",
        predict_fn=mlp_apply,
        lam_policy="tr",
        lam0=1e-2,
        lam_kwargs={
            "every_k": 1,
            "lower": 0.25,
            "upper": 0.75,
            "inc": 1.5,
            "dec": 0.5,
            "gain_ratio_clip": 5.0,
        },
        tol=1e-6,
        maxiter=10,
        learning_rate=0.2,
    )

    l0 = mse_loss(params, batch)
    params2, st2, _ = _run_steps(method, params, batch, key, n_steps=20)
    l1 = mse_loss(params2, batch)

    assert jnp.isfinite(l1)
    assert l1 < l0


# -------------------------
# FGN (row lane) damping
# -------------------------


def _linear_logits(params, x):
    return x @ params["w"] + params["b"]


@pytest.mark.parametrize(
    ("method_name", "expected_lane"),
    [("fgn_ce", "row"), ("sgn_ce", "param")],
)
def test_scheduled_damping_sequence_is_used_by_compiled_ce_methods(
        method_name,
        expected_lane,
        key,
):
    params = {
        "w": jnp.asarray(
            [[0.2, -0.1, 0.05], [-0.3, 0.15, 0.25]],
            jnp.float32,
        ),
        "b": jnp.asarray([0.05, -0.02, 0.01], jnp.float32),
    }
    batch = {
        "x": jnp.asarray(
            [[1.0, -0.5], [0.2, 0.7], [-0.8, 0.3], [0.4, -1.0]],
            jnp.float32,
        ),
        "y": jnp.asarray([0, 1, 2, 0], jnp.int32),
    }
    lam0 = 0.8

    def factor_schedule(step):
        return 1.0 / (step.astype(jnp.float32) + 1.0)

    method = somax.make(
        method_name,
        predict_fn=_linear_logits,
        lam_policy="scheduled",
        lam0=lam0,
        lam_kwargs={"factor_schedule": factor_schedule},
        tol=0.0,
        maxiter=1,
        warm_start=False,
        learning_rate=0.0,
        record_lam=True,
        record_cg_stats=True,
    )
    assert method.plan.lane == expected_lane
    state = method.init(params)

    @jax.jit
    def step(p, s, rng):
        return method.step(p, batch, s, rng)

    for outer_step in range(4):
        params, state, info = step(
            params,
            state,
            jax.random.fold_in(key, outer_step),
        )
        expected_used = lam0 / (outer_step + 1.0)
        expected_next = lam0 / (outer_step + 2.0)

        assert jnp.allclose(info[mx.LAM_USED], expected_used, rtol=1e-6, atol=1e-7)
        assert jnp.allclose(info[mx.LAM_NEXT], expected_next, rtol=1e-6, atol=1e-7)
        assert jnp.allclose(state.damp_state.lam, expected_next, rtol=1e-6, atol=1e-7)
        assert jnp.isfinite(info[mx.CG_RESID])
        assert int(state.step) == outer_step + 1
