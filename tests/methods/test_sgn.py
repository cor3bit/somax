import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

import somax
from somax import metrics as mx


def test_sgn_mse_smoke(key, medium_shapes, mlp_apply, make_teacher_regression, init_mlp_regression, mse_loss):
    B, D, H = medium_shapes["B"], medium_shapes["D"], medium_shapes["H"]
    _, batch = make_teacher_regression(key, B, D, H)
    params = init_mlp_regression(jax.random.PRNGKey(7), D, H)

    method = somax.make(
        "sgn_mse",
        predict_fn=mlp_apply,
        lam0=1e-2,
        tol=1e-3,
        maxiter=15,
        learning_rate=0.2,
    )
    st = method.init(params)

    @jax.jit
    def step(p, s, k):
        return method.step(p, batch, s, k)

    l0 = mse_loss(params, batch)
    for i in range(30):
        params, st, _ = step(params, st, jax.random.fold_in(key, i))
    l1 = mse_loss(params, batch)

    assert jnp.isfinite(l1)
    assert l1 < 0.1 * l0
    assert l1 < 1e-4
    assert method.plan.lane == "param"


def test_sgn_ce_smoke(key, medium_shapes, mlp_logits, make_teacher_classification,
                      init_mlp_classification, ce_loss, accuracy):
    B, D, H, C = medium_shapes["B"], medium_shapes["D"], medium_shapes["H"], medium_shapes["C"]
    _, batch = make_teacher_classification(key, B, D, H, C)
    params = init_mlp_classification(jax.random.PRNGKey(8), D, H, C)

    method = somax.make(
        "sgn_ce",
        predict_fn=mlp_logits,
        lam0=1e-2,
        tol=1e-3,
        maxiter=15,
        learning_rate=0.2,
    )
    st = method.init(params)

    @jax.jit
    def step(p, s, k):
        return method.step(p, batch, s, k)

    l0, a0 = ce_loss(params, batch), accuracy(params, batch)
    for i in range(40):
        params, st, _ = step(params, st, jax.random.fold_in(key, 100 + i))
    l1, a1 = ce_loss(params, batch), accuracy(params, batch)

    assert jnp.isfinite(l1)
    assert l1 < 0.1 * l0
    assert a1 > a0 + 0.4
    assert a1 > 0.95
    assert method.plan.lane == "param"


def test_sgn_ce_stability_saturated_logits(key, mlp_logits):
    # Stress: huge per-sample bias -> saturated softmax, ensure step stays finite.
    B, D, H, C = 32, 10, 8, 4
    x = jax.random.normal(key, (B, D))

    W1 = jax.random.normal(key, (D, H)) * 0.1
    b1 = jnp.zeros((H,), dtype=jnp.float32)
    W2 = jax.random.normal(key, (H, C)) * 0.1
    y = jax.random.randint(key, (B,), 0, C)

    onehot = jax.nn.one_hot(y, C)
    big_bias = 30.0 * onehot  # per-sample b2: (B,C)
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": big_bias}
    batch = {"x": x, "y": y}

    method = somax.make(
        "sgn_ce",
        predict_fn=mlp_logits,
        lam0=1e-2,
        tol=1e-6,
        maxiter=16,
        learning_rate=0.0,  # isolate direction computation; do not move params
    )
    st0 = method.init(params)

    # One step should not explode even if it does not improve objective.
    p1, st1, _ = method.step(params, batch, st0, rng=jax.random.PRNGKey(0))
    flat, _ = ravel_pytree(p1)
    assert jnp.isfinite(jnp.linalg.norm(flat))
    assert int(st1.step) == 1


def _small_linear_logits(params, x):
    return x @ params["w"] + params["b"]


def test_sgn_ce_gnb_diagonal_preconditioner_preserves_converged_solution(key):
    params0 = {
        "w": jnp.asarray(
            [[0.2, -0.1, 0.05], [-0.3, 0.15, 0.25]],
            jnp.float32,
        ),
        "b": jnp.asarray([0.05, -0.02, 0.01], jnp.float32),
    }
    batch = {
        "x": jnp.asarray(
            [
                [1.0, -0.5],
                [0.2, 0.7],
                [-0.8, 0.3],
                [0.4, -1.0],
                [0.7, 0.1],
                [-0.2, -0.4],
            ],
            jnp.float32,
        ),
        "y": jnp.asarray([0, 1, 2, 0, 1, 2], jnp.int32),
    }

    common = dict(
        predict_fn=_small_linear_logits,
        lam0=0.7,
        tol=1e-7,
        maxiter=20,
        warm_start=False,
        learning_rate=1.0,
        record_cg_stats=True,
    )
    reference = somax.make("sgn_ce", **common)
    preconditioned = somax.make(
        "sgn_ce",
        estimator="gnb",
        estimator_kwargs={"n_samples": 8},
        precond="diag_direct",
        precond_kwargs={"eps": 1e-6, "add_lambda": True},
        **common,
    )

    def run(method):
        state0 = method.init(params0)
        return jax.jit(method.step)(params0, batch, state0, key)

    params_ref, _, info_ref = run(reference)
    params_pre, _, info_pre = run(preconditioned)

    ref_flat, _ = ravel_pytree(params_ref)
    pre_flat, _ = ravel_pytree(params_pre)
    np.testing.assert_allclose(pre_flat, ref_flat, rtol=2e-6, atol=2e-7)

    for info in (info_ref, info_pre):
        assert bool(info[mx.CG_CONVERGED])
        assert float(info[mx.CG_RESID]) < 1e-6
        assert int(info[mx.CG_ITERS]) <= int(info[mx.CG_MAXITER])
