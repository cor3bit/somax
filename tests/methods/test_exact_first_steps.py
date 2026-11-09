import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import somax
from somax import metrics as mx


def _broadcast_logits(params, x):
    logits = params["logits"]
    return jnp.broadcast_to(logits[None, :], (x.shape[0], logits.shape[0]))


def _ce_loss_fn(params, batch):
    logits = _broadcast_logits(params, batch["x"])
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, batch["y"]))


def _expected_binary_ce_step(logits, *, target, lam):
    logits = jnp.asarray(logits)
    probs = jax.nn.softmax(logits)
    y = jax.nn.one_hot(jnp.asarray(target, jnp.int32), logits.shape[0], dtype=logits.dtype)
    grad = probs - y
    fisher = jnp.diag(probs) - jnp.outer(probs, probs)
    step = jnp.linalg.solve(fisher + lam * jnp.eye(logits.shape[0], dtype=logits.dtype), grad)
    return logits - step


@pytest.mark.parametrize(
    "name,make_method,rtol,atol",
    [
        (
                "fgn_ce",
                lambda lam: somax.make(
                    "fgn_ce",
                    predict_fn=_broadcast_logits,
                    lam0=lam,
                    tol=1e-12,
                    maxiter=8,
                    learning_rate=1.0,
                    record_lam=True,
                ),
                1e-6,
                1e-6,
        ),
        (
                "egn_ce",
                lambda lam: somax.make(
                    "egn_ce",
                    predict_fn=_broadcast_logits,
                    lam0=lam,
                    learning_rate=1.0,
                    record_lam=True,
                ),
                1e-5,
                2e-6,
        ),
        (
                "sgn_ce",
                lambda lam: somax.make(
                    "sgn_ce",
                    predict_fn=_broadcast_logits,
                    lam0=lam,
                    tol=1e-12,
                    maxiter=8,
                    learning_rate=1.0,
                    record_lam=True,
                ),
                1e-6,
                1e-6,
        ),
        (
                "newton_cg",
                lambda lam: somax.make(
                    "newton_cg",
                    loss_fn=_ce_loss_fn,
                    lam0=lam,
                    tol=1e-12,
                    maxiter=8,
                    learning_rate=1.0,
                    record_lam=True,
                ),
                1e-6,
                1e-6,
        ),
    ],
)
def test_binary_ce_one_step_matches_exact_damped_softmax_gn(name, make_method, rtol, atol, dtype):
    del name
    lam = jnp.asarray(0.3, dtype)
    params0 = {"logits": jnp.asarray([0.7, -0.4], dtype)}
    batch = {
        "x": jnp.zeros((1, 1), dtype),
        "y": jnp.asarray([0], jnp.int32),
    }

    method = make_method(float(lam))
    state0 = method.init(params0)
    params1, state1, info = method.step(params0, batch, state0, rng=jax.random.PRNGKey(0))

    expected_logits = _expected_binary_ce_step(params0["logits"], target=0, lam=lam)
    np.testing.assert_allclose(
        np.asarray(params1["logits"]),
        np.asarray(expected_logits),
        rtol=rtol,
        atol=atol,
    )

    assert int(state1.step) == 1
    assert int(info[mx.STEP]) == 0
    assert mx.LAM_NEXT in info
    assert mx.LAM_FACTOR in info
    np.testing.assert_allclose(np.asarray(info[mx.LAM_USED]), np.asarray(lam), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(info[mx.LAM_NEXT]), np.asarray(lam), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(info[mx.LAM_FACTOR]), np.asarray(jnp.asarray(1.0, dtype)), rtol=1e-6,
                               atol=1e-6)


@pytest.mark.parametrize("name", ["fgn_ce", "egn_mse_cg", "egn_ce_cg"])
def test_row_space_cg_presets_reject_warm_start(name, linear_predict):
    kwargs = {"lam0": 0.1, "warm_start": True}
    if name.startswith("egn_mse"):
        kwargs["predict_fn"] = linear_predict
    else:
        kwargs["predict_fn"] = _broadcast_logits if "ce" in name else linear_predict

    with pytest.raises(NotImplementedError, match="warm start"):
        somax.make(name, **kwargs)


def _linear_logits(params, x):
    return x @ params["w"] + params["b"]


@pytest.mark.parametrize("method_name", ["fgn_ce", "sgn_ce"])
def test_ce_method_mean_reduction_is_invariant_to_batch_duplication(
        method_name,
        key,
):
    params0 = {
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
    batch_twice = {
        "x": jnp.concatenate([batch["x"], batch["x"]], axis=0),
        "y": jnp.concatenate([batch["y"], batch["y"]], axis=0),
    }

    def one_step(step_batch):
        method = somax.make(
            method_name,
            predict_fn=_linear_logits,
            lam0=0.7,
            tol=1e-8,
            maxiter=32,
            warm_start=False,
            learning_rate=1.0,
            record_cg_stats=True,
        )
        state0 = method.init(params0)
        params1, _, info = jax.jit(method.step)(
            params0,
            step_batch,
            state0,
            key,
        )
        return params1, info

    params1, info1 = one_step(batch)
    params1_twice, info1_twice = one_step(batch_twice)

    for actual, expected in zip(
            jax.tree_util.tree_leaves(params1),
            jax.tree_util.tree_leaves(params1_twice),
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)

    assert float(info1[mx.CG_RESID]) < 1e-6
    assert float(info1_twice[mx.CG_RESID]) < 1e-6
