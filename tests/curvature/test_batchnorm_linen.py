import jax
import jax.numpy as jnp
import numpy as np
import pytest

nn = pytest.importorskip("flax.linen")

from somax.curvature.fgnha_ce import FGNHA
from somax.curvature.ggn_ce import GGNCE
from somax.curvature.ggn_mse import GGNMSE
from somax.methods.fgn import fgn_ce
from somax.methods.sgn import sgn_ce
from somax import metrics as mx


class _TinyBatchNormModel(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x, *, train: bool):
        x = nn.Dense(6, use_bias=False, name="hidden")(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-3,
            name="batch_norm",
        )(x)
        x = nn.tanh(x)
        return nn.Dense(self.output_dim, name="output")(x)


def _make_linen_predict_fn(model):
    def predict_fn(params, x, model_state):
        logits, updates = model.apply(
            {
                "params": params,
                "batch_stats": model_state["batch_stats"],
            },
            x,
            train=True,
            mutable=["batch_stats"],
        )
        return logits, {"batch_stats": updates["batch_stats"]}

    return predict_fn


def _assert_tree_allclose(actual, expected, *, rtol=1e-5, atol=1e-6):
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    assert actual_tree == expected_tree
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("operator_cls", [FGNHA, GGNCE])
def test_linen_batchnorm_ce_gradient_and_state_match_value_and_grad(
        operator_cls,
        key,
):
    x = jax.random.normal(key, (8, 5), jnp.float32)
    labels = jnp.asarray([0, 1, 2, 3, 1, 0, 2, 3], jnp.int32)
    model = _TinyBatchNormModel(output_dim=4)
    variables = model.init(key, x, train=True)
    params = variables["params"]
    model_state = {"batch_stats": variables["batch_stats"]}
    predict_fn = _make_linen_predict_fn(model)
    batch = {"x": x, "y": labels}

    def reference_loss(p):
        logits, model_state_new = predict_fn(p, x, model_state)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])
        return loss, model_state_new

    (_, model_state_ref), grad_ref = jax.value_and_grad(
        reference_loss,
        has_aux=True,
    )(params)

    op = operator_cls(
        predict_fn=predict_fn,
        x_key="x",
        y_key="y",
        reduction="mean",
        has_aux=True,
    )
    curvature_state, grad = op.init(
        params,
        batch,
        model_state=model_state,
        with_grad=True,
    )

    _assert_tree_allclose(grad, grad_ref, rtol=3e-5, atol=3e-6)
    _assert_tree_allclose(
        curvature_state.model_state_new,
        model_state_ref,
        rtol=1e-6,
        atol=1e-7,
    )


def test_linen_batchnorm_mse_gradient_and_state_match_value_and_grad(key):
    x = jax.random.normal(key, (8, 5), jnp.float32)
    targets = jnp.linspace(-1.0, 1.0, x.shape[0], dtype=jnp.float32)
    model = _TinyBatchNormModel(output_dim=1)
    variables = model.init(key, x, train=True)
    params = variables["params"]
    model_state = {"batch_stats": variables["batch_stats"]}
    predict_fn = _make_linen_predict_fn(model)
    batch = {"x": x, "y": targets}

    def reference_loss(p):
        predictions, model_state_new = predict_fn(p, x, model_state)
        predictions = predictions.squeeze(-1)
        loss = 0.5 * jnp.mean(jnp.square(predictions - targets))
        return loss, model_state_new

    (_, model_state_ref), grad_ref = jax.value_and_grad(
        reference_loss,
        has_aux=True,
    )(params)

    op = GGNMSE(
        predict_fn=predict_fn,
        x_key="x",
        y_key="y",
        reduction="mean",
        has_aux=True,
    )
    curvature_state, grad = op.init(
        params,
        batch,
        model_state=model_state,
        with_grad=True,
    )

    _assert_tree_allclose(grad, grad_ref, rtol=3e-5, atol=3e-6)
    _assert_tree_allclose(
        curvature_state.model_state_new,
        model_state_ref,
        rtol=1e-6,
        atol=1e-7,
    )



def _assert_tree_finite(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def _tree_squared_distance(a, b):
    return sum(
        float(jnp.sum(jnp.square(x - y)))
        for x, y in zip(
            jax.tree_util.tree_leaves(a),
            jax.tree_util.tree_leaves(b),
        )
    )


@pytest.mark.parametrize(
    ("method_factory", "expected_lane"),
    [(fgn_ce, "row"), (sgn_ce, "param")],
    ids=["fgn", "sgn"],
)
def test_linen_batchnorm_ce_method_step_commits_one_state_update(
        method_factory,
        expected_lane,
        key,
):
    x = jax.random.normal(key, (8, 5), jnp.float32)
    labels = jnp.asarray([0, 1, 2, 3, 1, 0, 2, 3], jnp.int32)
    model = _TinyBatchNormModel(output_dim=4)
    variables = model.init(key, x, train=True)
    params = variables["params"]
    model_state = {"batch_stats": variables["batch_stats"]}
    predict_fn = _make_linen_predict_fn(model)
    batch = {"x": x, "y": labels}

    method = method_factory(
        predict_fn=predict_fn,
        has_aux=True,
        lam0=1.0,
        tol=0.0,
        maxiter=1,
        warm_start=False,
        learning_rate=0.03,
        record_loss_every_k=1,
        record_cg_stats=True,
    )
    state0 = method.init(params, model_state=model_state)
    assert method.plan.lane == expected_lane

    _, expected_state1 = predict_fn(params, x, model_state)
    _, state_if_updated_twice = predict_fn(params, x, expected_state1)

    def step_fn(p, state, step_batch, rng):
        return method.step(p, step_batch, state, rng)

    compiled_step = jax.jit(step_fn).lower(params, state0, batch, key).compile()
    params1, state1, info1 = compiled_step(params, state0, batch, key)
    jax.block_until_ready((params1, state1, info1))

    _assert_tree_allclose(
        state1.model_state,
        expected_state1,
        rtol=1e-6,
        atol=1e-7,
    )
    assert _tree_squared_distance(state1.model_state, state_if_updated_twice) > 1e-12
    _assert_tree_finite(params1)
    _assert_tree_finite(state1.model_state)
    _assert_tree_finite(info1)
    assert int(state1.step) == 1
    assert int(info1[mx.CG_ITERS]) == 1
    assert bool(jnp.isfinite(info1[mx.CG_RESID]))
    assert bool(jnp.isfinite(info1[mx.LOSS_BEFORE]))
