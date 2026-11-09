import jax
import jax.numpy as jnp
import numpy as np
import pytest

from somax.curvature.base import linearize_predict
from somax.curvature.fgnha_ce import FGNHA
from somax.curvature.ggn_ce import GGNCE
from somax.curvature.ggn_mse import GGNMSE
from somax.utils import tree_vdot


def _random_like_tree(key, tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(
        treedef,
        [jax.random.normal(k, x.shape, x.dtype) for k, x in zip(keys, leaves)],
    )


def _make_stateful_problem(key, *, output_dim: int):
    batch_size, input_dim, hidden_dim = 7, 5, 6
    keys = jax.random.split(key, 10)

    params = {
        "W1": 0.2 * jax.random.normal(keys[0], (input_dim, hidden_dim)),
        "b1": 0.1 * jax.random.normal(keys[1], (hidden_dim,)),
        "scale": 1.0 + 0.05 * jax.random.normal(keys[2], (hidden_dim,)),
        "offset": 0.05 * jax.random.normal(keys[3], (hidden_dim,)),
        "W2": 0.2 * jax.random.normal(keys[4], (hidden_dim, output_dim)),
        "b2": 0.05 * jax.random.normal(keys[5], (output_dim,)),
    }
    x = 0.7 + 1.3 * jax.random.normal(keys[6], (batch_size, input_dim))
    model_state = {
        "batch_stats": {
            "mean": 0.2 * jax.random.normal(keys[7], (hidden_dim,)),
            "var": 0.8 + 0.2 * jax.random.uniform(keys[8], (hidden_dim,)),
        }
    }
    return params, x, model_state


def _stateful_batch_norm_predict(params, x, model_state):
    """Small functional analogue of training-mode BatchNorm.

    The primary output uses current minibatch moments. The auxiliary state is
    the running-statistics update that should be produced once per outer step.
    """
    preactivation = x @ params["W1"] + params["b1"]
    batch_mean = jnp.mean(preactivation, axis=0)
    centered = preactivation - batch_mean
    batch_var = jnp.mean(jnp.square(centered), axis=0)

    normalized = centered * jax.lax.rsqrt(batch_var + 1e-3)
    hidden = jnp.tanh(params["scale"] * normalized + params["offset"])
    predictions = hidden @ params["W2"] + params["b2"]

    momentum = jnp.asarray(0.9, predictions.dtype)
    old_stats = model_state["batch_stats"]
    model_state_new = {
        "batch_stats": {
            "mean": momentum * old_stats["mean"] + (1.0 - momentum) * batch_mean,
            "var": momentum * old_stats["var"] + (1.0 - momentum) * batch_var,
        }
    }
    return predictions, model_state_new


def _ce_loss(logits, labels):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])


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


def test_linearize_predict_evaluates_state_update_only_in_primal():
    calls = []
    params = {"w": jnp.asarray([0.2, -0.1, 0.4], jnp.float32)}
    x = jnp.asarray([1.0, 2.0, -1.0], jnp.float32)
    model_state = {"marker": jnp.asarray(0.3, jnp.float32)}

    def predict_fn(p, inputs, state):
        calls.append(None)
        predictions = p["w"] * inputs
        # ``sin`` is deliberately aux-only, making it easy to detect if the
        # state update leaks into the reusable tangent program.
        state_new = {
            "marker": jnp.sin(state["marker"] + jnp.sum(predictions))
        }
        return predictions, state_new

    predictions, jvp, model_state_new = linearize_predict(
        predict_fn=predict_fn,
        params=params,
        x=x,
        model_state=model_state,
        has_aux=True,
    )
    assert len(calls) == 1

    tangent = {"w": jnp.ones_like(params["w"])}
    tangent_out_1 = jvp(tangent)
    tangent_out_2 = jvp(tangent)
    transpose = jax.linear_transpose(jvp, params)
    transpose(jnp.ones_like(predictions))

    # Reusing the JVP/VJP does not invoke the model primal again.
    assert len(calls) == 1
    np.testing.assert_allclose(np.asarray(tangent_out_1), np.asarray(x))
    np.testing.assert_allclose(np.asarray(tangent_out_2), np.asarray(x))

    _, expected_state = predict_fn(params, x, model_state)
    _assert_tree_allclose(model_state_new, expected_state)

    tangent_jaxpr = jax.make_jaxpr(jvp)(tangent).jaxpr
    tangent_primitives = {eqn.primitive.name for eqn in tangent_jaxpr.eqns}
    assert "sin" not in tangent_primitives
    assert "cos" not in tangent_primitives


@pytest.mark.parametrize("operator_cls", [FGNHA, GGNCE])
def test_stateful_ce_curvature_matches_reference_gradient_and_state(
        operator_cls,
        key,
):
    params, x, model_state = _make_stateful_problem(key, output_dim=4)
    labels = jnp.asarray([0, 1, 3, 2, 1, 0, 2], jnp.int32)
    batch = {"x": x, "y": labels}

    def reference_loss(p):
        logits, model_state_new = _stateful_batch_norm_predict(
            p, x, model_state
        )
        return _ce_loss(logits, labels), model_state_new

    (loss_ref, model_state_ref), grad_ref = jax.value_and_grad(
        reference_loss,
        has_aux=True,
    )(params)

    op = operator_cls(
        predict_fn=_stateful_batch_norm_predict,
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

    _assert_tree_allclose(grad, grad_ref, rtol=2e-5, atol=2e-6)
    _assert_tree_allclose(
        curvature_state.model_state_new,
        model_state_ref,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(op.loss(params, curvature_state, batch)),
        np.asarray(loss_ref),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(op.loss_only(
            params,
            batch,
            model_state=model_state,
        )),
        np.asarray(loss_ref),
        rtol=1e-6,
        atol=1e-7,
    )

    # The cached linear operator is fixed, deterministic, and symmetric. No
    # model-state update is executed by repeated curvature products.
    u = _random_like_tree(jax.random.fold_in(key, 1), params)
    v = _random_like_tree(jax.random.fold_in(key, 2), params)
    Au_1 = op.matvec(params, curvature_state, u)
    Au_2 = op.matvec(params, curvature_state, u)
    Av = op.matvec(params, curvature_state, v)
    _assert_tree_allclose(Au_1, Au_2, rtol=1e-6, atol=1e-7)

    lhs = tree_vdot(u, Av)
    rhs = tree_vdot(Au_1, v)
    np.testing.assert_allclose(
        np.asarray(lhs),
        np.asarray(rhs),
        rtol=2e-4,
        atol=2e-5,
    )


def test_stateful_ggn_mse_matches_reference_gradient_and_state(key):
    params, x, model_state = _make_stateful_problem(key, output_dim=1)
    targets = jnp.linspace(-0.8, 0.9, x.shape[0], dtype=jnp.float32)
    batch = {"x": x, "y": targets}

    def reference_loss(p):
        predictions, model_state_new = _stateful_batch_norm_predict(
            p, x, model_state
        )
        predictions = predictions.squeeze(-1)
        loss = 0.5 * jnp.mean(jnp.square(predictions - targets))
        return loss, model_state_new

    (loss_ref, model_state_ref), grad_ref = jax.value_and_grad(
        reference_loss,
        has_aux=True,
    )(params)

    op = GGNMSE(
        predict_fn=_stateful_batch_norm_predict,
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

    _assert_tree_allclose(grad, grad_ref, rtol=2e-5, atol=2e-6)
    _assert_tree_allclose(
        curvature_state.model_state_new,
        model_state_ref,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(op.loss(params, curvature_state, batch)),
        np.asarray(loss_ref),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(op.loss_only(
            params,
            batch,
            model_state=model_state,
        )),
        np.asarray(loss_ref),
        rtol=1e-6,
        atol=1e-7,
    )
