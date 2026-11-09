import jax
import jax.numpy as jnp
import numpy as np
import pytest

from somax.methods.fgn import fgn_ce
from somax.methods.sgn import sgn_ce, sgn_mse
from somax.methods.sophia_g import sophia_g


def _make_problem(key, *, output_dim: int):
    batch_size, input_dim, hidden_dim = 6, 4, 5
    keys = jax.random.split(key, 9)
    params = {
        "W1": 0.15 * jax.random.normal(keys[0], (input_dim, hidden_dim)),
        "b1": 0.05 * jax.random.normal(keys[1], (hidden_dim,)),
        "scale": 1.0 + 0.03 * jax.random.normal(keys[2], (hidden_dim,)),
        "offset": 0.03 * jax.random.normal(keys[3], (hidden_dim,)),
        "W2": 0.15 * jax.random.normal(keys[4], (hidden_dim, output_dim)),
        "b2": 0.03 * jax.random.normal(keys[5], (output_dim,)),
    }
    x = 0.5 + jax.random.normal(keys[6], (batch_size, input_dim))
    model_state = {
        "batch_stats": {
            "mean": 0.1 * jax.random.normal(keys[7], (hidden_dim,)),
            "var": 0.9 + 0.1 * jax.random.uniform(keys[8], (hidden_dim,)),
        }
    }
    return params, x, model_state


def _predict(params, x, model_state):
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


def _assert_tree_finite(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def _build_fgn_ce(maxiter):
    return fgn_ce(
        predict_fn=_predict,
        has_aux=True,
        lam0=1.0,
        maxiter=maxiter,
        tol=1e-12,
        warm_start=False,
        learning_rate=0.05,
        record_cg_stats=False,
    )


def _build_sgn_ce(maxiter):
    return sgn_ce(
        predict_fn=_predict,
        has_aux=True,
        lam0=1.0,
        maxiter=maxiter,
        tol=1e-12,
        warm_start=False,
        learning_rate=0.05,
        record_cg_stats=False,
    )


def _build_sgn_mse(maxiter):
    return sgn_mse(
        predict_fn=_predict,
        has_aux=True,
        lam0=1.0,
        maxiter=maxiter,
        tol=1e-12,
        warm_start=False,
        learning_rate=0.05,
        record_cg_stats=False,
    )


@pytest.mark.parametrize(
    ("method_builder", "output_dim", "targets"),
    [
        (_build_fgn_ce, 3, jnp.asarray([0, 1, 2, 1, 0, 2], jnp.int32)),
        (_build_sgn_ce, 3, jnp.asarray([0, 1, 2, 1, 0, 2], jnp.int32)),
        (
                _build_sgn_mse,
                1,
                jnp.linspace(-0.5, 0.7, 6, dtype=jnp.float32),
        ),
    ],
    ids=[
        "fgnha-row-cg",
        "ggnce-param-cg",
        "ggnmse-param-cg",
    ],
)
def test_model_state_is_committed_once_and_reused_next_step(
        method_builder,
        output_dim,
        targets,
        key,
):
    params0, x, model_state0 = _make_problem(key, output_dim=output_dim)
    batch = {"x": x, "y": targets}
    method = method_builder(maxiter=2)
    state0 = method.init(params0, model_state=model_state0)

    _assert_tree_allclose(state0.model_state, model_state0)
    _, expected_model_state1 = _predict(params0, x, model_state0)

    @jax.jit
    def step(params, state, rng):
        return method.step(params, batch, state, rng)

    params1, state1, _ = step(params0, state0, key)
    jax.block_until_ready((params1, state1))
    _assert_tree_allclose(
        state1.model_state,
        expected_model_state1,
        rtol=1e-6,
        atol=1e-7,
    )
    _assert_tree_finite(params1)

    _, expected_model_state2 = _predict(
        params1,
        x,
        expected_model_state1,
    )
    params2, state2, _ = step(params1, state1, jax.random.fold_in(key, 1))
    jax.block_until_ready((params2, state2))
    _assert_tree_allclose(
        state2.model_state,
        expected_model_state2,
        rtol=1e-6,
        atol=1e-7,
    )
    _assert_tree_finite(params2)


@pytest.mark.parametrize(
    ("method_builder", "output_dim", "targets"),
    [
        (_build_fgn_ce, 3, jnp.asarray([0, 1, 2, 1, 0, 2], jnp.int32)),
        (_build_sgn_ce, 3, jnp.asarray([0, 1, 2, 1, 0, 2], jnp.int32)),
        (
                _build_sgn_mse,
                1,
                jnp.linspace(-0.5, 0.7, 6, dtype=jnp.float32),
        ),
    ],
    ids=[
        "fgnha-row-cg",
        "ggnce-param-cg",
        "ggnmse-param-cg",
    ],
)
def test_model_state_update_is_independent_of_cg_budget(
        method_builder,
        output_dim,
        targets,
        key,
):
    params, x, model_state = _make_problem(key, output_dim=output_dim)
    batch = {"x": x, "y": targets}
    expected = _predict(params, x, model_state)[1]

    updated_states = []
    for maxiter in (1, 4):
        method = method_builder(maxiter=maxiter)
        state = method.init(params, model_state=model_state)

        @jax.jit
        def step(p, s, rng):
            return method.step(p, batch, s, rng)

        params_new, state_new, _ = step(params, state, key)
        jax.block_until_ready((params_new, state_new))
        updated_states.append(state_new.model_state)
        _assert_tree_allclose(
            state_new.model_state,
            expected,
            rtol=1e-6,
            atol=1e-7,
        )

    _assert_tree_allclose(
        updated_states[0],
        updated_states[1],
        rtol=1e-7,
        atol=1e-8,
    )


def test_diag_lane_commits_model_state_once(key):
    params, x, model_state = _make_problem(key, output_dim=3)
    labels = jnp.asarray([0, 1, 2, 1, 0, 2], jnp.int32)
    batch = {"x": x, "y": labels}
    expected = _predict(params, x, model_state)[1]

    method = sophia_g(
        predict_fn=_predict,
        has_aux=True,
        beta1=0.0,
        beta2=0.0,
        gamma=1.0,
        eps=1e-8,
        n_samples=1,
        eval_every_k=1,
        learning_rate=0.0,
        weight_decay=0.0,
        clip_value=1e6,
    )
    state = method.init(params, model_state=model_state)

    params_new, state_new, _ = jax.jit(method.step)(
        params, batch, state, key
    )
    jax.block_until_ready((params_new, state_new))

    _assert_tree_allclose(
        state_new.model_state,
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


def test_stateless_method_init_keeps_empty_model_state(key):
    params, _, _ = _make_problem(key, output_dim=3)

    def stateless_predict(p, x):
        return x @ p["W1"] @ p["W2"] + p["b2"]

    method = fgn_ce(
        predict_fn=stateless_predict,
        has_aux=False,
        maxiter=1,
        record_cg_stats=False,
    )
    state = method.init(params)
    assert state.model_state == ()


def _iter_nested_jaxprs(value):
    if hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
        yield value.jaxpr
        return
    if hasattr(value, "eqns"):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_nested_jaxprs(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_nested_jaxprs(item)


def _primitive_names(jaxpr):
    names = []
    for eqn in jaxpr.eqns:
        names.append(eqn.primitive.name)
        for nested in _iter_nested_jaxprs(eqn.params):
            names.extend(_primitive_names(nested))
    return names


def _while_body_jaxprs(jaxpr):
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "while":
            yield from _iter_nested_jaxprs(eqn.params["body_jaxpr"])
        for nested in _iter_nested_jaxprs(eqn.params):
            yield from _while_body_jaxprs(nested)


def _array_shapes(jaxpr):
    shapes = set()
    for variable in (*jaxpr.constvars, *jaxpr.invars):
        shape = getattr(getattr(variable, "aval", None), "shape", None)
        if shape is not None:
            shapes.add(tuple(shape))
    return shapes


@pytest.mark.parametrize(
    ("method_factory", "output_dim", "targets"),
    [
        (fgn_ce, 3, jnp.asarray([0, 1, 2, 1], jnp.int32)),
        (sgn_ce, 3, jnp.asarray([0, 1, 2, 1], jnp.int32)),
        (
                sgn_mse,
                1,
                jnp.linspace(-0.4, 0.6, 4, dtype=jnp.float32),
        ),
    ],
    ids=[
        "fgnha-row-cg",
        "ggnce-param-cg",
        "ggnmse-param-cg",
    ],
)
def test_auxiliary_model_state_is_outside_cg_loop(
        method_factory,
        output_dim,
        targets,
):
    marker_shape = (11, 13)
    params = {
        "W": jnp.arange(3 * output_dim, dtype=jnp.float32).reshape(
            3, output_dim
        ) / 10.0,
        "b": jnp.zeros((output_dim,), jnp.float32),
    }
    x = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10.0
    batch = {"x": x, "y": targets}
    model_state = {
        "marker": jnp.full(marker_shape, 0.2, dtype=jnp.float32)
    }

    def predict_fn(p, inputs, state):
        predictions = inputs @ p["W"] + p["b"]
        # This operation is auxiliary-only. It must execute in the one outer
        # primal, never in a CG matvec or CG loop body.
        model_state_new = {
            "marker": jnp.sin(
                state["marker"] + jnp.sum(predictions)
            )
        }
        return predictions, model_state_new

    method_kwargs = dict(
        lam0=1.0,
        maxiter=3,
        tol=1e-12,
        warm_start=False,
        learning_rate=0.05,
        record_cg_stats=False,
    )
    method = method_factory(
        predict_fn=predict_fn,
        has_aux=True,
        **method_kwargs,
    )
    state = method.init(params, model_state=model_state)
    rng = jax.random.PRNGKey(0)

    closed_jaxpr = jax.make_jaxpr(
        lambda p, s: method.step(p, batch, s, rng)
    )(params, state)

    all_names = _primitive_names(closed_jaxpr.jaxpr)
    while_bodies = list(_while_body_jaxprs(closed_jaxpr.jaxpr))

    assert while_bodies, "Expected the CG implementation to contain a while loop."
    assert all_names.count("sin") == 1

    for body in while_bodies:
        body_names = _primitive_names(body)
        assert "sin" not in body_names
        assert marker_shape not in _array_shapes(body)

    def stateless_predict_fn(p, inputs):
        return inputs @ p["W"] + p["b"]

    stateless_method = method_factory(
        predict_fn=stateless_predict_fn,
        has_aux=False,
        **method_kwargs,
    )
    stateless_state = stateless_method.init(params)
    stateless_jaxpr = jax.make_jaxpr(
        lambda p, s: stateless_method.step(p, batch, s, rng)
    )(params, stateless_state)
    stateless_while_bodies = list(
        _while_body_jaxprs(stateless_jaxpr.jaxpr)
    )

    # Aux-aware state plumbing changes the outer primal only. The compiled CG
    # loop bodies are structurally identical to the stateless path.
    assert len(while_bodies) == len(stateless_while_bodies)
    assert [
               _primitive_names(body) for body in while_bodies
           ] == [
               _primitive_names(body) for body in stateless_while_bodies
           ]
