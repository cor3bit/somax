import jax
import jax.numpy as jnp
import chex

import somax


def _diag_quadratic_loss(d):
    d = jnp.asarray(d)

    def loss_fn(params, batch):
        w = params["w"]
        return 0.5 * jnp.sum(d * (w ** 2))

    return loss_fn


def _quartic_loss():
    def loss_fn(params, batch):
        w = params["w"]
        return 0.25 * jnp.sum(w ** 4)

    return loss_fn


def test_sophiah_first_step_matches_analytic_diagonal_quadratic(tol):
    d = jnp.array([4.0, 9.0, 25.0], dtype=jnp.float32)
    loss_fn = _diag_quadratic_loss(d)

    lr = 0.4
    method = somax.make(
        "sophia_h",
        loss_fn=loss_fn,
        beta1=0.0,
        beta2=0.0,
        gamma=1.0,
        eps=0.0,
        n_probes=1,
        use_abs=False,
        eval_every_k=1,
        learning_rate=lr,
        weight_decay=0.0,
        clip_value=1e9,
    )

    params0 = {"w": jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)}
    st0 = method.init(params0)

    params1, st1, _ = method.step(params0, batch={}, state=st0, rng=jax.random.PRNGKey(42))
    expected = {"w": (1.0 - lr) * params0["w"]}
    chex.assert_trees_all_close(params1, expected, **tol)
    assert int(st1.step) == 1


def test_sophiah_v_ema_raw_diag_accumulates():
    d = jnp.array([2.0, 3.0], dtype=jnp.float32)
    beta2 = 0.5
    loss_fn = _diag_quadratic_loss(d)

    method = somax.make(
        "sophia_h",
        loss_fn=loss_fn,
        beta1=0.0,
        beta2=beta2,
        gamma=1.0,
        eps=0.0,
        n_probes=1,
        use_abs=False,
        eval_every_k=1,
        learning_rate=0.0,
        weight_decay=0.0,
        clip_value=1e9,
    )
    params = {"w": jnp.array([1.0, -1.0], dtype=jnp.float32)}
    st0 = method.init(params)

    _, st1, _ = method.step(params, {}, st0, rng=jax.random.PRNGKey(1))
    _, st2, _ = method.step(params, {}, st1, rng=jax.random.PRNGKey(2))

    chex.assert_trees_all_close(st1.precond_state.v["w"], (1.0 - beta2) * d, atol=1e-6, rtol=0.0)
    chex.assert_trees_all_close(st2.precond_state.v["w"], (1.0 - beta2 ** 2) * d, atol=1e-6, rtol=0.0)
    assert int(st2.step) == 2


def test_sophiah_lazy_curvature_cadence_eval_every_k():
    d = jnp.array([3.0, 5.0], dtype=jnp.float32)
    loss_fn = _diag_quadratic_loss(d)
    params = {"w": jnp.array([1.0, -1.0], dtype=jnp.float32)}

    method = somax.make(
        "sophia_h",
        loss_fn=loss_fn,
        beta1=0.0,
        beta2=0.0,
        gamma=1.0,
        eps=0.0,
        n_probes=1,
        use_abs=False,
        eval_every_k=3,
        learning_rate=0.0,
        weight_decay=0.0,
        clip_value=1e9,
    )
    st = method.init(params)

    _, st1, _ = method.step(params, {}, st, rng=jax.random.PRNGKey(0))
    _, st2, _ = method.step(params, {}, st1, rng=jax.random.PRNGKey(0))
    _, st3, _ = method.step(params, {}, st2, rng=jax.random.PRNGKey(0))
    _, st4, _ = method.step(params, {}, st3, rng=jax.random.PRNGKey(0))

    chex.assert_trees_all_close(st2.precond_state.v, st1.precond_state.v, atol=0.0, rtol=0.0)
    chex.assert_trees_all_close(st3.precond_state.v, st2.precond_state.v, atol=0.0, rtol=0.0)
    chex.assert_tree_all_finite(st4.precond_state.v)


def test_sophiah_tx_clip_then_decay_semantics():
    method = somax.make(
        "sophia_h",
        loss_fn=_diag_quadratic_loss(jnp.array([1.0], dtype=jnp.float32)),
        beta1=0.0,
        beta2=0.0,
        gamma=1e-6,
        eps=0.0,
        n_probes=1,
        use_abs=False,
        eval_every_k=1,
        learning_rate=1.0,
        weight_decay=0.2,
        clip_value=0.5,
    )

    params = {"w": jnp.array([2.0, -3.0], dtype=jnp.float32)}
    s_raw = {"w": jnp.array([10.0, -10.0], dtype=jnp.float32)}
    opt_state = method.tx.init(params)
    updates, _ = method.tx.update(s_raw, opt_state, params)

    expected = {"w": -jnp.clip(s_raw["w"], -0.5, 0.5) - 0.2 * params["w"]}
    chex.assert_trees_all_close(updates, expected, atol=1e-6, rtol=0.0)


# -------------------------------------------------------------------------
# Additional paranoia tests: numerator momentum m_t and denom EMA v_t
# -------------------------------------------------------------------------

def test_sophiah_numerator_momentum_m_t_updates_exactly_with_beta1():
    beta1 = 0.9
    loss_fn = _diag_quadratic_loss(jnp.array([3.0, 7.0], dtype=jnp.float32))

    method = somax.make(
        "sophia_h",
        loss_fn=loss_fn,
        beta1=beta1,  # enables want_m in diag lane
        beta2=0.0,
        gamma=1.0,
        eps=1e-8,
        n_probes=1,
        use_abs=False,
        eval_every_k=1,
        learning_rate=0.0,
        weight_decay=0.0,
        clip_value=1e9,
    )

    params = {"w": jnp.array([1.5, -2.0], dtype=jnp.float32)}
    st0 = method.init(params)

    g = jax.grad(lambda p: loss_fn(p, {}))(params)

    _, st1, _ = method.step(params, {}, st0, rng=jax.random.PRNGKey(0))
    expected_m1 = jax.tree_util.tree_map(lambda gg: (1.0 - beta1) * gg, g)
    chex.assert_trees_all_close(st1.method_state, expected_m1, atol=1e-6, rtol=0.0)

    _, st2, _ = method.step(params, {}, st1, rng=jax.random.PRNGKey(1))
    expected_m2 = jax.tree_util.tree_map(lambda gg: (1.0 - beta1 ** 2) * gg, g)
    chex.assert_trees_all_close(st2.method_state, expected_m2, atol=1e-6, rtol=0.0)


def test_sophiah_denominator_v_t_matches_raw_diag_ema_update():
    # Sophia-H denom EMA stores v as EMA of diag_est (raw, not squared).
    beta2 = 0.6
    loss_fn = _quartic_loss()

    method = somax.make(
        "sophia_h",
        loss_fn=loss_fn,
        beta1=0.0,
        beta2=beta2,
        gamma=0.5,
        eps=0.0,
        n_probes=1,
        use_abs=False,
        eval_every_k=1,
        learning_rate=0.0,
        weight_decay=0.0,
        clip_value=1e9,
    )

    params = {"w": jnp.array([1.0, -2.0], dtype=jnp.float32)}
    st0 = method.init(params)

    rng0 = jax.random.PRNGKey(7)
    cstate0, _ = method.op.init(params, {}, with_grad=True)
    diag0 = method.op.diagonal(params, cstate0, jax.random.fold_in(rng0, 0))
    expected_v1 = jax.tree_util.tree_map(lambda d: (1.0 - beta2) * d, diag0)

    _, st1, _ = method.step(params, {}, st0, rng0)
    chex.assert_trees_all_close(st1.precond_state.v, expected_v1, atol=1e-6, rtol=0.0)
    assert int(st1.precond_state.t_ema) == 1

    rng1 = jax.random.PRNGKey(8)
    cstate1, _ = method.op.init(params, {}, with_grad=True)
    diag1 = method.op.diagonal(params, cstate1, jax.random.fold_in(rng1, 1))
    expected_v2 = jax.tree_util.tree_map(
        lambda v_old, d: beta2 * v_old + (1.0 - beta2) * d,
        expected_v1,
        diag1,
    )

    _, st2, _ = method.step(params, {}, st1, rng1)
    chex.assert_trees_all_close(st2.precond_state.v, expected_v2, atol=1e-6, rtol=0.0)
    assert int(st2.precond_state.t_ema) == 2
