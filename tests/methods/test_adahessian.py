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
    # f(w) = sum_i w_i^4 / 4
    # grad = w^3
    # Hess diag = 3 w^2 (diagonal operator -> Hutch diag is exact)
    def loss_fn(params, batch):
        w = params["w"]
        return 0.25 * jnp.sum(w ** 4)

    return loss_fn


def _tree_zeros_like(p):
    return jax.tree_util.tree_map(jnp.zeros_like, p)


def test_adahessian_first_step_matches_analytic_diagonal_quadratic(tol):
    d = jnp.array([4.0, 9.0, 25.0], dtype=jnp.float32)
    loss_fn = _diag_quadratic_loss(d)

    lr = 0.3
    method = somax.make(
        "adahessian",
        loss_fn=loss_fn,
        beta1=0.0,
        beta2=0.0,
        hessian_power=1.0,
        eps=0.0,
        n_probes=1,
        use_abs=False,
        spatial_averaging=False,
        eval_every_k=1,
        learning_rate=lr,
    )

    params0 = {"w": jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)}
    st0 = method.init(params0)

    params1, st1, info1 = method.step(params0, batch={}, state=st0, rng=jax.random.PRNGKey(42))

    expected = {"w": (1.0 - lr) * params0["w"]}
    chex.assert_trees_all_close(params1, expected, **tol)
    assert int(st1.step) == 1
    assert int(info1["step"]) == 0


def test_adahessian_diag_ema_state_updates_square_then_biascorr():
    d = jnp.array([2.0, 3.0], dtype=jnp.float32)
    beta2 = 0.5
    loss_fn = _diag_quadratic_loss(d)

    method = somax.make(
        "adahessian",
        loss_fn=loss_fn,
        beta1=0.0,
        beta2=beta2,
        hessian_power=1.0,
        eps=0.0,
        n_probes=1,
        use_abs=False,
        spatial_averaging=False,
        eval_every_k=1,
        learning_rate=0.0,
    )

    params0 = {"w": jnp.array([1.0, -1.0], dtype=jnp.float32)}
    st0 = method.init(params0)

    params1, st1, _ = method.step(params0, batch={}, state=st0, rng=jax.random.PRNGKey(1))
    chex.assert_trees_all_close(params1, params0, atol=0.0, rtol=0.0)

    expected_v_raw = (1.0 - beta2) * (d ** 2)
    chex.assert_trees_all_close(st1.precond_state.v["w"], expected_v_raw, atol=1e-6, rtol=0.0)
    assert int(st1.precond_state.t_ema) == 1

    b2c = 1.0 - (beta2 ** 1)
    v_hat = st1.precond_state.v["w"] / b2c
    chex.assert_trees_all_close(v_hat, d ** 2, atol=1e-6, rtol=0.0)


def test_adahessian_mse_smoke(make_teacher_regression, init_mlp_regression, mse_loss, medium_shapes, keys):
    B, D, H = medium_shapes["B"], medium_shapes["D"], medium_shapes["H"]
    _, batch = make_teacher_regression(keys(), B, D, H)
    params = init_mlp_regression(keys(), D, H)

    method = somax.make(
        "adahessian",
        loss_fn=mse_loss,
        beta1=0.9,
        beta2=0.999,
        hessian_power=1.0,
        eps=1e-8,
        n_probes=1,
        use_abs=False,
        spatial_averaging=True,
        eval_every_k=1,
        learning_rate=0.2,
    )
    st = method.init(params)

    @jax.jit
    def step(p, s, k):
        return method.step(p, batch, s, k)

    l0 = mse_loss(params, batch)
    for _ in range(120):
        params, st, _ = step(params, st, keys())
    l1 = mse_loss(params, batch)

    assert jnp.isfinite(l1)
    assert l1 < 0.5 * l0


def test_adahessian_jit_determinism_one_step():
    d = jnp.array([2.0, 8.0], dtype=jnp.float32)
    loss_fn = _diag_quadratic_loss(d)
    method = somax.make(
        "adahessian",
        loss_fn=loss_fn,
        beta1=0.9,
        beta2=0.99,
        hessian_power=1.0,
        eps=1e-8,
        n_probes=4,
        use_abs=False,
        spatial_averaging=False,
        eval_every_k=1,
        learning_rate=0.0,
    )

    params0 = {"w": jnp.array([0.5, -1.0], dtype=jnp.float32)}
    st0 = method.init(params0)
    rng = jax.random.PRNGKey(123)

    @jax.jit
    def once(p, s, r):
        return method.step(p, {}, s, r)

    p1, s1, _ = once(params0, st0, rng)
    p2, s2, _ = once(params0, st0, rng)

    chex.assert_trees_all_close(p1, p2, atol=0.0, rtol=0.0)
    assert int(s1.step) == int(s2.step) == 1


def test_adahessian_numerator_momentum_m_t_updates_exactly_with_beta1():
    # Validate stored m_t recurrence:
    # m_t = beta1 * m_{t-1} + (1-beta1) * g_t
    # Here we keep params fixed (lr=0), so g_t is constant.
    beta1 = 0.8
    loss_fn = _diag_quadratic_loss(jnp.array([3.0, 7.0], dtype=jnp.float32))

    method = somax.make(
        "adahessian",
        loss_fn=loss_fn,
        beta1=beta1,  # enables want_m and numerator EMA
        beta2=0.0,
        hessian_power=1.0,
        eps=1e-8,
        n_probes=1,
        use_abs=False,
        spatial_averaging=False,
        eval_every_k=1,
        learning_rate=0.0,  # keep params fixed so g constant
    )

    params = {"w": jnp.array([1.5, -2.0], dtype=jnp.float32)}
    st0 = method.init(params)

    g = jax.grad(lambda p: loss_fn(p, {}))(params)

    rng0 = jax.random.PRNGKey(0)
    _, st1, _ = method.step(params, {}, st0, rng0)
    expected_m1 = jax.tree_util.tree_map(lambda gg: (1.0 - beta1) * gg, g)
    chex.assert_trees_all_close(st1.method_state, expected_m1, atol=1e-6, rtol=0.0)

    rng1 = jax.random.PRNGKey(1)
    _, st2, _ = method.step(params, {}, st1, rng1)
    expected_m2 = jax.tree_util.tree_map(lambda gg: (1.0 - beta1 ** 2) * gg, g)
    chex.assert_trees_all_close(st2.method_state, expected_m2, atol=1e-6, rtol=0.0)


def test_adahessian_denominator_v_t_matches_diag_square_ema_update():
    # For diagonal separable losses, Hutchinson diagonal is exact.
    # AdaHessian denom EMA stores v as EMA of (diag_est)^2.
    beta2 = 0.6
    loss_fn = _quartic_loss()

    method = somax.make(
        "adahessian",
        loss_fn=loss_fn,
        beta1=0.0,
        beta2=beta2,
        hessian_power=1.0,
        eps=0.0,
        n_probes=1,
        use_abs=False,
        spatial_averaging=False,
        eval_every_k=1,
        learning_rate=0.0,
    )

    params = {"w": jnp.array([1.0, -2.0], dtype=jnp.float32)}
    st0 = method.init(params)

    # Reconstruct exact diag_est that precond.build sees:
    # precond uses k = fold_in(rng, step_precond) with step_precond=0 at first call.
    rng0 = jax.random.PRNGKey(42)
    cstate0, _ = method.op.init(params, {}, with_grad=True)
    diag0 = method.op.diagonal(params, cstate0, jax.random.fold_in(rng0, 0))
    contrib0 = jax.tree_util.tree_map(lambda d: d * d, diag0)
    expected_v1 = jax.tree_util.tree_map(lambda c: (1.0 - beta2) * c, contrib0)

    _, st1, _ = method.step(params, {}, st0, rng0)
    chex.assert_trees_all_close(st1.precond_state.v, expected_v1, atol=1e-5, rtol=0.0)
    assert int(st1.precond_state.t_ema) == 1

    # Second step: step_precond was incremented to 1, so fold_in(...,1)
    rng1 = jax.random.PRNGKey(43)
    cstate1, _ = method.op.init(params, {}, with_grad=True)
    diag1 = method.op.diagonal(params, cstate1, jax.random.fold_in(rng1, 1))
    contrib1 = jax.tree_util.tree_map(lambda d: d * d, diag1)
    expected_v2 = jax.tree_util.tree_map(
        lambda v_old, c: beta2 * v_old + (1.0 - beta2) * c,
        expected_v1,
        contrib1,
    )

    _, st2, _ = method.step(params, {}, st1, rng1)
    chex.assert_trees_all_close(st2.precond_state.v, expected_v2, atol=1e-5, rtol=0.0)
    assert int(st2.precond_state.t_ema) == 2
