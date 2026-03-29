import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import chex
import optax

import somax


def _linear_logits(params, x):
    return x @ params["W"] + params["b"]


def _ce_loss(predict_fn, params, batch):
    x, y = batch["x"], batch["y"]
    logits = predict_fn(params, x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))


def test_sophiag_first_step_reconstructs_true_grad_when_gamma1_beta0(key):
    B, d, C = 6, 5, 3
    kx, kw, kb, krng = jax.random.split(key, 4)
    x = jax.random.normal(kx, (B, d))
    params = {
        "W": jax.random.normal(kw, (d, C), dtype=jnp.float32) * 0.3,
        "b": jax.random.normal(kb, (C,), dtype=jnp.float32) * 0.1,
    }
    y = jnp.argmax(_linear_logits(params, x), axis=-1).astype(jnp.int32)
    batch = {"x": x, "y": y}

    method = somax.make(
        "sophia_g",
        predict_fn=_linear_logits,
        beta1=0.0,
        beta2=0.0,
        gamma=1.0,
        eps=0.0,
        n_samples=1,
        eval_every_k=1,
        learning_rate=0.0,
        weight_decay=0.0,
        clip_value=1e9,
    )
    st0 = method.init(params)

    params1, st1, _ = method.step(params, batch, st0, rng=krng)
    chex.assert_trees_all_close(params1, params, atol=0.0, rtol=0.0)

    g_true = jax.grad(lambda p: _ce_loss(_linear_logits, p, batch))(params)

    v = st1.precond_state.v
    chex.assert_tree_all_finite(v)

    s_hat = jtu.tree_map(lambda g, vv: g / vv, g_true, v)
    chex.assert_tree_all_finite(s_hat)
    assert int(st1.step) == 1
    assert method.plan.lane == "diag"


def test_sophiag_lazy_curvature_updates_every_k(key):
    B, d, C = 5, 4, 3
    x = jax.random.normal(key, (B, d))
    params = {
        "W": jax.random.normal(key, (d, C)) * 0.5,
        "b": jax.random.normal(key, (C,)) * 0.2,
    }
    y = (jnp.arange(B) % C).astype(jnp.int32)
    batch = {"x": x, "y": y}

    method = somax.make(
        "sophia_g",
        predict_fn=_linear_logits,
        beta1=0.0,
        beta2=0.0,
        gamma=1.0,
        eps=0.0,
        n_samples=1,
        eval_every_k=3,
        learning_rate=0.0,
        weight_decay=0.0,
        clip_value=1e9,
    )
    st0 = method.init(params)

    _, st1, _ = method.step(params, batch, st0, rng=key)
    _, st2, _ = method.step(params, batch, st1, rng=key)
    _, st3, _ = method.step(params, batch, st2, rng=key)
    _, st4, _ = method.step(params, batch, st3, rng=key)

    chex.assert_trees_all_close(st2.precond_state.v, st1.precond_state.v, atol=0.0, rtol=0.0)
    chex.assert_trees_all_close(st3.precond_state.v, st2.precond_state.v, atol=0.0, rtol=0.0)
    chex.assert_tree_all_finite(st4.precond_state.v)


def test_sophiag_tx_clip_then_decay_semantics():
    method = somax.make(
        "sophia_g",
        predict_fn=_linear_logits,
        beta1=0.0,
        beta2=0.0,
        gamma=1e-6,
        eps=0.0,
        n_samples=1,
        eval_every_k=1,
        learning_rate=1.0,
        weight_decay=0.2,
        clip_value=0.5,
    )

    params = {"W": jnp.ones((2, 2), dtype=jnp.float32), "b": jnp.ones((2,), dtype=jnp.float32)}
    s_raw = {"W": jnp.full((2, 2), 10.0, dtype=jnp.float32), "b": jnp.array([10.0, -10.0], dtype=jnp.float32)}
    opt_state = method.tx.init(params)
    updates, _ = method.tx.update(s_raw, opt_state, params)

    expected = jtu.tree_map(lambda sr, p: -jnp.clip(sr, -0.5, 0.5) - 0.2 * p, s_raw, params)
    chex.assert_trees_all_close(updates, expected, atol=1e-6, rtol=0.0)


def test_sophiag_ce_smoke(
        medium_shapes, key, mlp_logits, ce_loss, accuracy, make_teacher_classification, init_mlp_classification
):
    B, D, H, C = medium_shapes["B"], medium_shapes["D"], medium_shapes["H"], medium_shapes["C"]
    _, batch = make_teacher_classification(key, B, D, H, C)
    params = init_mlp_classification(jax.random.PRNGKey(9), D, H, C)

    method = somax.make(
        "sophia_g",
        predict_fn=mlp_logits,
        beta1=0.965,
        beta2=0.99,
        gamma=0.05,
        eps=1e-12,
        n_samples=1,
        eval_every_k=5,
        learning_rate=0.1,
        weight_decay=0.2,
        clip_value=1.0,
    )
    st = method.init(params)

    @jax.jit
    def step(p, s, k):
        return method.step(p, batch, s, k)

    l0, a0 = ce_loss(params, batch), accuracy(params, batch)
    for i in range(80):
        params, st, _ = step(params, st, jax.random.fold_in(key, i))
    l1, a1 = ce_loss(params, batch), accuracy(params, batch)

    assert jnp.isfinite(l1)
    assert l1 < 0.7 * l0
    assert a1 > a0 + 0.10


# -------------------------------------------------------------------------
# Additional paranoia tests: numerator momentum m_t and denom EMA v_t
# -------------------------------------------------------------------------

def test_sophiag_numerator_momentum_m_t_updates_exactly_with_beta1(key):
    # Keep params fixed (lr=0) so grad is constant; validate stored m_t.
    beta1 = 0.85
    B, d, C = 8, 6, 4
    kx, kw, kb, kr = jax.random.split(key, 4)
    x = jax.random.normal(kx, (B, d))
    params = {
        "W": jax.random.normal(kw, (d, C), dtype=jnp.float32) * 0.1,
        "b": jax.random.normal(kb, (C,), dtype=jnp.float32) * 0.1,
    }
    y = (jnp.arange(B) % C).astype(jnp.int32)
    batch = {"x": x, "y": y}

    method = somax.make(
        "sophia_g",
        predict_fn=_linear_logits,
        beta1=beta1,
        beta2=0.0,
        gamma=1.0,
        eps=1e-8,
        n_samples=1,
        eval_every_k=1,
        learning_rate=0.0,
        weight_decay=0.0,
        clip_value=1e9,
    )
    st0 = method.init(params)

    g = jax.grad(lambda p: _ce_loss(_linear_logits, p, batch))(params)

    _, st1, _ = method.step(params, batch, st0, rng=kr)
    expected_m1 = jtu.tree_map(lambda gg: (1.0 - beta1) * gg, g)
    chex.assert_trees_all_close(st1.method_state, expected_m1, atol=1e-6, rtol=0.0)

    _, st2, _ = method.step(params, batch, st1, rng=jax.random.PRNGKey(999))
    expected_m2 = jtu.tree_map(lambda gg: (1.0 - beta1 ** 2) * gg, g)
    chex.assert_trees_all_close(st2.method_state, expected_m2, atol=1e-6, rtol=0.0)


def test_sophiag_denominator_v_t_matches_raw_diag_ema_update(key):
    # Validate precond_state.v tracks EMA of diag_est (raw) using same fold_in(rng, step).
    beta2 = 0.7
    B, d, C = 10, 5, 3
    kx, kw, kb, kr0, kr1 = jax.random.split(key, 5)
    x = jax.random.normal(kx, (B, d))
    params = {
        "W": jax.random.normal(kw, (d, C), dtype=jnp.float32) * 0.2,
        "b": jax.random.normal(kb, (C,), dtype=jnp.float32) * 0.1,
    }
    y = (jnp.arange(B) % C).astype(jnp.int32)
    batch = {"x": x, "y": y}

    method = somax.make(
        "sophia_g",
        predict_fn=_linear_logits,
        beta1=0.0,
        beta2=beta2,
        gamma=0.5,
        eps=0.0,
        n_samples=1,
        eval_every_k=1,
        learning_rate=0.0,
        weight_decay=0.0,
        clip_value=1e9,
    )
    st0 = method.init(params)

    # step_precond=0
    cstate0, _ = method.op.init(params, batch, with_grad=True)
    diag0 = method.op.diagonal(params, cstate0, jax.random.fold_in(kr0, 0))
    expected_v1 = jtu.tree_map(lambda d0: (1.0 - beta2) * d0, diag0)

    _, st1, _ = method.step(params, batch, st0, rng=kr0)
    chex.assert_trees_all_close(st1.precond_state.v, expected_v1, atol=1e-6, rtol=0.0)

    # step_precond=1
    cstate1, _ = method.op.init(params, batch, with_grad=True)
    diag1 = method.op.diagonal(params, cstate1, jax.random.fold_in(kr1, 1))
    expected_v2 = jtu.tree_map(
        lambda v_old, d1: beta2 * v_old + (1.0 - beta2) * d1,
        expected_v1,
        diag1,
    )

    _, st2, _ = method.step(params, batch, st1, rng=kr1)
    chex.assert_trees_all_close(st2.precond_state.v, expected_v2, atol=1e-6, rtol=0.0)
