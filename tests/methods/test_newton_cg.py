import jax
import jax.numpy as jnp

import somax
from somax import metrics as mx


def test_newtoncg_mse_smoke(key, medium_shapes, init_mlp_regression, mse_loss, make_teacher_regression):
    B, D, H = medium_shapes["B"], medium_shapes["D"], medium_shapes["H"]
    _, batch = make_teacher_regression(key, B, D, H)
    params = init_mlp_regression(jax.random.PRNGKey(5), D, H)

    method = somax.make(
        "newton_cg",
        loss_fn=mse_loss,
        lam0=10.0,
        tol=1e-6,
        maxiter=15,
        learning_rate=0.1,
    )
    st = method.init(params)

    @jax.jit
    def step(p, s, k):
        return method.step(p, batch, s, k)

    l0 = mse_loss(params, batch)
    for i in range(15):
        params, st, info = step(params, st, jax.random.fold_in(key, i))
    l1 = mse_loss(params, batch)

    assert jnp.isfinite(l1)
    assert l1 < 0.9 * l0
    # assert l1 < 1e-4
    assert method.plan.lane == "param"
    assert int(info[mx.CG_ITERS]) >= 0


def test_newtoncg_ce_smoke(key, medium_shapes, init_mlp_classification, mlp_logits, ce_loss,
                           accuracy, make_teacher_classification):
    B, D, H, C = medium_shapes["B"], medium_shapes["D"], medium_shapes["H"], medium_shapes["C"]
    _, batch = make_teacher_classification(key, B, D, H, C)
    params = init_mlp_classification(jax.random.PRNGKey(6), D, H, C)

    method = somax.make(
        "newton_cg",
        loss_fn=ce_loss,
        lam0=1.0,
        tol=1e-6,
        maxiter=15,
        learning_rate=1.0,
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
    assert l1 < 0.5 * l0
    assert a1 > a0 + 0.2
    assert method.plan.lane == "param"
