import pytest

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import somax
from somax import metrics as mx
from somax.solvers.row_cg import RowConjugateGradient
from somax.solvers.row_cg_lifted import LiftedRowConjugateGradient


def test_fgn_ce_smoke(key, medium_shapes, mlp_logits, make_teacher_classification,
                      init_mlp_classification, ce_loss, accuracy):
    B, D, H, C = medium_shapes["B"], medium_shapes["D"], medium_shapes["H"], medium_shapes["C"]
    _, batch = make_teacher_classification(key, B, D, H, C)
    params = init_mlp_classification(jax.random.PRNGKey(4), D, H, C)

    method = somax.make(
        "fgn_ce",
        predict_fn=mlp_logits,
        lam0=1e-2,
        tol=1e-6,
        maxiter=10,
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
    assert method.plan.lane == "row"


def _broadcast_parameter_logits(params, x):
    return jnp.broadcast_to(
        params["logits"][None, :],
        (x.shape[0], params["logits"].shape[0]),
    )


@pytest.mark.parametrize("lifted", [False, True])
@pytest.mark.parametrize(
    "logits",
    [
        jnp.asarray([-80.0, 20.0, 10.0, 0.0], jnp.float32),
        jnp.asarray([80.0, 0.0, 0.0, 0.0], jnp.float32),
    ],
    ids=["true-class-underflow", "true-class-saturation"],
)
def test_fgn_ce_extreme_true_class_probabilities_remain_finite(logits, lifted, key):
    params0 = {"logits": logits}
    batch = {
        "x": jnp.zeros((4, 1), jnp.float32),
        "y": jnp.zeros((4,), jnp.int32),
    }
    method = somax.make(
        "fgn_ce",
        predict_fn=_broadcast_parameter_logits,
        lifted=lifted,
        lam0=1e-2,
        tol=1e-8,
        maxiter=8,
        learning_rate=0.1,
        record_loss_every_k=1,
        record_cg_stats=True,
    )
    state0 = method.init(params0)
    params1, state1, info = jax.jit(method.step)(
        params0,
        batch,
        state0,
        key,
    )

    assert bool(jnp.all(jnp.isfinite(params1["logits"])))
    assert bool(jnp.isfinite(info[mx.LOSS_BEFORE]))
    assert bool(jnp.isfinite(info[mx.CG_RESID]))
    assert int(info[mx.CG_ITERS]) <= int(info[mx.CG_MAXITER])
    assert int(state1.step) == 1


@pytest.mark.parametrize("maxiter", [2, 5])
def test_fgn_ce_lifted_matches_standard_step(
        maxiter, key, small_shapes, mlp_logits,
        make_teacher_classification, init_mlp_classification,
        monkeypatch,
):
    """Lifted and ordinary row CG produce equivalent optimizer updates."""
    B, D, H, C = (small_shapes[k] for k in ("B", "D", "H", "C"))
    _, batch = make_teacher_classification(key, B, D, H, C)
    params = init_mlp_classification(jax.random.PRNGKey(22), D, H, C)

    options = dict(
        predict_fn=mlp_logits,
        lam0=0.3,
        maxiter=maxiter,
        tol=0.0,
        learning_rate=1.0,
        record_cg_stats=True,
    )
    ordinary = somax.make("fgn_ce", lifted=False, **options)
    lifted = somax.make("fgn_ce", lifted=True, **options)

    assert type(ordinary.solver) is RowConjugateGradient
    assert type(lifted.solver) is LiftedRowConjugateGradient
    assert ordinary.plan.lane == lifted.plan.lane == "row"

    def forbidden_solve(*args, **kwargs):
        pytest.fail("Lifted solver called ordinary solve()")

    monkeypatch.setattr(lifted.solver, "solve", forbidden_solve)

    p0, s0, i0 = jax.jit(ordinary.step)(
        params, batch, ordinary.init(params), key
    )
    p1, s1, i1 = jax.jit(lifted.step)(
        params, batch, lifted.init(params), key
    )

    # Compare actual parameter updates, not parameter values.
    flat, _ = ravel_pytree(params)
    flat0, _ = ravel_pytree(p0)
    flat1, _ = ravel_pytree(p1)

    delta0 = flat0 - flat
    delta1 = flat1 - flat

    assert jnp.all(jnp.isfinite(delta0))
    assert jnp.all(jnp.isfinite(delta1))

    rel_error = jnp.linalg.norm(delta1 - delta0) / jnp.maximum(
        jnp.linalg.norm(delta0), 1e-12
    )
    assert float(rel_error) < 2e-4

    # Same CG behavior and optimizer-step progression.
    assert int(s0.step) == int(s1.step) == 1
    assert 0 < int(i0[mx.CG_ITERS]) == int(i1[mx.CG_ITERS]) <= maxiter
    assert bool(i0[mx.CG_CONVERGED]) == bool(i1[mx.CG_CONVERGED])

    assert jnp.isfinite(i0[mx.CG_RESID])
    assert jnp.isfinite(i1[mx.CG_RESID])
    assert jnp.allclose(
        i0[mx.CG_RESID], i1[mx.CG_RESID],
        rtol=1e-5, atol=1e-7,
    )
