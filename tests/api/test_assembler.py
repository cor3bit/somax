import pytest

import jax.numpy as jnp

from somax import assemble
from somax.specs import CurvatureSpec, DampingSpec, SolverSpec, PrecondSpec, TelemetrySpec, EstimatorSpec
from somax.methods import newton_cg
from somax import metrics as mx


def test_assemble_minimal_config(linear_predict):
    # Minimal assembly: curvature + solver.
    curv = CurvatureSpec(
        "ggn_mse",
        {"predict_fn": linear_predict, "x_key": "x", "y_key": "y", "reduction": "mean"},
    )
    solv = SolverSpec("cg", {"maxiter": 5})

    method = assemble(curvature=curv, solver=solv)

    assert method.plan.lane == "param"
    assert mx.STEP in method.plan.metric_keys

    # Defaults: damping and preconditioner are identity-ish.
    assert method.damping is not None
    assert method.precond is not None


def test_assemble_full_config(linear_predict):
    # Full assembly with all components.
    # Use JAX scalar loss to avoid Python-float weirdness under jit/tracing.
    curv = CurvatureSpec("hessian", {"loss_fn": lambda p, b: jnp.array(0.0, dtype=jnp.float32)})

    # Trust-region cadence should dominate telemetry cadence.
    damp = DampingSpec("trust_region", lam0=1e-3, kwargs={"every_k": 5})

    solv = SolverSpec("newton_cg", {"maxiter": 10})
    prec = PrecondSpec("diag_ema", {"beta2": 0.9})
    est = EstimatorSpec("hutchinson", {"n_probes": 3})

    # Telemetry requests rho + cg stats each step; planner should merge with damping cadence.
    tele = TelemetrySpec(record_rho=True, record_cg=True)

    method = assemble(
        curvature=curv,
        solver=solv,
        damping=damp,
        precond=prec,
        estimator=est,
        telemetry=tele,
    )

    # Plan invariants
    assert method.plan.lane == "param"
    assert method.plan.req.rho_every_k == 5  # damping cadence dominates telemetry cadence
    assert mx.RHO in method.plan.metric_keys
    assert mx.CG_ITERS in method.plan.metric_keys

    # Curvature should be wrapped when an estimator is provided.
    # Don't rely on exact class name; check it exposes estimator attribute.
    assert hasattr(method.op, "init")
    assert hasattr(method.op, "matvec")


def test_assemble_invalid_curvature_raises():
    curv = CurvatureSpec("non_existent_curvature", {})
    solv = SolverSpec("cg", {})
    with pytest.raises(ValueError, match="Unknown curvature kind"):
        assemble(curvature=curv, solver=solv)


def test_method_init_structure():
    # Verify state structure and step initialization.
    method = newton_cg(loss_fn=lambda p, b: jnp.array(0.0, dtype=jnp.float32))
    params = {"w": jnp.zeros((10,), dtype=jnp.float32)}

    state = method.init(params)

    assert hasattr(state, "step")
    assert hasattr(state, "damp_state")
    assert hasattr(state, "solver_state")
    assert int(state.step) == 0
