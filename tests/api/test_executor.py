from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import struct
from flax.core import FrozenDict

from somax import metrics as mx
from somax.damping.base import DampingState
from somax.executor import step_param
from somax.planner import Requirements, build_plan


# ---------------- Tiny fake components ----------------


class QuadOp:
    """Quadratic objective: loss = 0.5 * ||params||^2, grad = params, curvature = I."""

    def init(self, params, batch, *, with_grad: bool):
        # cstate can be anything; executor keeps it alive for matvec.
        cstate = ()
        grad = params if with_grad else ()
        return cstate, grad

    def loss(self, params, cstate, batch):
        leaves = jax.tree_util.tree_leaves(params)
        v = jnp.concatenate([jnp.ravel(x) for x in leaves], axis=0) if len(leaves) > 1 else jnp.ravel(leaves[0])
        return jnp.asarray(0.5, v.dtype) * jnp.vdot(v, v)

    def loss_only(self, params, batch):
        return self.loss(params, (), batch)

    def matvec(self, params, cstate, v):
        # I * v
        return v


class DirectSolver:
    """Solve (I + lam I) s = grad -> s = grad / (1+lam)."""

    def solve(self, A_mv, rhs, *, state, precond):
        # rhs is a PyTree
        # figure out lam by probing A_mv on rhs: A(rhs) = rhs + lam*rhs for our A_mv in executor
        Arhs = A_mv(rhs)
        # Estimate scalar lam from first leaf ratio: Arhs = (1+lam)*rhs
        leaf_rhs = jax.tree_util.tree_leaves(rhs)[0]
        leaf_Arhs = jax.tree_util.tree_leaves(Arhs)[0]
        ratio = (leaf_Arhs / jnp.maximum(leaf_rhs, jnp.asarray(1e-12, leaf_rhs.dtype))).mean()
        lam = jnp.maximum(ratio - jnp.asarray(1.0, ratio.dtype), jnp.asarray(0.0, ratio.dtype))

        denom = jnp.asarray(1.0, lam.dtype) + lam
        s = jax.tree_util.tree_map(lambda x: x / denom, rhs)

        info = {}  # executor will default-fill cg stats if not requested
        return s, info, state


class NullPrecond:
    def needed_metrics(self):
        return frozenset()

    def build(self, *, params, op, cstate, rng, lam, state):
        return None, state


class DummyDampingNoOp:
    """No-op damping. Exposes every_k only for planner; update returns same lam."""
    every_k: int = 1

    def needed_metrics(self):
        # TR-like: needs rho to decide, but update is no-op here.
        return frozenset({mx.STEP, mx.RHO, mx.RHO_VALID})

    def init(self):
        return DampingState(lam=jnp.asarray(1.0, jnp.float32), aux=())

    def update(self, state, info):
        return state


@struct.dataclass
class ExecState:
    step: jax.Array
    damp_state: Any
    precond_state: Any
    solver_state: Any
    opt_state: Any
    method_state: Any

    # Match flax.struct dataclass API used by executor.
    def replace(self, **kwargs):
        return self.replace(**kwargs)  # pragma: no cover


# Use Flax struct dataclass to get .replace method.
@struct.dataclass
class State:
    step: jax.Array
    damp_state: Any
    precond_state: Any
    solver_state: Any
    opt_state: Any
    method_state: Any


# ---------------- Tests ----------------


def _init_state(params, damping, tx):
    return State(
        step=jnp.asarray(0, jnp.int32),
        damp_state=damping.init(),
        precond_state=(),
        solver_state=(),
        opt_state=tx.init(params),
        method_state=(),
    )


def test_executor_step_param_jit_and_schema(dtype):
    op = QuadOp()
    solver = DirectSolver()
    precond = NullPrecond()
    damping = DummyDampingNoOp()

    # Request metrics that exercise base + rho pack + damping next.
    telemetry = Requirements(metrics=frozenset({mx.RHO, mx.LAM_NEXT, mx.LAM_FACTOR, mx.LOSS_BEFORE}))
    plan = build_plan(damping=damping, solver=solver, precond=precond, telemetry_reqs=telemetry)

    # Use optax scale as "tx" (post-direction transform).
    tx = optax.scale(-0.1)

    params = {"w": jnp.asarray([1.0, -2.0, 3.0], dtype)}
    batch = {"x": jnp.zeros((1,), dtype), "y": jnp.zeros((1,), dtype)}
    rng = jax.random.PRNGKey(0)
    state = _init_state(params, damping, tx)

    @jax.jit
    def run(p, st):
        return step_param(
            plan=plan,
            op=op,
            solver=solver,
            precond=precond,
            damping=damping,
            tx=tx,
            params=p,
            batch=batch,
            rng=rng,
            state=st,
        )

    params2, state2, info = run(params, state)

    assert isinstance(info, FrozenDict)
    assert set(info.keys()) == set(plan.metric_keys)

    # Step increments.
    assert int(state2.step) == 1

    # Some invariants: lam_next exists because telemetry requested it.
    assert jnp.isfinite(info[mx.LAM_NEXT])
    assert jnp.isfinite(info[mx.LAM_FACTOR])

    # Rho pack requested -> rho_valid should be bool scalar.
    assert info[mx.RHO_VALID].dtype == jnp.bool_
    assert info[mx.STEP].dtype == jnp.int32


def test_executor_rho_pack_respects_cadence(dtype):
    op = QuadOp()
    solver = DirectSolver()
    precond = NullPrecond()
    damping = DummyDampingNoOp()

    # Ask for rho metrics so want_rho_pack=True.
    telemetry = Requirements(metrics=frozenset({mx.RHO, mx.RHO_VALID, mx.LOSS_AFTER, mx.PRED_DEC, mx.ACT_DEC}))
    plan = build_plan(damping=damping, solver=solver, precond=precond, telemetry_reqs=telemetry)

    # Force cadence to every 2 steps (0,2,4,...). This is planner-owned; safe to overwrite here.
    plan = plan.__class__(
        lane=plan.lane,
        req=plan.req.__class__(metrics=plan.req.metrics, loss_every_k=plan.req.loss_every_k, rho_every_k=2),
        metric_keys=plan.metric_keys,
        want=plan.want,
    )

    tx = optax.scale(-0.1)
    params = {"w": jnp.asarray([1.0, 2.0, 3.0], dtype)}
    batch = {"x": jnp.zeros((1,), dtype), "y": jnp.zeros((1,), dtype)}
    rng = jax.random.PRNGKey(0)
    state = _init_state(params, damping, tx)

    @jax.jit
    def run(p, st):
        return step_param(
            plan=plan,
            op=op,
            solver=solver,
            precond=precond,
            damping=damping,
            tx=tx,
            params=p,
            batch=batch,
            rng=rng,
            state=st,
        )

    # step 0 -> compute rho pack (finite)
    p1, s1, info0 = run(params, state)
    assert bool(info0[mx.RHO_VALID])  # should be valid on compute steps
    assert jnp.isfinite(info0[mx.RHO])

    # step 1 -> cadence off -> should be NaN/False
    p2, s2, info1 = run(p1, s1)
    assert not bool(info1[mx.RHO_VALID])
    assert jnp.isnan(info1[mx.RHO])

    # step 2 -> cadence on again
    p3, s3, info2 = run(p2, s2)
    assert bool(info2[mx.RHO_VALID])
    assert jnp.isfinite(info2[mx.RHO])
