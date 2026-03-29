"""Assembler: build a runnable SecondOrderMethod (init/step) from specs.

Contract:
- Public API is init()/step() only.
- Somax owns the post-direction transform tx and its opt_state.
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from .planner import Plan, build_plan, Requirements
from .specs import (
    CurvatureSpec,
    DampingSpec,
    EstimatorSpec,
    PrecondSpec,
    SolverSpec,
    TelemetrySpec,
    build_curvature,
    build_damping,
    build_precond,
    build_solver,
    build_telemetry_reqs,
)
from .optax import build_optax_tx
from .types import Array, Batch, Params, PRNGKey, StepInfo, ScalarOrSchedule, GradientTransformation


@struct.dataclass
class SecondOrderState:
    """Single state tree owned by Somax.

    opt_state: post-direction tx state (Optax state or custom).
    method_state: optional extra state for method-specific algorithms (eg AdaHessian EMAs).
    """
    step: Array
    damp_state: Any
    precond_state: Any
    solver_state: Any
    opt_state: Any
    method_state: Any  # optional extension point (eg AdaHessian)


class SecondOrderMethod:
    """Runnable Somax method: init/update.

    update(...) performs one fused step:
      init cstate -> grad/row_op -> solve direction -> tx.update -> apply -> (optional rho) -> damping.update
    """

    def __init__(
            self,
            *,
            op: Any,
            damping: Any,
            solver: Any,
            precond: Any,
            plan: Plan,
            tx: Any,
    ):
        self.op = op
        self.damping = damping
        self.solver = solver
        self.precond = precond
        self.plan = plan
        self.tx = tx

    def init(self, params: Params) -> SecondOrderState:
        m_t = jax.tree_util.tree_map(jnp.zeros_like, params) if self.plan.want.want_m else ()

        return SecondOrderState(
            step=jnp.asarray(0, jnp.int32),
            damp_state=self.damping.init(),
            precond_state=self.precond.init(params),
            solver_state=self.solver.init(params),
            opt_state=self.tx.init(params),
            method_state=m_t,  # empty by default; can be extended
        )

    def step(
            self,
            params: Params,
            batch: Batch,
            state: SecondOrderState,
            rng: PRNGKey,
    ) -> Tuple[Params, SecondOrderState, StepInfo]:
        from . import executor

        lane_fn = {
            "diag": executor.step_diag,
            "param": executor.step_param,
            "row": executor.step_row,
        }[self.plan.lane]

        return lane_fn(
            plan=self.plan,
            op=self.op,
            solver=self.solver,
            precond=self.precond,
            damping=self.damping,
            tx=self.tx,
            params=params,
            batch=batch,
            rng=rng,
            state=state,
        )


def assemble(
        *,
        curvature: CurvatureSpec,
        solver: SolverSpec,
        damping: Optional[DampingSpec] = None,
        precond: Optional[PrecondSpec] = None,
        estimator: Optional[EstimatorSpec] = None,
        telemetry: Optional[TelemetrySpec] = None,

        # post-direction application
        tx: GradientTransformation | None = None,  # full optax chain, if None, we build from knobs below
        learning_rate: ScalarOrSchedule = 1.0,
        clip_norm: float | None = None,
        weight_decay: float | None = None,
        weight_decay_mask: Any | None = None,
        direction_momentum: float | None = None,
        nesterov: bool = False,
) -> SecondOrderMethod:
    # Build components from specs
    op = build_curvature(curvature, estimator)
    dp = build_damping(damping)
    pp = build_precond(precond)
    slv = build_solver(solver, pp)
    tlm_reqs = build_telemetry_reqs(telemetry) if telemetry is not None else Requirements()

    # Build plan from components
    plan = build_plan(damping=dp, solver=slv, precond=pp, telemetry_reqs=tlm_reqs)

    # Build owned post-direction tx
    optax_chain = build_optax_tx(
        tx=tx,
        learning_rate=learning_rate,
        clip_norm=clip_norm,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        direction_momentum=direction_momentum,
        nesterov=nesterov,
    )

    return SecondOrderMethod(op=op, damping=dp, solver=slv, precond=pp, plan=plan, tx=optax_chain)
