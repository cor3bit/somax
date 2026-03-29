"""Executor: fused init/solve/tx/apply/rho/damping update.

Contract:
- One step function per lane: step_param / step_row / step_diag.
- Keeps cstate alive until after delta is known, enabling exact pred_dec(delta).
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from . import metrics as mx
from .planner import Plan
from .types import Batch, Params, PRNGKey, Scalar, StepInfo, Updates
from .utils import (
    as_float,
    as_int,
    cadence,
    nan_scalar,
    tree_norm2,
    tree_rms,
    tree_vdot,
)
from .solvers.row_common import bind_row_system
from .preconditioners.diag_direct import DiagDirectPrecond


# ---------------------------------------------------------------------------
# StepInfo packing and builder
# ---------------------------------------------------------------------------

def _pack_stepinfo(plan: Plan, partial: Dict[str, jax.Array]) -> StepInfo:
    """Pack a partial dict into a full StepInfo with static schema."""
    out: Dict[str, jax.Array] = {}
    for k in plan.metric_keys:
        out[k] = partial.get(k, mx.default(k))
    return FrozenDict(out)


@dataclass
class StepInfoBuilder:
    plan: Plan
    _d: Dict[str, jax.Array]

    @classmethod
    def create(cls, plan: Plan) -> "StepInfoBuilder":
        return cls(plan=plan, _d={})

    def add(self, key: str, value: jax.Array) -> "StepInfoBuilder":
        self._d[key] = value
        return self

    def add_base(self, *, step: jax.Array, lam_used: Scalar, loss_before: Scalar) -> "StepInfoBuilder":
        if mx.STEP in self.plan.metric_keys:
            self._d[mx.STEP] = as_int(step)
        if mx.LAM_USED in self.plan.metric_keys:
            self._d[mx.LAM_USED] = as_float(lam_used)
        if mx.LOSS_BEFORE in self.plan.metric_keys:
            self._d[mx.LOSS_BEFORE] = as_float(loss_before)
        return self

    def add_cg_stats(self, solver_info: Dict[str, Any]) -> "StepInfoBuilder":
        if not self.plan.want.want_cg_stats:
            return self

        if mx.CG_ITERS in self.plan.metric_keys:
            self._d[mx.CG_ITERS] = as_int(solver_info.get(mx.CG_ITERS, -1))
        if mx.CG_MAXITER in self.plan.metric_keys:
            self._d[mx.CG_MAXITER] = as_int(solver_info.get(mx.CG_MAXITER, -1))
        if mx.CG_RESID in self.plan.metric_keys:
            self._d[mx.CG_RESID] = as_float(solver_info.get(mx.CG_RESID, nan_scalar()))
        if mx.CG_CONVERGED in self.plan.metric_keys:
            self._d[mx.CG_CONVERGED] = jnp.asarray(solver_info.get(mx.CG_CONVERGED, False), jnp.bool_)
        return self

    def add_s_stats(self, grad: Updates, s: Updates) -> "StepInfoBuilder":
        if not self.plan.want.want_s_stats:
            return self

        if mx.S_NORM2 in self.plan.metric_keys:
            self._d[mx.S_NORM2] = as_float(tree_norm2(s))
        if mx.S_RMS in self.plan.metric_keys:
            self._d[mx.S_RMS] = as_float(tree_rms(s))

        if mx.G_DOT_S in self.plan.metric_keys:
            # grad can be () depending on lane/planner. Keep schema stable.
            self._d[mx.G_DOT_S] = nan_scalar() if grad == () else as_float(tree_vdot(grad, s))

        return self

    def add_delta_stats(self, delta: Updates) -> "StepInfoBuilder":
        if not self.plan.want.want_delta_rms:
            return self
        if mx.DELTA_RMS in self.plan.metric_keys:
            self._d[mx.DELTA_RMS] = as_float(tree_rms(delta))
        return self

    def add_rho_pack(
            self,
            *,
            loss_after: Scalar,
            act_dec: Scalar,
            pred_dec: Scalar,
            rho: Scalar,
            rho_valid: jax.Array,
            step_accepted: jax.Array,
    ) -> "StepInfoBuilder":
        if not self.plan.want.want_rho_pack:
            return self
        if mx.LOSS_AFTER in self.plan.metric_keys:
            self._d[mx.LOSS_AFTER] = as_float(loss_after)
        if mx.ACT_DEC in self.plan.metric_keys:
            self._d[mx.ACT_DEC] = as_float(act_dec)
        if mx.PRED_DEC in self.plan.metric_keys:
            self._d[mx.PRED_DEC] = as_float(pred_dec)
        if mx.RHO in self.plan.metric_keys:
            self._d[mx.RHO] = as_float(rho)
        if mx.RHO_VALID in self.plan.metric_keys:
            self._d[mx.RHO_VALID] = jnp.asarray(rho_valid, jnp.bool_)
        if mx.STEP_ACCEPTED in self.plan.metric_keys:
            self._d[mx.STEP_ACCEPTED] = jnp.asarray(step_accepted, jnp.bool_)
        return self

    def add_damping_next(self, *, damp_state2: Any, lam_used: Scalar) -> "StepInfoBuilder":
        lam_next = as_float(getattr(damp_state2, "lam"))

        if mx.LAM_NEXT in self.plan.metric_keys:
            self._d[mx.LAM_NEXT] = lam_next

        if mx.LAM_FACTOR in self.plan.metric_keys:
            eps = jnp.asarray(1e-16, dtype=lam_next.dtype)
            denom = jnp.maximum(as_float(lam_used), eps)
            self._d[mx.LAM_FACTOR] = lam_next / denom

        # Optional extras: only include if already in schema.
        if mx.TR_ACTION in self.plan.metric_keys:
            aux = getattr(damp_state2, "aux", None)
            v = getattr(aux, "last_action", None) if aux is not None else None
            if v is not None:
                self._d[mx.TR_ACTION] = v

        if mx.LAM_CLIPPED in self.plan.metric_keys:
            aux = getattr(damp_state2, "aux", None)
            v = getattr(aux, "lam_clipped", None) if aux is not None else None
            if v is not None:
                self._d[mx.LAM_CLIPPED] = v

        return self

    def build(self) -> StepInfo:
        return _pack_stepinfo(self.plan, self._d)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _tree_add_scaled(a: Scalar, x: Updates, y: Updates) -> Updates:
    """Return y + a * x, leafwise."""
    return jax.tree_util.tree_map(lambda xi, yi: yi + a * xi, x, y)


# ---------------------------------------------------------------------------
# Loss and rho helpers
# ---------------------------------------------------------------------------

def _compute_loss_before(plan: Plan, step: jax.Array, loss_fn: Callable[[], Scalar]) -> Scalar:
    if not plan.want.want_loss_before:
        return nan_scalar()
    do_loss = cadence(step, plan.req.loss_every_k)
    return jax.lax.cond(
        do_loss,
        lambda _: loss_fn(),
        lambda _: nan_scalar(),
        operand=None,
    )


def _ensure_loss_before(loss_before: Scalar, loss_fn: Callable[[], Scalar]) -> Scalar:
    return jax.lax.cond(
        jnp.isfinite(loss_before),
        lambda _: loss_before,
        lambda _: loss_fn(),
        operand=None,
    )


def _rho_pack(
        *,
        step: jax.Array,
        plan: Plan,
        lam: Scalar,
        loss_before: Scalar,
        loss_before_fn: Callable[[], Scalar],
        loss_after_fn: Callable[[], Scalar],
        grad: Updates,
        delta: Updates,
        matvec_delta_fn: Callable[[], Updates],
) -> Tuple[Scalar, Scalar, Scalar, Scalar, jax.Array, jax.Array]:
    """Return (loss_after, act_dec, pred_dec, rho, rho_valid, step_accepted) or NaNs/False."""
    nan = nan_scalar(jnp.float32)
    f = jnp.asarray(False, jnp.bool_)

    if not plan.want.want_rho_pack:
        return nan, nan, nan, nan, f, f

    # Robustness: rho needs grad for g * delta. If grad is (), treat rho as unavailable.
    if grad == ():
        return nan, nan, nan, nan, f, f

    do_rho = cadence(step, plan.req.rho_every_k)

    def _compute():
        lb = _ensure_loss_before(loss_before, loss_before_fn)
        la = as_float(loss_after_fn())
        act = as_float(lb - la)

        g_dot_d = as_float(tree_vdot(grad, delta))
        Cd = matvec_delta_fn()
        d_C_d = as_float(tree_vdot(delta, Cd))
        d_n2 = as_float(tree_norm2(delta))
        quad = as_float(d_C_d + as_float(lam) * d_n2)

        pred = as_float(-(g_dot_d + as_float(0.5) * quad))

        tiny = jnp.asarray(1e-16, dtype=pred.dtype)
        rho = as_float(act / jnp.maximum(pred, tiny))
        rho_valid = jnp.logical_and(jnp.isfinite(rho), pred > tiny)
        step_acc = jnp.logical_and(rho_valid, rho >= jnp.asarray(0.0, dtype=rho.dtype))
        return la, act, pred, rho, rho_valid, step_acc

    def _skip():
        return nan, nan, nan, nan, f, f

    return jax.lax.cond(do_rho, lambda _: _compute(), lambda _: _skip(), operand=None)


# ---------------------------------------------------------------------------
# Lane: param-space
# ---------------------------------------------------------------------------

def step_param(
        *,
        plan: Plan,
        op: Any,
        solver: Any,
        precond: Any,
        damping: Any,
        tx: Any,
        params: Params,
        batch: Batch,
        rng: PRNGKey,
        state: Any,
) -> Tuple[Params, Any, StepInfo]:
    step = state.step
    lam = state.damp_state.lam

    cstate, grad = op.init(params, batch, with_grad=True)

    loss_before = _compute_loss_before(plan, step, lambda: as_float(op.loss(params, cstate, batch)))

    # TODO: move to plan
    if isinstance(precond, DiagDirectPrecond):
        diag_override = jax.tree_util.tree_map(lambda g: g * g, grad)
        pre_fn, precond_state2 = precond.build(
            params=params,
            op=op,
            cstate=cstate,
            rng=rng,
            lam=lam,
            state=state.precond_state,
            diag=diag_override,
        )
    else:
        pre_fn, precond_state2 = precond.build(
            params=params,
            op=op,
            cstate=cstate,
            rng=rng,
            lam=lam,
            state=state.precond_state,
        )

    def A_mv(v: Updates) -> Updates:
        Cv = op.matvec(params, cstate, v)
        return _tree_add_scaled(lam, v, Cv)  # Cv + lam * v

    s, solver_info, solver_state2 = solver.solve(
        A_mv, grad, state=state.solver_state, precond=pre_fn
    )

    delta, opt_state2 = tx.update(s, state.opt_state, params)
    params_new = optax.apply_updates(params, delta)

    b = StepInfoBuilder.create(plan)
    b.add_base(step=step, lam_used=lam, loss_before=loss_before)
    b.add_cg_stats(solver_info)
    b.add_s_stats(grad, s)
    b.add_delta_stats(delta)

    loss_after, act_dec, pred_dec, rho, rho_valid, step_acc = _rho_pack(
        step=step,
        plan=plan,
        lam=as_float(lam),
        loss_before=as_float(loss_before),
        loss_before_fn=lambda: as_float(op.loss(params, cstate, batch)),
        loss_after_fn=lambda: op.loss_only(params_new, batch),
        grad=grad,
        delta=delta,
        matvec_delta_fn=lambda: op.matvec(params, cstate, delta),
    )
    b.add_rho_pack(
        loss_after=loss_after,
        act_dec=act_dec,
        pred_dec=pred_dec,
        rho=rho,
        rho_valid=rho_valid,
        step_accepted=step_acc,
    )

    info_mid = b.build()
    damp_state2 = damping.update(state.damp_state, info_mid)

    b.add_damping_next(damp_state2=damp_state2, lam_used=lam)
    info_out = b.build()

    state_out = state.replace(
        step=step + jnp.asarray(1, step.dtype),
        damp_state=damp_state2,
        precond_state=precond_state2,
        solver_state=solver_state2,
        opt_state=opt_state2,
        method_state=state.method_state,
    )
    return params_new, state_out, info_out


# ---------------------------------------------------------------------------
# Lane: row-space
# ---------------------------------------------------------------------------

def step_row(
        *,
        plan: Plan,
        op: Any,
        solver: Any,
        precond: Any,  # unused by design (row lane currently does not support preconditioning)
        damping: Any,
        tx: Any,
        params: Params,
        batch: Batch,
        rng: PRNGKey,
        state: Any,
) -> Tuple[Params, Any, StepInfo]:
    with_grad = plan.want.want_grad
    step = state.step
    lam = state.damp_state.lam

    cstate, grad = op.init(params, batch, with_grad=with_grad)

    loss_before = _compute_loss_before(plan, step, lambda: as_float(op.loss(params, cstate, batch)))

    row_op = op.row_op(params, cstate, batch)
    A_mv_row, rhs_row, backproject_fn, _ = bind_row_system(
        row_op=row_op, lam=lam, reduction="mean"
    )

    u, solver_info, solver_state2 = solver.solve(
        A_mv_row, rhs_row, state=state.solver_state, precond=None
    )
    s = backproject_fn(u)

    delta, opt_state2 = tx.update(s, state.opt_state, params)
    params_new = optax.apply_updates(params, delta)

    b = StepInfoBuilder.create(plan)
    b.add_base(step=step, lam_used=lam, loss_before=loss_before)
    b.add_cg_stats(solver_info)

    # Only compute if requested; schema gates inclusion.
    if plan.want.want_s_stats:
        b.add_s_stats(grad, s)
    b.add_delta_stats(delta)

    loss_after, act_dec, pred_dec, rho, rho_valid, step_acc = _rho_pack(
        step=step,
        plan=plan,
        lam=as_float(lam),
        loss_before=as_float(loss_before),
        loss_before_fn=lambda: as_float(op.loss(params, cstate, batch)),
        loss_after_fn=lambda: op.loss_only(params_new, batch),
        grad=grad,
        delta=delta,
        matvec_delta_fn=lambda: op.matvec(params, cstate, delta),
    )
    b.add_rho_pack(
        loss_after=loss_after,
        act_dec=act_dec,
        pred_dec=pred_dec,
        rho=rho,
        rho_valid=rho_valid,
        step_accepted=step_acc,
    )

    info_mid = b.build()
    damp_state2 = damping.update(state.damp_state, info_mid)

    b.add_damping_next(damp_state2=damp_state2, lam_used=lam)
    info_out = b.build()

    state_out = state.replace(
        step=step + jnp.asarray(1, step.dtype),
        damp_state=damp_state2,
        precond_state=state.precond_state,
        solver_state=solver_state2,
        opt_state=opt_state2,
        method_state=state.method_state,
    )
    return params_new, state_out, info_out


# ---------------------------------------------------------------------------
# Lane: diagonal (diag preconditioner lane)
# ---------------------------------------------------------------------------

def step_diag(
        *,
        plan: Plan,
        op: Any,
        solver: Any,
        precond: Any,
        damping: Any,
        tx: Any,
        params: Params,
        batch: Batch,
        rng: PRNGKey,
        state: Any,
) -> Tuple[Params, Any, StepInfo]:
    step = state.step
    lam = state.damp_state.lam

    cstate, grad = op.init(params, batch, with_grad=True)

    loss_before = _compute_loss_before(plan, step, lambda: as_float(op.loss(params, cstate, batch)))

    # Numerator momentum (AdaHessian/Sophia-style).
    if plan.want.want_m:
        beta1_val = getattr(precond, "beta1", None)
        if beta1_val is None:
            raise ValueError("plan.want.want_m=True but precond has no attribute 'beta1'.")
        beta1 = jnp.asarray(min(float(beta1_val), 1.0 - 1e-12), jnp.float32)

        step_next = step + jnp.asarray(1, step.dtype)

        m_new = jax.tree_util.tree_map(
            lambda m, g: beta1 * m + (1.0 - beta1) * g,
            state.method_state,
            grad,
        )
        b1c = 1.0 - jnp.power(beta1, step_next.astype(jnp.float32))
        m_hat = jax.tree_util.tree_map(lambda mm: mm / b1c, m_new)
        method_state2 = m_new
    else:
        m_hat = grad
        method_state2 = state.method_state

    pre_fn, precond_state2 = precond.build(
        params=params,
        op=op,
        cstate=cstate,
        rng=rng,
        lam=lam,
        state=state.precond_state,
    )

    # Diag lane: solver uses precond directly (no A_mv).
    s, solver_info, solver_state2 = solver.solve(
        None, m_hat, state=state.solver_state, precond=pre_fn
    )

    delta, opt_state2 = tx.update(s, state.opt_state, params)
    params_new = optax.apply_updates(params, delta)

    b = StepInfoBuilder.create(plan)
    b.add_base(step=step, lam_used=lam, loss_before=loss_before)
    b.add_cg_stats(solver_info)
    b.add_s_stats(grad, s)
    b.add_delta_stats(delta)

    loss_after, act_dec, pred_dec, rho, rho_valid, step_acc = _rho_pack(
        step=step,
        plan=plan,
        lam=as_float(lam),
        loss_before=as_float(loss_before),
        loss_before_fn=lambda: as_float(op.loss(params, cstate, batch)),
        loss_after_fn=lambda: op.loss_only(params_new, batch),
        grad=grad,
        delta=delta,
        matvec_delta_fn=lambda: op.matvec(params, cstate, delta),
    )
    b.add_rho_pack(
        loss_after=loss_after,
        act_dec=act_dec,
        pred_dec=pred_dec,
        rho=rho,
        rho_valid=rho_valid,
        step_accepted=step_acc,
    )

    info_mid = b.build()
    damp_state2 = damping.update(state.damp_state, info_mid)

    b.add_damping_next(damp_state2=damp_state2, lam_used=lam)
    info_out = b.build()

    state_out = state.replace(
        step=step + jnp.asarray(1, step.dtype),
        damp_state=damp_state2,
        precond_state=precond_state2,
        solver_state=solver_state2,
        opt_state=opt_state2,
        method_state=method_state2,
    )
    return params_new, state_out, info_out
