"""Executor: fused init/solve/tx/apply/gain-ratio/damping update.

Contract:
- One step function per lane: step_param / step_row / step_diag.
- Keeps cstate alive until after delta is known, enabling exact pred_dec(delta).
- Gain-ratio damping never accepts/rejects here; the candidate step is
  always committed and only the next-step damping state is updated.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

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


# ---------------------------------------------------------------------------
# StepInfo packing and builder
# ---------------------------------------------------------------------------


def _split_model_state_update(cstate: Any) -> tuple[Any, Any]:
    """Remove the model-state update from the solver-visible curvature state.

    The update is an outer-step output, not part of the fixed linear operator.
    Keeping it in a separate local prevents row/parameter CG closures from
    carrying BatchNorm state as loop invariants.
    """
    model_state_new = cstate.model_state_new
    return cstate.replace(model_state_new=()), model_state_new


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
            grad_missing = isinstance(grad, tuple) and len(grad) == 0
            self._d[mx.G_DOT_S] = nan_scalar() if grad_missing else as_float(tree_vdot(grad, s))
        return self

    def add_delta_stats(self, delta: Updates) -> "StepInfoBuilder":
        if not self.plan.want.want_delta_rms:
            return self
        if mx.DELTA_RMS in self.plan.metric_keys:
            self._d[mx.DELTA_RMS] = as_float(tree_rms(delta))
        return self

    def add_gain_ratio_pack(
            self,
            *,
            loss_after: Scalar,
            act_dec: Scalar,
            pred_dec: Scalar,
            gain_ratio: Scalar,
            gain_ratio_valid: jax.Array,
            step_improved: jax.Array,
    ) -> "StepInfoBuilder":
        if not self.plan.want.want_gain_ratio_pack:
            return self
        if mx.LOSS_AFTER in self.plan.metric_keys:
            self._d[mx.LOSS_AFTER] = as_float(loss_after)
        if mx.ACT_DEC in self.plan.metric_keys:
            self._d[mx.ACT_DEC] = as_float(act_dec)
        if mx.PRED_DEC in self.plan.metric_keys:
            self._d[mx.PRED_DEC] = as_float(pred_dec)
        if mx.GAIN_RATIO in self.plan.metric_keys:
            self._d[mx.GAIN_RATIO] = as_float(gain_ratio)
        if mx.GAIN_RATIO_VALID in self.plan.metric_keys:
            self._d[mx.GAIN_RATIO_VALID] = jnp.asarray(gain_ratio_valid, jnp.bool_)
        if mx.STEP_IMPROVED in self.plan.metric_keys:
            self._d[mx.STEP_IMPROVED] = jnp.asarray(step_improved, jnp.bool_)
        return self

    def add_damping_next(self, *, damp_state2: Any, lam_used: Scalar) -> "StepInfoBuilder":
        lam_next = as_float(getattr(damp_state2, "lam"))

        if mx.LAM_NEXT in self.plan.metric_keys:
            self._d[mx.LAM_NEXT] = lam_next

        if mx.LAM_FACTOR in self.plan.metric_keys:
            eps = jnp.asarray(1e-16, dtype=lam_next.dtype)
            denom = jnp.maximum(as_float(lam_used), eps)
            self._d[mx.LAM_FACTOR] = lam_next / denom

        aux = getattr(damp_state2, "aux", None)
        if aux is not None and mx.TR_ACTION in self.plan.metric_keys:
            v = getattr(aux, "action", None)
            if v is not None:
                self._d[mx.TR_ACTION] = v

        if aux is not None and mx.LAM_CLIPPED in self.plan.metric_keys:
            v = getattr(aux, "lam_clipped", None)
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
# Loss and gain-ratio helpers
# ---------------------------------------------------------------------------

def _compute_loss_before(plan: Plan, step: jax.Array, loss_fn: Callable[[], Scalar]) -> Scalar:
    if not plan.want.want_loss_before:
        return nan_scalar()

    loss_dtype = jax.eval_shape(loss_fn).dtype
    do_loss = cadence(step, plan.req.loss_every_k)

    return jax.lax.cond(
        do_loss,
        lambda _: loss_fn(),
        lambda _: nan_scalar(loss_dtype),
        operand=None,
    )


def _ensure_loss_before(loss_before: Scalar, loss_fn: Callable[[], Scalar]) -> Scalar:
    return jax.lax.cond(
        jnp.isfinite(loss_before),
        lambda _: loss_before,
        lambda _: loss_fn(),
        operand=None,
    )


def _gain_ratio_from_parts(
        *,
        loss_before: Scalar,
        loss_after: Scalar,
        pred_dec: Scalar,
) -> Tuple[Scalar, Scalar, jax.Array, jax.Array]:
    act_dec = loss_before - loss_after
    tiny = jnp.asarray(1e-16, dtype=pred_dec.dtype)
    gain_ratio = as_float(act_dec / jnp.maximum(pred_dec, tiny))
    gain_ratio_valid = jnp.logical_and(jnp.isfinite(gain_ratio), pred_dec > tiny)
    # Diagnostic only: the current executor does not reject candidates.
    step_improved = jnp.logical_and(
        gain_ratio_valid,
        act_dec >= jnp.asarray(0.0, dtype=act_dec.dtype),
    )
    return as_float(act_dec), gain_ratio, gain_ratio_valid, step_improved


def _gain_ratio_pack_param(
        *,
        step: jax.Array,
        plan: Plan,
        loss_before: Scalar,
        loss_before_fn: Callable[[], Scalar],
        loss_after_fn: Callable[[], Scalar],
        grad: Updates,
        delta: Updates,
        matvec_delta_fn: Callable[[], Updates],
) -> Tuple[Scalar, Scalar, Scalar, Scalar, Scalar, jax.Array, jax.Array]:
    """Return loss/gain-ratio diagnostics for a parameter-space candidate."""
    nan = nan_scalar(jnp.float32)
    f = jnp.asarray(False, jnp.bool_)

    if not plan.want.want_gain_ratio_pack:
        return as_float(loss_before), nan, nan, nan, nan, f, f

    grad_missing = isinstance(grad, tuple) and len(grad) == 0
    if grad_missing:
        return as_float(loss_before), nan, nan, nan, nan, f, f

    do_gain_ratio = cadence(step, plan.req.gain_ratio_every_k)

    def _compute():
        lb = _ensure_loss_before(loss_before, loss_before_fn)
        la = loss_after_fn()

        g_dot_d = tree_vdot(grad, delta)
        Cd = matvec_delta_fn()
        d_C_d = tree_vdot(delta, Cd)
        # Compare the actual objective decrease against the undamped local
        # quadratic model.  Damping regularizes the linear solve; it is not a
        # term in the measured data objective.  Including 0.5 * lam * ||d||^2
        # in the denominator systematically inflates the gain ratio.
        pred = -(g_dot_d + as_float(0.5) * d_C_d)

        act, gain_ratio, gain_ratio_valid, step_improved = _gain_ratio_from_parts(
            loss_before=lb,
            loss_after=la,
            pred_dec=pred,
        )
        return as_float(lb), as_float(la), act, as_float(pred), gain_ratio, gain_ratio_valid, step_improved

    def _skip():
        return as_float(loss_before), nan, nan, nan, nan, f, f

    return jax.lax.cond(do_gain_ratio, lambda _: _compute(), lambda _: _skip(), operand=None)


def _gain_ratio_pack_row(
        *,
        step: jax.Array,
        plan: Plan,
        loss_before: Scalar,
        loss_before_fn: Callable[[], Scalar],
        loss_after_fn: Callable[[], Scalar],
        row_op: Any,
        delta: Updates,
) -> Tuple[Scalar, Scalar, Scalar, Scalar, Scalar, jax.Array, jax.Array]:
    """Return loss/gain-ratio diagnostics using row_op.{rhs,jvp,b}."""
    nan = nan_scalar(jnp.float32)
    f = jnp.asarray(False, jnp.bool_)

    if not plan.want.want_gain_ratio_pack:
        return as_float(loss_before), nan, nan, nan, nan, f, f

    do_gain_ratio = cadence(step, plan.req.gain_ratio_every_k)

    def _compute():
        lb = _ensure_loss_before(loss_before, loss_before_fn)
        la = loss_after_fn()

        z = row_op.jvp(delta)
        alpha = jnp.asarray(1.0 / row_op.b, dtype=z.dtype)
        g_dot_d = alpha * jnp.vdot(row_op.rhs, z)
        d_C_d = alpha * jnp.vdot(z, z)
        # As in the parameter-space lane, evaluate the undamped objective
        # model; lambda belongs to the solve, not to the measured objective.
        pred = -(g_dot_d + as_float(0.5) * d_C_d)

        act, gain_ratio, gain_ratio_valid, step_improved = _gain_ratio_from_parts(
            loss_before=lb,
            loss_after=la,
            pred_dec=pred,
        )
        return as_float(lb), as_float(la), act, as_float(pred), gain_ratio, gain_ratio_valid, step_improved

    def _skip():
        return as_float(loss_before), nan, nan, nan, nan, f, f

    return jax.lax.cond(do_gain_ratio, lambda _: _compute(), lambda _: _skip(), operand=None)


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

    cstate, grad = op.init(
        params,
        batch,
        model_state=state.model_state,
        with_grad=True,
    )
    cstate, model_state_new = _split_model_state_update(cstate)

    loss_before = _compute_loss_before(plan, step, lambda: op.loss(params, cstate, batch))

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
        return _tree_add_scaled(lam, v, Cv)

    s, solver_info, solver_state2 = solver.solve(
        A_mv, grad, state=state.solver_state, precond=pre_fn
    )

    delta, opt_state2 = tx.update(s, state.opt_state, params)
    params_new = optax.apply_updates(params, delta)

    loss_before_used, loss_after, act_dec, pred_dec, gain_ratio, gain_ratio_valid, step_improved = _gain_ratio_pack_param(
        step=step,
        plan=plan,
        loss_before=loss_before,
        loss_before_fn=lambda: op.loss(params, cstate, batch),
        loss_after_fn=lambda: op.loss_only(
            params_new,
            batch,
            # Keep the auxiliary model state fixed when measuring the actual
            # reduction of the local model. The state update from the primal
            # is committed independently below.
            model_state=state.model_state,
        ),
        grad=grad,
        delta=delta,
        matvec_delta_fn=lambda: op.matvec(params, cstate, delta),
    )

    b = StepInfoBuilder.create(plan)
    b.add_base(step=step, lam_used=lam, loss_before=loss_before_used)
    b.add_cg_stats(solver_info)
    b.add_s_stats(grad, s)
    b.add_delta_stats(delta)
    b.add_gain_ratio_pack(
        loss_after=loss_after,
        act_dec=act_dec,
        pred_dec=pred_dec,
        gain_ratio=gain_ratio,
        gain_ratio_valid=gain_ratio_valid,
        step_improved=step_improved,
    )

    info_mid = b.build()
    damp_state2 = damping.update(state.damp_state, info_mid)

    b.add_damping_next(damp_state2=damp_state2, lam_used=lam)
    info_out = b.build()

    state_out = state.replace(
        step=step + jnp.asarray(1, step.dtype),
        model_state=model_state_new,
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
        precond: Any,
        damping: Any,
        tx: Any,
        params: Params,
        batch: Batch,
        rng: PRNGKey,
        state: Any,
) -> Tuple[Params, Any, StepInfo]:
    del precond, rng
    with_grad = plan.want.want_grad
    step = state.step
    lam = state.damp_state.lam

    cstate, grad = op.init(
        params,
        batch,
        model_state=state.model_state,
        with_grad=with_grad,
    )
    cstate, model_state_new = _split_model_state_update(cstate)

    loss_before = _compute_loss_before(plan, step, lambda: op.loss(params, cstate, batch))

    row_op = op.row_op(params, cstate, batch)
    # Static capability selection; generic solve() continues to return row u.
    solve_row_direction = getattr(solver, "solve_row_direction", None)
    if solve_row_direction is not None:
        s, solver_info, solver_state2 = solve_row_direction(
            row_op, lam, params=params, state=state.solver_state, reduction="mean"
        )
    else:
        A_mv_row, rhs_row, backproject_fn, _ = bind_row_system(
            row_op=row_op, lam=lam, reduction="mean"
        )
        u, solver_info, solver_state2 = solver.solve(
            A_mv_row, rhs_row, state=state.solver_state, precond=None
        )
        s = backproject_fn(u)

    delta, opt_state2 = tx.update(s, state.opt_state, params)
    params_new = optax.apply_updates(params, delta)

    loss_before_used, loss_after, act_dec, pred_dec, gain_ratio, gain_ratio_valid, step_improved = _gain_ratio_pack_row(
        step=step,
        plan=plan,
        loss_before=loss_before,
        loss_before_fn=lambda: op.loss(params, cstate, batch),
        loss_after_fn=lambda: op.loss_only(
            params_new,
            batch,
            model_state=state.model_state,
        ),
        row_op=row_op,
        delta=delta,
    )

    b = StepInfoBuilder.create(plan)
    b.add_base(step=step, lam_used=lam, loss_before=loss_before_used)
    b.add_cg_stats(solver_info)
    if plan.want.want_s_stats:
        b.add_s_stats(grad, s)
    b.add_delta_stats(delta)
    b.add_gain_ratio_pack(
        loss_after=loss_after,
        act_dec=act_dec,
        pred_dec=pred_dec,
        gain_ratio=gain_ratio,
        gain_ratio_valid=gain_ratio_valid,
        step_improved=step_improved,
    )

    info_mid = b.build()
    damp_state2 = damping.update(state.damp_state, info_mid)

    b.add_damping_next(damp_state2=damp_state2, lam_used=lam)
    info_out = b.build()

    state_out = state.replace(
        step=step + jnp.asarray(1, step.dtype),
        model_state=model_state_new,
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

    cstate, grad = op.init(
        params,
        batch,
        model_state=state.model_state,
        with_grad=True,
    )
    cstate, model_state_new = _split_model_state_update(cstate)

    loss_before = _compute_loss_before(
        plan, step, lambda: op.loss(params, cstate, batch)
    )

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

    s, solver_info, solver_state2 = solver.solve(
        None, m_hat, state=state.solver_state, precond=pre_fn
    )

    delta, opt_state2 = tx.update(s, state.opt_state, params)
    params_new = optax.apply_updates(params, delta)

    loss_before_used, loss_after, act_dec, pred_dec, gain_ratio, gain_ratio_valid, step_improved = _gain_ratio_pack_param(
        step=step,
        plan=plan,
        loss_before=loss_before,
        loss_before_fn=lambda: op.loss(params, cstate, batch),
        loss_after_fn=lambda: op.loss_only(
            params_new,
            batch,
            model_state=state.model_state,
        ),
        grad=grad,
        delta=delta,
        matvec_delta_fn=lambda: op.matvec(params, cstate, delta),
    )

    b = StepInfoBuilder.create(plan)
    b.add_base(step=step, lam_used=lam, loss_before=loss_before_used)
    b.add_cg_stats(solver_info)
    b.add_s_stats(grad, s)
    b.add_delta_stats(delta)
    b.add_gain_ratio_pack(
        loss_after=loss_after,
        act_dec=act_dec,
        pred_dec=pred_dec,
        gain_ratio=gain_ratio,
        gain_ratio_valid=gain_ratio_valid,
        step_improved=step_improved,
    )

    info_mid = b.build()
    damp_state2 = damping.update(state.damp_state, info_mid)

    b.add_damping_next(damp_state2=damp_state2, lam_used=lam)
    info_out = b.build()

    state_out = state.replace(
        step=step + jnp.asarray(1, step.dtype),
        model_state=model_state_new,
        damp_state=damp_state2,
        precond_state=precond_state2,
        solver_state=solver_state2,
        opt_state=opt_state2,
        method_state=method_state2,
    )
    return params_new, state_out, info_out
