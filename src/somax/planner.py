from dataclasses import dataclass
import math
from typing import FrozenSet, Literal, Tuple

from . import metrics as mx
from .damping.base import DampingPolicy
from .preconditioners.base import PreconditionerPolicy
from .solvers.base import LinearSolver


def _gcd_pos(a: int, b: int) -> int:
    """Return gcd(a,b) over positive ints, treating <1 as 'unset'."""
    a = int(a)
    b = int(b)
    if a < 1:
        return b
    if b < 1:
        return a
    return math.gcd(a, b)


def _wants_gain_ratio_pack(keys: FrozenSet[str]) -> bool:
    return (
            (mx.GAIN_RATIO in keys)
            or (mx.LOSS_AFTER in keys)
            or (mx.PRED_DEC in keys)
            or (mx.ACT_DEC in keys)
            or (mx.GAIN_RATIO_VALID in keys)
            or (mx.STEP_IMPROVED in keys)
    )


@dataclass(frozen=True)
class Requirements:
    """Planner-time static requirements.

    metrics: keys that must be present in StepInfo (FrozenDict) every step.
    loss_every_k: cadence for computing loss_before. <1 means 'unset'.
    gain_ratio_every_k: cadence for gain-ratio pack. <1 means 'unset'.
    """
    metrics: FrozenSet[str] = frozenset()
    loss_every_k: int = -1
    gain_ratio_every_k: int = -1

    def merge(self, other: "Requirements") -> "Requirements":
        return Requirements(
            metrics=frozenset(set(self.metrics) | set(other.metrics)),
            loss_every_k=_gcd_pos(self.loss_every_k, other.loss_every_k),
            gain_ratio_every_k=_gcd_pos(self.gain_ratio_every_k, other.gain_ratio_every_k),
        )


@dataclass(frozen=True)
class Wants:
    want_loss_before: bool
    want_grad: bool
    want_gain_ratio_pack: bool
    want_delta_rms: bool
    want_s_stats: bool
    want_cg_stats: bool
    want_m: bool


@dataclass(frozen=True)
class Plan:
    lane: Literal["diag", "param", "row"]
    req: Requirements
    metric_keys: Tuple[str, ...]  # ordered schema
    want: Wants


def _choose_lane(solver: LinearSolver) -> Literal["diag", "param", "row"]:
    space = getattr(solver, "space", None)
    if space == "diag":
        return "diag"
    if space == "row":
        return "row"
    return "param"


def _requirements_from_damping(damping: DampingPolicy) -> Requirements:
    keys = frozenset(damping.needed_metrics())
    gain_ratio_every_k = -1
    if (mx.GAIN_RATIO in keys) or (mx.GAIN_RATIO_VALID in keys):
        gain_ratio_every_k = int(getattr(damping, "every_k", -1))
    return Requirements(metrics=keys, gain_ratio_every_k=gain_ratio_every_k)


def _requirements_from_precond(precond: PreconditionerPolicy) -> Requirements:
    return Requirements(metrics=frozenset(precond.needed_metrics()))


def _augment_dependencies(keys_in: FrozenSet[str]) -> FrozenSet[str]:
    keys = set(keys_in)

    keys.add(mx.STEP)
    keys.add(mx.LAM_USED)

    if _wants_gain_ratio_pack(frozenset(keys)):
        keys.add(mx.LOSS_BEFORE)
        keys.add(mx.LOSS_AFTER)
        keys.add(mx.PRED_DEC)
        keys.add(mx.ACT_DEC)
        keys.add(mx.GAIN_RATIO)
        keys.add(mx.GAIN_RATIO_VALID)
        keys.add(mx.STEP_IMPROVED)

    if mx.LAM_NEXT in keys:
        keys.add(mx.LAM_FACTOR)

    return frozenset(keys)


def _compute_wants(
        keys: FrozenSet[str],
        precond: PreconditionerPolicy,
        *,
        lane: Literal["diag", "param", "row"],
) -> Wants:
    want_loss_before = mx.LOSS_BEFORE in keys
    want_gain_ratio_pack = _wants_gain_ratio_pack(keys)

    want_delta_rms = mx.DELTA_RMS in keys
    want_s_stats = (mx.G_DOT_S in keys) or (mx.S_NORM2 in keys) or (mx.S_RMS in keys)
    want_cg_stats = (
            (mx.CG_ITERS in keys)
            or (mx.CG_RESID in keys)
            or (mx.CG_MAXITER in keys)
            or (mx.CG_CONVERGED in keys)
    )

    want_grad = (mx.G_DOT_S in keys) or (want_gain_ratio_pack and lane != "row")

    # beta1 is consumed only by the diagonal optimizer lane. Parameter-space
    # PCG never updates method_state, so allocating a full parameter-sized
    # first-moment tree there is pure memory overhead.
    want_m = False
    precond_beta1 = getattr(precond, "beta1", None)
    if lane == "diag" and precond_beta1 is not None and precond_beta1 > 0.0:
        want_m = True

    return Wants(
        want_loss_before=want_loss_before,
        want_grad=want_grad,
        want_gain_ratio_pack=want_gain_ratio_pack,
        want_delta_rms=want_delta_rms,
        want_s_stats=want_s_stats,
        want_cg_stats=want_cg_stats,
        want_m=want_m,
    )


def build_plan(
        *,
        damping: DampingPolicy,
        solver: LinearSolver,
        precond: PreconditionerPolicy,
        telemetry_reqs: Requirements,
) -> Plan:
    lane = _choose_lane(solver)

    req = Requirements()
    req = req.merge(_requirements_from_damping(damping))
    req = req.merge(_requirements_from_precond(precond))
    req = req.merge(telemetry_reqs)

    metrics = _augment_dependencies(req.metrics)
    want = _compute_wants(metrics, precond, lane=lane)

    loss_every_k = int(req.loss_every_k)
    gain_ratio_every_k = int(req.gain_ratio_every_k)

    if not want.want_loss_before:
        loss_every_k = -1
    if not want.want_gain_ratio_pack:
        gain_ratio_every_k = -1

    if want.want_loss_before and loss_every_k < 1:
        loss_every_k = 1
    if want.want_gain_ratio_pack and gain_ratio_every_k < 1:
        gain_ratio_every_k = 1

    if want.want_gain_ratio_pack:
        loss_every_k = _gcd_pos(loss_every_k, gain_ratio_every_k)

    req2 = Requirements(
        metrics=metrics,
        loss_every_k=loss_every_k,
        gain_ratio_every_k=gain_ratio_every_k,
    )
    metric_keys = tuple(sorted(req2.metrics))
    return Plan(lane=lane, req=req2, metric_keys=metric_keys, want=want)
