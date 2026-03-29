import pytest

import jax.numpy as jnp

import somax
from somax import metrics as mx
from somax.planner import Requirements, build_plan
from somax.solvers.identity import IdentitySolve


class DummyDamping:
    def __init__(self, *, every_k: int, keys):
        self.every_k = every_k
        self._keys = frozenset(keys)

    def needed_metrics(self):
        return self._keys


class DummyPrecond:
    def __init__(self, keys=(), beta1=None):
        self._keys = frozenset(keys)
        if beta1 is not None:
            self.beta1 = beta1

    def needed_metrics(self):
        return self._keys


class DummySolver:
    def __init__(self, *, space=None):
        self.space = space


def test_planner_lane_diag_identity_solver():
    plan = build_plan(
        damping=DummyDamping(every_k=1, keys=()),
        solver=IdentitySolve(),
        precond=DummyPrecond(),
        telemetry_reqs=Requirements(metrics=frozenset()),
    )
    assert plan.lane == "diag"


def test_planner_lane_row_space_solver():
    plan = build_plan(
        damping=DummyDamping(every_k=1, keys=()),
        solver=DummySolver(space="row"),
        precond=DummyPrecond(),
        telemetry_reqs=Requirements(metrics=frozenset()),
    )
    assert plan.lane == "row"


def test_planner_lane_param_default():
    plan = build_plan(
        damping=DummyDamping(every_k=1, keys=()),
        solver=DummySolver(space=None),
        precond=DummyPrecond(),
        telemetry_reqs=Requirements(metrics=frozenset()),
    )
    assert plan.lane == "param"


def test_planner_always_includes_step_and_lam_used():
    plan = build_plan(
        damping=DummyDamping(every_k=1, keys=()),
        solver=DummySolver(),
        precond=DummyPrecond(),
        telemetry_reqs=Requirements(metrics=frozenset()),
    )
    keys = set(plan.metric_keys)
    assert mx.STEP in keys
    assert mx.LAM_USED in keys


def test_planner_rho_pack_dependency_augmentation():
    # Request any rho-related key; planner should augment to full rho pack + loss_before.
    telemetry = Requirements(metrics=frozenset({mx.RHO}))
    plan = build_plan(
        damping=DummyDamping(every_k=3, keys=()),  # cadence may come from telemetry/damping
        solver=DummySolver(),
        precond=DummyPrecond(),
        telemetry_reqs=telemetry,
    )
    keys = set(plan.metric_keys)
    assert plan.want.want_rho_pack
    # Full rho pack:
    for k in [mx.LOSS_BEFORE, mx.LOSS_AFTER, mx.PRED_DEC, mx.ACT_DEC, mx.RHO, mx.RHO_VALID, mx.STEP_ACCEPTED]:
        assert k in keys


def test_planner_lam_next_implies_lam_factor():
    telemetry = Requirements(metrics=frozenset({mx.LAM_NEXT}))
    plan = build_plan(
        damping=DummyDamping(every_k=1, keys=()),
        solver=DummySolver(),
        precond=DummyPrecond(),
        telemetry_reqs=telemetry,
    )
    keys = set(plan.metric_keys)
    assert mx.LAM_NEXT in keys
    assert mx.LAM_FACTOR in keys


def test_planner_want_m_detects_precond_beta1():
    plan = build_plan(
        damping=DummyDamping(every_k=1, keys=()),
        solver=DummySolver(),
        precond=DummyPrecond(beta1=0.9),
        telemetry_reqs=Requirements(metrics=frozenset()),
    )
    assert plan.want.want_m


def test_planner_want_loss_before_implied_by_rho_pack():
    telemetry = Requirements(metrics=frozenset({mx.RHO}))
    plan = build_plan(
        damping=DummyDamping(every_k=2, keys=()),
        solver=DummySolver(),
        precond=DummyPrecond(),
        telemetry_reqs=telemetry,
    )
    assert plan.want.want_loss_before
    assert mx.LOSS_BEFORE in set(plan.metric_keys)


def test_planner_cadence_invariants_for_unset():
    # If no rho output keys requested, want_rho_pack must be False and rho_every_k should be <1 (unset).
    plan = build_plan(
        damping=DummyDamping(every_k=7, keys=()),  # should NOT force rho cadence unless rho metrics are needed
        solver=DummySolver(),
        precond=DummyPrecond(),
        telemetry_reqs=Requirements(metrics=frozenset()),
    )
    assert not plan.want.want_rho_pack
    assert plan.req.rho_every_k < 1


def test_planner_rho_every_k_set_only_if_rho_metrics_needed():
    # Damping that does NOT need rho should not force rho cadence.
    # Keys mimic PI step-norm style requirements: STEP + S_RMS.
    pi_like = DummyDamping(every_k=5, keys={mx.STEP, mx.S_RMS})
    plan = build_plan(
        damping=pi_like,
        solver=DummySolver(),
        precond=DummyPrecond(),
        telemetry_reqs=Requirements(metrics=frozenset({mx.S_RMS})),
    )
    assert not plan.want.want_rho_pack
    assert plan.req.rho_every_k < 1
