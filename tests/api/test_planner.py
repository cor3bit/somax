import pytest

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


def _build(*, solver=None, damping=None, precond=None, telemetry=None):
    return build_plan(
        damping=damping or DummyDamping(every_k=1, keys=()),
        solver=solver or DummySolver(),
        precond=precond or DummyPrecond(),
        telemetry_reqs=telemetry or Requirements(metrics=frozenset()),
    )


def test_planner_lane_diag_identity_solver():
    assert _build(solver=IdentitySolve()).lane == "diag"


def test_planner_lane_row_space_solver():
    assert _build(solver=DummySolver(space="row")).lane == "row"


def test_planner_lane_param_default():
    assert _build(solver=DummySolver(space=None)).lane == "param"


def test_planner_always_includes_step_and_lam_used():
    keys = set(_build().metric_keys)
    assert mx.STEP in keys
    assert mx.LAM_USED in keys


def test_planner_gain_ratio_pack_dependency_augmentation():
    plan = _build(
        damping=DummyDamping(every_k=3, keys=()),
        telemetry=Requirements(metrics=frozenset({mx.GAIN_RATIO})),
    )
    keys = set(plan.metric_keys)
    assert plan.want.want_gain_ratio_pack
    for key in (
            mx.LOSS_BEFORE,
            mx.LOSS_AFTER,
            mx.PRED_DEC,
            mx.ACT_DEC,
            mx.GAIN_RATIO,
            mx.GAIN_RATIO_VALID,
            mx.STEP_IMPROVED,
    ):
        assert key in keys


def test_step_accepted_is_only_a_legacy_alias_for_step_improved():
    assert mx.STEP_ACCEPTED == mx.STEP_IMPROVED


def test_planner_lam_next_implies_lam_factor():
    plan = _build(
        telemetry=Requirements(metrics=frozenset({mx.LAM_NEXT}))
    )
    keys = set(plan.metric_keys)
    assert mx.LAM_NEXT in keys
    assert mx.LAM_FACTOR in keys


@pytest.mark.parametrize(
    ("space", "expected"),
    [("diag", True), ("param", False), ("row", False), (None, False)],
)
def test_planner_preconditioner_beta1_allocates_method_moment_only_in_diag_lane(space, expected):
    plan = _build(
        solver=DummySolver(space=space),
        precond=DummyPrecond(beta1=0.9),
    )
    assert plan.want.want_m is expected


def test_planner_gain_ratio_needs_gradient_only_in_parameter_lane():
    telemetry = Requirements(metrics=frozenset({mx.GAIN_RATIO}))
    param_plan = _build(solver=DummySolver(space="param"), telemetry=telemetry)
    row_plan = _build(solver=DummySolver(space="row"), telemetry=telemetry)

    assert param_plan.want.want_grad
    assert not row_plan.want.want_grad


def test_planner_want_loss_before_implied_by_gain_ratio_pack():
    plan = _build(
        damping=DummyDamping(every_k=2, keys=()),
        telemetry=Requirements(metrics=frozenset({mx.GAIN_RATIO})),
    )
    assert plan.want.want_loss_before
    assert mx.LOSS_BEFORE in set(plan.metric_keys)


def test_planner_cadence_invariants_for_unset():
    plan = _build(damping=DummyDamping(every_k=7, keys=()))
    assert not plan.want.want_gain_ratio_pack
    assert plan.req.gain_ratio_every_k < 1


def test_non_gain_ratio_damping_does_not_force_gain_ratio_cadence():
    step_norm_like = DummyDamping(every_k=5, keys={mx.STEP, mx.S_RMS})
    plan = _build(
        damping=step_norm_like,
        telemetry=Requirements(metrics=frozenset({mx.S_RMS})),
    )
    assert not plan.want.want_gain_ratio_pack
    assert plan.req.gain_ratio_every_k < 1


def test_planner_gain_ratio_cadence_uses_gcd_and_loss_covers_ratio_steps():
    telemetry = Requirements(metrics=frozenset({mx.GAIN_RATIO}), gain_ratio_every_k=6)
    plan = _build(
        damping=DummyDamping(
            every_k=10,
            keys={mx.STEP, mx.GAIN_RATIO, mx.GAIN_RATIO_VALID},
        ),
        telemetry=telemetry,
    )

    # Controller cadence 10 and telemetry cadence 6 merge to gcd(10, 6)=2.
    assert plan.want.want_gain_ratio_pack
    assert plan.req.gain_ratio_every_k == 2

    # Every ratio step must have a cached loss, possibly at a higher cadence.
    assert plan.req.loss_every_k > 0
    assert plan.req.gain_ratio_every_k % plan.req.loss_every_k == 0


def test_explicit_loss_cadence_merges_with_gain_ratio_cadence():
    telemetry = Requirements(
        metrics=frozenset({mx.GAIN_RATIO, mx.LOSS_BEFORE}),
        loss_every_k=6,
        gain_ratio_every_k=10,
    )
    plan = _build(telemetry=telemetry)

    assert plan.req.gain_ratio_every_k == 10
    assert plan.req.loss_every_k == 2


def test_planner_standalone_loss_defaults_to_every_step():
    plan = _build(
        damping=DummyDamping(every_k=10, keys=()),
        telemetry=Requirements(metrics=frozenset({mx.LOSS_BEFORE})),
    )

    assert plan.want.want_loss_before
    assert not plan.want.want_gain_ratio_pack
    assert plan.req.loss_every_k == 1
    assert plan.req.gain_ratio_every_k < 1
