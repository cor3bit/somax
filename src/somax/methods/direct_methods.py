from typing import Any, Optional

from ..presets import register
from ..assembler import assemble
from ..optax import build_optax_tx
from ..specs import CurvatureSpec, DampingSpec, SolverSpec, PrecondSpec, EstimatorSpec, TelemetrySpec
from ..types import LossFn, PredictFn, ScalarOrSchedule


@register("newton_direct", desc="Regularized True Hessian (mean loss) + dense direct SPD solve (for tests).")
def newton_direct(
        *,
        loss_fn: LossFn,
        lam0: float = 1.0,

        # telemetry
        record_loss_every_k: int = -1,

        # optax chain knobs
        learning_rate: ScalarOrSchedule = 1.0,
        clip_norm: float | None = None,
        weight_decay: float | None = None,
        weight_decay_mask: Any | None = None,
        direction_momentum: float | None = None,
        nesterov: bool = False,

        # catch all
        **kwargs: dict,
):
    # optax chain
    tx = build_optax_tx(
        tx=None,
        learning_rate=learning_rate,
        clip_norm=clip_norm,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        direction_momentum=direction_momentum,
        nesterov=nesterov,
    )

    # telemetry
    tlm = TelemetrySpec(
        record_loss_before=(record_loss_every_k > 0),
        loss_every_k=(
            record_loss_every_k if record_loss_every_k > 0 else None
        ),
    )

    return assemble(
        curvature=CurvatureSpec(kind="hessian", kwargs=dict(loss_fn=loss_fn)),
        damping=DampingSpec(kind="const", lam0=lam0, kwargs={}),
        solver=SolverSpec(kind="direct", kwargs={}),
        telemetry=tlm,
        tx=tx,
    )


@register("ggn_direct_mse", desc="GGN Hessian (MSE) + dense direct SPD solve (for tests).")
def ggn_direct_mse(
        *,
        predict_fn: PredictFn,
        lam0: float = 1.0,

        # telemetry
        record_loss_every_k: int = -1,

        # optax chain knobs
        learning_rate: ScalarOrSchedule = 1.0,
        clip_norm: float | None = None,
        weight_decay: float | None = None,
        weight_decay_mask: Any | None = None,
        direction_momentum: float | None = None,
        nesterov: bool = False,

        # catch all
        **kwargs: dict,
):
    # optax chain
    tx = build_optax_tx(
        tx=None,
        learning_rate=learning_rate,
        clip_norm=clip_norm,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        direction_momentum=direction_momentum,
        nesterov=nesterov,
    )

    # telemetry
    tlm = TelemetrySpec(
        record_loss_before=(record_loss_every_k > 0),
        loss_every_k=(
            record_loss_every_k if record_loss_every_k > 0 else None
        ),
    )

    return assemble(
        curvature=CurvatureSpec(kind="ggn_mse", kwargs=dict(predict_fn=predict_fn)),
        damping=DampingSpec(kind="const", lam0=lam0, kwargs={}),
        solver=SolverSpec(kind="direct", kwargs={}),
        telemetry=tlm,
        tx=tx,
    )


@register("ggn_direct_ce", desc="GGN Hessian (CE) + dense direct SPD solve (for tests).")
def ggn_direct_ce(
        *,
        predict_fn: PredictFn,
        lam0: float = 1.0,

        # telemetry
        record_loss_every_k: int = -1,

        # optax chain knobs
        learning_rate: ScalarOrSchedule = 1.0,
        clip_norm: float | None = None,
        weight_decay: float | None = None,
        weight_decay_mask: Any | None = None,
        direction_momentum: float | None = None,
        nesterov: bool = False,

        # catch all
        **kwargs: dict,
):
    # optax chain
    tx = build_optax_tx(
        tx=None,
        learning_rate=learning_rate,
        clip_norm=clip_norm,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        direction_momentum=direction_momentum,
        nesterov=nesterov,
    )

    # telemetry
    tlm = TelemetrySpec(
        record_loss_before=(record_loss_every_k > 0),
        loss_every_k=(
            record_loss_every_k if record_loss_every_k > 0 else None
        ),
    )

    return assemble(
        curvature=CurvatureSpec(kind="ggn_ce", kwargs=dict(predict_fn=predict_fn)),
        damping=DampingSpec(kind="const", lam0=lam0, kwargs={}),
        solver=SolverSpec(kind="direct", kwargs={}),
        telemetry=tlm,
        tx=tx,
    )
