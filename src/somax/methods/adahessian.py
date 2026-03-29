from typing import Any, Optional

from ..presets import register
from ..assembler import assemble
from ..optax import build_optax_tx
from ..specs import CurvatureSpec, DampingSpec, SolverSpec, PrecondSpec, EstimatorSpec, TelemetrySpec
from ..types import LossFn, ScalarOrSchedule


@register("adahessian", desc="AdaHessian: True Hessian + Diagonal EMA Preconditioner w/ Hutchinson Estimator.")
def adahessian(
        *,
        loss_fn: LossFn,
        x_key="x",
        y_key="y",
        beta1: float = 0.9,
        beta2: float = 0.999,
        hessian_power: float = 1.0,
        eps: float = 1e-8,
        n_probes: int = 1,
        use_abs: bool = False,
        spatial_averaging: bool = True,
        eval_every_k: int = 1,

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
    curv = CurvatureSpec(
        "hessian", {
            "loss_fn": loss_fn,
            "x_key": x_key,
            "y_key": y_key,
            "reduction": "mean",
        },
    )

    solv = SolverSpec("identity", {})

    pre = PrecondSpec("diag_ema", {
        "beta1": beta1,
        "beta2": beta2,
        "hessian_power": hessian_power,
        "eps": eps,
        "spatial_averaging": spatial_averaging,
        "eval_every_k": eval_every_k,
    })

    est = EstimatorSpec("hutchinson", {
        "n_probes": n_probes,
        "use_abs": use_abs,
    })

    # telemetry
    tlm = TelemetrySpec(
        record_loss_before=(record_loss_every_k > 0),
        loss_every_k=(
            record_loss_every_k if record_loss_every_k > 0 else None
        ),
    )

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

    return assemble(curvature=curv, solver=solv, precond=pre, estimator=est, telemetry=tlm, tx=tx)
