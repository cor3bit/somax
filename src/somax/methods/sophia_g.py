from typing import Any, Optional

from ..presets import register
from ..assembler import assemble
from ..optax import sophia_tx
from ..specs import CurvatureSpec, DampingSpec, SolverSpec, PrecondSpec, EstimatorSpec, TelemetrySpec
from ..types import PredictFn, ScalarOrSchedule


@register("sophia_g", desc="Sophia-G: True Hessian + Diagonal EMA Preconditioner w/ GNB Estimator.")
def sophia_g(
        *,
        predict_fn: PredictFn,
        x_key="x",
        y_key="y",
        beta1: float = 0.965,
        beta2: float = 0.99,
        gamma: float = 0.05,
        eps: float = 1e-8,
        n_samples: int = 1,
        eval_every_k: int = 10,

        # telemetry
        record_loss_every_k: int = -1,

        # optax chain knobs
        learning_rate: ScalarOrSchedule = 1.0,
        weight_decay: float = 0.2,
        clip_value: float = 1.0,
        mask=None,

        # catch all
        **kwargs: dict,
):
    curv = CurvatureSpec(
        "ggn_ce", {
            "predict_fn": predict_fn,
            "x_key": x_key,
            "y_key": y_key,
            "reduction": "mean",
        },
    )

    solv = SolverSpec("identity", {})

    pre = PrecondSpec("diag_ema", {
        "beta1": beta1,  # consumed in diagonal lane as numerator EMA
        "beta2": beta2,  # EMA of diag(est)
        "gamma": gamma,  # enables Sophia-H denom
        "eps": eps,
        "eval_every_k": eval_every_k,
    })

    est = EstimatorSpec("gnb_ce", kwargs={
        "n_samples": n_samples,
    })

    # telemetry
    tlm = TelemetrySpec(
        record_loss_before=(record_loss_every_k > 0),
        loss_every_k=(
            record_loss_every_k if record_loss_every_k > 0 else None
        ),
    )

    # optax chain
    tx = sophia_tx(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clip_value=clip_value,
        mask=mask,
    )

    return assemble(curvature=curv, solver=solv, precond=pre, estimator=est, telemetry=tlm, tx=tx)
