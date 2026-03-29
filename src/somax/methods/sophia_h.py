from typing import Any, Optional

from ..presets import register
from ..assembler import assemble
from ..optax import sophia_tx
from ..specs import CurvatureSpec, DampingSpec, SolverSpec, PrecondSpec, EstimatorSpec, TelemetrySpec
from ..types import LossFn, ScalarOrSchedule


@register("sophia_h", desc="Sophia-H: True Hessian + Diagonal Hutchinson EMA Preconditioner w/ Hutchinson Estimator.")
def sophia_h(
        *,
        loss_fn: LossFn,
        x_key="x",
        y_key="y",
        beta1: float = 0.965,  # Note: slightly different default from paper (0.96)
        beta2: float = 0.99,
        gamma: float = 0.01,
        eps: float = 1e-8,
        n_probes: int = 1,
        use_abs: bool = False,  # Hutchinson usually unbiased -> False
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
        "hessian", kwargs={
            "loss_fn": loss_fn,
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

    est = EstimatorSpec("hutchinson", kwargs={
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
    tx = sophia_tx(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clip_value=clip_value,
        mask=mask,
    )

    return assemble(curvature=curv, solver=solv, precond=pre, estimator=est, telemetry=tlm, tx=tx)
