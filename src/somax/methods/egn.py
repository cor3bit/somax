from typing import Any, Optional

from ..presets import register
from ..assembler import assemble
from ..optax import build_optax_tx
from ..specs import CurvatureSpec, DampingSpec, SolverSpec, PrecondSpec, EstimatorSpec, TelemetrySpec
from ..types import PredictFn, ScalarOrSchedule


@register(
    "egn_mse",
    desc="EGN (MSE): GGN + Exact solver (Cholesky in observation space)."
)
def egn_mse(
        *,
        predict_fn: PredictFn,
        x_key: str = "x",
        y_key: str = "y",

        # damping params
        lam_policy: str = "const",
        lam0: float = 1.0,
        lam_kwargs: Optional[dict] = None,

        # telemetry
        record_loss_every_k: int = -1,
        record_rho_every_k: int = -1,
        record_lam: bool = False,

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
        "ggn_mse", {
            "predict_fn": predict_fn,
            "x_key": x_key,
            "y_key": y_key,
            "reduction": "mean"},
    )

    damp = DampingSpec(lam_policy, lam0=lam0, kwargs=(lam_kwargs or {}))

    solv = SolverSpec("row_cholesky", {})

    # telemetry
    tlm = TelemetrySpec(
        record_loss_before=(record_loss_every_k > 0),
        loss_every_k=(
            record_loss_every_k if record_loss_every_k > 0 else None
        ),
        record_rho=(record_rho_every_k > 0),
        rho_every_k=(
            record_rho_every_k if record_rho_every_k > 0 else None
        ),
        record_lam=record_lam,
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

    return assemble(curvature=curv, damping=damp, solver=solv, telemetry=tlm, tx=tx)


@register(
    "egn_mse_cg",
    desc="EGN (MSE): GGN + Approximate solver (CG in observation space)."
)
def egn_mse_cg(
        *,
        predict_fn: PredictFn,
        x_key: str = "x",
        y_key: str = "y",

        # damping params
        lam_policy: str = "const",
        lam0: float = 1.0,
        lam_kwargs: Optional[dict] = None,

        # CG params
        tol: float = 1e-5,
        maxiter: int = 20,
        warm_start: bool = True,
        stabilise_every: int = 10,
        record_cg_stats: bool = True,

        # telemetry
        record_loss_every_k: int = -1,
        record_rho_every_k: int = -1,
        record_lam: bool = False,

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
        "ggn_mse", {
            "predict_fn": predict_fn,
            "x_key": x_key,
            "y_key": y_key,
            "reduction": "mean"},
    )

    damp = DampingSpec(lam_policy, lam0=lam0, kwargs=(lam_kwargs or {}))

    solv = SolverSpec(
        "row_cg", {
            "tol": tol,
            "maxiter": maxiter,
            "warm_start": warm_start,
            "stabilise_every": stabilise_every,
            "backend": "pcg",
        },
    )

    # telemetry
    tlm = TelemetrySpec(
        record_loss_before=(record_loss_every_k > 0),
        loss_every_k=(
            record_loss_every_k if record_loss_every_k > 0 else None
        ),
        record_rho=(record_rho_every_k > 0),
        rho_every_k=(
            record_rho_every_k if record_rho_every_k > 0 else None
        ),
        record_lam=record_lam,
        record_cg=record_cg_stats,
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

    return assemble(curvature=curv, damping=damp, solver=solv, telemetry=tlm, tx=tx)


@register(
    "egn_ce",
    desc="EGN (CE): GGN + Exact solver (Cholesky in observation space))."
)
def egn_ce(
        *,
        predict_fn: PredictFn,
        x_key: str = "x",
        y_key: str = "y",

        # damping params
        lam_policy: str = "const",
        lam0: float = 1.0,
        lam_kwargs: Optional[dict] = None,

        # telemetry
        record_loss_every_k: int = -1,
        record_rho_every_k: int = -1,
        record_lam: bool = False,

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
        "ggn_ce", {
            "predict_fn": predict_fn,
            "x_key": x_key,
            "y_key": y_key,
            "reduction": "mean",
        })

    damp = DampingSpec(lam_policy, lam0=lam0, kwargs=(lam_kwargs or {}))

    solv = SolverSpec("row_cholesky", {})

    # telemetry
    tlm = TelemetrySpec(
        record_loss_before=(record_loss_every_k > 0),
        loss_every_k=(
            record_loss_every_k if record_loss_every_k > 0 else None
        ),
        record_rho=(record_rho_every_k > 0),
        rho_every_k=(
            record_rho_every_k if record_rho_every_k > 0 else None
        ),
        record_lam=record_lam,
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

    return assemble(curvature=curv, damping=damp, solver=solv, telemetry=tlm, tx=tx)
