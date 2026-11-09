from typing import Any, Optional, Literal

from ..presets import register
from ..assembler import assemble
from ..optax import build_optax_tx
from ..specs import CurvatureSpec, DampingSpec, SolverSpec, PrecondSpec, EstimatorSpec, TelemetrySpec
from ..types import PredictFnLike, ScalarOrSchedule

GramMode = Literal["none", "linear_head"]


@register("fgn_ce", desc="FGN (CE): FGNHA + Row-space CG.")
def fgn_ce(
        *,
        predict_fn: PredictFnLike,
        x_key="x",
        y_key="y",
        has_aux: bool = False,

        # damping params
        lam_policy: str = "const",
        lam0: float = 1.0,
        lam_kwargs: Optional[dict] = None,

        # CG params
        tol: float = 1e-5,
        maxiter: int = 10,
        warm_start: bool = False,  # Row-space warm starts are unsupported.
        stabilise_every: int = 0,
        lifted: bool = False,

        # Optional cached Gram for frozen-feature linear heads.
        gram_mode: GramMode = "none",
        linear_head_has_bias: bool = True,  # Used when gram_mode == "linear_head".

        # telemetry
        record_loss_every_k: int = -1,
        record_gain_ratio_every_k: int = -1,
        record_lam: bool = False,
        record_cg_stats: bool = True,

        # optax chain knobs
        learning_rate: ScalarOrSchedule = 1.0,
        clip_norm: float | None = None,
        weight_decay: float | None = None,
        weight_decay_mask: Any | None = None,
        direction_momentum: float | None = None,
        nesterov: bool = False,

        **kwargs: dict,
):
    curv = CurvatureSpec(
        "fgnha", {
            "predict_fn": predict_fn,
            "has_aux": has_aux,
            "x_key": x_key,
            "y_key": y_key,
            "reduction": "mean",
            "gram_mode": gram_mode,
            "linear_head_has_bias": linear_head_has_bias,
        })

    damp = DampingSpec(lam_policy, lam0=lam0, kwargs=(lam_kwargs or {}))

    solv = SolverSpec(
        "row_cg_lifted" if lifted else "row_cg", {
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
        record_gain_ratio=(record_gain_ratio_every_k > 0),
        gain_ratio_every_k=(
            record_gain_ratio_every_k if record_gain_ratio_every_k > 0 else None
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


@register("fgn_ce_exact", desc="FGN (CE): FGNHA + Row-space Cholesky.")
def fgn_ce_exact(
        *,
        predict_fn: PredictFnLike,
        x_key="x",
        y_key="y",
        has_aux: bool = False,

        # damping params
        lam_policy: str = "const",
        lam0: float = 1.0,
        lam_kwargs: Optional[dict] = None,

        # telemetry
        record_loss_every_k: int = -1,
        record_gain_ratio_every_k: int = -1,
        record_lam: bool = False,

        # optax chain knobs
        learning_rate: ScalarOrSchedule = 1.0,
        clip_norm: float | None = None,
        weight_decay: float | None = None,
        weight_decay_mask: Any | None = None,
        direction_momentum: float | None = None,
        nesterov: bool = False,

        **kwargs: dict,
):
    curv = CurvatureSpec(
        "fgnha", {
            "predict_fn": predict_fn,
            "has_aux": has_aux,
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
        record_gain_ratio=(record_gain_ratio_every_k > 0),
        gain_ratio_every_k=(
            record_gain_ratio_every_k if record_gain_ratio_every_k > 0 else None
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
