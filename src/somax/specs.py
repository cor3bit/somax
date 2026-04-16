from dataclasses import dataclass
from typing import Any, Optional

from .planner import Requirements
from . import metrics as mx
from .preconditioners.base import PreconditionerPolicy
from .preconditioners.identity import IdentityPrecond


@dataclass(frozen=True)
class CurvatureSpec:
    kind: str
    kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class EstimatorSpec:
    kind: str
    kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class DampingSpec:
    kind: str
    lam0: float
    kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class SolverSpec:
    kind: str
    kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class PrecondSpec:
    kind: str
    kwargs: dict[str, Any] | None = None


# IDEA: can force calculation of certain additional metrics during the step
# e.g., train set loss every k, gradient norm, etc.
# None = unspecified (defer to planner); -1 is an internal sentinel and
# must not appear in user-facing TelemetrySpec.
@dataclass(frozen=True)
class TelemetrySpec:
    record_step: bool = False
    record_lam: bool = False

    record_loss_before: bool = False
    loss_every_k: int | None = None

    record_rho: bool = False
    rho_every_k: int | None = None

    record_delta_rms: bool = False
    record_cg: bool = False


def build_curvature(spec: CurvatureSpec, estimator: Optional[EstimatorSpec] = None) -> Any:
    k = spec.kind.lower()
    kw = dict(spec.kwargs or {})

    if k == "ggn_ce":
        from .curvature.ggn_ce import GGNCE

        op = GGNCE(**kw)
    elif k == "ggn_mse":
        from .curvature.ggn_mse import GGNMSE

        op = GGNMSE(**kw)
    elif k in ("hessian", "exact_hessian"):
        from .curvature.hessian import ExactHessian

        op = ExactHessian(**kw)
    else:
        raise ValueError(f"Unknown curvature kind: {spec.kind}")

    # Wrap with estimator if specified
    if estimator is not None:
        from .estimators import make_estimator
        from .curvature.with_estimators import CurvatureOpWithEstimators

        est = make_estimator(estimator.kind, **(estimator.kwargs or {}))
        op = CurvatureOpWithEstimators(op, est)

    # Enforce the post_step invariant: loss_only must exist.
    if not hasattr(op, "loss_only"):
        raise ValueError("CurvatureOperator must implement loss_only(params, batch).")
    return op


def build_damping(spec: Optional[DampingSpec]) -> Any:
    if spec is None:
        from .damping.constant import ConstantDamping

        return ConstantDamping(lam0=0.0)

    k = spec.kind.lower()
    kw = dict(spec.kwargs or {})
    lam0 = float(spec.lam0)

    if k in ("const", "constant", "fixed"):
        from .damping.constant import ConstantDamping

        return ConstantDamping(lam0=lam0, **kw)
    if k in ("trust_region", "tr", "lm", "levenberg_marquardt"):
        from .damping.trust_region import TrustRegionDamping

        return TrustRegionDamping(lam0=lam0, **kw)

    raise ValueError(f"Unknown damping kind: {spec.kind}")


def build_solver(spec: SolverSpec, precond: PreconditionerPolicy) -> Any:
    k = spec.kind.lower()
    kw = dict(spec.kwargs or {})

    if k in ("cg", "param_cg", "newton_cg"):
        from .solvers.cg import ConjugateGradient

        # Compile the correct solver variant at assembly time.
        kw["preconditioned"] = not precond.is_identity
        return ConjugateGradient(**kw)

    if k in ("row_cg", "cg_row"):
        from .solvers.row_cg import RowConjugateGradient

        # Row CG currently does not support preconditioning.
        if not precond.is_identity:
            raise ValueError(
                "RowConjugateGradient does not support preconditioning. "
                "Use IdentityPrecond (or omit preconditioner) for row-space solves, "
                "or switch to param-space CG if you need preconditioning."
            )

        kw["preconditioned"] = False
        return RowConjugateGradient(**kw)

    if k in ("row_cholesky", "cholesky_row", "row_direct"):
        from .solvers.row_cholesky import RowCholesky

        return RowCholesky(**kw)

    if k in ("direct", "solve"):
        from .solvers.direct import DirectSPD

        return DirectSPD()

    if k in ("identity", "diag", "none"):
        from .solvers.identity import IdentitySolve

        return IdentitySolve()

    raise ValueError(f"Unknown solver kind: {spec.kind}")


def build_precond(spec: Optional[PrecondSpec]) -> Any | None:
    if spec is None:
        from .preconditioners.identity import IdentityPrecond

        return IdentityPrecond()

    k = spec.kind.lower()
    kw = dict(spec.kwargs or {})

    if k in ("diag_direct", "direct"):
        from .preconditioners.diag_direct import DiagDirectPrecond
        return DiagDirectPrecond(**kw)

    if k in ("diag_ema", "ema"):
        from .preconditioners.diag_ema import DiagEMAPrecond

        return DiagEMAPrecond(**kw)

    if k in ("none", "identity"):
        from .preconditioners.identity import IdentityPrecond

        return IdentityPrecond()

    raise ValueError(f"Unknown preconditioner kind: {spec.kind}")


def build_telemetry_reqs(spec: TelemetrySpec) -> Requirements:
    """Return planner Requirements derived from telemetry spec.

    - metrics: concrete metric keys to include in StepInfo schema
    - loss_every_k: cadence for loss_before
    - rho_every_k: cadence for rho/loss_after/pred/act-dec pack
    """
    keys: set[str] = set()

    # Optional: allow power users to force additional keys (string list)
    extra = getattr(spec, "extra_metrics", None)
    if extra is not None:
        keys |= set(extra)

    if spec.record_loss_before:
        keys.add(mx.LOSS_BEFORE)

    if spec.record_rho:
        keys |= {
            mx.LOSS_AFTER,
            mx.PRED_DEC,
            mx.ACT_DEC,
            mx.RHO,
            mx.RHO_VALID,
            mx.STEP_ACCEPTED,
        }

    if spec.record_cg:
        keys |= {
            mx.CG_ITERS,
            mx.CG_RESID,
            mx.CG_MAXITER,
            mx.CG_CONVERGED,
        }

    if spec.record_delta_rms:
        keys.add(mx.DELTA_RMS)

    rho_every_k = getattr(spec, "rho_every_k", None)
    if spec.record_rho:
        if rho_every_k is None:
            rho_every_k_req = 0
        else:
            rho_every_k_req = int(rho_every_k)
            assert rho_every_k_req > 0
    else:
        rho_every_k_req = 0

    loss_every_k = getattr(spec, "loss_every_k", None)
    if getattr(spec, "record_loss_before", False) or getattr(spec, "record_loss_after", False):
        if loss_every_k is None:
            loss_every_k_req = 0
        else:
            loss_every_k_req = int(loss_every_k)
            assert loss_every_k_req > 0
    else:
        loss_every_k_req = 0

    return Requirements(
        metrics=frozenset(keys),
        loss_every_k=loss_every_k_req,
        rho_every_k=rho_every_k_req,
    )
