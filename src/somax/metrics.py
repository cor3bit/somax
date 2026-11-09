"""Metric keys and defaults.

All metrics are keyed by stable ASCII strings. In jitted code, the key set
(StepInfo schema) must be planner-time static.

Defaults:
- float metrics: NaN (float32)
- int metrics: -1 (int32)
- bool metrics: False (bool)
"""

from typing import Dict

import jax
import jax.numpy as jnp

# -------------------------
# Metric keys (stable strings)
# -------------------------

STEP = "step"

LAM_USED = "lam_used"
LAM_NEXT = "lam_next"
LAM_FACTOR = "lam_factor"
LAM_CLIPPED = "lam_clipped"

LOSS_BEFORE = "loss_before"
LOSS_AFTER = "loss_after"

G_DOT_S = "g_dot_s"
S_NORM2 = "s_norm2"
S_RMS = "s_rms"

PRED_DEC = "pred_dec"
ACT_DEC = "act_dec"
GAIN_RATIO = "gain_ratio"
GAIN_RATIO_VALID = "gain_ratio_valid"
# Diagnostic only: the candidate lowered the same-batch loss.  The executor
# always commits candidates; there is no accept/reject transaction.  Keep the
# legacy wire key for compatibility with existing logs.
STEP_IMPROVED = "step_accepted"
STEP_ACCEPTED = STEP_IMPROVED  # deprecated Python alias

TR_ACTION = "tr_action"  # int32: -1 dec, 0 keep/skip, +1 inc

DELTA_RMS = "delta_rms"

CG_ITERS = "cg_iters"
CG_MAXITER = "cg_maxiter"
CG_RESID = "cg_resid"
CG_CONVERGED = "cg_converged"

MU = "mu"  # row-space damping
LANE = "lane"  # 0=diag,1=param,2=row

# -------------------------
# Kind map and defaults
# -------------------------

KIND: Dict[str, str] = {
    # int
    STEP: "i32",
    TR_ACTION: "i32",
    CG_ITERS: "i32",
    CG_MAXITER: "i32",
    LANE: "i32",

    # bool
    GAIN_RATIO_VALID: "bool",
    STEP_IMPROVED: "bool",
    LAM_CLIPPED: "bool",
    CG_CONVERGED: "bool",
}


def default(key: str) -> jax.Array:
    kind = KIND.get(key, "f32")
    if kind == "bool":
        return jnp.asarray(False, jnp.bool_)
    if kind == "i32":
        return jnp.asarray(-1, jnp.int32)
    return jnp.asarray(jnp.nan, jnp.float32)
