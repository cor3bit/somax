from typing import Any, Dict, Optional, Tuple
import collections

import jax
import jax.numpy as jnp
from jax import lax
from flax import struct

from .base import LinearSolver, NullSolverState
from ..types import MatVec, Updates, PrecondFn, Scalar
from ..utils import (tree_sub, tree_vdot, tree_norm2, tree_add_scaled,
                     maybe_cast_tree, tree_zeros_like, tree_dot_pair)
from .. import metrics as mx

try:
    import lineax as lx

    _HAS_LINEAX = True
except Exception:
    _HAS_LINEAX = False


@struct.dataclass
class CGState:
    # Always-present buffer to avoid Optional/None in traced pytrees.
    last_x: Updates


class ConjugateGradient(LinearSolver):
    """Fast CG/PCG with optional stabilization and warm start.

    Backends:
      - "pcg": internal JAX while_loop implementation (fast path)
      - "scipy": jax.scipy.sparse.linalg.cg reference
      - "lineax": lineax CG reference (if installed)

    Static (per-instance) flags:
      - preconditioned: chooses PCG vs CG core and enforces precond presence.
      - warm_start: uses CGState(last_x) as x0 and updates it after solve.
      - do_stabilise: enables periodic restart via exact residual recompute (pcg only).
    """

    space = "param"

    def __init__(
            self,
            *,
            maxiter: int = 50,
            tol: float = 1e-4,
            warm_start: bool = True,
            stabilise_every: int = 10,
            preconditioned: bool = False,
            assume_spd: bool = True,
            solve_dtype: Optional[jnp.dtype] = None,
            backend: str = "pcg",  # "pcg", "scipy", "lineax"
            telemetry_residual: bool = False,
    ):
        self._backend = str(backend)
        self._maxiter = int(maxiter)
        self._tol = float(tol)
        self._warm_start = bool(warm_start)
        self._preconditioned = bool(preconditioned)
        self._assume_spd = bool(assume_spd)
        self._solve_dtype = solve_dtype
        self._telemetry_residual = bool(telemetry_residual)

        se = int(stabilise_every)
        self._do_stabilise = 0 < se < self._maxiter
        self._stabilise_every = se

        # pcg emits a telemetry residual metric for free, scipy and lineax do not
        self._needs_telemetry_residual = False

        # based on the above flags, bind the appropriate solve() implementation to self._solve_fn
        if self._backend == "pcg":
            if self._preconditioned:
                core = _make_pcg_core(self._do_stabilise, self._stabilise_every)
            else:
                core = _make_cg_core(self._do_stabilise, self._stabilise_every)

            # Bind all hyperparams now, so solve() is tiny.
            self._solve_fn = lambda A_mv, rhs, precond, x0: core(
                A_mv, rhs, precond, x0, self._tol, self._maxiter,
            )

        elif self._backend == "scipy":
            self._needs_telemetry_residual = True
            self._solve_fn = lambda A_mv, rhs, precond, x0: _solve_jax_scipy(
                A_mv, rhs,
                precond=precond, x0=x0,
                tol=self._tol,
                maxiter=self._maxiter,
                solve_dtype=self._solve_dtype,
            )

        elif self._backend == "lineax":
            self._needs_telemetry_residual = True
            self._lx_cache = collections.OrderedDict()
            self._solve_fn = lambda A_mv, rhs, precond, x0: _solve_lineax(
                A_mv, rhs,
                precond=precond, x0=x0,
                tol=self._tol, maxiter=self._maxiter,
                solve_dtype=self._solve_dtype,
                stabilise_every=self._stabilise_every,
                assume_spd=self._assume_spd,
                _cache=self._lx_cache, _cache_cap=8,
            )
        else:
            raise ValueError(f"Unknown backend={backend}")

    def init(self, params: Updates) -> Any:
        if self._warm_start:
            return CGState(last_x=tree_zeros_like(params))
        return NullSolverState()

    def solve(
            self,
            A_mv: MatVec,
            rhs: Updates,
            *,
            state: Any,
            precond: Optional[PrecondFn] = None,
    ) -> Tuple[Updates, Dict[str, Any], Any]:
        # Enforce static has_precond axis across all backends (keeps experiments honest).
        if self._preconditioned and precond is None:
            raise ValueError("ConjugateGradient(preconditioned=True) requires precond != None.")
        if (not self._preconditioned) and precond is not None:
            raise ValueError("ConjugateGradient(preconditioned=False) requires precond=None.")

        x0: Optional[Updates] = None
        if self._warm_start:
            x0 = state.last_x

        x, info = self._solve_fn(A_mv, rhs, precond, x0)

        if self._telemetry_residual and self._needs_telemetry_residual:
            info[mx.CG_RESID] = _telemetry_residual(A_mv, x, rhs)

        new_state = CGState(last_x=x) if self._warm_start else state

        return x, info, new_state


# -----------------------
# PCG/CG core builders
# -----------------------

def _make_cg_core(do_stabilise: bool, stabilise_every: int):
    if not do_stabilise:
        return _cg_core_nostab
    se = jnp.asarray(stabilise_every, jnp.int32)
    return lambda A_mv, b, _precond, x0, tol, maxiter: _cg_core_stab(A_mv, b, x0, tol, maxiter, se)


def _make_pcg_core(do_stabilise: bool, stabilise_every: int):
    if not do_stabilise:
        return _pcg_core_nostab
    se = jnp.asarray(stabilise_every, jnp.int32)
    return lambda A_mv, b, precond, x0, tol, maxiter: _pcg_core_stab(A_mv, b, precond, x0, tol, maxiter, se)


# -----------------------
# CG (no preconditioner)
# -----------------------

def _cg_init(A_mv: MatVec, b: Updates, x0: Optional[Updates]) -> Tuple[Updates, Updates]:
    if x0 is None:
        x = tree_zeros_like(b)
        r = b
    else:
        x = x0
        r = tree_sub(b, A_mv(x))
    return x, r


def _cg_core_nostab(
        A_mv: MatVec,
        b: Updates,
        _precond: Any,
        x0: Optional[Updates],
        tol: float,
        maxiter: int,
) -> Tuple[Updates, Dict[str, Any]]:
    leaf_dtype = jnp.result_type(*jax.tree_util.tree_leaves(b))
    finfo = jnp.finfo(leaf_dtype)
    tiny, eps = finfo.tiny, finfo.eps

    b2 = jnp.maximum(tree_norm2(b), tiny)
    tol2 = (tol * tol) * b2

    x, r = _cg_init(A_mv, b, x0)
    p = r
    rTr = tree_vdot(r, r)

    k0 = jnp.asarray(0, jnp.int32)
    init = (x, r, p, rTr, k0)

    def cond(c):
        _x, _r, _p, rTr_c, k = c
        return jnp.logical_and(rTr_c > tol2, k < maxiter)

    def body(c):
        x, r, p, rTr, k = c
        Ap = A_mv(p)
        pAp = tree_vdot(p, Ap)
        alpha = rTr / jnp.maximum(pAp, eps)

        x_new = tree_add_scaled(x, alpha, p)
        r_new = tree_add_scaled(r, -alpha, Ap)

        rTr_new = tree_vdot(r_new, r_new)
        beta = rTr_new / jnp.maximum(rTr, eps)
        p_new = tree_add_scaled(r_new, beta, p)

        return (x_new, r_new, p_new, rTr_new, k + 1)

    x_fin, _r_fin, _p_fin, rTr_fin, k_fin = lax.while_loop(cond, body, init)

    info: Dict[str, jax.Array] = {
        mx.CG_ITERS: k_fin,
        mx.CG_MAXITER: jnp.asarray(maxiter, k_fin.dtype),
        mx.CG_RESID: jnp.sqrt(rTr_fin / b2),
        mx.CG_CONVERGED: rTr_fin <= tol2,
    }
    return x_fin, info


def _cg_core_stab(
        A_mv: MatVec,
        b: Updates,
        x0: Optional[Updates],
        tol: float,
        maxiter: int,
        se: jax.Array,
) -> Tuple[Updates, Dict[str, Any]]:
    leaf_dtype = jnp.result_type(*jax.tree_util.tree_leaves(b))
    finfo = jnp.finfo(leaf_dtype)
    tiny, eps = finfo.tiny, finfo.eps

    b2 = jnp.maximum(tree_norm2(b), tiny)
    tol2 = (tol * tol) * b2

    x, r = _cg_init(A_mv, b, x0)
    p = r
    rTr = tree_vdot(r, r)

    k0 = jnp.asarray(0, jnp.int32)
    t0 = se
    init = (x, r, p, rTr, k0, t0)

    def cond(c):
        _x, _r, _p, rTr_c, k, _t = c
        return jnp.logical_and(rTr_c > tol2, k < maxiter)

    def body(c):
        x, r, p, rTr, k, t = c
        Ap = A_mv(p)
        pAp = tree_vdot(p, Ap)
        alpha = rTr / jnp.maximum(pAp, eps)

        x_new = tree_add_scaled(x, alpha, p)
        r_tmp = tree_add_scaled(r, -alpha, Ap)

        k_next = k + 1
        t_next = t - 1
        do_replace = (t_next == 0)

        def _replace(_):
            r_exact = tree_sub(b, A_mv(x_new))
            rTr_exact = tree_vdot(r_exact, r_exact)
            p_new = r_exact
            return (r_exact, p_new, rTr_exact)

        def _cheap(_):
            r_new = r_tmp
            rTr_new = tree_vdot(r_new, r_new)
            beta = rTr_new / jnp.maximum(rTr, eps)
            p_new = tree_add_scaled(r_new, beta, p)
            return (r_new, p_new, rTr_new)

        r_new, p_new, rTr_new = lax.cond(do_replace, _replace, _cheap, operand=None)
        t_out = lax.select(do_replace, se, t_next)

        return (x_new, r_new, p_new, rTr_new, k_next, t_out)

    x_fin, _r_fin, _p_fin, rTr_fin, k_fin, _t_fin = lax.while_loop(cond, body, init)

    info: Dict[str, jax.Array] = {
        mx.CG_ITERS: k_fin,
        mx.CG_MAXITER: jnp.asarray(maxiter, k_fin.dtype),
        mx.CG_RESID: jnp.sqrt(rTr_fin / b2),
        mx.CG_CONVERGED: rTr_fin <= tol2,
    }
    return x_fin, info


# -----------------------
# PCG (preconditioned)
# -----------------------


def _pcg_core_nostab(
        A_mv: MatVec,
        b: Updates,
        precond: PrecondFn,
        x0: Optional[Updates],
        tol: float,
        maxiter: int,
) -> Tuple[Updates, Dict[str, Any]]:
    leaf_dtype = jnp.result_type(*jax.tree_util.tree_leaves(b))
    finfo = jnp.finfo(leaf_dtype)
    tiny, eps = finfo.tiny, finfo.eps

    b2 = jnp.maximum(tree_norm2(b), tiny)
    tol2 = (tol * tol) * b2

    x, r = _cg_init(A_mv, b, x0)
    z = precond(r)
    r2, rz = tree_dot_pair(r, z)
    p = z

    k0 = jnp.asarray(0, jnp.int32)
    init = (x, r, z, p, rz, r2, k0)

    def cond(c):
        _x, _r, _z, _p, _rz, r2_c, k = c
        return jnp.logical_and(r2_c > tol2, k < maxiter)

    def body(c):
        x, r, z, p, rz, r2, k = c
        Ap = A_mv(p)
        pAp = tree_vdot(p, Ap)
        alpha = rz / jnp.maximum(pAp, eps)

        x_new = tree_add_scaled(x, alpha, p)
        r_new = tree_add_scaled(r, -alpha, Ap)

        z_new = precond(r_new)
        r2_new, rz_new = tree_dot_pair(r_new, z_new)

        beta = rz_new / jnp.maximum(rz, eps)
        p_new = tree_add_scaled(z_new, beta, p)

        return (x_new, r_new, z_new, p_new, rz_new, r2_new, k + 1)

    x_fin, _r_fin, _z_fin, _p_fin, _rz_fin, r2_fin, k_fin = lax.while_loop(cond, body, init)

    info: Dict[str, jax.Array] = {
        mx.CG_ITERS: k_fin,
        mx.CG_MAXITER: jnp.asarray(maxiter, k_fin.dtype),
        mx.CG_RESID: jnp.sqrt(r2_fin / b2),
        mx.CG_CONVERGED: r2_fin <= tol2,
    }
    return x_fin, info


def _pcg_core_stab(
        A_mv: MatVec,
        b: Updates,
        precond: PrecondFn,
        x0: Optional[Updates],
        tol: float,
        maxiter: int,
        se: jax.Array,
) -> Tuple[Updates, Dict[str, Any]]:
    leaf_dtype = jnp.result_type(*jax.tree_util.tree_leaves(b))
    finfo = jnp.finfo(leaf_dtype)
    tiny, eps = finfo.tiny, finfo.eps

    b2 = jnp.maximum(tree_norm2(b), tiny)
    tol2 = (tol * tol) * b2

    x, r = _cg_init(A_mv, b, x0)
    z = precond(r)
    r2, rz = tree_dot_pair(r, z)
    p = z

    k0 = jnp.asarray(0, jnp.int32)
    t0 = se
    init = (x, r, z, p, rz, r2, k0, t0)

    def cond(c):
        _x, _r, _z, _p, _rz, r2_c, k, _t = c
        return jnp.logical_and(r2_c > tol2, k < maxiter)

    def body(c):
        x, r, z, p, rz, r2, k, t = c
        Ap = A_mv(p)
        pAp = tree_vdot(p, Ap)
        alpha = rz / jnp.maximum(pAp, eps)

        x_new = tree_add_scaled(x, alpha, p)
        r_tmp = tree_add_scaled(r, -alpha, Ap)

        k_next = k + 1
        t_next = t - 1
        do_replace = (t_next == 0)

        def _replace(_):
            r_exact = tree_sub(b, A_mv(x_new))
            z_exact = precond(r_exact)
            r2_exact, rz_exact = tree_dot_pair(r_exact, z_exact)
            p_new = z_exact
            return (r_exact, z_exact, p_new, rz_exact, r2_exact)

        def _cheap(_):
            r_new = r_tmp
            z_new = precond(r_new)
            r2_new, rz_new = tree_dot_pair(r_new, z_new)
            beta = rz_new / jnp.maximum(rz, eps)
            p_new = tree_add_scaled(z_new, beta, p)
            return (r_new, z_new, p_new, rz_new, r2_new)

        r_new, z_new, p_new, rz_new, r2_new = lax.cond(do_replace, _replace, _cheap, operand=None)
        t_out = lax.select(do_replace, se, t_next)

        return (x_new, r_new, z_new, p_new, rz_new, r2_new, k_next, t_out)

    x_fin, _r_fin, _z_fin, _p_fin, _rz_fin, r2_fin, k_fin, _t_fin = lax.while_loop(cond, body, init)

    info: Dict[str, jax.Array] = {
        mx.CG_ITERS: k_fin,
        mx.CG_MAXITER: jnp.asarray(maxiter, k_fin.dtype),
        mx.CG_RESID: jnp.sqrt(r2_fin / b2),
        mx.CG_CONVERGED: r2_fin <= tol2,
    }
    return x_fin, info


# -----------------------
# Reference backends
# -----------------------


def _telemetry_residual(A_mv: MatVec, sol: Updates, rhs: Updates) -> Scalar:
    r = jax.tree_util.tree_map(lambda Ax, bb: Ax - bb, A_mv(sol), rhs)
    b2 = tree_norm2(rhs)
    leaf_dtype = jnp.result_type(*jax.tree_util.tree_leaves(rhs))
    tiny = jnp.asarray(jnp.finfo(leaf_dtype).tiny, leaf_dtype)
    return jnp.sqrt(tree_norm2(r) / jnp.maximum(b2, tiny))


def _solve_jax_scipy(
        A_mv: MatVec,
        rhs: Updates,
        *,
        precond: Optional[PrecondFn],
        x0: Optional[Updates],
        tol: float,
        maxiter: int,
        solve_dtype: Optional[jnp.dtype],
) -> Tuple[Updates, Dict[str, Any]]:
    # jax.scipy.sparse.linalg.cg supports M=preconditioner and x0.
    A0 = A_mv
    b_ref = rhs

    if solve_dtype is not None:

        def A_s(v):
            return A0(maybe_cast_tree(v, solve_dtype))

        if precond is None:
            M_s = None
        else:
            P0 = precond

            def M_s(v):
                return P0(maybe_cast_tree(v, solve_dtype))

        b = maybe_cast_tree(rhs, solve_dtype)
        x0_s = None if x0 is None else maybe_cast_tree(x0, solve_dtype)
    else:
        A_s = A_mv
        M_s = precond
        b = rhs
        x0_s = x0

    x_s, _info = jax.scipy.sparse.linalg.cg(A=A_s, b=b, x0=x0_s, tol=tol, maxiter=maxiter, M=M_s)
    del _info

    if solve_dtype is not None:
        x = jax.tree_util.tree_map(lambda a, ref: a.astype(ref.dtype), x_s, b_ref)
    else:
        x = x_s

    info: Dict[str, Any] = {mx.CG_MAXITER: maxiter}
    return x, info


def _solve_lineax(
        A_mv: MatVec,
        rhs: Updates,
        *,
        precond: Optional[PrecondFn],
        x0: Optional[Updates],
        tol: float,
        maxiter: int,
        solve_dtype: Optional[jnp.dtype],
        stabilise_every: int,
        assume_spd: bool,
        _cache: "collections.OrderedDict[Any, Any]",
        _cache_cap: int,
) -> Tuple[Updates, Dict[str, Any]]:
    if not _HAS_LINEAX:
        raise RuntimeError("backend='lineax' requested but lineax is not installed.")

    A0 = A_mv
    b_ref = rhs

    if solve_dtype is not None:
        A_mv = lambda v: A0(maybe_cast_tree(v, solve_dtype))
        b = maybe_cast_tree(rhs, solve_dtype)
        x0 = None if x0 is None else maybe_cast_tree(x0, solve_dtype)
        if precond is not None:
            P0 = precond
            precond = lambda v: P0(maybe_cast_tree(v, solve_dtype))
    else:
        b = rhs

    # Cache linear operators by (A identity, precond identity, treedef, dtype, SPD tag).
    treedef = jax.tree_util.tree_structure(b)
    rhs_dtype = jnp.result_type(*jax.tree_util.tree_leaves(b))
    key = ("lineax", id(A0), id(precond) if precond is not None else 0, treedef, rhs_dtype, assume_spd)

    entry = _cache.get(key)
    if entry is None:
        tags = (lx.symmetric_tag, lx.positive_semidefinite_tag) if assume_spd else ()
        A_op = lx.FunctionLinearOperator(A_mv, b, tags=tags)
        M_op = None if precond is None else lx.FunctionLinearOperator(precond, b, tags=(lx.positive_semidefinite_tag,))
        _cache[key] = (A_op, M_op)
        while len(_cache) > _cache_cap:
            _cache.popitem(last=False)
    else:
        A_op, M_op = entry

    # lineax CG supports stabilise_every and max_steps.
    solver = lx.CG(rtol=tol, atol=0.0, stabilise_every=stabilise_every, max_steps=maxiter)

    options: Dict[str, Any] = {}
    if M_op is not None:
        options["preconditioners"] = M_op
    if x0 is not None:
        options["y0"] = x0

    st = solver.init(A_op, options)
    x, _result, _stats = solver.compute(st, b, options)
    del _result, _stats

    if solve_dtype is not None:
        x = jax.tree_util.tree_map(lambda a, ref: a.astype(ref.dtype), x, b_ref)

    info: Dict[str, Any] = {mx.CG_MAXITER: maxiter}

    return x, info
