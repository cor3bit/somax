import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from somax.solvers.direct import DirectSPD
from somax.solvers.row_common import bind_row_system, mu_from_lambda


def _rel_resid_array(A_mv, x, b):
    Ax = A_mv(x)
    num = jnp.linalg.norm(Ax - b)
    den = jnp.maximum(jnp.linalg.norm(b), jnp.asarray(1e-12, b.dtype))
    return num / den


def _rel_resid_pytree(A_mv, x, b):
    Ax = A_mv(x)
    Ax_f, _ = ravel_pytree(Ax)
    b_f, _ = ravel_pytree(b)
    num = jnp.linalg.norm(Ax_f - b_f)
    den = jnp.maximum(jnp.linalg.norm(b_f), jnp.asarray(1e-12, b_f.dtype))
    return num / den


def _rel_fro(A, B):
    num = jnp.linalg.norm(A - B)
    den = jnp.maximum(jnp.linalg.norm(B), jnp.asarray(1e-12, B.dtype))
    return num / den


def _dense_from_array_matvec(A_mv, n, dtype):
    eye = jnp.eye(n, dtype=dtype)
    return jax.vmap(A_mv, in_axes=1, out_axes=1)(eye)


def _effective_dense_matrix(A_mv, n, dtype, *, symmetrize, jitter, solve_dtype):
    A = _dense_from_array_matvec(A_mv, n, dtype)

    if symmetrize:
        A = 0.5 * (A + A.T)

    if solve_dtype is not None:
        A = A.astype(solve_dtype)

    if jitter > 0.0:
        A = A + jnp.asarray(jitter, dtype=A.dtype) * jnp.eye(n, dtype=A.dtype)

    return A


def _make_full_rank_J(m: int, n: int, key, dtype=jnp.float32):
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (m, m), dtype=dtype)
    Q, _ = jnp.linalg.qr(A)
    B = jax.random.normal(k2, (n, n), dtype=dtype)
    R, _ = jnp.linalg.qr(B)
    D = jnp.linspace(0.5, 2.0, min(m, n), dtype=dtype)
    return (Q[:, :n] * D) @ R.T


def _normal_eq_solution(J: jax.Array, rhs_row: jax.Array, mu: float):
    n = J.shape[1]
    H = J.T @ J + mu * jnp.eye(n, dtype=J.dtype)
    g = J.T @ rhs_row
    return jnp.linalg.solve(H, g)


def _row_value_rtol(tol):
    return max(float(tol["rtol"]), 5e-3)


def _row_value_atol(tol):
    return max(float(tol["atol"]), 5e-4)


class _LinearRowOp:
    """Duck-typed row operator for production bind_row_system tests."""

    def __init__(self, J, rhs, b):
        self.J = J
        self.rhs = rhs
        self.b = int(b)

    def jvp(self, p):
        return self.J @ p

    def vjp(self, u):
        return self.J.T @ u


@pytest.mark.parametrize("solve_method", ["scipy", "chol"])
@pytest.mark.parametrize("symmetrize", [False, True])
def test_direct_spd_param_pytree_matches_dense_solve(
        key,
        make_spd,
        mv_from_dense,
        make_block_pytree,
        tol,
        solve_method,
        symmetrize,
):
    template, pack, _zeros_like = make_block_pytree(shape_a=(4,), shape_b=(3,))
    flat0, _ = ravel_pytree(template)
    n = int(flat0.shape[0])

    kb, kA = jax.random.split(key)
    b_flat = jax.random.normal(kb, (n,), dtype=jnp.float32)
    b = pack(b_flat)

    A = make_spd(n=n, key=kA, cond=20.0, eps=1e-4)
    A_mv = mv_from_dense(A, pack=pack)

    solver = DirectSPD(
        space="param",
        vector_space="pytree",
        solve_method=solve_method,
        symmetrize=symmetrize,
        jitter=0.0,
        solve_dtype=None,
    )

    state_in = object()
    x, info, state_out = solver.solve(A_mv, b, state=state_in, precond=None)

    x_flat, _ = ravel_pytree(x)
    x_ref = jnp.linalg.solve(A, b_flat)
    rel = _rel_resid_pytree(A_mv, x, b)

    assert state_out is state_in
    assert info["space"] == "param"
    assert info["vector_space"] == "pytree"
    assert info["solve_method"] == solve_method
    assert int(info["dim"]) == n
    assert int(info["n"]) == n

    np.testing.assert_allclose(
        np.asarray(x_flat),
        np.asarray(x_ref),
        rtol=tol["rtol"],
        atol=tol["atol"],
    )
    assert float(rel) < 1e-5


@pytest.mark.parametrize("solve_method", ["scipy", "chol"])
def test_direct_spd_row_array_diagonal_operator_matches_analytic_solution(
        solve_method,
):
    diag = jnp.asarray([2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
    b = jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float32)

    def A_mv(u):
        return diag * u

    A_dense = _dense_from_array_matvec(A_mv, diag.shape[0], b.dtype)
    A_ref = jnp.diag(diag)

    # Materializing this particular diagonal operator should be exact.
    np.testing.assert_allclose(
        np.asarray(A_dense),
        np.asarray(A_ref),
        rtol=0.0,
        atol=0.0,
    )

    solver = DirectSPD(
        space="row",
        vector_space="array",
        solve_method=solve_method,
        symmetrize=False,
        jitter=0.0,
        solve_dtype=None,
    )
    u, info, _ = solver.solve(A_mv, b, state=None, precond=None)
    u_ref = b / diag

    eps = np.finfo(np.asarray(u).dtype).eps

    np.testing.assert_allclose(
        np.asarray(u),
        np.asarray(u_ref),
        rtol=16.0 * eps,
        atol=16.0 * eps,
    )
    assert float(_rel_resid_array(A_mv, u, b)) <= 32.0 * eps

    assert info["space"] == "row"
    assert info["vector_space"] == "array"
    assert info["solve_method"] == solve_method


@pytest.mark.parametrize("solve_method", ["scipy", "chol"])
@pytest.mark.parametrize("symmetrize", [False, True])
def test_direct_spd_row_array_matches_effective_materialized_matrix(
        key,
        make_spd,
        tol,
        solve_method,
        symmetrize,
):
    n = 12
    kb, kA = jax.random.split(key)
    b = jax.random.normal(kb, (n,), dtype=jnp.float32)
    A = make_spd(n=n, key=kA, cond=30.0, eps=1e-4)

    def A_mv(u):
        return A @ u

    A_dense = _dense_from_array_matvec(A_mv, n, b.dtype)

    # Reconstruction from a matmul-backed A_mv is approximate in float32.
    assert float(_rel_fro(A_dense, A)) < 1e-3

    A_eff = _effective_dense_matrix(
        A_mv,
        n,
        b.dtype,
        symmetrize=symmetrize,
        jitter=0.0,
        solve_dtype=None,
    )

    solver = DirectSPD(
        space="row",
        vector_space="array",
        solve_method=solve_method,
        symmetrize=symmetrize,
        jitter=0.0,
        solve_dtype=None,
    )

    state_in = object()
    u, info, state_out = solver.solve(A_mv, b, state=state_in, precond=None)
    u_ref = jnp.linalg.solve(A_eff, b)
    rel_eff = _rel_resid_array(lambda x: A_eff @ x, u, b)

    assert state_out is state_in
    assert info["space"] == "row"
    assert info["vector_space"] == "array"
    assert info["solve_method"] == solve_method
    assert int(info["dim"]) == n
    assert int(info["row_dim"]) == n

    np.testing.assert_allclose(
        np.asarray(u),
        np.asarray(u_ref),
        rtol=_row_value_rtol(tol),
        atol=_row_value_atol(tol),
    )
    assert float(rel_eff) < 5e-5


@pytest.mark.parametrize("reduction,expected_scale", [("mean", 8.0), ("sum", 1.0)])
def test_bind_row_system_production_contract_matches_dense_row_matrix(key, reduction, expected_scale, tol):
    m, n = 16, 10
    kJ, krhs = jax.random.split(key)
    J = _make_full_rank_J(m, n, kJ, dtype=jnp.float32)
    rhs_row = jax.random.normal(krhs, (m,), dtype=jnp.float32)

    lam = jnp.asarray(0.25, dtype=jnp.float32)
    bsz = 8

    row_op = _LinearRowOp(J=J, rhs=rhs_row, b=bsz)
    A_mv, rhs, backproject, mu = bind_row_system(row_op, lam, reduction=reduction)

    A_dense = _dense_from_array_matvec(A_mv, m, rhs.dtype)
    A_ref = J @ J.T + mu * jnp.eye(m, dtype=J.dtype)

    np.testing.assert_allclose(
        np.asarray(rhs),
        np.asarray(rhs_row),
        rtol=tol["rtol"],
        atol=tol["atol"],
    )
    np.testing.assert_allclose(
        np.asarray(mu),
        np.asarray(expected_scale * lam),
        rtol=tol["rtol"],
        atol=tol["atol"],
    )

    # Production binder contract: A_mv is close to JJ^T + mu I in float32.
    assert float(_rel_fro(A_dense, A_ref)) < 2e-3

    u_probe = jax.random.normal(jax.random.PRNGKey(7), (m,), dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(backproject(u_probe)),
        np.asarray(J.T @ u_probe),
        rtol=tol["rtol"],
        atol=tol["atol"],
    )


@pytest.mark.parametrize("solve_method", ["scipy", "chol"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_direct_spd_row_production_binder_backprojects_to_param_normal_equation_solution(
        key,
        tol,
        solve_method,
        reduction,
):
    m, n = 16, 10
    kJ, krhs = jax.random.split(key)
    J = _make_full_rank_J(m, n, kJ, dtype=jnp.float32)
    rhs_row = jax.random.normal(krhs, (m,), dtype=jnp.float32)

    # Use stronger damping to make the float32 row solve less brittle.
    lam = jnp.asarray(0.25, dtype=jnp.float32)
    bsz = 8

    row_op = _LinearRowOp(J=J, rhs=rhs_row, b=bsz)
    A_mv, rhs, backproject, mu = bind_row_system(row_op, lam, reduction=reduction)

    solver = DirectSPD(
        space="row",
        vector_space="array",
        solve_method=solve_method,
        symmetrize=True,
        jitter=1e-6,
        solve_dtype=None,
    )

    u, info, st = solver.solve(A_mv, rhs, state=None, precond=None)
    assert st is None

    # Gross row-space failure check only.
    row_rel = _rel_resid_array(A_mv, u, rhs)
    assert float(row_rel) < 1e-2

    # Main invariant: backprojected parameter-space direction solves the damped normal equations.
    s = backproject(u)
    s_ref = _normal_eq_solution(J, rhs_row, float(mu))

    np.testing.assert_allclose(
        np.asarray(s),
        np.asarray(s_ref),
        rtol=_row_value_rtol(tol),
        atol=_row_value_atol(tol),
    )

    assert info["space"] == "row"
    assert info["vector_space"] == "array"
    assert info["solve_method"] == solve_method
    assert int(info["row_dim"]) == m


def test_direct_spd_symmetrize_solves_effective_symmetric_materialized_matrix(key, tol):
    n = 8
    kS, kN, kb = jax.random.split(key, 3)

    M = jax.random.normal(kS, (n, n), dtype=jnp.float32)
    S = M.T @ M + jnp.asarray(1e-2, jnp.float32) * jnp.eye(n, dtype=jnp.float32)

    N = jax.random.normal(kN, (n, n), dtype=jnp.float32)
    skew = N - N.T
    A_raw = S + jnp.asarray(5e-3, jnp.float32) * skew
    b = jax.random.normal(kb, (n,), dtype=jnp.float32)

    def A_mv(u):
        return A_raw @ u

    A_eff = _effective_dense_matrix(
        A_mv,
        n,
        b.dtype,
        symmetrize=True,
        jitter=0.0,
        solve_dtype=None,
    )

    solver = DirectSPD(
        space="row",
        vector_space="array",
        solve_method="chol",
        symmetrize=True,
        jitter=0.0,
        solve_dtype=None,
    )

    u, info, _ = solver.solve(A_mv, b, state=None, precond=None)
    u_ref = jnp.linalg.solve(A_eff, b)
    rel_eff = _rel_resid_array(lambda x: A_eff @ x, u, b)

    np.testing.assert_allclose(
        np.asarray(u),
        np.asarray(u_ref),
        rtol=_row_value_rtol(tol),
        atol=max(_row_value_atol(tol), 1e-2),
    )
    assert float(rel_eff) < 5e-4
    assert info["solve_method"] == "chol"
    assert info["space"] == "row"


@pytest.mark.parametrize("solve_method", ["scipy", "chol"])
def test_direct_spd_jitter_stabilizes_semidefinite_operator(key, tol, solve_method):
    del key
    A = jnp.diag(jnp.asarray([2.0, 0.0, 1.0, 3.0], dtype=jnp.float32))
    b = jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float32)
    eps = jnp.asarray(1e-3, dtype=jnp.float32)
    A_reg = A + eps * jnp.eye(A.shape[0], dtype=A.dtype)

    def A_mv(u):
        return A @ u

    solver = DirectSPD(
        space="row",
        vector_space="array",
        solve_method=solve_method,
        symmetrize=True,
        jitter=float(eps),
        solve_dtype=None,
    )

    u, info, _ = solver.solve(A_mv, b, state=None, precond=None)
    rel_reg = _rel_resid_array(lambda x: A_reg @ x, u, b)
    u_ref = jnp.linalg.solve(A_reg, b)

    assert float(rel_reg) < 5e-5
    np.testing.assert_allclose(
        np.asarray(u),
        np.asarray(u_ref),
        rtol=_row_value_rtol(tol),
        atol=_row_value_atol(tol),
    )
    assert info["solve_method"] == solve_method


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(space="diag"), "space"),
        (dict(space="param", vector_space="bad"), "vector_space"),
        (dict(space="param", solve_method="bad"), "solve_method"),
        (dict(space="row", vector_space="pytree"), "row"),
    ],
)
def test_direct_spd_rejects_invalid_constructor_configs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DirectSPD(**kwargs)


def test_direct_spd_array_mode_rejects_non_vector_rhs():
    solver = DirectSPD(space="row", vector_space="array", solve_method="scipy")
    rhs = jnp.zeros((2, 3), dtype=jnp.float32)

    def A_mv(u):
        return u

    with pytest.raises(ValueError, match="shape"):
        solver.solve(A_mv, rhs, state=None, precond=None)


def test_direct_spd_param_mode_rejects_operator_output_size_mismatch(make_block_pytree):
    template, pack, _zeros_like = make_block_pytree(shape_a=(4,), shape_b=(3,))
    rhs = template

    other_template, other_pack, _ = make_block_pytree(shape_a=(5,), shape_b=(3,))
    flat_other, _ = ravel_pytree(other_template)

    solver = DirectSPD(space="param", vector_space="pytree", solve_method="scipy")

    def bad_A_mv(x_tree):
        del x_tree
        return other_pack(jnp.zeros_like(flat_other))

    with pytest.raises(ValueError):
        solver.solve(bad_A_mv, rhs, state=None, precond=None)


def test_direct_spd_default_constructor_is_param_pytree_scipy():
    solver = DirectSPD()
    assert solver.space == "param"
    assert solver.vector_space == "pytree"
    assert solver.solve_method == "scipy"


def test_mu_from_lambda_matches_reduction_contract():
    lam = jnp.asarray(0.25, dtype=jnp.float32)

    mu_mean = mu_from_lambda(lam, b=8, reduction="mean")
    mu_sum = mu_from_lambda(lam, b=8, reduction="sum")

    np.testing.assert_allclose(np.asarray(mu_mean), np.asarray(2.0), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(mu_sum), np.asarray(0.25), rtol=0.0, atol=0.0)
