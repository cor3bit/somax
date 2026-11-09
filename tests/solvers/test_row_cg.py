import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from somax import metrics as mx
from somax.solvers.base import NullSolverState
from somax.solvers.row_cg import RowConjugateGradient
from somax.solvers.row_common import bind_row_system


@pytest.fixture
def problem(make_block_pytree, make_row_operator_from_J):
    params, pack, _ = make_block_pytree(shape_a=(8,), shape_b=(3,))
    L = 0.25 * jax.random.normal(jax.random.PRNGKey(17), (7, 11))
    rhs = jnp.asarray([1.0, -0.6, 0.4, 0.7, -0.2, 1.1, 0.3])
    # Row dimension m=7 differs from batch size B=3.
    return params, make_row_operator_from_J(L, rhs, pack, b=3), L


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_row_system_scaling_and_solution(problem, reduction):
    _, op, L = problem
    lam = jnp.asarray(0.3, L.dtype)
    A, rhs, backproject, mu = bind_row_system(op, lam, reduction=reduction)
    factor = 1 / op.b if reduction == "mean" else 1.0

    np.testing.assert_allclose(mu, lam / factor, rtol=1e-6)
    probe = jnp.linspace(-0.8, 0.7, rhs.size)
    np.testing.assert_allclose(A(probe), L @ (L.T @ probe) + mu * probe, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(ravel_pytree(backproject(probe))[0], L.T @ probe, rtol=2e-6, atol=1e-6)

    solver = RowConjugateGradient(maxiter=40, tol=1e-6)
    u, info, state = solver.solve(A, rhs, state=solver.init(None))
    direction = np.asarray(ravel_pytree(backproject(u))[0])
    L64, rhs64 = np.asarray(L, np.float64), np.asarray(rhs, np.float64)
    H = factor * L64.T @ L64 + float(lam) * np.eye(L.shape[1])
    g = factor * L64.T @ rhs64

    np.testing.assert_allclose(direction, np.linalg.solve(H, g), rtol=1e-4, atol=2e-6)
    assert isinstance(state, NullSolverState)
    assert bool(info[mx.CG_CONVERGED])
    assert int(info[mx.CG_ITERS]) <= 40
    assert np.isfinite(np.asarray(info[mx.CG_RESID]))


def test_gram_shortcut_does_not_use_jvp(problem):
    _, op, L = problem
    gram = L @ L.T

    def unavailable(_):
        pytest.fail("Gram matvec should bypass row JVP")

    gram_op = op.replace(jvp=unavailable, gram_mv=lambda u: gram @ u)
    A, rhs, backproject, _ = bind_row_system(gram_op, jnp.asarray(0.3))
    solver = RowConjugateGradient(maxiter=40, tol=1e-6)
    u, _, _ = solver.solve(A, rhs, state=solver.init(None))
    expected = np.linalg.solve(np.asarray(gram, np.float64) + 0.3 * op.b * np.eye(rhs.size),
                               np.asarray(rhs, np.float64))
    np.testing.assert_allclose(u, expected, rtol=1e-4, atol=2e-6)
    np.testing.assert_allclose(ravel_pytree(backproject(u))[0], L.T @ u, rtol=1e-5)


def test_stabilized_row_cg(problem):
    _, op, _ = problem
    A, rhs, _, _ = bind_row_system(op, jnp.asarray(0.3))
    solver = RowConjugateGradient(maxiter=40, tol=1e-5, stabilise_every=2)
    u, info, _ = solver.solve(A, rhs, state=solver.init(None))
    true_relative_residual = jnp.linalg.norm(A(u) - rhs) / jnp.linalg.norm(rhs)
    assert np.isfinite(np.asarray(u)).all()
    assert true_relative_residual < 1e-4
    assert int(info[mx.CG_ITERS]) <= 40


def test_stateless_under_jit(problem):
    _, op, _ = problem
    solver = RowConjugateGradient(maxiter=10, tol=1e-5)
    A, rhs, _, _ = bind_row_system(op, jnp.asarray(0.3))

    @jax.jit
    def run(b, state):
        u, info, next_state = solver.solve(A, b, state=state)
        return u, info[mx.CG_RESID], next_state

    u1, resid1, state1 = run(rhs, solver.init(None))
    u2, resid2, state2 = run(0.8 * rhs, state1)
    assert isinstance(state2, NullSolverState)
    assert not jax.tree_util.tree_leaves(state2)
    assert u1.shape == u2.shape == rhs.shape
    assert np.isfinite(np.asarray([resid1, resid2])).all()


@pytest.mark.parametrize("option, error", [
    ({"warm_start": True}, NotImplementedError),
    ({"preconditioned": True}, NotImplementedError),
    ({"solve_dtype": jnp.float32}, ValueError),
])
def test_rejects_unsupported_options(option, error):
    with pytest.raises(error):
        RowConjugateGradient(**option)


def test_rejects_invalid_rhs_and_preconditioner(problem):
    _, op, _ = problem
    solver = RowConjugateGradient()
    A, rhs, _, _ = bind_row_system(op, jnp.asarray(0.3))
    with pytest.raises(ValueError, match="shape"):
        solver.solve(A, rhs[:, None], state=solver.init(None))
    with pytest.raises(NotImplementedError, match="preconditioning"):
        solver.solve(A, rhs, state=solver.init(None), precond=lambda x: x)
