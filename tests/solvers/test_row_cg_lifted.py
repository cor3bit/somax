import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from somax import metrics as mx
from somax.curvature.base import RowOperator
from somax.solvers.base import NullSolverState
from somax.solvers.row_cg import RowConjugateGradient
from somax.solvers.row_cg_lifted import LiftedRowConjugateGradient
from somax.solvers.row_common import bind_row_system


@pytest.fixture
def problem(make_block_pytree, make_row_operator_from_J):
    params, pack, _ = make_block_pytree(shape_a=(8,), shape_b=(3,))
    L = 0.25 * jax.random.normal(jax.random.PRNGKey(17), (7, 11))
    rhs = jnp.asarray([1.0, -0.6, 0.4, 0.7, -0.2, 1.1, 0.3])
    return params, make_row_operator_from_J(L, rhs, pack, b=3), L


def ordinary(op, lam, *, maxiter, tol=0.0, reduction="mean"):
    solver = RowConjugateGradient(maxiter=maxiter, tol=tol)
    A, rhs, backproject, _ = bind_row_system(op, jnp.asarray(lam), reduction=reduction)
    u, info, _ = solver.solve(A, rhs, state=solver.init(None))
    return backproject(u), info


def lifted(params, op, lam, *, maxiter, tol=0.0, reduction="mean"):
    solver = LiftedRowConjugateGradient(maxiter=maxiter, tol=tol)
    return solver.solve_row_direction(
        op, lam, params=params, state=solver.init(params), reduction=reduction
    )


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_matches_ordinary_at_every_budget(problem, assert_allclose_tree, budget, reduction):
    params, op, _ = problem
    expected, ref_info = ordinary(op, 0.3, maxiter=budget, reduction=reduction)
    direction, info, state = lifted(params, op, 0.3, maxiter=budget, reduction=reduction)

    assert_allclose_tree(direction, expected, rtol=2e-5, atol=2e-7)
    assert set(info) == set(ref_info)
    for name in (mx.CG_ITERS, mx.CG_MAXITER, mx.CG_CONVERGED):
        np.testing.assert_array_equal(info[name], ref_info[name])
    np.testing.assert_allclose(info[mx.CG_RESID], ref_info[mx.CG_RESID], rtol=2e-5, atol=1e-7)
    assert np.isfinite(np.asarray(ravel_pytree(direction)[0])).all()
    assert isinstance(state, NullSolverState)


def test_converges_to_dense_parameter_solution(problem):
    params, op, L = problem
    d, info, _ = lifted(params, op, 0.3, maxiter=40, tol=1e-6)
    L, rhs = np.asarray(L, np.float64), np.asarray(op.rhs, np.float64)
    H = L.T @ L / op.b + 0.3 * np.eye(L.shape[1])
    g = L.T @ rhs / op.b
    np.testing.assert_allclose(ravel_pytree(d)[0], np.linalg.solve(H, g), rtol=1e-4, atol=2e-6)
    assert bool(info[mx.CG_CONVERGED])


@pytest.mark.parametrize("case", ["zero_rhs", "zero_budget", "initial_tolerance"])
def test_no_iteration(problem, case):
    params, op, _ = problem
    if case == "zero_rhs":
        op = op.replace(rhs=jnp.zeros_like(op.rhs))
    budget = 0 if case == "zero_budget" else 5
    tol = 1.0 if case == "initial_tolerance" else 1e-5
    d, info, _ = lifted(params, op, 0.3, maxiter=budget, tol=tol)
    np.testing.assert_array_equal(ravel_pytree(d)[0], jnp.zeros_like(ravel_pytree(params)[0]))
    assert int(info[mx.CG_ITERS]) == 0
    assert np.isfinite(np.asarray(info[mx.CG_RESID]))


def test_exact_one_step_convergence():
    rhs = jnp.asarray([1.0, -2.0, 3.0])
    op = RowOperator(rhs=rhs, b=3, jvp=lambda v: v, vjp=lambda u: u)
    d, info, _ = lifted(jnp.zeros_like(rhs), op, 0.0, maxiter=5, tol=1e-6)
    np.testing.assert_array_equal(d, rhs)
    assert int(info[mx.CG_ITERS]) == 1
    assert bool(info[mx.CG_CONVERGED])


def test_jit_scan_with_empty_state(problem, assert_allclose_tree):
    params, op, _ = problem
    solver = LiftedRowConjugateGradient(maxiter=3, tol=0.0)
    sequence = jnp.stack([op.rhs, 0.8 * op.rhs, jnp.zeros_like(op.rhs)])

    @jax.jit
    def run(seq, state):
        def body(st, b):
            d, info, next_st = solver.solve_row_direction(
                op.replace(rhs=b), 0.3, params=params, state=st
            )
            return next_st, (d, info[mx.CG_ITERS])

        return jax.lax.scan(body, state, seq)

    state, (directions, iters) = run(sequence, solver.init(params))
    assert isinstance(state, NullSolverState)
    assert not jax.tree_util.tree_leaves(state)
    for i in range(len(sequence)):
        expected, _ = ordinary(op.replace(rhs=sequence[i]), 0.3, maxiter=3)
        actual = jax.tree_util.tree_map(lambda x: x[i], directions)
        assert_allclose_tree(actual, expected, rtol=2e-5, atol=2e-7)
    assert int(iters[-1]) == 0


def test_three_iterations_skip_final_vjp(problem):
    params, op, _ = problem
    calls = []

    def vjp(u):
        jax.debug.callback(lambda: calls.append(1), ordered=True)
        return op.vjp(u)

    counted = op.replace(vjp=vjp)
    solver = LiftedRowConjugateGradient(maxiter=3, tol=0.0)

    @jax.jit
    def run():
        return solver.solve_row_direction(
            counted, 0.3, params=params, state=solver.init(params)
        )[:2]

    result = run()
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert int(result[1][mx.CG_ITERS]) == 3
    assert len(calls) == 3  # Ordinary row CG needs four VJPs.


def test_inherited_solve_returns_row_solution(problem):
    params, op, _ = problem
    lifted_solver = LiftedRowConjugateGradient(maxiter=3, tol=0.0)
    A, rhs, _, _ = bind_row_system(op, jnp.asarray(0.3))
    u, info, _ = lifted_solver.solve(A, rhs, state=lifted_solver.init(params))
    base = RowConjugateGradient(maxiter=3, tol=0.0)
    ref, ref_info, _ = base.solve(A, rhs, state=base.init(None))
    np.testing.assert_allclose(u, ref, rtol=0, atol=0)
    assert int(info[mx.CG_ITERS]) == int(ref_info[mx.CG_ITERS])


@pytest.mark.parametrize("option", [
    {"warm_start": True},
    {"preconditioned": True},
    {"stabilise_every": 2},
    {"backend": "scipy"},
    {"solve_dtype": jnp.float32},
])
def test_rejects_unsupported_modes(option):
    with pytest.raises(NotImplementedError):
        LiftedRowConjugateGradient(**option)


def test_rejects_unsupported_operator_options(problem):
    params, op, _ = problem
    solver = LiftedRowConjugateGradient()
    kwargs = dict(params=params, state=solver.init(params))
    with pytest.raises(NotImplementedError, match="Gram"):
        solver.solve_row_direction(op.replace(gram_mv=lambda x: x), 0.3, **kwargs)
    with pytest.raises(NotImplementedError, match="preconditioning"):
        solver.solve_row_direction(op, 0.3, precond=lambda x: x, **kwargs)
    with pytest.raises(ValueError, match="shape"):
        solver.solve_row_direction(op.replace(rhs=op.rhs[:, None]), 0.3, **kwargs)
    with pytest.raises(TypeError, match="damping"):
        solver.solve_row_direction(op, jnp.asarray(0.3, jnp.float16), **kwargs)
