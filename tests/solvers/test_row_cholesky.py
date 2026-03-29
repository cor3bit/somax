import pytest
import jax
import jax.numpy as jnp

from somax.solvers.row_cholesky import RowCholesky
from tests.conftest import make_full_rank_J, normal_eq_solution


@pytest.mark.parametrize("m,n", [(16, 10), (24, 6)])
@pytest.mark.parametrize("symmetrize", [True, False])
def test_row_cholesky_matches_normal_equations(keys, m, n, symmetrize, make_block_pytree, bind_row_system_from_J, tol):
    key = keys()
    template, pack, _zeros_like = make_block_pytree(shape_a=(n,), shape_b=(0,))
    # pack expects flat size n; template leaf "b" is empty, harmless.

    J = make_full_rank_J(m, n, key, dtype=jnp.float32)
    rhs_row = jax.random.normal(keys(), (m,), dtype=jnp.float32)

    lam = jnp.asarray(1e-3, dtype=jnp.float32)
    bsz = 8

    A_mv, rhs, backproject, mu = bind_row_system_from_J(
        J, rhs_row, pack, lam=lam, b=bsz, reduction="mean")

    solver = RowCholesky(symmetrize=symmetrize, jitter=1e-6, solve_dtype=None)
    u, info, st = solver.solve(A_mv, rhs, state=None, precond=None)
    assert st is None
    assert info["mode"] == "row_cholesky"

    s = backproject(u)
    s_ref = normal_eq_solution(J, rhs_row, float(mu))

    # Solve is float32; use conftest tol.
    assert jnp.allclose(jax.flatten_util.ravel_pytree(s)[0], s_ref, **tol)
