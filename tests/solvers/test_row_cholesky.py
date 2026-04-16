import pytest
import jax
import jax.numpy as jnp

from somax.solvers.row_cholesky import RowCholesky
from helpers import make_full_rank_J


@pytest.mark.parametrize("m,n", [(16, 10), (24, 6)])
@pytest.mark.parametrize("symmetrize", [True, False])
def test_row_cholesky_matches_dense_row_solve(
        keys, m, n, symmetrize, make_block_pytree, bind_row_system_from_J
):
    key = keys()
    _template, pack, _zeros_like = make_block_pytree(shape_a=(n,), shape_b=(0,))

    J = make_full_rank_J(m, n, key, dtype=jnp.float32)
    rhs_row = jax.random.normal(keys(), (m,), dtype=jnp.float32)

    lam = jnp.asarray(1e-3, dtype=jnp.float32)
    bsz = 8

    A_mv, rhs, backproject, mu = bind_row_system_from_J(
        J, rhs_row, pack, lam=lam, b=bsz, reduction="mean"
    )

    solver = RowCholesky(symmetrize=symmetrize, jitter=0.0, solve_dtype=None)
    u, info, st = solver.solve(A_mv, rhs, state=None, precond=None)

    assert st is None
    assert info["mode"] == "row_cholesky"

    A_row = J @ J.T + mu * jnp.eye(m, dtype=J.dtype)
    u_ref = jnp.linalg.solve(A_row, rhs_row)

    # Main correctness check: residual of the solved system.
    resid = jnp.linalg.norm(A_row @ u - rhs_row)
    rhs_norm = jnp.maximum(jnp.linalg.norm(rhs_row), jnp.asarray(1e-12, rhs_row.dtype))
    rel_resid = resid / rhs_norm
    assert rel_resid < 5e-5

    # Secondary sanity check against dense float32 reference.
    assert jnp.allclose(u, u_ref, rtol=1e-3, atol=5e-4)

    s = backproject(u)
    s_ref = J.T @ u_ref
    assert jnp.allclose(jax.flatten_util.ravel_pytree(s)[0], s_ref, rtol=1e-3, atol=5e-4)
