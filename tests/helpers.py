import pytest

import jax
import jax.numpy as jnp


def make_full_rank_J(m: int, n: int, key, dtype=jnp.float32):
    """Dense J (m x n) with full column rank when m >= n and well-spread spectrum."""
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (m, m), dtype=dtype)
    Q, _ = jnp.linalg.qr(A)
    B = jax.random.normal(k2, (n, n), dtype=dtype)
    R, _ = jnp.linalg.qr(B)
    D = jnp.linspace(0.5, 2.0, min(m, n), dtype=dtype)
    return (Q[:, :n] * D) @ R.T  # (m, n)


def normal_eq_solution(J: jax.Array, rhs_row: jax.Array, mu: float):
    """Solve (J^T J + mu I) s = J^T rhs_row in param space (reference)."""
    n = J.shape[1]
    JT = J.T
    H = JT @ J + mu * jnp.eye(n, dtype=J.dtype)
    g = JT @ rhs_row
    L = jnp.linalg.cholesky(H)
    y = jax.lax.linalg.triangular_solve(L, g, left_side=True, lower=True)
    s = jax.lax.linalg.triangular_solve(L, y, left_side=True, lower=True, transpose_a=True)
    return s
