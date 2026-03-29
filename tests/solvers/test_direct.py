import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from somax.solvers.direct import DirectSPD


def test_direct_spd_matches_dense_solve(key, make_spd, mv_from_dense, make_block_pytree):
    template, pack, _zeros_like = make_block_pytree(shape_a=(4,), shape_b=(3,))
    template_flat, _ = ravel_pytree(template)
    n = int(template_flat.shape[0])

    b_flat = jax.random.normal(key, (n,), dtype=jnp.float32)
    b = pack(b_flat)

    A = make_spd(n=n, key=key, cond=20.0, eps=1e-6)
    A_mv = mv_from_dense(A, pack=pack)

    solver = DirectSPD()
    x, info, _ = solver.solve(A_mv, b, state=None)

    Ax = A_mv(x)
    Ax_f, _ = ravel_pytree(Ax)
    b_f, _ = ravel_pytree(b)
    rel = jnp.linalg.norm(Ax_f - b_f) / jnp.maximum(jnp.linalg.norm(b_f), jnp.array(1e-12, b_f.dtype))

    assert info["mode"] == "dense_direct"
    assert float(rel) < 1e-5
