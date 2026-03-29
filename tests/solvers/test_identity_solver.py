import pytest
import jax.numpy as jnp

from somax.solvers.identity import IdentitySolve


def test_identity_requires_precond():
    solver = IdentitySolve()

    def bad_A_mv(_):
        raise AssertionError("IdentitySolve must not call A_mv")

    b = {"x": jnp.ones((3,), dtype=jnp.float32)}
    with pytest.raises(ValueError):
        solver.solve(bad_A_mv, b, state=None, precond=None)


def test_identity_applies_precond_pytree():
    solver = IdentitySolve()

    def bad_A_mv(_):
        raise AssertionError("IdentitySolve must not call A_mv")

    b = {
        "x": jnp.arange(3.0, dtype=jnp.float32),
        "y": jnp.array([-2.0, 1.0], dtype=jnp.float32),
    }

    def precond(u):
        return {"x": 2.0 * u["x"], "y": -u["y"]}

    s, info, st = solver.solve(bad_A_mv, b, state=None, precond=precond)

    assert info["mode"] == "direct_precond"
    assert st is None
    assert jnp.allclose(s["x"], 2.0 * b["x"])
    assert jnp.allclose(s["y"], -b["y"])
