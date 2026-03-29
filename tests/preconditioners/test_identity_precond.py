import jax
import jax.numpy as jnp
import chex

from somax.preconditioners.identity import IdentityPrecond
from somax.solvers.cg import ConjugateGradient
from somax.solvers.base import NullSolverState
from somax.curvature.base import CurvatureState


def _rand_tree(keys, dtype):
    k1, k2, k3 = keys(3)
    return {
        "w": jax.random.normal(k1, (7,), dtype=dtype),
        "b": jax.random.normal(k2, (3, 5), dtype=dtype),
        "ten": jax.random.normal(k3, (2, 2, 3), dtype=dtype),
    }


def test_identity_init_and_build_are_state_stable(keys, dtype):
    params = _rand_tree(keys, dtype)

    pre = IdentityPrecond()
    st0 = pre.init(params)

    assert pre.is_identity

    class _Op:
        pass

    op = _Op()
    cst = CurvatureState(cache=())

    pre_fn, st1 = pre.build(params, op, cst, rng=keys(), lam=None, state=st0)

    # New contract: identity preconditioner returns None (= no preconditioning).
    assert pre_fn is None

    # State must be returned as-is (no churn, stable PyTree).
    assert st1 is st0
