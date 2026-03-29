import pytest

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from somax import metrics as mx
from somax.damping.base import DampingState
from somax.damping.trust_region import TrustRegionDamping


def _info(step, rho, rho_valid=True):
    return FrozenDict(
        {
            mx.STEP: jnp.asarray(step, jnp.int32),
            mx.RHO: jnp.asarray(rho, jnp.float32),
            mx.RHO_VALID: jnp.asarray(rho_valid, jnp.bool_),
        }
    )


def test_trust_region_init_sets_lam(dtype):
    pol = TrustRegionDamping(lam0=1.23, dtype=dtype)
    st = pol.init()
    assert isinstance(st, DampingState)
    assert jnp.allclose(st.lam, jnp.asarray(1.23, dtype))


def test_trust_region_invalid_rho_is_noop(dtype):
    pol = TrustRegionDamping(lam0=1.0, every_k=1, dtype=dtype)
    st = pol.init()

    st2 = pol.update(st, _info(0, 0.1, rho_valid=False))
    assert jnp.allclose(st2.lam, st.lam)

    st3 = pol.update(st, _info(0, jnp.nan, rho_valid=True))
    assert jnp.allclose(st3.lam, st.lam)


def test_trust_region_increase_decrease(dtype):
    pol = TrustRegionDamping(
        lam0=1.0,
        lower=0.4,
        upper=0.6,
        inc=1.5,
        dec=0.5,
        every_k=1,
        rho_clip=10.0,
        dtype=dtype,
    )
    st = pol.init()

    # bad ratio -> increase
    st = pol.update(st, _info(0, 0.1))
    assert jnp.allclose(st.lam, jnp.asarray(1.5, dtype), atol=1e-6)

    # good ratio -> decrease
    st = pol.update(st, _info(1, 0.9))
    assert jnp.allclose(st.lam, jnp.asarray(0.75, dtype), atol=1e-6)


def test_trust_region_cadence_every_k(dtype):
    pol = TrustRegionDamping(
        lam0=2.0,
        lower=0.4,
        upper=0.6,
        inc=2.0,
        dec=0.5,
        every_k=3,
        rho_clip=10.0,
        dtype=dtype,
    )
    st = pol.init()

    # cadence: step % 3 == 0 updates at steps 0,3,6,...
    st = pol.update(st, _info(0, 0.1))  # update -> *2
    assert jnp.allclose(st.lam, jnp.asarray(4.0, dtype), atol=1e-6)

    st2 = pol.update(st, _info(1, 0.1))  # no update
    assert jnp.allclose(st2.lam, st.lam)

    st3 = pol.update(st2, _info(2, 0.1))  # no update
    assert jnp.allclose(st3.lam, st.lam)

    st4 = pol.update(st3, _info(3, 0.1))  # update -> *2 again
    assert jnp.allclose(st4.lam, jnp.asarray(8.0, dtype), atol=1e-6)


def test_trust_region_jit(dtype):
    pol = TrustRegionDamping(lam0=3.0, every_k=2, dtype=dtype)
    st = pol.init()

    @jax.jit
    def step_fn(state, step_idx, rho_val, rho_ok):
        return pol.update(state, _info(step_idx, rho_val, rho_ok))

    st0 = step_fn(st, jnp.asarray(0, jnp.int32), jnp.asarray(0.1, dtype), jnp.asarray(True))
    # step 0 updates (0 % 2 == 0)
    assert st0.lam > st.lam

    st1 = step_fn(st0, jnp.asarray(1, jnp.int32), jnp.asarray(0.1, dtype), jnp.asarray(True))
    # step 1 no update
    assert jnp.allclose(st1.lam, st0.lam)
