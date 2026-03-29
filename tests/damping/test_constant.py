import pytest

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from somax.damping.base import DampingState
from somax.damping.constant import ConstantDamping


def test_constant_init_sets_lam_scalar_array(dtype):
    pol = ConstantDamping(lam0=3.5, dtype=dtype)
    st = pol.init()
    assert isinstance(st, DampingState)
    assert st.aux == ()
    assert st.lam.shape == ()
    assert st.lam.dtype == dtype
    assert jnp.allclose(st.lam, jnp.asarray(3.5, dtype))


def test_constant_update_is_noop_with_empty_info(dtype):
    pol = ConstantDamping(lam0=2.0, dtype=dtype)
    st = pol.init()
    info = FrozenDict({})
    st2 = pol.update(st, info)
    assert jnp.allclose(st2.lam, st.lam)
    assert st2.aux == st.aux


def test_constant_multiple_updates_do_not_change_lam(dtype):
    pol = ConstantDamping(lam0=0.125, dtype=dtype)
    st = pol.init()
    info = FrozenDict({})
    for _ in range(10):
        st = pol.update(st, info)
    assert jnp.allclose(st.lam, jnp.asarray(0.125, dtype))


def test_constant_jit_compatibility(dtype):
    pol = ConstantDamping(lam0=5.0, dtype=dtype)
    st = pol.init()
    info = FrozenDict({})

    @jax.jit
    def f(state):
        return pol.update(state, info)

    st2 = f(st)
    assert jnp.allclose(st2.lam, jnp.asarray(5.0, dtype))
