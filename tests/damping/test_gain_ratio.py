import pytest

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from somax import metrics as mx
from somax.damping.base import DampingState
from somax.damping.trust_region import GainRatioDamping, TrustRegionDamping


def _info(step, gain_ratio, gain_ratio_valid=True):
    return FrozenDict(
        {
            mx.STEP: jnp.asarray(step, jnp.int32),
            mx.GAIN_RATIO: jnp.asarray(gain_ratio, jnp.float32),
            mx.GAIN_RATIO_VALID: jnp.asarray(gain_ratio_valid, jnp.bool_),
        }
    )


def test_gain_ratio_damping_requests_only_required_metrics():
    policy = GainRatioDamping()
    assert policy.needed_metrics() == frozenset(
        {mx.STEP, mx.GAIN_RATIO, mx.GAIN_RATIO_VALID}
    )


def test_gain_ratio_damping_init_sets_and_clamps_lambda(dtype):
    policy = GainRatioDamping(
        lam0=1.23,
        min_lam=0.5,
        max_lam=2.0,
        dtype=dtype,
    )
    state = policy.init()
    assert isinstance(state, DampingState)
    assert jnp.allclose(state.lam, jnp.asarray(1.23, dtype))
    assert int(state.aux.action) == 0
    assert not bool(state.aux.lam_clipped)

    low = GainRatioDamping(lam0=0.1, min_lam=0.5, max_lam=2.0, dtype=dtype).init()
    high = GainRatioDamping(lam0=3.0, min_lam=0.5, max_lam=2.0, dtype=dtype).init()
    assert jnp.allclose(low.lam, jnp.asarray(0.5, dtype))
    assert jnp.allclose(high.lam, jnp.asarray(2.0, dtype))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"every_k": 0},
        {"lower": 0.7, "upper": 0.6},
        {"inc": 1.0},
        {"dec": 0.0},
        {"dec": 1.0},
        {"min_lam": 0.0},
        {"min_lam": 2.0, "max_lam": 1.0},
        {"gain_ratio_clip": 0.0},
        {"lam0": jnp.inf},
    ],
)
def test_gain_ratio_damping_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        GainRatioDamping(**kwargs)


def test_gain_ratio_damping_invalid_ratio_is_noop_and_resets_aux(dtype):
    policy = GainRatioDamping(lam0=1.0, every_k=1, dtype=dtype)
    state = policy.update(policy.init(), _info(0, 0.1))
    assert int(state.aux.action) == 1

    invalid_flag = policy.update(state, _info(1, 0.1, gain_ratio_valid=False))
    assert jnp.allclose(invalid_flag.lam, state.lam)
    assert int(invalid_flag.aux.action) == 0
    assert not bool(invalid_flag.aux.lam_clipped)

    nonfinite = policy.update(state, _info(1, jnp.nan, gain_ratio_valid=True))
    assert jnp.allclose(nonfinite.lam, state.lam)
    assert int(nonfinite.aux.action) == 0
    assert not bool(nonfinite.aux.lam_clipped)


@pytest.mark.parametrize(
    ("ratio", "expected_factor", "expected_action"),
    [
        (0.1, 1.5, 1),
        (0.4, 1.0, 0),  # thresholds are strict
        (0.5, 1.0, 0),
        (0.6, 1.0, 0),  # thresholds are strict
        (0.9, 0.5, -1),
    ],
)
def test_gain_ratio_damping_increase_keep_decrease(dtype, ratio, expected_factor, expected_action):
    policy = GainRatioDamping(
        lam0=2.0,
        lower=0.4,
        upper=0.6,
        inc=1.5,
        dec=0.5,
        every_k=1,
        gain_ratio_clip=10.0,
        dtype=dtype,
    )
    state_t = policy.init()
    state_next = policy.update(state_t, _info(0, ratio))

    # The input is lambda_t; the returned state contains lambda_{t+1}.
    assert jnp.allclose(state_t.lam, jnp.asarray(2.0, dtype))
    assert jnp.allclose(
        state_next.lam,
        jnp.asarray(2.0 * expected_factor, dtype),
        atol=1e-6,
    )
    assert int(state_next.aux.action) == expected_action


def test_gain_ratio_damping_cadence_updates_next_lambda_only(dtype):
    policy = GainRatioDamping(
        lam0=2.0,
        lower=0.4,
        upper=0.6,
        inc=2.0,
        dec=0.5,
        every_k=3,
        gain_ratio_clip=10.0,
        dtype=dtype,
    )
    state = policy.init()

    state1 = policy.update(state, _info(0, 0.1))
    assert jnp.allclose(state1.lam, jnp.asarray(4.0, dtype), atol=1e-6)

    state2 = policy.update(state1, _info(1, 0.1))
    state3 = policy.update(state2, _info(2, 0.1))
    assert jnp.allclose(state2.lam, state1.lam)
    assert jnp.allclose(state3.lam, state1.lam)
    assert int(state2.aux.action) == 0
    assert int(state3.aux.action) == 0

    state4 = policy.update(state3, _info(3, 0.1))
    assert jnp.allclose(state4.lam, jnp.asarray(8.0, dtype), atol=1e-6)


def test_gain_ratio_damping_clamps_proposals_and_reports_clipping(dtype):
    increase = GainRatioDamping(
        lam0=1.0,
        lower=0.4,
        upper=0.6,
        inc=3.0,
        dec=0.5,
        min_lam=0.5,
        max_lam=2.0,
        dtype=dtype,
    )
    high = increase.update(increase.init(), _info(0, 0.0))
    assert jnp.allclose(high.lam, jnp.asarray(2.0, dtype))
    assert int(high.aux.action) == 1
    assert bool(high.aux.lam_clipped)

    decrease = GainRatioDamping(
        lam0=1.0,
        lower=0.4,
        upper=0.6,
        inc=2.0,
        dec=0.1,
        min_lam=0.5,
        max_lam=2.0,
        dtype=dtype,
    )
    low = decrease.update(decrease.init(), _info(0, 1.0))
    assert jnp.allclose(low.lam, jnp.asarray(0.5, dtype))
    assert int(low.aux.action) == -1
    assert bool(low.aux.lam_clipped)


def test_gain_ratio_damping_finite_ratio_is_clipped_before_comparison(dtype):
    policy = GainRatioDamping(
        lam0=1.0,
        lower=0.4,
        upper=0.6,
        gain_ratio_clip=0.5,
        dtype=dtype,
    )
    # 100 is finite, but clips to 0.5, which lies in the keep interval.
    state = policy.update(policy.init(), _info(0, 100.0))
    assert jnp.allclose(state.lam, jnp.asarray(1.0, dtype))
    assert int(state.aux.action) == 0


def test_gain_ratio_damping_jit_sequence(dtype):
    policy = GainRatioDamping(lam0=3.0, every_k=2, dtype=dtype)
    state = policy.init()

    @jax.jit
    def step_fn(current, step_idx, ratio, valid):
        return policy.update(current, _info(step_idx, ratio, valid))

    state0 = step_fn(
        state,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0.1, dtype),
        jnp.asarray(True),
    )
    assert state0.lam > state.lam

    state1 = step_fn(
        state0,
        jnp.asarray(1, jnp.int32),
        jnp.asarray(0.1, dtype),
        jnp.asarray(True),
    )
    assert jnp.allclose(state1.lam, state0.lam)
    assert int(state1.aux.action) == 0

    state2 = step_fn(
        state1,
        jnp.asarray(2, jnp.int32),
        jnp.asarray(0.9, dtype),
        jnp.asarray(True),
    )
    assert state2.lam < state1.lam


def test_trust_region_name_is_backwards_compatibility_alias():
    assert TrustRegionDamping is GainRatioDamping
