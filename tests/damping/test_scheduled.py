import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest
from flax.core import FrozenDict

from somax import metrics as mx
from somax.damping import DampingState, ScheduledDamping
from somax.specs import DampingSpec, build_damping


def _info(step):
    return FrozenDict({mx.STEP: jnp.asarray(step, jnp.int32)})


def test_scheduled_init_uses_factor_at_step_zero(dtype):
    schedule = lambda step: jnp.asarray(0.25, dtype)
    policy = ScheduledDamping(
        lam0=8.0,
        factor_schedule=schedule,
        dtype=dtype,
    )

    state = policy.init()

    assert isinstance(state, DampingState)
    assert state.aux == ()
    assert state.lam.dtype == jnp.dtype(dtype)
    np.testing.assert_allclose(np.asarray(state.lam), 2.0)


def test_scheduled_update_stores_next_outer_steps_lambda(dtype):
    schedule = lambda step: 1.0 / (jnp.asarray(step, dtype) + 1.0)
    policy = ScheduledDamping(
        lam0=6.0,
        factor_schedule=schedule,
        dtype=dtype,
    )

    state0 = policy.init()                    # lambda_0 = 6
    state1 = policy.update(state0, _info(0)) # stores lambda_1 = 3
    state2 = policy.update(state1, _info(1)) # stores lambda_2 = 2

    np.testing.assert_allclose(np.asarray(state0.lam), 6.0, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(state1.lam), 3.0, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(state2.lam), 2.0, rtol=1e-6)

    # Functional update: the input state remains lambda_t.
    np.testing.assert_allclose(np.asarray(state0.lam), 6.0, rtol=1e-6)


def test_scheduled_reuses_optax_cosine_schedule_directly(dtype):
    ratio = 10.0
    transition_steps = 8
    factor_schedule = optax.cosine_decay_schedule(
        init_value=1.0,
        decay_steps=transition_steps,
        alpha=1.0 / ratio,
    )
    policy = ScheduledDamping(
        lam0=5.0,
        factor_schedule=factor_schedule,
        dtype=dtype,
    )

    state = policy.init()
    for step in range(transition_steps + 2):
        expected = 5.0 * factor_schedule(jnp.asarray(step, jnp.int32))
        np.testing.assert_allclose(
            np.asarray(state.lam),
            np.asarray(expected, dtype=np.dtype(dtype)),
            rtol=2e-6,
            atol=2e-6,
        )
        state = policy.update(state, _info(step))


def test_scheduled_reuses_composed_optax_schedule(dtype):
    hold_steps = 3
    transition_steps = 6
    ratio = 3.0

    decay = optax.cosine_decay_schedule(
        init_value=1.0,
        decay_steps=transition_steps,
        alpha=1.0 / ratio,
    )
    factor_schedule = optax.join_schedules(
        schedules=[optax.constant_schedule(1.0), decay],
        boundaries=[hold_steps],
    )
    policy = ScheduledDamping(
        lam0=9.0,
        factor_schedule=factor_schedule,
        dtype=dtype,
    )

    state = policy.init()
    observed = []
    for step in range(hold_steps + transition_steps + 1):
        observed.append(float(state.lam))
        expected = 9.0 * factor_schedule(jnp.asarray(step, jnp.int32))
        np.testing.assert_allclose(
            np.asarray(state.lam),
            np.asarray(expected, dtype=np.dtype(dtype)),
            rtol=2e-6,
            atol=2e-6,
        )
        state = policy.update(state, _info(step))

    np.testing.assert_allclose(observed[: hold_steps + 1], 9.0, rtol=2e-6)
    assert observed[-1] < observed[hold_steps]


def test_scheduled_needed_metrics_are_minimal():
    policy = ScheduledDamping(
        lam0=1.0,
        factor_schedule=optax.constant_schedule(1.0),
    )

    assert policy.needed_metrics() == frozenset({mx.STEP})
    assert mx.GAIN_RATIO not in policy.needed_metrics()
    assert mx.LOSS_AFTER not in policy.needed_metrics()


def test_scheduled_is_jittable_across_multiple_updates(dtype):
    schedule = optax.cosine_decay_schedule(
        init_value=1.0,
        decay_steps=5,
        alpha=0.2,
    )
    policy = ScheduledDamping(
        lam0=4.0,
        factor_schedule=schedule,
        dtype=dtype,
    )

    @jax.jit
    def advance(state, first_step):
        def body(carry, offset):
            step = first_step + offset
            new_state = policy.update(carry, _info(step))
            return new_state, new_state.lam

        return jax.lax.scan(
            body,
            state,
            jnp.arange(4, dtype=jnp.int32),
        )

    state0 = policy.init()
    state4, lams = advance(state0, jnp.asarray(0, jnp.int32))

    expected = jnp.stack(
        [
            4.0 * schedule(jnp.asarray(step, jnp.int32))
            for step in range(1, 5)
        ]
    ).astype(dtype)
    np.testing.assert_allclose(np.asarray(lams), np.asarray(expected), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(state4.lam), np.asarray(expected[-1]), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("lam0", [-1.0, float("nan"), float("inf")])
def test_scheduled_rejects_invalid_reference_damping(lam0):
    with pytest.raises(ValueError, match="lam0"):
        ScheduledDamping(
            lam0=lam0,
            factor_schedule=optax.constant_schedule(1.0),
        )


def test_scheduled_rejects_non_callable_schedule():
    with pytest.raises(TypeError, match="factor_schedule"):
        ScheduledDamping(lam0=1.0, factor_schedule=1.0)


def test_build_damping_constructs_scheduled_policy():
    schedule = optax.constant_schedule(0.5)
    policy = build_damping(
        DampingSpec(
            kind="scheduled",
            lam0=4.0,
            kwargs={"factor_schedule": schedule},
        )
    )

    assert isinstance(policy, ScheduledDamping)
    np.testing.assert_allclose(np.asarray(policy.init().lam), 2.0)


def test_build_damping_schedule_alias_is_supported():
    policy = build_damping(
        DampingSpec(
            kind="schedule",
            lam0=3.0,
            kwargs={"factor_schedule": optax.constant_schedule(2.0)},
        )
    )

    assert isinstance(policy, ScheduledDamping)
    np.testing.assert_allclose(np.asarray(policy.init().lam), 6.0)


def test_build_damping_requires_factor_schedule():
    with pytest.raises(ValueError, match="factor_schedule"):
        build_damping(DampingSpec(kind="scheduled", lam0=1.0, kwargs={}))
