import jax
import jax.numpy as jnp

from somax import optax as sx_optax


def test_build_optax_tx_sign_flip():
    # learning_rate=1.0 -> scale(-1.0) in optax for descent.
    tx = sx_optax.build_optax_tx(learning_rate=1.0)
    params = {"w": jnp.array([1.0], dtype=jnp.float32)}
    updates = {"w": jnp.array([2.0], dtype=jnp.float32)}  # "s" (direction)
    state = tx.init(params)

    delta, _ = tx.update(updates, state, params)
    assert jnp.allclose(delta["w"], jnp.array([-2.0], dtype=jnp.float32))


def test_build_optax_tx_with_lr():
    tx = sx_optax.build_optax_tx(learning_rate=0.1)
    updates = {"w": jnp.array([10.0], dtype=jnp.float32)}
    state = tx.init({})
    delta, _ = tx.update(updates, state, {})
    assert jnp.allclose(delta["w"], jnp.array([-1.0], dtype=jnp.float32))


def test_build_optax_tx_clipping_and_decay():
    # Clip -> Decay -> Scale(-LR)
    tx = sx_optax.build_optax_tx(learning_rate=1.0, clip_norm=1.0, weight_decay=0.1)

    # updates s = 10.0, params = 2.0:
    # clip: s -> 1.0
    # decay: s -> 1.0 + 0.1*2.0 = 1.2
    # scale: delta = -1.2
    updates = {"w": jnp.array([10.0], dtype=jnp.float32)}
    params = {"w": jnp.array([2.0], dtype=jnp.float32)}
    state = tx.init(params)

    delta, _ = tx.update(updates, state, params)
    assert jnp.allclose(delta["w"], jnp.array([-1.2], dtype=jnp.float32))


def test_build_optax_tx_direction_momentum_recurrence_under_jit():
    tx = sx_optax.build_optax_tx(
        learning_rate=0.1,
        direction_momentum=0.5,
        nesterov=False,
    )
    params = {"w": jnp.zeros((1,), dtype=jnp.float32)}
    state0 = tx.init(params)

    @jax.jit
    def update(direction, state):
        return tx.update({"w": direction}, state, params)

    delta1, state1 = update(jnp.asarray([1.0], jnp.float32), state0)
    delta2, _ = update(jnp.asarray([2.0], jnp.float32), state1)

    # trace_1 = 1; trace_2 = 2 + 0.5 * 1 = 2.5.
    assert jnp.allclose(delta1["w"], jnp.asarray([-0.1], jnp.float32))
    assert jnp.allclose(delta2["w"], jnp.asarray([-0.25], jnp.float32))


def test_build_optax_tx_momentum_precedes_global_norm_clipping():
    tx = sx_optax.build_optax_tx(
        learning_rate=1.0,
        direction_momentum=0.5,
        clip_norm=1.0,
    )
    params = {"w": jnp.zeros((1,), dtype=jnp.float32)}
    state = tx.init(params)

    delta1, state = tx.update(
        {"w": jnp.asarray([10.0], jnp.float32)},
        state,
        params,
    )
    delta2, _ = tx.update(
        {"w": jnp.asarray([0.0], jnp.float32)},
        state,
        params,
    )

    # Momentum retains 5.0 on the second call, then clipping maps it to 1.0.
    assert jnp.allclose(delta1["w"], jnp.asarray([-1.0], jnp.float32))
    assert jnp.allclose(delta2["w"], jnp.asarray([-1.0], jnp.float32))


def test_build_optax_tx_learning_rate_schedule_advances_once_per_update():
    def schedule(count):
        return jnp.asarray(0.2, jnp.float32) / (
            count.astype(jnp.float32) + 1.0
        )

    tx = sx_optax.build_optax_tx(learning_rate=schedule)
    params = {"w": jnp.zeros((1,), dtype=jnp.float32)}
    state0 = tx.init(params)
    direction = {"w": jnp.asarray([2.0], jnp.float32)}

    @jax.jit
    def update(state):
        return tx.update(direction, state, params)

    delta1, state1 = update(state0)
    delta2, _ = update(state1)

    assert jnp.allclose(delta1["w"], jnp.asarray([-0.4], jnp.float32))
    assert jnp.allclose(delta2["w"], jnp.asarray([-0.2], jnp.float32))
