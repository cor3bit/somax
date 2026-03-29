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
