import jax
import jax.numpy as jnp
import flax.linen as nn

from somax import EGN

if __name__ == '__main__':
    jax.config.update('jax_platform_name', 'cpu')


    @jax.jit
    def mse(params, x, y):
        residuals = y - model.apply(params, x)
        return 0.5 * jnp.mean(jnp.square(residuals))


    # create a synthetic sin wave dataset
    n = 200
    x = jnp.linspace(-3, 3, n)
    y = jnp.sin(x) + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (n,))

    x = x[:, None]

    # define a function approximator, eg. a neural network
    model = nn.Sequential([
        nn.Dense(8), nn.relu,
        nn.Dense(8), nn.relu,
        nn.Dense(1), lambda _: jnp.squeeze(_),
    ])

    # initialize parameters
    params = model.init(jax.random.PRNGKey(1), x)

    # define the solver
    solver = EGN(
        predict_fun=model.apply,
        loss_type='mse',
        learning_rate=0.1,
        regularizer=1.0,
        batch_size=16,
    )
    opt_state = solver.init_state(params)
    update_fn = jax.jit(solver.update)

    # train the model
    batch_size = 16
    for i in range(10):
        # print the loss on the whole training data
        loss = mse(params, x, y)
        print(f'T={i}, loss: {loss:.4f}')

        # make a step
        batch_x = x[i * batch_size:(i + 1) * batch_size]
        batch_y = y[i * batch_size:(i + 1) * batch_size]
        params, opt_state = update_fn(
            params, opt_state, batch_x, targets=batch_y)

    # print the loss on the whole training data
    loss = mse(params, x, y)
    print(f'T={i + 1}, loss: {loss:.4f}')
