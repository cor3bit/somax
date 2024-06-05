import jax
import jax.numpy as jnp
import flax.linen as nn

# sklearn
from sklearn import datasets as skl_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from somax import EGN


class MLPRegressorSmall(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)

        x = jnp.squeeze(x)

        return x


if __name__ == '__main__':

    @jax.jit
    def mse(params, x, y):
        residuals = y - model.apply(params, x)
        return 0.5 * jnp.mean(jnp.square(residuals))


    # ------------------- START HERE -------------------
    # jax.config.update('jax_platform_name', 'cpu')

    # load the dataset
    X, Y = skl_datasets.fetch_california_housing(return_X_y=True)
    X_scaled = StandardScaler(copy=False).fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.1, random_state=1337)
    batch_size = 64

    # define a neural network
    model = MLPRegressorSmall()

    # initialize parameters
    params = model.init(jax.random.PRNGKey(1), X_train[:batch_size])

    # define the solver
    solver = EGN(
        predict_fun=model.apply,
        loss_type='mse',
        learning_rate=0.1,
        regularizer=1.0,
        batch_size=batch_size,
    )

    opt_state = solver.init_state(params)
    update_fn = jax.jit(solver.update)

    # train the model
    for i in range(10):
        loss = mse(params, X_test, Y_test)
        print(f'T={i}, loss: {loss:.4f}')

        # make a step
        batch_x = X_train[i * batch_size:(i + 1) * batch_size]
        batch_y = Y_train[i * batch_size:(i + 1) * batch_size]
        params, opt_state = update_fn(params, opt_state, batch_x, targets=batch_y)

    # print the loss on the whole training data
    loss = mse(params, X_test, Y_test)
    print(f'T={i + 1}, loss: {loss:.4f}')
