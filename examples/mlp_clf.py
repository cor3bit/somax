import jax
import jax.numpy as jnp
import flax.linen as nn

# sklearn
from sklearn import datasets as skl_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from somax import EGN


class MLPClassifierSmall(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(8)(x)
        x = nn.relu(x)
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(8)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)

        return x


if __name__ == '__main__':

    @jax.jit
    def accuracy(params, features, labels):
        logits = predict_fn(params, features)
        predicted_classes = jnp.argmax(logits, axis=1)
        correct_predictions = predicted_classes == labels
        return jnp.mean(correct_predictions)


    @jax.jit
    def ce(params, features, ohe_labels):
        logits = predict_fn(params, features)
        log_probs = jax.nn.log_softmax(logits)
        residuals = jnp.sum(ohe_labels * log_probs, axis=1)
        ce_loss = -jnp.mean(residuals)
        return ce_loss


    # ------------------- START HERE -------------------
    # jax.config.update('jax_platform_name', 'cpu')

    # load the dataset
    X, test_labels = skl_datasets.load_iris(return_X_y=True)
    X_scaled = StandardScaler(copy=False).fit_transform(X)
    X_train, X_test, Y_train_labels, Y_test_labels = train_test_split(
        X_scaled, test_labels, test_size=0.1, random_state=1337)

    n_classes = 3

    # transfer to device
    X_train = jnp.array(X_train)
    Y_train = jax.nn.one_hot(Y_train_labels, n_classes)
    X_test = jnp.array(X_test)
    Y_test = jnp.array(Y_test_labels)

    batch_size = 32

    # define a neural network
    model = MLPClassifierSmall(num_classes=n_classes)

    # initialize parameters
    params = model.init(jax.random.PRNGKey(1), X_train[:batch_size])

    # define the solver
    predict_fn = model.apply

    solver = EGN(
        predict_fun=predict_fn,
        loss_type='ce',
        learning_rate=0.1,
        regularizer=1.0,
        batch_size=batch_size,
        n_classes=n_classes,
    )

    opt_state = solver.init_state(params)
    update_fn = jax.jit(solver.update)

    # train the model
    num_samples = X_train.shape[0]  # Total number of samples in the training dataset
    for i in range(10):
        loss = accuracy(params, X_test, Y_test)
        print(f'T={i}, loss: {loss:.4f}')

        # compute start index using modulo to wrap around
        start_index = (i * batch_size) % num_samples
        end_index = (start_index + batch_size) % num_samples

        # handle the wrap-around case
        if start_index + batch_size <= num_samples:
            batch_x = X_train[start_index:end_index]
            batch_y = Y_train[start_index:end_index]
        else:
            # concatenate the end of the dataset with the beginning to form a complete batch
            batch_x = jnp.concatenate((X_train[start_index:], X_train[:end_index]), axis=0)
            batch_y = jnp.concatenate((Y_train[start_index:], Y_train[:end_index]), axis=0)

        # update model parameters
        params, opt_state = update_fn(params, opt_state, batch_x, targets=batch_y)

    # print the loss on the whole training data
    loss = accuracy(params, X_test, Y_test)
    print(f'T={i + 1}, loss: {loss:.4f}')
