from sklearn import datasets as skl_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.flatten_util import ravel_pytree


def flatten_2d_jacobian(jac_tree):
    return jax.vmap(lambda _: ravel_pytree(_)[0], in_axes=(0,))(jac_tree)


def flatten_3d_jacobian(jac_tree):
    flattened_jacobians = jax.vmap(flatten_2d_jacobian)(jac_tree)
    return flattened_jacobians.reshape(-1, flattened_jacobians.shape[-1])


def load_california():
    X, Y = skl_datasets.fetch_california_housing(return_X_y=True)
    X_scaled = StandardScaler(copy=False).fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.1, random_state=1337)
    return X_train, X_test, Y_train, Y_test


def load_iris():
    X, test_labels = skl_datasets.load_iris(return_X_y=True)
    X_scaled = StandardScaler(copy=False).fit_transform(X)
    X_train, X_test, Y_train_labels, Y_test_labels = train_test_split(
        X_scaled, test_labels, test_size=0.1, random_state=1337)

    n_classes = 3

    # transfer to device
    X_train = jnp.array(X_train)
    Y_train = jax.nn.one_hot(Y_train_labels, n_classes)
    X_test = jnp.array(X_test)
    Y_test = jax.nn.one_hot(Y_test_labels, n_classes)

    return (X_train, X_test, Y_train, Y_test), True, n_classes


class MLPRegressorMini(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(4)(x)
        x = nn.relu(x)
        x = nn.Dense(4)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)

        x = jnp.squeeze(x)

        return x


class MLPClassifierMini(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(4)(x)
        x = nn.relu(x)
        x = nn.Dense(4)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)

        return x
