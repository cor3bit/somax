import jax
import jax.numpy as jnp
import flax
from flax import linen as nn


def batch_agnostic_reshape(x, x_dims=3):
    # no batching
    # eg. (H, W, C) or (L, C)
    if len(x.shape) == x_dims:
        return x.reshape(-1)
    # assume the first dimension is the batch dimension
    # eg. (B, H, W, C) or (B, L, C)
    else:
        return x.reshape((x.shape[0], -1))


def batch_agnostic_transpose(x, x_dims=3):
    # no batching
    # eg. (H, W, C) or (L, C)
    if len(x.shape) == x_dims:
        return jnp.transpose(x, (1, 2, 0))
    # assume the first dimension is the batch dimension
    # eg. (B, H, W, C) or (B, L, C)
    else:
        return jnp.transpose(x, (0, 2, 3, 1))


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


class MLPRegressorSmall(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(8)(x)
        x = nn.relu(x)
        x = nn.Dense(8)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)

        x = jnp.squeeze(x)

        return x


class MLPRegressorMedium(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)

        x = jnp.squeeze(x)

        return x


class MLPRegressorLarge(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)

        x = jnp.squeeze(x)

        return x


class ImageRegressorSmall(nn.Module):
    x_dims: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # normalize pixel values
        x = x / 255.

        x = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = batch_agnostic_reshape(x, self.x_dims)

        x = nn.Dense(64)(x)
        x = nn.relu(x)

        x = nn.Dense(1)(x)
        x = jnp.squeeze(x)

        return x


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


class MLPClassifierMedium(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)

        x = nn.Dense(self.num_classes)(x)

        return x


class MLPClassifierLarge(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)

        x = nn.Dense(self.num_classes)(x)

        return x


class ImageClassifierMLP(nn.Module):
    num_classes: int
    x_dims: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # Flatten the input
        x = batch_agnostic_reshape(x, self.x_dims)

        # Normalize
        x = x / 255.

        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        # x = nn.Dense(64)(x)
        # x = nn.relu(x)

        x = nn.Dense(self.num_classes)(x)

        return x


class CNNClassifierSmall(nn.Module):
    num_classes: int
    x_dims: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = x / 255.  # Normalize pixel values

        x = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten the output
        x = batch_agnostic_reshape(x, self.x_dims)

        x = nn.Dense(32)(x)
        x = nn.relu(x)

        x = nn.Dense(self.num_classes)(x)

        return x


class CNNClassifierMedium(nn.Module):
    num_classes: int
    x_dims: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # normalize pixel values
        x = x / 255.

        x = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten the output
        x = batch_agnostic_reshape(x, self.x_dims)

        x = nn.Dense(64)(x)
        x = nn.relu(x)

        x = nn.Dense(self.num_classes)(x)

        return x


class CNNClassifierLarge(nn.Module):
    num_classes: int
    x_dims: int = 3

    @nn.compact
    def __call__(self, x):
        # normalize pixel values
        x = x / 255.

        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten the output
        x = batch_agnostic_reshape(x, self.x_dims)

        x = nn.Dense(features=128)(x)
        x = nn.relu(x)

        x = nn.Dense(features=10)(x)

        return x


# Classic Control
class QNetworkSmall(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(16)(x)
        x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        return x


class QNetworkMedium(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)

        return x


class QNetworkLarge(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)

        return x


# MinAtar
class QNetworkCNNSmall(nn.Module):
    action_dim: int
    x_dims: int = 3

    @nn.compact
    def __call__(self, x):
        # reorders the input channels to be the last dimension
        # x = jnp.transpose(x, (0, 2, 3, 1))
        # x = x / 255.

        # self.conv = flax.linen.Conv(features=16, kernel_size=3, strides=(1, 1), padding='SAME', use_bias=False,
        #                             dtype=jnp.float32, kernel_init=flax.linen.initializers.xavier_uniform())

        x = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)

        # Flatten the output
        x = batch_agnostic_reshape(x, self.x_dims)

        x = nn.Dense(64)(x)

        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)

        return x


# Atari
class QNetworkCNNMedium(nn.Module):
    action_dim: int
    x_dims: int = 3

    @nn.compact
    def __call__(self, x):
        x = batch_agnostic_transpose(x, self.x_dims)
        x = x / 255.

        x = nn.Conv(16, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(32, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)

        # x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        # x = nn.relu(x)
        # x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        # x = nn.relu(x)
        # x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        # x = nn.relu(x)

        # Flatten the output
        x = batch_agnostic_reshape(x, self.x_dims)

        x = nn.Dense(256)(x)
        # x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)

        return x
