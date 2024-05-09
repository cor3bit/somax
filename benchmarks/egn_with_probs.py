"""
Designing a classification loss function for second-order stochastic optimization
"""

import os.path
import time
from itertools import product
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

import jax
from jax import tree_map
import jax.numpy as jnp
import optax
from jaxopt import OptaxSolver
from flax import linen as nn

from benchmarks.utils.data_loader import load_data
import somax

TASK = 'covtype'  # imdb_reviews covtype cifar10 mnist fashion_mnist wine_quality iris a1a sensit

GPU = True
JIT = True

STEPS = 2_000  # 10_000
EVAL_EVERY = 50

N_SEEDS = 10

# grid
# 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001

SOLVER_HPS = {
    'iris': {
        'adam': {
            'b': [32, ],
            'lr': [0.01, ],
        },

        'egn-probs': {
            'b': [32, ],
            'lr': [0.5, ],
            'reg': [1.0, ],
            'ls': [False, ],
            'al': [False, ],
            'm': [0.0, ],
        },
    },

    'covtype': {
        'adam': {
            'b': [32, ],
            'lr': [5e-5, ],
        },

        'egn-probs': {
            'b': [32, ],
            'lr': [0.01,],
            'reg': [1.0, ],
            'ls': [False, ],
            'al': [False, ],
            'm': [0.0, ],
        },
    },

    'mnist': {
        'adam': {
            'b': [32, ],
            'lr': [0.001, ],
        },

        'egn-probs': {
            'b': [32, ],
            'lr': [ 0.1, ],
            'reg': [1.0, ],
            'ls': [False, ],
            'al': [False, ],
            'm': [0.0, ],
        },
    },

    'imdb_reviews': {
    },

    'sensit': {
    },

    'wine_quality': {
    },

    'fashion_mnist': {

    },

    'cifar10': {

    },

}


def batch_agnostic_reshape(x, x_dims=3):
    # no batching
    # eg. (H, W, C) or (L, C)
    if len(x.shape) == x_dims:
        return x.reshape(-1)
    # assume the first dimension is the batch dimension
    # eg. (B, H, W, C) or (B, L, C)
    else:
        return x.reshape((x.shape[0], -1))


class MLPClassifierSmallProbs(nn.Module):
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

        # softmax as part of the model
        x = jax.nn.softmax(x)

        return x


class MLPClassifierMediumProbs(nn.Module):
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

        # softmax as part of the model
        x = jax.nn.softmax(x)

        return x


class CNNClassifierMediumProbs(nn.Module):
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

        # softmax as part of the model
        x = jax.nn.softmax(x)

        return x


def create_solver(
        solver_type,
        solver_config,
        w_params,
        X_train,
        y_train,
        n_classes,
):
    opt_params = tree_map(lambda x: x, w_params)

    if solver_type == 'sgd':
        b, lr = solver_config
        solver_id = f'{solver_type}_b{b}_lr{lr}'
        solver = OptaxSolver(ce if n_classes > 1 else ce_binary, opt=optax.sgd(lr))
        opt_state = solver.init_state(opt_params, X_train[0:5], y_train[0:5])
    elif solver_type == 'adam':
        b, lr = solver_config
        solver_id = f'{solver_type}_b{b}_lr{lr}'
        solver = OptaxSolver(ce if n_classes > 1 else ce_binary, opt=optax.adam(lr))
        opt_state = solver.init_state(opt_params, X_train[0:5], y_train[0:5])

    elif solver_type == 'egn-probs':
        b, lr, reg, ls, al, m = solver_config

        ls_str = 'Y' if ls else 'N'
        al_str = 'Y' if al else 'N'

        solver_id = f'{solver_type}_b{b}_lr{lr}_reg{reg}_{ls_str}_{al_str}_m{m}'

        solver = somax.EGNProb(
            loss_fun=ce,
            predict_fun=predict_fn,

            loss_type='ce',
            learning_rate=lr,
            regularizer=reg,
            line_search=ls,
            adaptive_lambda=al,
            momentum=m,
            batch_size=b,
        )
        opt_state = solver.init_state(opt_params)

    else:
        raise ValueError(f'Unknown solver: {solver_type}')

    update_fn = jax.jit(solver.update)

    return update_fn, opt_params, opt_state, solver_id


def resolve_model(dataset_id: str, n_classes: Optional[int], is_cnn=True):
    # first handle special cases
    # very small datasets
    if dataset_id in ['iris', 'a1a', ]:
        model = MLPClassifierSmallProbs(num_classes=n_classes)
    # smallest CNNs
    elif dataset_id in ['covtype', 'wine_quality']:
        model = MLPClassifierMediumProbs(num_classes=n_classes)
    elif dataset_id in ['mnist', 'fashion_mnist']:
        model = CNNClassifierMediumProbs(num_classes=n_classes)
    # larger CNNs
    elif dataset_id in ['cifar10', ]:
        raise NotImplementedError
    # default model
    else:
        raise NotImplementedError

    # params = model.init(jax.random.PRNGKey(seed), X_sample)
    # predict_fn = jax.jit(model.apply)

    return model


def plot_metric(results):
    # Define common time points for interpolation
    time_limit = min([values[-1][0] for values in results.values()])
    n_points = 100  # Number of points for interpolation
    common_time_points = np.linspace(0, time_limit, n_points)

    # Organize data by optimizer
    runs_by_optimizer = {}
    for (optimizer_name, seed), values in results.items():
        if optimizer_name not in runs_by_optimizer:
            runs_by_optimizer[optimizer_name] = []
        times, run_values = zip(*values)
        runs_by_optimizer[optimizer_name].append((np.array(times), np.array(run_values)))

    # Interpolate runs for each optimizer
    interpolated_stats = {}
    for optimizer_name, runs in runs_by_optimizer.items():
        n_seeds = len(runs)
        interpolated_values = np.zeros((n_points, n_seeds))
        for i, (times, values) in enumerate(runs):
            interp_fn = interp1d(times, values, kind='linear', bounds_error=True)
            interpolated_values[:, i] = interp_fn(common_time_points)
        interpolated_stats[optimizer_name] = interpolated_values.mean(axis=1), interpolated_values.std(axis=1)

    # plotting
    fig, ax = plt.subplots()

    for optimizer_name, (avg_values, std_values) in interpolated_stats.items():
        ax.plot(common_time_points, avg_values, label=f'{optimizer_name}', linewidth=1)
        ax.fill_between(common_time_points, avg_values - std_values, avg_values + std_values, alpha=0.25)

    # save plot on disk
    ax.set_ylabel(f'Accuracy on Test Set')

    # !!HACK: log xi for ignd
    # ax.set_ylabel(r'$\xi$ during training')
    # ax.set_yscale('log')
    # ax.legend(loc='upper left')

    ax.legend(loc='lower right')

    ax.set_xlabel('Wall Time (s)')

    # ax.set_yscale('log')
    # ax.set_xlabel(f'Environment Steps ($\\times {scale_str}%$)')
    # ax.set_title(env_name)

    # TODO limiter
    # if TASK == 'california_housing':
    #     ax.set_ylim(None, 1.1)
    # elif TASK == 'superconduct':
    #     ax.set_ylim(None, 20.1)

    ax.set_title(f'{TASK}, Iterations={STEPS}')

    # modifier = os.path.basename(__file__).replace('.py', '')
    foldname = os.path.join('..', 'artifacts', 'examples', TASK)
    if not os.path.exists(foldname):
        os.makedirs(foldname)

    fname = os.path.join(foldname, f'{TASK}.png')
    fig.savefig(fname, bbox_inches='tight', dpi=600)


def calc_size(steps_in_run, configs):
    n_runs = sum(len(c) for _, c in configs.items())
    print(f'Number of runs: {n_runs}')
    return n_runs * steps_in_run * N_SEEDS


def create_configs(solvers):
    configs = {}

    for solver_type, solver_params in solvers.items():
        solver_configs = list(product(*solver_params.values()))
        configs[solver_type] = solver_configs

    return configs


if __name__ == '__main__':
    # force jax to use CPU
    if not GPU:
        jax.config.update('jax_platform_name', 'cpu')

    # for debugging JAX-related issues
    # jax.config.update('jax_enable_x64', True)
    # jax.config.update('jax_debug_nans', True)

    jax.config.update('jax_disable_jit', not JIT)


    @jax.jit
    def ce(params, features, targets):
        probs = predict_fn(params, features)
        log_probs = jnp.log(probs)
        residuals = jnp.sum(targets * log_probs, axis=-1)
        return -jnp.mean(residuals)


    @jax.jit
    def accuracy(params, X, Y_true):
        # b x C
        probs = predict_fn(params, X)

        # b x 1
        predicted_classes = jnp.argmax(probs, axis=1)
        correct_predictions = predicted_classes == Y_true

        # scalar
        return jnp.mean(correct_predictions)


    def ce_binary(params, x, y):
        # TODO from probs

        # b x 1
        logits = predict_fn(params, x)

        # b x 1
        losses = optax.sigmoid_binary_cross_entropy(logits.ravel(), y)

        # 1,
        # average over the batch
        return jnp.mean(losses)


    @jax.jit
    def accuracy_binary(params, X, Y_true):
        # TODO from probs

        # b x 1
        logits = predict_fn(params, X)

        # b x 1
        # probs = jax.nn.sigmoid(logits).ravel()
        # predicted_classes = probs > 0.5

        # Convert logits directly to class predictions by checking if they are >= 0
        # This step leverages the fact that the sigmoid function outputs values in the range (0, 1),
        # and its output is >= 0.5 (class 1) when the input logit is >= 0.
        predicted_classes = logits >= 0

        # scalar
        accuracy = jnp.mean(predicted_classes.ravel() == Y_true)

        return accuracy


    # --------------- dataset & models ---------------
    print(f'Running task: {TASK}')

    task_hps = SOLVER_HPS[TASK]

    configs = create_configs(task_hps)
    total_size = calc_size(STEPS, configs)

    # loads data
    # (X_train, X_test, Y_train, Y_test), is_clf, n_classes = load_data(
    #     dataset_id=TASK, test_size=0.1, seed=SEED)
    #

    # n_classes = egn.get_n_classes(TASK)

    #

    results = {}

    with tqdm(total=total_size) as pbar:
        for solver_type, solver_params in task_hps.items():
            for config in configs[solver_type]:
                for seed in range(N_SEEDS):
                    # --------------- random part ---------------
                    # ! RANDOMNESS (1): the seed is used to split and shuffle the dataset
                    (X_train, X_test, Y_train, Y_test), is_clf, n_classes = load_data(
                        dataset_id=TASK, test_size=0.1, seed=seed)

                    # ! RANDOMNESS (2): the seed is used to initialize the model
                    model = resolve_model(TASK, n_classes)
                    params = model.init(jax.random.PRNGKey(seed), X_train[0])
                    predict_fn = jax.jit(model.apply)

                    # --------------- init solvers ---------------
                    is_egn_like = solver_type in ['egn', 'hfo', 'sgn', 'egn-probs', ] or solver_type.startswith('ignd')
                    b = config[0]

                    if n_classes == 1:
                        raise NotImplementedError('Binary classification is not supported')

                    accuracy_fn = accuracy if n_classes > 1 else accuracy_binary
                    loss_fn = ce if n_classes > 1 else ce_binary

                    update_fn, opt_params, opt_state, solver_id = create_solver(
                        solver_type, config, params, X_train, Y_train, n_classes)

                    results[(solver_id, seed)] = []

                    # --------------- warm up ---------------
                    n_warmup = 2
                    fake_params = jax.tree_map(lambda _: _, params)
                    fake_opt_state = jax.tree_map(lambda _: _, opt_state)
                    for i in range(n_warmup):
                        # on Test
                        fake_loss3 = accuracy_fn(fake_params, X_test, Y_test)

                        # on batches
                        batch_X = X_train[i * b:(i + 1) * b, :]
                        batch_y = Y_train[i * b:(i + 1) * b]

                        fake_loss2 = loss_fn(fake_params, batch_X, batch_y)
                        fake_preds = predict_fn(fake_params, batch_X)

                        if is_egn_like:
                            fake_params, fake_opt_state = update_fn(
                                fake_params, fake_opt_state, batch_X, targets=batch_y)
                        else:
                            fake_update = update_fn(
                                fake_params, fake_opt_state, batch_X, batch_y)

                    # --------------- training loop ---------------

                    num_batches = X_train.shape[0] // b

                    step = 0

                    while True:
                        if step >= STEPS:
                            break

                        for i in range(0, num_batches):
                            if step >= STEPS:
                                break

                            if step % EVAL_EVERY == 0:
                                # on Test
                                loss = accuracy_fn(opt_params, X_test, Y_test)

                                # !!HACK: log xi for ignd
                                # loss = opt_state.xi

                                # first time is zero
                                if not results[(solver_id, seed)]:
                                    start_time = time.time()
                                    results[(solver_id, seed)].append((0, loss))
                                else:
                                    delta_t = time.time() - start_time
                                    results[(solver_id, seed)].append((delta_t, loss))

                            # ignore the last batch if it's smaller than the batch size
                            batch_X = X_train[i * b:(i + 1) * b, :]
                            batch_y = Y_train[i * b:(i + 1) * b]

                            # UPDATE PARAMS
                            if not is_egn_like:
                                opt_params, opt_state = update_fn(
                                    opt_params, opt_state, batch_X, batch_y)
                            else:
                                opt_params, opt_state = update_fn(
                                    opt_params, opt_state, batch_X, targets=batch_y)

                            step += 1
                            pbar.update(1)

    # --------------- plotting ---------------
    plt.style.use('bmh')
    plot_metric(results)
