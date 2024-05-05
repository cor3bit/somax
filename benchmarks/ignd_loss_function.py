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
from optax._src.loss import poly_loss_cross_entropy
from flax import linen as nn

from benchmarks.utils.data_loader import load_data
import somax

TASK = 'iris'  # imdb_reviews covtype cifar10 mnist fashion_mnist wine_quality iris a1a sensit

GPU = True
JIT = True

# STEPS = 10_000
# STEPS = 2_000
STEPS = 100  # 10_000
EVAL_EVERY = 1
# EVAL_EVERY = 1

N_SEEDS = 1

# grid
# 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001

SOLVER_HPS = {
    'iris': {
        # 'adam': {
        #     'b': [32, ],
        #     'lr': [0.01, ],
        # },

        'ignd': {
            'b': [32, ],
            'lr': [0.5, ],
            'm': [(0, 0), ],
        },

        'ignd-logeps': {
            'b': [32, ],
            'lr': [0.5, ],
            'm': [(0, 0), ],
        },

        'ignd-poly-a': {
            'b': [32, ],
            'lr': [0.5, ],
            'm': [(0, 0), ],
        },

        'ignd-logbarrier': {
            'b': [32, ],
            'lr': [0.5, ],
            'm': [(0, 0), ],
        },

    },

    'mnist': {
    },

    'imdb_reviews': {
    },

    'sensit': {
    },

    'covtype': {
    },

    'wine_quality': {
    },

    'fashion_mnist': {

    },

    'cifar10': {

    },

}


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

        # softmax
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
    elif solver_type == 'adam-poly-a':
        b, lr = solver_config
        solver_id = f'{solver_type}_b{b}_lr{lr}'
        solver = OptaxSolver(ce_poly, opt=optax.adam(lr))
        opt_state = solver.init_state(opt_params, X_train[0:5], y_train[0:5])
    elif solver_type == 'adam-logeps':
        b, lr = solver_config
        solver_id = f'{solver_type}_b{b}_lr{lr}'
        solver = OptaxSolver(ce_logeps, opt=optax.adam(lr))
        opt_state = solver.init_state(opt_params, X_train[0:5], y_train[0:5])
    elif solver_type == 'adam-poly-1':
        b, lr = solver_config
        solver_id = f'{solver_type}_b{b}_lr{lr}'
        solver = OptaxSolver(ce_poly1, opt=optax.adam(lr))
        opt_state = solver.init_state(opt_params, X_train[0:5], y_train[0:5])


    elif solver_type == 'ignd':
        b, lr, m = solver_config

        solver_id = f'{solver_type}_b{b}_lr{lr}_m{m[0]}_{m[1]}'

        solver = somax.IGND(
            predict_fun=predict_fn,
            loss_type='ce',
            learning_rate=lr,
            momentum=m[0],
            beta2=m[1],
            batch_size=b,

            loss_fun=ce,
            loss_grad_fun=lambda p: -1 / p,
            loss_hessian_fun=lambda p: 1 / (p * p),
        )
        opt_state = solver.init_state(opt_params)
    elif solver_type == 'ignd-logeps':
        b, lr, m = solver_config

        solver_id = f'{solver_type}_b{b}_lr{lr}_m{m[0]}_{m[1]}'

        solver = somax.IGND(
            predict_fun=predict_fn,
            loss_type='ce',
            learning_rate=lr,
            momentum=m[0],
            beta2=m[1],
            batch_size=b,

            loss_fun=ce_logeps,
            loss_grad_fun=lambda p: -1 / (p + 1e-3),
            loss_hessian_fun=lambda p: 1 / ((p + 1e-3) * (p + 1e-3)),
        )
        opt_state = solver.init_state(opt_params)
    elif solver_type == 'ignd-poly-a':
        b, lr, m = solver_config

        solver_id = f'{solver_type}_b{b}_lr{lr}_m{m[0]}_{m[1]}'

        solver = somax.IGND(
            predict_fun=predict_fn,
            loss_type='ce',
            learning_rate=lr,
            momentum=m[0],
            beta2=m[1],
            batch_size=b,

            loss_fun=ce_poly,  # 1.4002 * (1 - p) + 3.5593 * (1 - p) ** 10
            loss_grad_fun=lambda p: -1.4002 - 35.593 * (1 - p) ** 9,
            loss_hessian_fun=lambda p: 320.337 * (1 - p) ** 8,
        )
        opt_state = solver.init_state(opt_params)
    elif solver_type == 'ignd-logbarrier':
        b, lr, m = solver_config

        solver_id = f'{solver_type}_b{b}_lr{lr}_m{m[0]}_{m[1]}'

        solver = somax.IGND(
            predict_fun=predict_fn,
            loss_type='ce',
            learning_rate=lr,
            momentum=m[0],
            beta2=m[1],
            batch_size=b,

            loss_fun=ce_logbarrier,
            loss_grad_fun=lambda p: -1 / (p + 1e-3) - 1 / (p + 1e-3) ** 2,
            loss_hessian_fun=lambda p: 1 / (p + 1e-3) ** 2 + 2 / (p + 1e-3) ** 3,
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
    elif dataset_id in ['mnist', 'fashion_mnist']:
        raise NotImplementedError
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
    def ce_poly(params, features, targets):
        def poly_approx(p):
            return 1.4002 * (1 - p) + 3.5593 * (1 - p) ** 10

        probs = predict_fn(params, features)
        log_probs = poly_approx(probs)
        residuals = jnp.sum(targets * log_probs, axis=-1)
        ce_loss = jnp.mean(residuals)
        return ce_loss


    @jax.jit
    def ce_poly1(params, features, targets, epsilon=2.0):
        probs = predict_fn(params, features)
        log_probs_smoothed = -jax.nn.log(probs) + epsilon * (1 - probs)
        residuals = jnp.sum(targets * log_probs_smoothed, axis=-1)
        ce_loss = jnp.mean(residuals)
        return ce_loss


    @jax.jit
    def ce_logeps(params, features, targets, epsilon=1e-3):
        probs = predict_fn(params, features)
        log_probs = jnp.log(probs + epsilon)
        residuals = jnp.sum(targets * log_probs, axis=1)
        ce_loss = -jnp.mean(residuals)
        return ce_loss


    @jax.jit
    def ce_logbarrier(params, features, targets, epsilon=1e-3):
        probs = predict_fn(params, features)
        log_probs = jnp.log(probs + epsilon)

        residuals1 = -jnp.sum(targets * log_probs, axis=1)
        residuals2 = 1 / (jnp.sum(targets * probs, axis=1) + epsilon)

        ce_loss = jnp.mean(residuals1 + residuals2)
        return ce_loss


    @jax.jit
    def accuracy(params, X, Y_true):
        # b x C
        logits = predict_fn(params, X)

        # b x 1
        predicted_classes = jnp.argmax(logits, axis=1)
        correct_predictions = predicted_classes == Y_true

        # scalar
        return jnp.mean(correct_predictions)


    def ce_binary(params, x, y):
        # b x 1
        logits = predict_fn(params, x)

        # b x 1
        losses = optax.sigmoid_binary_cross_entropy(logits.ravel(), y)

        # 1,
        # average over the batch
        return jnp.mean(losses)


    @jax.jit
    def accuracy_binary(params, X, Y_true):
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
                    is_egn_like = solver_type in ['egn', 'hfo', 'sgn', ] or solver_type.startswith('ignd')
                    b = config[0]

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
