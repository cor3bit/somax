import os.path
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

import jax
from jax import tree_map
import jax.numpy as jnp
import optax
from jaxopt import OptaxSolver

from benchmarks.utils.data_loader import load_data
from benchmarks.utils import model_zoo as zoo
from somax import EGN, HFO, SGN

TASK = 'california_housing'  # california_housing, diamonds, superconduct

GPU = True
JIT = True

# STEPS = 100_000
STEPS = 5_000
EVAL_EVERY = 50

N_SEEDS = 1

# grid
# 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001


SOLVER_HPS = {
    'california_housing': {
        # 'sgd': {
        #     'b': [64, ],
        #     'lr': [0.05, ],
        # },

        # 'adam': {
        #     'b': [64, ],
        #     'lr': [0.0005, 0.001, ],
        # },
        #
        # 'egn': {
        #     'b': [64, ],
        #     'lr': [0.1, ],
        #
        #     # ~fixed
        #     'reg': [1.0, ],
        #     'ls': [False, ],
        #     'al': [False, ],
        #     'm': [0.0, ],
        # },

        'hfo': {
            'b': [64, ],
            'maxcg': [20, ],
            'lr': [0.05, ],
            'reg': [1.0, ],
        },

        # 'sgn': {
        #     'b': [64, ],
        #     'maxcg': [10, ],
        #     'lr': [0.1, ],
        #     'reg': [1.0, ],
        # },
    },

    'superconduct': {
        # 'sgd': {
        #     'b': [64, ],
        #     'lr': [0.0003, ],
        # },

        'adam': {
            'b': [256, ],
            'lr': [0.01, 0.005, ],  # 0.005,
        },

        'egn': {
            'b': [256, ],
            'lr': [0.05, ],

            # ~fixed
            'reg': [1.0, ],
            'ls': [False, ],
            'al': [False, ],
            'm': [0.0, ],
        },
    },

    'diamonds': {
        # 'sgd': {
        #     'b': [32, 64, 128, ],
        #     'lr': [0.0000003, 0.0000001, ],
        # },
        #
        'adam': {
            'b': [128, ],
            'lr': [0.0001, 0.00005, ],  # 0.005, 0.01, 0.05
        },

        'egn': {
            'b': [128, ],
            'lr': [0.001, 0.005, ],  # 0.007,
            'reg': [1.0, ],
            'ls': [False, True, ],
            'al': [False, ],
            'm': [0.0, ],
        },
    },
}


def create_solver(
        solver_type,
        solver_config,
        w_params,
        X_train,
        y_train,
):
    opt_params = tree_map(lambda x: x, w_params)

    if solver_type == 'sgd':
        b, lr = solver_config
        solver_id = f'{solver_type}_b{b}_lr{lr}'
        solver = OptaxSolver(mse, opt=optax.sgd(lr))
        opt_state = solver.init_state(opt_params, X_train[0:5], y_train[0:5])
    elif solver_type == 'adam':
        b, lr = solver_config
        solver_id = f'{solver_type}_b{b}_lr{lr}'
        solver = OptaxSolver(mse, opt=optax.adam(lr))
        opt_state = solver.init_state(opt_params, X_train[0:5], y_train[0:5])
    elif solver_type == 'egn':
        b, lr, reg, ls, al, m = solver_config

        ls_str = 'Y' if ls else 'N'
        al_str = 'Y' if al else 'N'

        solver_id = f'{solver_type}_b{b}_lr{lr}_reg{reg}_{ls_str}_{al_str}_m{m}'

        solver = EGN(
            predict_fun=model_fn,
            loss_type='mse',
            learning_rate=lr,
            regularizer=reg,
            line_search=ls,
            adaptive_lambda=al,
            momentum=m,
            # beta2=m[1],
            batch_size=b,
            # total_iterations=STEPS,
        )
        opt_state = solver.init_state(opt_params)
    elif solver_type == 'hfo':
        b, maxcg, lr, reg = solver_config

        # ls_str = 'Y' if ls else 'N'
        # al_str = 'Y' if al else 'N'

        # solver_id = f'{solver_type}_b{b}_lr{lr}_reg{reg}_{ls_str}_{al_str}_m{m}'
        solver_id = f'{solver_type}_b{b}_cg{maxcg}_lr{lr}_reg{reg}'

        solver = HFO(
            loss_fun=mse,
            maxcg=maxcg,
            learning_rate=lr,
            regularizer=reg,
            # adaptive_lambda=al,
            batch_size=b,
        )
        opt_state = solver.init_state(opt_params)
    elif solver_type == 'sgn':
        b, maxcg, lr, reg = solver_config

        # ls_str = 'Y' if ls else 'N'
        # al_str = 'Y' if al else 'N'
        #
        # solver_id = f'{solver_type}_b{b}_lr{lr}_reg{reg}_{ls_str}_{al_str}_m{m}'

        solver_id = f'{solver_type}_b{b}_cg{maxcg}_lr{lr}_reg{reg}'

        solver = SGN(
            predict_fun=model_fn,
            loss_type='mse',
            maxcg=maxcg,
            learning_rate=lr,
            regularizer=reg,
            # adaptive_lambda=al,
            batch_size=b,
        )
        opt_state = solver.init_state(opt_params)
    else:
        raise ValueError(f'Unknown solver: {solver_type}')

    update_fn = jax.jit(solver.update)

    return update_fn, opt_params, opt_state, solver_id


def plot_metric(results, results_lambda):
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
    ax.set_ylabel(f'RMSE on Test Set')
    ax.set_xlabel('Wall Time (s)')
    ax.legend()
    # ax.set_yscale('log')
    # ax.set_xlabel(f'Environment Steps ($\\times {scale_str}%$)')
    # ax.set_title(env_name)

    if TASK == 'california_housing':
        ax.set_ylim(None, 1.1)
    elif TASK == 'superconduct':
        ax.set_ylim(None, 20.1)

    ax.set_title(f'{TASK}, Iterations={STEPS}')

    # modifier = os.path.basename(__file__).replace('.py', '')
    foldname = os.path.join('..', 'artifacts', 'examples', TASK)
    if not os.path.exists(foldname):
        os.makedirs(foldname)

    fname = os.path.join(foldname, f'{TASK}.png')
    fig.savefig(fname, bbox_inches='tight', dpi=600)

    # Lambda(t)
    # if False:
    # if results_lambda:
    #     fig, ax = plt.subplots()
    #
    #     for (optimizer_name, seed), values in results_lambda.items():
    #         if optimizer_name.startswith('egn'):
    #             times, run_values = zip(*values)
    #             ax.plot(times, run_values, label=f'{optimizer_name}', linewidth=1)
    #
    #     fname = os.path.join(foldname, f'lambda.png')
    #     fig.savefig(fname, bbox_inches='tight', dpi=600)


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
    def mse(params, X, y):
        residuals = y - model_fn(params, X)
        return 0.5 * jnp.mean(jnp.square(residuals))


    @jax.jit
    def rmse(params, X, y):
        residuals_ = y - model_fn(params, X)
        mse_ = jnp.mean(jnp.square(residuals_))
        return jnp.sqrt(mse_)


    @jax.jit
    def mape(params, X, y):
        predictions = model_fn(params, X)
        percentage_errors = jnp.abs((y - predictions) / jnp.abs(y + 1e-8))
        return jnp.mean(percentage_errors)


    # --------------- dataset & models ---------------
    print(f'Running task: {TASK}')

    task_hps = SOLVER_HPS[TASK]
    configs = create_configs(task_hps)
    total_size = calc_size(STEPS, configs)

    # model definition
    model = zoo.MLPRegressorMedium()

    model_fn = jax.jit(model.apply)

    results = {}
    results_lambda = {}

    with tqdm(total=total_size) as pbar:
        for solver_type, solver_params in task_hps.items():
            for config in configs[solver_type]:
                for seed in range(N_SEEDS):
                    # --------------- random part ---------------
                    # ! RANDOMNESS (1): the seed is used to split and shuffle the dataset
                    (X_train, X_test, Y_train, Y_test), is_clf, n_classes = load_data(
                        dataset_id=TASK, test_size=0.1, seed=seed)

                    # ! RANDOMNESS (2): the seed is used to initialize the model
                    params = model.init(jax.random.PRNGKey(seed), X_train[0])

                    # --------------- init solver ---------------
                    is_egn_like = solver_type in ['egn', 'hfo', 'sgn', ]
                    b = config[0]

                    update_fn, opt_params, opt_state, solver_id = create_solver(
                        solver_type, config, params, X_train, Y_train)

                    results[(solver_id, seed)] = []
                    results_lambda[(solver_id, seed)] = []

                    # --------------- warm up ---------------
                    n_warmup = 2
                    fake_params = jax.tree_map(lambda _: _, params)
                    fake_opt_state = jax.tree_map(lambda _: _, opt_state)
                    for i in range(n_warmup):
                        # on Test
                        fake_loss3 = rmse(fake_params, X_test, Y_test)
                        fake_loss4 = mape(fake_params, X_test, Y_test)

                        # on Batch
                        batch_X = X_train[i * b:(i + 1) * b, :]
                        batch_y = Y_train[i * b:(i + 1) * b]

                        fake_loss2 = mse(fake_params, batch_X, batch_y)
                        fake_preds = model_fn(fake_params, batch_X)

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
                                loss = rmse(opt_params, X_test, Y_test)

                                # first time is zero
                                if not results[(solver_id, seed)]:
                                    start_time = time.time()
                                    results[(solver_id, seed)].append((0, loss))
                                else:
                                    delta_t = time.time() - start_time
                                    results[(solver_id, seed)].append((delta_t, loss))

                                # also log lambda
                                # if solver_id.startswith('egn'):
                                #     results_lambda[(solver_id, seed)].append((step, opt_state.regularizer))

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
    plot_metric(results, results_lambda)
