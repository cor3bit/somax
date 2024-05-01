import os
from collections import OrderedDict

import yaml
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import sqlite3

CHART_SETTINGS = {
    # Regression
    'california_housing': {
        # iterp for wall time
        'iterp_points': 100,

        # naming
        'title': 'California Housing',
        'y_axis': 'Test Set RMSE',

        # constraints on X and Y axes
        'time_limit': 7.4,
        'y_limiter': (0.4, 1.1),
    },

    'superconduct': {
        # iterp for wall time
        'iterp_points': 100,

        # naming
        'title': 'Superconductivity',
        'y_axis': 'Test Set RMSE',

        # constraints on X and Y axes
        'time_limit': 7.4,
        'y_limiter': (10.0, 21.0),
    },

    'diamonds': {
        # iterp for wall time
        'iterp_points': 100,

        # naming
        'title': 'Diamonds',
        'y_axis': 'Test Set RMSE',

        # constraints on X and Y axes
        'time_limit': 21.1,
        'y_limiter': (500, 6000),
    },

    # Classification
    'imdb_reviews': {
        # iterp for wall time
        'iterp_points': 100,

        # naming
        'title': 'IMDB Reviews',
        'y_axis': 'Test Set Accuracy',

        # constraints on X and Y axes
        'time_limit': 21.1,
        'y_limiter': (0.6, 0.85),
    },
    'covtype': {
        # iterp for wall time
        'iterp_points': 100,

        # naming
        'title': 'Covertype',
        'y_axis': 'Test Set Accuracy',

        # constraints on X and Y axes
        'y_limiter': (0.4, 1.1),
        'time_limit': 11.1,
    },
    'sensit': {
        # iterp for wall time
        'iterp_points': 100,

        # naming
        'title': 'SensIT',
        'y_axis': 'Test Set Accuracy',

        # constraints on X and Y axes
        'y_limiter': (0.7, 0.9),
        'time_limit': 11.1,
    },

    'wine_quality': {
        'iterp_points': 100,

        'title': 'Wine Quality',
        'y_axis': 'Test Set Accuracy',

        'time_limit': 19.0,
        'y_limiter': (None, None),
    },
    'mnist': {
        'iterp_points': 100,

        'title': 'MNIST',
        'y_axis': 'Test Set Accuracy',

        'time_limit': 26.0,
        'y_limiter': (None, None),
    },
    'fashion_mnist': {
        'iterp_points': 100,

        'title': 'Fashion MNIST',
        'y_axis': 'Test Set Accuracy',

        'time_limit': 51.0,
        'y_limiter': (None, None),
    },

    'cifar10': {
        'iterp_points': 100,

        'title': 'CIFAR10',
        'y_axis': 'Test Set Accuracy',

        'time_limit': 61.0,
        'y_limiter': (None, None),
    },

    # LQR
    'BDT1': {
        'iterp_points': 100,

        'title': 'BDT',
        'y_axis': r'$\left\Vert \mathbf{H}^{*}-\mathbf{H}_{i}\right\Vert $',

        'time_limit': 15.0,
        'y_limiter': (0, 90),
    },
    'AC1': {
        'iterp_points': 100,

        'title': 'UAV',
        'y_axis': r'$\left\Vert \mathbf{H}^{*}-\mathbf{H}_{i}\right\Vert $',

        'time_limit': 17.0,
        'y_limiter': (0, 90),
    },

    # Classic Control
    'Acrobot-v1': {
        'iterp_points': 100,

        'title': 'Acrobot-v1',
        'y_axis': 'Episodic Return',

        'time_limit': 260.0,
        'y_limiter': (-550.0, 20.0),
        'x_scaler': 60.0,  # min
    },
    'CartPole-v1': {
        'iterp_points': 100,

        'title': 'CartPole-v1',
        'y_axis': 'Episodic Return',

        'time_limit': 505.0,
        'y_limiter': (None, None),
    },

    # MinAtar
    'Asterix-v1': {
        'iterp_points': 100,

        'title': 'Asterix-v1',
        'y_axis': 'Episodic Return',

        'time_limit': 3060.0,  # 51 min
        'y_limiter': (None, None),
        'x_scaler': 60.0,  # min
    },
    'Breakout-v1': {
        'iterp_points': 100,

        'title': 'Breakout-v1',
        'y_axis': 'Episodic Return',

        'time_limit': 3060.0,  # 51 min
        'y_limiter': (None, None),
        'x_scaler': 60.0,  # min
    },
    'Freeway-v1': {
        'iterp_points': 100,

        'title': 'Freeway-v1',
        'y_axis': 'Episodic Return',

        'time_limit': 3500.0,  # 51 min
        'y_limiter': (None, 65.0),
        'x_scaler': 60.0,  # min
    },
    'Seaquest-v1': {
        'iterp_points': 100,

        'title': 'Seaquest-v1',
        'y_axis': 'Episodic Return',

        'time_limit': 3600.0,  # 51 min
        'y_limiter': (None, 80),
        'x_scaler': 60.0,  # min
    },

    # Atari
    'BeamRiderNoFrameskip-v4': {
        'iterp_points': 100,

        'title': 'BeamRiderNoFrameskip-v4',
        'y_axis': 'Episodic Return',

        'time_limit': 6060.0,
        'y_limiter': (0, 8050),
        'x_scaler': 60.0,  # min
    },
    'PongNoFrameskip-v4': {
        'iterp_points': 100,

        'title': 'PongNoFrameskip-v4',
        'y_axis': 'Episodic Return',

        'time_limit': 3060.0,
        'y_limiter': (None, None),
        'x_scaler': 60.0,  # min
    },
}
PATH_CHARTS = os.path.join(os.path.dirname(__file__), "..", "artifacts", "charts")


# --------------------- DB TOOLS ---------------------
def get_connection():
    db_folder = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'db')
    db_path = os.path.join(db_folder, 'wandb.db')
    conn = sqlite3.connect(db_path)
    return conn


def load_runs(task_id, opt_id, opt_params, conn):
    # get initial df
    query = """
        select * from Run
        where task_id = ?
          and batch_size = ?
          and optimizer = ?
          and learning_rate = ?
        """

    if opt_id == 'egn':
        query += '\n and line_search = ? \n and reg_lambda = ? \n and momentum = ?'
        params = [
            task_id,
            opt_params['batch_size'],
            opt_id,
            opt_params['learning_rate'],
            opt_params['line_search'],
            opt_params['reg_lambda'],
            opt_params['momentum'],
        ]
    elif opt_id in ['hfo', 'sgn']:
        query += '\n and maxcg = ? \n and reg_lambda = ?'

        params = [
            task_id,
            opt_params['batch_size'],
            opt_id,
            opt_params['learning_rate'],
            opt_params['maxcg'],
            opt_params['reg_lambda'],
        ]
    else:
        params = [
            task_id,
            opt_params['batch_size'],
            opt_id,
            opt_params['learning_rate'],
        ]

    max_steps = opt_params.get('max_steps')
    if max_steps is not None:
        query += '\n and max_steps = ?'
        params.append(max_steps)

    total_timesteps = opt_params.get('total_timesteps')
    if total_timesteps is not None:
        query += '\n and total_timesteps = ?'
        params.append(total_timesteps)

    run_df = pd.read_sql(sql=query, con=conn, params=params)

    # analysis
    n_rows = run_df.shape[0]
    if n_rows == 0:
        print(f"No runs found for {opt_id} with params {opt_params}")
        return []

    run_names = run_df['Run'].values.tolist()

    # if same seeds
    n_seeds = run_df['seed'].nunique()
    if n_rows != n_seeds:
        print(f"Multiple seeds found for {opt_id} with params {opt_params}")

    return run_names


def load_steps(run_names, conn):
    placeholders = ', '.join(['?'] * len(run_names))
    query = f"SELECT Run, Step, Time, Value FROM Step WHERE Run IN ({placeholders})"
    step_df = pd.read_sql(sql=query, con=conn, params=run_names)
    return step_df


def build_query(task_id, opt_id, n_steps, is_sl):
    # SELECT
    query = 'SELECT DISTINCT batch_size, learning_rate'
    if opt_id == 'egn':
        query += ', line_search, reg_lambda, momentum'
    elif opt_id in ['hfo', 'sgn']:
        query += ', maxcg, reg_lambda'

    # FROM
    query += '\nFROM Run\n'

    # WHERE
    query += f"WHERE task_id = \"{task_id}\"\n"

    if is_sl:
        query += f"  AND max_steps = \"{n_steps}\"\n"
    else:
        query += f"  AND total_timesteps = \"{n_steps}\"\n"

    query += f"  AND optimizer = \"{opt_id}\"\n"

    return query


def load_trials_with_filter(db_filter, conn):
    settings = []

    cursor = conn.cursor()

    is_sl = 'max_steps' in db_filter
    assert is_sl or 'total_timesteps' in db_filter, "Either max_steps or total_timesteps must be provided."

    n_steps = db_filter.get('max_steps') if is_sl else db_filter.get('total_timesteps')

    task_id = db_filter['task_id']

    try:
        for opt_id in db_filter['optimizer']:
            # load unique from DB
            query = build_query(task_id, opt_id, n_steps, is_sl)
            cursor.execute(query)
            results = cursor.fetchall()

            # load all runs
            if opt_id in ['sgd', 'adam']:

                for result in results:
                    b, lr = result[0], result[1]

                    # apply filtering
                    if 'batch_size' in db_filter:
                        if b not in db_filter['batch_size']:
                            continue

                    if 'learning_rate' in db_filter:
                        if lr not in db_filter['learning_rate']:
                            continue

                    opt_params = {
                        'batch_size': b,
                        'learning_rate': lr,
                    }
                    if is_sl:
                        opt_params['max_steps'] = n_steps
                    else:
                        opt_params['total_timesteps'] = n_steps

                    trial_id = f'{opt_id}_b{b}_lr{lr}'

                    settings.append((trial_id, opt_id, opt_params))

            elif opt_id == 'egn':
                for result in results:
                    b, lr, ls, reg, m = result[0], result[1], result[2], result[3], result[4]

                    # apply filtering
                    if 'batch_size' in db_filter:
                        if b not in db_filter['batch_size']:
                            continue

                    if 'learning_rate' in db_filter:
                        if lr not in db_filter['learning_rate']:
                            continue

                    if 'line_search' in db_filter:
                        if ls not in db_filter['line_search']:
                            continue

                    if 'reg_lambda' in db_filter:
                        if reg not in db_filter['reg_lambda']:
                            continue

                    if 'momentum' in db_filter:
                        if m not in db_filter['momentum']:
                            continue

                    opt_params = {
                        'batch_size': b,
                        'learning_rate': lr,
                        'line_search': ls,
                        'reg_lambda': reg,
                        'momentum': m,
                    }
                    if is_sl:
                        opt_params['max_steps'] = n_steps
                    else:
                        opt_params['total_timesteps'] = n_steps

                    ls_str = 'Y' if ls else 'N'
                    trial_id = f'{opt_id}_b{b}_lr{lr}_reg{reg}_{ls_str}_m{m}'

                    settings.append((trial_id, opt_id, opt_params))

            elif opt_id in ['hfo', 'sgn']:
                for result in results:
                    b, lr, mxcg, reg = result[0], result[1], result[2], result[3]

                    # apply filtering
                    if 'batch_size' in db_filter:
                        if b not in db_filter['batch_size']:
                            continue

                    if 'learning_rate' in db_filter:
                        if lr not in db_filter['learning_rate']:
                            continue

                    if 'maxcg' in db_filter:
                        if mxcg not in db_filter['maxcg']:
                            continue

                    if 'reg_lambda' in db_filter:
                        if reg not in db_filter['reg_lambda']:
                            continue

                    opt_params = {
                        'batch_size': b,
                        'learning_rate': lr,
                        'maxcg': mxcg,
                        'reg_lambda': reg,
                    }
                    if is_sl:
                        opt_params['max_steps'] = n_steps
                    else:
                        opt_params['total_timesteps'] = n_steps

                    trial_id = f'{opt_id}_b{b}_lr{lr}_cg{mxcg}_reg{reg}'

                    settings.append((trial_id, opt_id, opt_params))

            else:
                raise ValueError(f'Unknown optimizer: {opt_id}')
    except Exception as e:
        print(f'Error: {e}.')
    finally:
        cursor.close()

    return settings


def load_processed_trials(db_filter, conn):
    trial_configs = load_trials_with_filter(db_filter, conn)

    task_id = db_filter['task_id']

    optimizer_dfs = []

    for trial_id, opt_id, opt_params in trial_configs:
        # collect run names, initial analysis
        run_names = load_runs(task_id, opt_id, opt_params, conn)
        if not run_names:
            continue

        # load steps
        step_df = load_steps(run_names, conn)

        print(f"{trial_id}: loaded {len(run_names)} Runs with {step_df.shape[0]} Steps In total.")

        optimizer_dfs.append((trial_id, step_df))

    return optimizer_dfs


def load_processed_runs(task_id, settings, conn):
    optimizer_to_df_map = {}

    for opt_id, opt_params in settings.items():
        # collect run names, initial analysis
        run_names = load_runs(task_id, opt_id, opt_params, conn)
        if not run_names:
            continue

        # load steps
        step_df = load_steps(run_names, conn)

        print(f"{opt_id}: loaded {len(run_names)} Runs with {step_df.shape[0]} Steps In total.")

        optimizer_to_df_map[opt_id] = step_df

    return optimizer_to_df_map


# --------------------- CHARTING TOOLS ---------------------
def get_plot_skeleton(task_ids, ignore_time_limit, ignore_loss_limit, dims=(1, 1), figsize=None, skip_y_axis=()):
    plt.style.use('bmh')

    # canvas
    fig, axs = plt.subplots(
        nrows=dims[0],
        ncols=dims[1],
        figsize=figsize,
        dpi=600,
    )

    assert dims[0] == 1, "Only one row of charts is supported."

    for i, task_id in enumerate(task_ids):
        if task_id not in CHART_SETTINGS:
            raise ValueError(f"Chart settings for task \'{task_id}\' not found.")

        task_chart_settings = CHART_SETTINGS[task_id]

        # selected axes
        if dims[0] == 1 and dims[1] == 1:
            ax = axs
        else:
            ax = axs[i]

        # naming
        ax.set_title(task_chart_settings['title'], fontsize=11)

        if 'x_scaler' not in task_chart_settings:
            ax.set_xlabel(f'Wall Time (sec)', fontsize=10)
        else:
            x_scale = task_chart_settings['x_scaler']
            if x_scale == 60.0:
                formatter = FuncFormatter(lambda x, pos: f'{x / x_scale:.0f}')
                ax.xaxis.set_major_formatter(formatter)
                ax.set_xlabel(f'Wall Time (min)', fontsize=10)
            else:
                raise ValueError(f"Unknown x_scaler: {x_scale}")

        if i not in skip_y_axis:
            ax.set_ylabel(task_chart_settings['y_axis'], fontsize=10)

        # limits on X and Y axes
        if not ignore_loss_limit:
            limiter_y = task_chart_settings.get('y_limiter')
            ax.set_ylim(*limiter_y)

        if not ignore_time_limit:
            limiter_x = task_chart_settings.get('time_limit')
            ax.set_xlim(None, limiter_x)

    fig.tight_layout()

    return fig, axs


def interpolate_data(df, time_limit, n_points):
    # prep df
    distinct_run_names = set(x.replace('_Time', '') for x in df.columns if 'Time' in x)
    time_cols = [x for x in df.columns if 'Time' in x]
    val_cols = [x for x in df.columns if 'Value' in x]
    n_seeds = len(time_cols)

    # sanity check
    assert len(time_cols) == len(val_cols), "Number of time and value columns must be the same."
    assert len(time_cols) == len(distinct_run_names), "Number of time and value columns must be the same."

    # --------- Approach 1: interpolate each run separately
    common_time_points = np.linspace(0, time_limit, n_points)
    interpolated_values = np.zeros((n_points, n_seeds))
    for i, run_name in enumerate(distinct_run_names):
        times = df[f'{run_name}_Time'].values
        values = df[f'{run_name}_Value'].values

        interp_fn = interp1d(times, values, kind='linear',
                             bounds_error=True)  # bounds_error=False, fill_value=values[-1]
        interpolated_values[:, i] = interp_fn(common_time_points)

    # --------- Approach 2: interpolate all runs together
    # TODO takes too long
    # common_time_points = np.linspace(0, time_limit, n_points)
    # interpolated_values = griddata(df[time_cols].values, df[val_cols].values,
    #                                common_time_points, method='linear')

    return common_time_points, interpolated_values


def add_trial_to_chart(task_id, trial_id, step_df, ax, ignore_time_limit):
    # convert to stacked columns df
    step_df['ID'] = step_df.groupby('Run').cumcount()
    pivoted_df = step_df.pivot(index='ID', columns='Run')
    pivoted_df.columns = ['_'.join([str(col), str(run)]) for run, col in pivoted_df.columns]
    full_df = pivoted_df.reset_index(drop=True)

    # always assumes Time vs Loss chart
    df = full_df[[x for x in full_df.columns if 'Time' in x or 'Value' in x]]

    # time limit for interpolation
    shortest_time_across_runs = df[[x for x in df.columns if 'Time' in x]].max().min()
    if ignore_time_limit:
        time_limit = shortest_time_across_runs
    else:
        prescribed_time_limit = CHART_SETTINGS[task_id]['time_limit']
        if prescribed_time_limit > shortest_time_across_runs:
            time_limit = shortest_time_across_runs
        else:
            time_limit = prescribed_time_limit

    # interpolate data to get values for the common time points
    n_points = CHART_SETTINGS[task_id]['iterp_points']
    common_time_points, interpolated_values = interpolate_data(df, time_limit, n_points)

    # average
    avg_value = interpolated_values.mean(axis=1)
    # avg_time = np.linspace(0, shortest_time_across_runs, n_points)

    ax.plot(common_time_points, avg_value, label=f'{trial_id}', linewidth=1)

    std_res = interpolated_values.std(axis=1)
    ax.fill_between(common_time_points, avg_value - std_res, avg_value + std_res, alpha=0.25)


# --------------------- glue code ---------------------


def create_chart_paper_sl(save_on_disk):
    print('Building SL chart for paper...')

    # fixed hyper-parameters
    hp_path = os.path.join(os.path.dirname(__file__), "..", "config", "hyperparams.yaml")
    with open(hp_path, 'r') as file:
        task_to_hps_map = yaml.safe_load(file)

    # fixed selection of tasks and order
    # selected_tasks = ['california_housing', 'superconduct', 'diamonds', 'mnist', ]
    selected_tasks = ['california_housing', 'superconduct', 'diamonds', 'imdb_reviews', ]

    # initialize canvas
    fig, axs = get_plot_skeleton(
        selected_tasks,
        ignore_time_limit=False,
        ignore_loss_limit=False,
        dims=(1, 4),
        figsize=(10, 2.5),
        skip_y_axis=(1, 2),
    )

    # connect to the DB
    conn = get_connection()

    # try-catch block to close the connection safely
    try:
        # load and form the needed pandas dfs
        data_cache = {}
        for i, task_id in enumerate(selected_tasks):
            print(f"\nProcessing task: {task_id}")

            settings = task_to_hps_map[task_id]
            optimizer_to_df_map = load_processed_runs(task_id, settings, conn)
            data_cache[task_id] = optimizer_to_df_map

            # sort items in the dict in order to have the same order of lines in the chart
            # i.e. EGN is always a red line, ADAM is blue, etc.
            optimizer_to_df_map = OrderedDict(sorted(optimizer_to_df_map.items(), key=lambda x: x[0]))

            for opt_id, step_df in optimizer_to_df_map.items():
                add_trial_to_chart(task_id, opt_id, step_df, axs[i], ignore_time_limit=False)

    except Exception as e:
        print(f'Error: {e}.')
    finally:
        conn.close()

    # finish the chart
    # legend magic
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.09))

    if save_on_disk:
        fname = os.path.join(PATH_CHARTS, 'fig_sl.png')
        fig.savefig(fname, bbox_inches='tight', dpi=600)
    else:
        plt.tight_layout()
        plt.show()


def create_chart_paper_rl(save_on_disk):
    print('Building RL chart for paper...')

    # fixed hyper-parameters
    hp_path = os.path.join(os.path.dirname(__file__), "..", "config", "hyperparams.yaml")
    with open(hp_path, 'r') as file:
        task_to_hps_map = yaml.safe_load(file)

    # fixed selection of tasks and order
    selected_tasks = ['BDT1', 'AC1', 'Acrobot-v1', 'Freeway-v1', ]

    # initialize canvas
    fig, axs = get_plot_skeleton(
        selected_tasks,
        ignore_time_limit=False,
        ignore_loss_limit=False,
        dims=(1, 4),
        figsize=(10, 2.5),
        skip_y_axis=(1, 3),
    )

    # connect to the DB
    conn = get_connection()

    # try-catch block to close the connection safely
    try:
        # load and form the needed pandas dfs
        data_cache = {}
        for i, task_id in enumerate(selected_tasks):
            print(f"\nProcessing task: {task_id}")

            settings = task_to_hps_map[task_id]
            optimizer_to_df_map = load_processed_runs(task_id, settings, conn)
            data_cache[task_id] = optimizer_to_df_map

            # sort items in the dict in order to have the same order of lines in the chart
            # i.e. EGN is always a red line, ADAM is blue, etc.
            optimizer_to_df_map = OrderedDict(sorted(optimizer_to_df_map.items(), key=lambda x: x[0]))

            for opt_id, step_df in optimizer_to_df_map.items():
                add_trial_to_chart(task_id, opt_id, step_df, axs[i], ignore_time_limit=False)

    except Exception as e:
        print(f'Error: {e}.')
    finally:
        conn.close()

    # finish the chart
    # legend magic
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.09))

    if save_on_disk:
        fname = os.path.join(PATH_CHARTS, 'fig_rl.png')
        fig.savefig(fname, bbox_inches='tight', dpi=600)
    else:
        plt.tight_layout()
        plt.show()


def create_hpopt_chart(db_filter, ignore_time_limit, ignore_loss_limit, save_on_disk):
    # connect to the DB
    conn = get_connection()

    try:
        # load data from the DB based on the filter
        optimizer_dfs = load_processed_trials(db_filter, conn)
        if not optimizer_dfs:
            raise ValueError("No data found for the given filter.")

        task_id = db_filter["task_id"]

        # initialize chart
        fig, ax = get_plot_skeleton([task_id], ignore_time_limit, ignore_loss_limit, dims=(1, 1))

        # !! each trial represents one line (with confidence bands) in the chart
        for trial_id, opt_df in optimizer_dfs:
            add_trial_to_chart(task_id, trial_id, opt_df, ax, ignore_time_limit)

        # legend magic
        # fig.legend()
        fig.legend(loc='lower right')

        # save chart on disk
        if save_on_disk:
            fname = os.path.join(PATH_CHARTS, f'hp_opt_{task_id}.png')
            fig.savefig(fname, bbox_inches='tight', dpi=600)
        else:
            plt.tight_layout()
            plt.show()

    finally:
        conn.close()


# ------------------------------------------------------------

if __name__ == '__main__':
    # Stand-alone chart
    ignore_time_limit = True
    ignore_loss_limit = True
    save_on_disk = False

    db_filter = {
        # 'task_id': 'california_housing',
        # 'task_id': 'diamonds',
        # 'task_id': 'superconduct',
        # 'task_id': 'covtype',
        # 'task_id': 'sensit',
        # 'task_id': 'imdb_reviews',

        # 'task_id': 'Acrobot-v1',
        # 'task_id': 'CartPole-v1',

        # 'task_id': 'Asterix-v1',
        # 'task_id': 'Breakout-v1',
        # 'task_id': 'Freeway-v1',
        # 'task_id': 'Seaquest-v1',

        # 'task_id': 'BeamRiderNoFrameskip-v4',
        'task_id': 'PongNoFrameskip-v4',
        # 'task_id': 'BDT1',
        # 'task_id': 'AC1',

        # 'max_steps': 50_000,
        # 'max_steps': 20_000,
        # 'max_steps': 10_000,

        # 'total_timesteps': 3_000_000,
        'total_timesteps': 2_000_000,
        # 'total_timesteps': 500_000,
        # 'total_timesteps': 200_000,
        # 'total_timesteps': 20_000,

        # 'batch_size': [128, ],  # Optional
        # 'batch_size': [64, ],  # Optional
        'batch_size': [32, ],  # Optional

        'optimizer': ['adam', 'egn'],  # Optional
        # 'optimizer': ['egn', 'adam', ],  # Optional
        # 'optimizer': ['hfo', 'sgn', 'egn', ],  # Optional

        # 'learning_rate': [3e-4, 0.3, ],  # Optional
        # 'learning_rate': [ 0.0007, 0.0005, ],  # Optional
        # 'learning_rate': [0.5, 0.3, 0.4, 0.6, ],  # Optional

        # 'line_search': [False, ],  # Optional
        # 'momentum': [0.0, ],  # Optional
        # 'maxcg': [3, 5, 10, ],  # Optional

    }

    create_hpopt_chart(db_filter, ignore_time_limit, ignore_loss_limit, save_on_disk)

    # # Paper SL chart x4
    # save_on_disk = True
    # create_chart_paper_sl(save_on_disk)

    # # Paper RL chart x4
    # save_on_disk = True
    # create_chart_paper_rl(save_on_disk)
