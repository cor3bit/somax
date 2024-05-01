import os
import sqlite3
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import wandb

METRICS = {
    # SL
    "california_housing": "test_loss",
    "diamonds": "test_loss",
    "superconduct": "test_loss",

    "imdb_reviews": "test_loss",
    "covtype": "test_loss",
    "sensit": "test_loss",

    "mnist": "test_loss",
    "fashion_mnist": "test_loss",
    "wine_quality": "test_loss",
    "cifar10": "test_loss",

    # RL
    "BDT1": "charts/value_fn_diff",
    "AC1": "charts/value_fn_diff",

    "Acrobot-v1": "charts/episodic_return",
    "CartPole-v1": "charts/episodic_return",

    "Asterix-v1": "charts/episodic_return",
    "Breakout-v1": "charts/episodic_return",
    "Freeway-v1": "charts/episodic_return",
    "Seaquest-v1": "charts/episodic_return",

    "PongNoFrameskip-v4": "charts/episodic_return",
    "BeamRiderNoFrameskip-v4": "charts/episodic_return",
    "BreakoutNoFrameskip-v4": "charts/episodic_return",
}


def process_run(run, conn, already_exists):
    task_id = run.config["task_id"]
    if task_id not in METRICS:
        raise ValueError(f"Metric for task \'{task_id}\' not found.")

    metric = METRICS[task_id]
    run_name = run.name

    # extract data from runs (Web request)
    df = run.history(keys=[metric, '_timestamp'], samples=100_000)

    # add a time difference column instead of absolute timestamps
    df[f'Time'] = df['_timestamp'] - df['_timestamp'].iloc[0]

    # loss_df[f'time_{seed}'] = pd.to_datetime(loss_df['_timestamp'])
    df = df.drop(columns=['_timestamp'])

    # rename metric column to a unified name
    df = df.rename(columns={metric: f'Value', '_step': f'Step'})
    df['Run'] = run_name
    df['Metric'] = metric

    # if a given run already exists in DB
    if already_exists:
        existing_step_df = pd.read_sql_query("SELECT * FROM Step WHERE Run = ?", conn, params=[run_name])
        # TODO compare
        dfs_equal = False
        if not dfs_equal:
            print(f"Run \'{run_name}\' already exists in DB. Skipping.")
    else:
        # save to DB: Table "Step"
        df.to_sql('Step', conn, if_exists='append', index=False)


def create_tables(conn):
    # create a cursor object using the cursor() method
    cursor = conn.cursor()

    # SQL command to create the first table
    step_table_cmd = """
    CREATE TABLE IF NOT EXISTS Step (
        Run TEXT,
        Metric TEXT,
        Step INTEGER,
        Time REAL,
        Value REAL
    );
    """
    cursor.execute(step_table_cmd)

    # SQL command to create the second table
    run_table_cmd = """
    CREATE TABLE IF NOT EXISTS Run (
        Run TEXT PRIMARY KEY,
        task_id TEXT,
        seed INTEGER,
        optimizer TEXT,
        batch_size INTEGER,
        learning_rate REAL,
        maxcg INTEGER,
        line_search BOOLEAN,
        reset_option TEXT,
        adaptive_lambda BOOLEAN,
        reg_lambda REAL,
        regularizer_eps REAL,
        momentum REAL,
        total_timesteps INTEGER,
        buffer_size INTEGER,
        gamma REAL,
        tau REAL,
        target_network_frequency INTEGER,
        start_e REAL,
        end_e REAL,
        exploration_fraction REAL,
        learning_starts INTEGER,
        train_frequency INTEGER,
        frame_stacking INTEGER,
        track BOOLEAN,
        wandb_project_name TEXT,
        wandb_entity TEXT,
        capture_video BOOLEAN,
        save_model BOOLEAN,
        debug BOOLEAN,
        dt REAL,
        deterministic BOOLEAN,
        reset_noise_sigma REAL,
        reset_every_n INTEGER,
        update_policy_every_n INTEGER,
        action_noise_sigma REAL,
        test_size REAL,
        n_epochs INTEGER,
        max_steps INTEGER,
        evaluate_every_n INTEGER
    );
    """
    cursor.execute(run_table_cmd)

    # commit & close
    conn.commit()
    cursor.close()


def create_db(selected_runs, verbose):
    # load runs (Web request)
    print("Retrieving runs from W&B...")

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key is None:
        raise ValueError("W&B API key not found. Please check .env for the WANDB_API_KEY variable.")

    api = wandb.Api(api_key=api_key)
    runs = api.runs("dysco/egn", filters=selected_runs if selected_runs else None)
    print(f"Found {len(runs)} runs.")
    if not runs:
        print("No runs found. Exiting.")
        return

    # connect to SQLite database (will be created if not exists)
    print("Loading local DB...")
    db_folder = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'db')
    os.makedirs(db_folder, exist_ok=True)
    db_path = os.path.join(db_folder, 'wandb.db')
    conn = sqlite3.connect(db_path)
    create_tables(conn)

    print("Processing runs...")

    # cache runs already saved in the DB
    existing_runs = set(pd.read_sql_query("SELECT Run FROM Run", conn)['Run'].values)

    run_configs = {}

    for run in tqdm(runs):
        # save to DB: Table "Step"
        try:
            # TODO also verify integrity of the Step table
            already_exists = run.name in existing_runs
            if already_exists:
                if verbose:
                    print(f"Run \'{run.name}\' already exists in DB. Skipping.")

                continue

            process_run(run, conn, already_exists)

            run_configs[run.name] = run.config
        except Exception as e:
            print(f"WARNING: {e}. Run \'{run.name}\' skipped.")

    # save to DB: Table "Run"
    pd.DataFrame.from_dict(run_configs, orient='index').to_sql(
        'Run', conn, if_exists='append', index=True, index_label='Run')

    # DB: close connection
    conn.close()


# ------------------------------------------------------------

if __name__ == '__main__':
    # load the necessary API keys from .env
    load_dotenv()

    # settings
    verbose = False

    selected_runs = {
        "$and": [
            {"state": "Finished"},
            # {"$or": [
            #     # {"config.task_id": "california_housing"},
            #     # {"config.task_id": "diamonds"},
            #     # {"config.task_id": "Freeway-v1"},
            #     # {"config.task_id": "BeamRiderNoFrameskip-v4"},
            #     {"config.task_id": "BDT1"},
            # ]},
            # # {"config.total_timesteps": total_timesteps},
            # {"config.max_steps": 5000},
            # {"config.optimizer": 'egn'},
            # {"config.batch_size": 64},
            # {"config.learning_rate": 0.1},
            # {"config.reg_lambda": 1.0},
        ]
    }

    # run DB creation
    create_db(selected_runs, verbose)
