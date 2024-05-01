'''
Adapted from
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/benchmark.py
'''

import argparse
import os
import re
import shlex
import subprocess
from distutils.util import strtobool


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    # FIXED part
    parser.add_argument("--task-id", type=str, default="diabetes",
                        help="the id of the dataset")

    # Related to Supervised Learning
    parser.add_argument("--test-size", type=float, default=0.1,
                        help="the size of the test set")
    parser.add_argument("--n-epochs", type=int, default=1,
                        help="the number of epochs")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="the maximum number of steps")
    parser.add_argument("--evaluate-every-n", type=int, default=10,
                        help="evaluate performance every n-th step")

    # Related to RL (Common)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
                        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=1_000_000,
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
                        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=1_000,
                        help="the timesteps it takes to update the target network")
    parser.add_argument("--start-e", type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
                        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=80_000,
                        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
                        help="the frequency of training")

    # Related to RL (Atari)
    parser.add_argument('--frame-stacking', type=int, default=3,
                        help='the number of frames to stack for Atari games')

    # Related to RL (LQR)
    parser.add_argument("--dt", type=float, default=0.1,
                        help="the discretization time step")
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggle Ax+Bu vs. Ax+Bu+Gw")
    parser.add_argument("--reset-noise-sigma", type=float, default=1.,
                        help="sigma of the env reset")
    parser.add_argument("--reset-every-n", type=int, default=10,
                        help="the timesteps it takes to reset the env")
    parser.add_argument("--update-policy-every-n", type=int, default=100,
                        help="the timesteps it takes to update the policy")
    parser.add_argument("--action-noise-sigma", type=float, default=100.,
                        help="sigma of the exploratory action")

    # Logging
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="egn",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will write additional parameters at each step")

    # SEARCH SPACE
    # learning rates should be equal to the number of optimizers
    parser.add_argument('--optimizers', nargs='+', default=['sgd', 'adam', 'egn'],
                        help='the optimization algorithms to benchmark')

    parser.add_argument('--learning-rates', nargs='+', default=[3e-4, 3e-4, 0.1],
                        help='the optimization algorithms to benchmark')

    parser.add_argument("--batch-sizes", nargs='+', default=[32, 128],
                        help="the batch size of sample from the reply memory")

    parser.add_argument("--line-search", nargs='+', default=[False, ],
                        help="enable line search")

    # NOTE! LS, AL and reset-option are excluded from the search space
    parser.add_argument("--reset-option", type=str, default='increase',
                        help="the reset option for Line Search")
    parser.add_argument("--adaptive-lambda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="enables adaptive regularization for EGN")
    parser.add_argument("--regularizer-eps", type=float, default=1e-5,
                        help="the epsilon for the regularizer")

    # EGN specific
    parser.add_argument("--reg-lambdas", nargs='+', default=[1., ],
                        help="Levenberg-Marquardt style regularization")
    parser.add_argument("--momenta", nargs='+', default=[0, ],
                        help="the momentum of EGN")

    # HFO, SGN specific
    parser.add_argument("--maxcgs", nargs='+', default=[10, ],
                        help="the maximum number of CG iterations")

    parser.add_argument('--num-seeds', type=int, default=1,
                        help='the number of random seeds')
    parser.add_argument('--start-seed', type=int, default=1,
                        help='the number of the starting seed')

    parser.add_argument('--workers', type=int, default=1,
                        help='the number of workers to run benchmark experiments')

    args = parser.parse_args()

    return args


def identify_task_group(task_id):
    # Heuristic to identify the script to call
    if task_id in ['Acrobot-v1', 'CartPole-v1']:
        return 'rl_control'
    elif task_id in ['Asterix-v1', 'Breakout-v1', 'Freeway-v1', 'Seaquest-v1', 'SpaceInvaders-v1']:
        return 'rl_minatar'
    elif 'NoFrameskip' in task_id:
        return 'rl_atari'
    elif re.match(r'[A-Z]{2,3}\d', task_id):
        return 'rl_lqr'
    else:
        return 'sl'


def build_core_command(args):
    comand_pieces = []

    task_group_id = identify_task_group(args.task_id)
    if task_group_id == 'sl':
        comand_pieces.extend([
            'python3 egn/scripts/sl.py',

            '--test-size', str(args.test_size),
            '--n-epochs', str(args.n_epochs),
            '--max-steps', str(args.max_steps),
            '--evaluate-every-n', str(args.evaluate_every_n),
        ])
    elif task_group_id == 'rl_atari':
        comand_pieces.extend([
            'python3 egn/scripts/rl_atari.py',

            '--total-timesteps', str(args.total_timesteps),
            '--buffer-size', str(args.buffer_size),
            '--gamma', str(args.gamma),
            '--tau', str(args.tau),
            '--target-network-frequency', str(args.target_network_frequency),
            '--start-e', str(args.start_e),
            '--end-e', str(args.end_e),
            '--exploration-fraction', str(args.exploration_fraction),
            '--learning-starts', str(args.learning_starts),
            '--train-frequency', str(args.train_frequency),
            '--frame-stacking', str(args.frame_stacking),
        ])
    elif task_group_id == 'rl_minatar':
        comand_pieces.extend([
            'python3 egn/scripts/rl_minatar.py',

            '--total-timesteps', str(args.total_timesteps),
            '--buffer-size', str(args.buffer_size),
            '--gamma', str(args.gamma),
            '--tau', str(args.tau),
            '--target-network-frequency', str(args.target_network_frequency),
            '--start-e', str(args.start_e),
            '--end-e', str(args.end_e),
            '--exploration-fraction', str(args.exploration_fraction),
            '--learning-starts', str(args.learning_starts),
            '--train-frequency', str(args.train_frequency),
        ])
    elif task_group_id == 'rl_control':
        comand_pieces.extend([
            'python3 egn/scripts/rl_control.py',

            '--total-timesteps', str(args.total_timesteps),
            '--buffer-size', str(args.buffer_size),
            '--gamma', str(args.gamma),
            '--tau', str(args.tau),
            '--target-network-frequency', str(args.target_network_frequency),
            '--start-e', str(args.start_e),
            '--end-e', str(args.end_e),
            '--exploration-fraction', str(args.exploration_fraction),
            '--learning-starts', str(args.learning_starts),
            '--train-frequency', str(args.train_frequency),
        ])
    elif task_group_id == 'rl_lqr':
        comand_pieces.extend([
            'python3 egn/scripts/rl_lqr.py',

            '--dt', str(args.dt),
            '--deterministic', str(args.deterministic),
            '--reset-noise-sigma', str(args.reset_noise_sigma),

            '--total-timesteps', str(args.total_timesteps),
            '--buffer-size', str(args.buffer_size),
            '--gamma', str(args.gamma),
            '--learning-starts', str(args.learning_starts),
            '--train-frequency', str(args.train_frequency),

            '--reset-every-n', str(args.reset_every_n),
            '--update-policy-every-n', str(args.update_policy_every_n),
            '--action-noise-sigma', str(args.action_noise_sigma),
        ])
    else:
        raise NotImplementedError(f'Unknown task group id: {task_group_id}')

    # Common for all tasks
    comand_pieces.extend([
        '--task-id', args.task_id,

        '--track', str(args.track),
        '--wandb-project-name', str(args.wandb_project_name),
        '--wandb-entity', str(args.wandb_entity),
        '--debug', str(args.debug),
    ])

    return ' '.join(comand_pieces)


def run_experiment(command: str):
    command_list = shlex.split(command)
    print(f'running {command}')
    fd = subprocess.Popen(command_list)
    return_code = fd.wait()
    assert return_code == 0


if __name__ == '__main__':
    args = parse_args()

    # Create the core command depending on the task (i.e. there are specifics for SL, RL, etc.)
    core_command = build_core_command(args)

    # Collect the jobs
    jobs = set()

    optimizers = args.optimizers
    lrs = args.learning_rates
    assert len(optimizers) == len(lrs)

    # Add "search space" arguments
    for opt, lr in zip(optimizers, lrs):
        for batch_size in args.batch_sizes:
            for seed in range(args.start_seed, args.start_seed + args.num_seeds):
                params = ' '.join([
                    '--optimizer', opt,
                    '--seed', str(seed),
                    '--batch-size', str(batch_size),
                    '--learning-rate', str(lr),
                ])

                # only for EGN: additional loop over lambdas and momentum
                if opt == 'egn':
                    for reg_lambda in args.reg_lambdas:
                        for momentum in args.momenta:
                            for ls in args.line_search:
                                egn_params = ' '.join([
                                    '--reg-lambda', str(reg_lambda),
                                    '--momentum', str(momentum),
                                    '--line-search', str(ls),
                                    '--reset-option', str(args.reset_option),
                                    '--adaptive-lambda', str(args.adaptive_lambda),
                                    '--regularizer-eps', str(args.regularizer_eps),
                                ])
                                job_id = core_command + ' ' + params + ' ' + egn_params
                                jobs.add(job_id)
                elif opt in ['hfo', 'sgn']:
                    for reg_lambda in args.reg_lambdas:
                        for maxcg in args.maxcgs:
                            hfo_sgn_params = ' '.join([
                                '--reg-lambda', str(reg_lambda),
                                '--maxcg', str(maxcg),
                                # '--adaptive-lambda', str(args.adaptive_lambda),
                                # '--regularizer-eps', str(args.regularizer_eps),
                            ])
                            job_id = core_command + ' ' + params + ' ' + hfo_sgn_params
                            jobs.add(job_id)
                else:
                    job_id = core_command + ' ' + params
                    jobs.add(job_id)

    print(f'Total of {len(jobs)} jobs to run.')

    # Start the processes
    if args.workers > 0:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix='egn-bm-worker-')
        for job in jobs:
            executor.submit(run_experiment, job)
        executor.shutdown(wait=True)
    else:
        print('not running the experiments because --workers is set to 0; just printing the commands to run')
