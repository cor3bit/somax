import argparse
import os
from typing import Callable, Dict, List, Tuple, Union, Optional
import random
import time
from distutils.util import strtobool

import numpy as np
from tqdm.autonotebook import tqdm
from tensorboardX import SummaryWriter

# JAX ecosystem
import jax
import jax.numpy as jnp
from optax import adam, sgd, sigmoid_binary_cross_entropy
from jaxopt import OptaxSolver

# EGN
from benchmarks.utils.data_loader import load_data
from benchmarks.utils import model_zoo as zoo
import somax


def parse_args():
    parser = argparse.ArgumentParser()

    # ====== Dataset ======
    # 'diabetes', 'diamonds', 'california_housing', 'superconduct', 'iris', 'wine_quality',
    # 'mnist', 'fashion_mnist', 'cifar10'
    # dataset determines the type of the loss function as well as the model architecture
    parser.add_argument("--task-id", type=str, default="iris",
                        help="the id of the dataset")
    parser.add_argument("--test-size", type=float, default=0.1,
                        help="the size of the test set")
    parser.add_argument("--n-epochs", type=int, default=1,
                        help="the number of epochs")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="the maximum number of steps (not used if equals to 0)")
    parser.add_argument("--evaluate-every-n", type=int, default=10,
                        help="evaluate performance every n-th step")

    # ====== Seeding ======
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")

    # ====== Optimization ======
    # Common
    parser.add_argument('-o', '--optimizer', type=str, default='egn',  # 'sgd', 'adam', 'egn', 'hfo', 'sgn'
                        help='the optimization algorithm')
    parser.add_argument('-b', "--batch-size", type=int, default=32,
                        help="the batch size of sample from the dataset")
    parser.add_argument('-a', "--learning-rate", type=float, default=0.1,
                        help="the learning rate of the optimizer")

    # EGN specific
    parser.add_argument("--line-search", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="enables line search for EGN")
    parser.add_argument("--reset-option", type=str, default='increase',
                        help="the reset option for Line Search")

    parser.add_argument("--adaptive-lambda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="enables adaptive regularization for EGN")
    parser.add_argument('-l', "--reg-lambda", type=float, default=1.0,
                        help="Levenberg-Marquardt style regularization")
    parser.add_argument("--regularizer-eps", type=float, default=1e-5,
                        help="the epsilon for the regularizer")

    parser.add_argument("--momentum", type=float, default=0.0,
                        help="momentum acceleration of the EGN direction")

    # HFO, SGN specific
    parser.add_argument("--maxcg", type=int, default=10,
                        help="the maximum number of CG iterations")

    # ====== Experiment Tracking ======
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="egn",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will write additional parameters at each step")

    args = parser.parse_args()

    return args


def resolve_model(
        dataset_id: str,
        n_classes: Optional[int],
        X_sample: jnp.ndarray,
        seed: int,
        is_clf: bool,
) -> Tuple[dict, Callable]:
    # assign a model (flax nn.Module) for each dataset
    # initialize parameters by passing a sample from the train set

    if dataset_id == 'diabetes':
        model = zoo.MLPRegressorSmall()
    elif dataset_id == 'iris':
        model = zoo.MLPClassifierSmall(num_classes=n_classes)
    elif dataset_id in ['mnist', 'fashion_mnist']:
        model = zoo.CNNClassifierMedium(num_classes=n_classes)
    elif dataset_id in ['cifar10']:
        model = zoo.CNNClassifierLarge(num_classes=n_classes)
    # default choice
    else:
        if is_clf:
            model = zoo.MLPClassifierMedium(num_classes=n_classes)
        else:
            model = zoo.MLPRegressorMedium()

    params = model.init(jax.random.PRNGKey(seed), X_sample)
    predict_fn = jax.jit(model.apply)

    return params, predict_fn


if __name__ == '__main__':
    # force jax to use CPU
    # jax.config.update('jax_platform_name', 'cpu')

    # for debugging JAX-related issues
    # jax.config.update('jax_enable_x64', True)
    # jax.config.update('jax_debug_nans', True)
    # jax.config.update('jax_disable_jit', True)

    @jax.jit
    def mse(params, X, Y_true):
        residuals = Y_true - predict_fn(params, X)
        return 0.5 * jnp.mean(jnp.square(residuals))


    @jax.jit
    def rmse(params, X, y):
        residuals_ = y - predict_fn(params, X)
        mse_ = jnp.mean(jnp.square(residuals_))
        return jnp.sqrt(mse_)


    @jax.jit
    def mape(params, X, Y_true):
        predictions = predict_fn(params, X)
        percentage_errors = jnp.abs((Y_true - predictions) / jnp.abs(Y_true + 1e-8))
        return jnp.mean(percentage_errors)


    @jax.jit
    def mse_mape(params, X, Y_true, eps=1e-8):
        # common part
        predictions = predict_fn(params, X)
        residuals = Y_true - predictions

        # MSE
        mse_loss = 0.5 * jnp.mean(jnp.square(residuals))

        # MAPE
        percentage_errors = jnp.abs(residuals) / (jnp.abs(Y_true) + eps)
        mape_loss = jnp.mean(percentage_errors)

        return mse_loss, mape_loss


    @jax.jit
    def accuracy(params, X, Y_true):
        # b x C
        logits = predict_fn(params, X)

        # b x 1
        predicted_classes = jnp.argmax(logits, axis=1)
        correct_predictions = predicted_classes == Y_true

        # scalar
        return jnp.mean(correct_predictions)


    @jax.jit
    def ce(params, X, Y_true):
        # b x C
        logits = predict_fn(params, X)

        # Using softmax followed by log can lead to numerical instability.
        # Instead, we use jax.nn.log_softmax, which combines these operations in a numerically stable way.
        log_probs = jax.nn.log_softmax(logits)

        # Compute residuals
        # If y is one-hot encoded, this operation picks the log probability of the correct class
        residuals = jnp.sum(Y_true * log_probs, axis=1)

        # scalar
        ce_loss = -jnp.mean(residuals)

        return ce_loss


    @jax.jit
    def ce_binary(params, x, y):
        # b x 1
        logits = predict_fn(params, x)

        # b x 1
        losses = sigmoid_binary_cross_entropy(logits.ravel(), y)

        # 1,
        # average over the batch
        return jnp.mean(losses)


    @jax.jit
    def accuracy_binary(params, X, Y_true):
        # b x 1
        logits = predict_fn(params, X)

        # b x 1
        # Convert logits directly to class predictions by checking if they are >= 0
        # This step leverages the fact that the sigmoid function outputs values in the range (0, 1),
        # and its output is >= 0.5 (class 1) when the input logit is >= 0.
        predicted_classes = logits >= 0

        # scalar
        accuracy = jnp.mean(predicted_classes.ravel() == Y_true)

        return accuracy


    # --------------- parse args, set loggers ---------------
    # jax.config.update('jax_disable_jit', True)

    args = parse_args()
    run_name = f"{args.task_id}__{args.optimizer}__{args.seed}__{int(time.time())}"
    task_id = args.task_id

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )

    writer = SummaryWriter(os.path.join('..', 'artifacts', 'tensorboard', run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # --------------- dataset, model & loss ---------------
    seed = args.seed

    # ! RANDOMNESS: the seed is used to split and shuffle the dataset
    (X_train, X_test, Y_train, Y_test), is_clf, n_classes = load_data(
        args.task_id, test_size=args.test_size, seed=seed)

    # ! RANDOMNESS: the seed is used to initialize the model
    params, predict_fn = resolve_model(args.task_id, n_classes, X_train[0], seed, is_clf)

    batch_size = args.batch_size

    if is_clf:
        loss_fn = ce if n_classes > 2 else ce_binary
        reporting_fn = accuracy if n_classes > 2 else accuracy_binary
    else:
        loss_fn = mse
        reporting_fn = rmse

    # --------------- initialize solver ---------------
    opt_id = args.optimizer
    is_egn_like = False
    if opt_id == 'sgd':
        solver = OptaxSolver(loss_fn, opt=sgd(args.learning_rate))
    elif opt_id == 'adam':
        solver = OptaxSolver(loss_fn, opt=adam(args.learning_rate))
    elif opt_id == 'egn':
        is_egn_like = True
        solver = somax.EGN(
            predict_fun=predict_fn,
            loss_type='ce' if is_clf else 'mse',
            learning_rate=args.learning_rate,
            regularizer=args.reg_lambda,
            line_search=args.line_search,
            reset_option=args.reset_option,
            adaptive_lambda=args.adaptive_lambda,
            momentum=args.momentum,
            batch_size=args.batch_size,
            n_classes=n_classes,
        )
    elif opt_id == 'hfo':
        is_egn_like = True
        solver = somax.NewtonCG(
            loss_fun=loss_fn,
            maxcg=args.maxcg,
            learning_rate=args.learning_rate,
            regularizer=args.reg_lambda,
            # adaptive_lambda=args.adaptive_lambda,
            batch_size=args.batch_size,
        )
    elif opt_id == 'sgn':
        is_egn_like = True
        solver = somax.SGN(
            predict_fun=predict_fn,
            loss_type='ce' if is_clf else 'mse',
            maxcg=args.maxcg,
            learning_rate=args.learning_rate,
            regularizer=args.reg_lambda,
            # adaptive_lambda=al,
            # adaptive_lambda=args.adaptive_lambda,
            batch_size=args.batch_size,
            n_classes=n_classes,
        )
    else:
        raise ValueError(f'Unrecognized optimizer: {opt_id}')

    opt_state = solver.init_state(params, X_train[:batch_size], Y_train[:batch_size])

    update_fn = jax.jit(solver.update)

    # --------------- warm up jitted functions for better Wall Time tracking ---------------
    n_warmup = 2
    fake_params = jax.tree_map(lambda _: _, params)
    fake_opt_state = jax.tree_map(lambda _: _, opt_state)

    for i in range(n_warmup):
        batch_x = X_train[i * batch_size:(i + 1) * batch_size, :]
        batch_y = Y_train[i * batch_size:(i + 1) * batch_size]

        fake_preds = predict_fn(fake_params, batch_x)

        # on full test dataset
        fake_loss = reporting_fn(fake_params, X_test, Y_test)

        # on batch
        fake_loss = loss_fn(fake_params, batch_x, batch_y)

        if is_egn_like:
            fake_params, fake_opt_state = update_fn(fake_params, fake_opt_state, batch_x, targets=batch_y)
        else:
            fake_update = update_fn(fake_params, fake_opt_state, batch_x, batch_y)

    # --------------- training loop ---------------
    n_epochs = args.n_epochs
    num_batches = X_train.shape[0] // batch_size
    assert num_batches > 0, 'batch size must be smaller than the number of training samples'
    total_timesteps = n_epochs * num_batches

    check_max_steps = args.max_steps > 0

    evaluate_every_n = args.evaluate_every_n

    step = 0

    with tqdm(total=total_timesteps if not check_max_steps else args.max_steps) as pbar:

        for epoch in range(n_epochs):
            for i in range(0, num_batches):

                # evaluate on the full test set
                if step % evaluate_every_n == 0:
                    # Accuracy for Classification, RMSE for Regression
                    test_loss = reporting_fn(params, X_test, Y_test)
                    writer.add_scalar('test_loss', jax.device_get(test_loss), step)

                # ignore the last batch if it's smaller than the batch size
                batch_x = X_train[i * batch_size:(i + 1) * batch_size, :]
                batch_y = Y_train[i * batch_size:(i + 1) * batch_size]

                # update step
                if is_egn_like:
                    params, opt_state = update_fn(params, opt_state, batch_x, targets=batch_y)
                else:
                    params, opt_state = update_fn(params, opt_state, batch_x, batch_y)

                step += 1

                # debug mode: write opt_state params in the StreamWriter at each step
                # TODO: consider writing only each evaluate_every_n step
                # Note: will actually start at T=1, since updates after the first step
                if args.debug and is_egn_like:
                    writer.add_scalar('alpha', opt_state.stepsize, step)
                    writer.add_scalar('lambda', opt_state.regularizer, step)

                pbar.update(1)

                if check_max_steps and step >= args.max_steps:
                    break

    # bookkeeping
    writer.close()
    if args.track:
        wandb.finish()
