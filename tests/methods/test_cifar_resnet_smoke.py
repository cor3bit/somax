from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from somax import metrics as mx
from somax.methods.fgn import fgn_ce
from somax.methods.sgn import sgn_ce


def _conv2d(x, kernel, stride=1):
    return jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(stride, stride),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )


def _batch_norm_train(x, scale, offset, stats):
    mean = jnp.mean(x, axis=(0, 1, 2))
    centered = x - mean
    var = jnp.mean(jnp.square(centered), axis=(0, 1, 2))
    normalized = centered * jax.lax.rsqrt(
        var + jnp.asarray(1e-5, x.dtype)
    )
    y = normalized * scale + offset

    momentum = jnp.asarray(0.9, x.dtype)
    stats_new = {
        "mean": momentum * stats["mean"] + (1.0 - momentum) * mean,
        "var": momentum * stats["var"] + (1.0 - momentum) * var,
    }
    return y, stats_new


def _option_a_shortcut(x, stride, out_channels):
    if stride == 2:
        x = x[:, ::2, ::2, :]
    in_channels = x.shape[-1]
    if in_channels < out_channels:
        total = out_channels - in_channels
        before = total // 2
        after = total - before
        x = jnp.pad(
            x,
            ((0, 0), (0, 0), (0, 0), (before, after)),
        )
    return x


def _he_kernel(key, shape):
    fan_in = shape[0] * shape[1] * shape[2]
    return jax.random.normal(key, shape, jnp.float32) * jnp.sqrt(
        jnp.asarray(2.0 / fan_in, jnp.float32)
    )


def _init_resnet20(key, *, n_classes, base_width=1):
    keys = iter(jax.random.split(key, 64))
    params: dict[str, Any] = {}
    batch_stats = []

    params["stem"] = {
        "kernel": _he_kernel(next(keys), (3, 3, 3, base_width)),
        "scale": jnp.ones((base_width,), jnp.float32),
        "offset": jnp.zeros((base_width,), jnp.float32),
    }
    batch_stats.append(
        {
            "mean": jnp.zeros((base_width,), jnp.float32),
            "var": jnp.ones((base_width,), jnp.float32),
        }
    )

    blocks = []
    in_channels = base_width
    for stage, out_channels in enumerate(
            (base_width, 2 * base_width, 4 * base_width)
    ):
        for block_index in range(3):
            stride = 2 if stage > 0 and block_index == 0 else 1
            block = {
                "conv1": {
                    "kernel": _he_kernel(
                        next(keys),
                        (3, 3, in_channels, out_channels),
                    ),
                    "scale": jnp.ones((out_channels,), jnp.float32),
                    "offset": jnp.zeros((out_channels,), jnp.float32),
                },
                "conv2": {
                    "kernel": _he_kernel(
                        next(keys),
                        (3, 3, out_channels, out_channels),
                    ),
                    "scale": jnp.ones((out_channels,), jnp.float32),
                    "offset": jnp.zeros((out_channels,), jnp.float32),
                },
            }
            blocks.append(block)
            batch_stats.extend(
                [
                    {
                        "mean": jnp.zeros((out_channels,), jnp.float32),
                        "var": jnp.ones((out_channels,), jnp.float32),
                    },
                    {
                        "mean": jnp.zeros((out_channels,), jnp.float32),
                        "var": jnp.ones((out_channels,), jnp.float32),
                    },
                ]
            )
            in_channels = out_channels

    params["blocks"] = tuple(blocks)
    params["head"] = {
        "kernel": 0.01 * jax.random.normal(
            next(keys),
            (in_channels, n_classes),
            jnp.float32,
        ),
        "bias": jnp.zeros((n_classes,), jnp.float32),
    }
    model_state = {
        "batch_stats": tuple(batch_stats),
        "primal_calls": jnp.asarray(0, jnp.int32),
    }
    return params, model_state


def _resnet20_predict(params, x, model_state):
    stats_new = []
    stats_index = 0

    stem = params["stem"]
    h = _conv2d(x, stem["kernel"])
    h, updated = _batch_norm_train(
        h,
        stem["scale"],
        stem["offset"],
        model_state["batch_stats"][stats_index],
    )
    stats_new.append(updated)
    stats_index += 1
    h = jax.nn.relu(h)

    for block_index, block in enumerate(params["blocks"]):
        stride = 2 if block_index in (3, 6) else 1
        residual = _option_a_shortcut(
            h,
            stride,
            block["conv2"]["kernel"].shape[-1],
        )

        conv1 = block["conv1"]
        z = _conv2d(h, conv1["kernel"], stride)
        z, updated = _batch_norm_train(
            z,
            conv1["scale"],
            conv1["offset"],
            model_state["batch_stats"][stats_index],
        )
        stats_new.append(updated)
        stats_index += 1
        z = jax.nn.relu(z)

        conv2 = block["conv2"]
        z = _conv2d(z, conv2["kernel"])
        z, updated = _batch_norm_train(
            z,
            conv2["scale"],
            conv2["offset"],
            model_state["batch_stats"][stats_index],
        )
        stats_new.append(updated)
        stats_index += 1
        h = jax.nn.relu(z + residual)

    features = jnp.mean(h, axis=(1, 2))
    logits = features @ params["head"]["kernel"] + params["head"]["bias"]
    return logits, {
        "batch_stats": tuple(stats_new),
        "primal_calls": model_state["primal_calls"] + jnp.asarray(1, jnp.int32),
    }


def _tree_all_finite(tree):
    return all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _tree_allclose(actual, expected, *, rtol=2e-5, atol=2e-5):
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    return actual_tree == expected_tree and all(
        bool(jnp.allclose(a, b, rtol=rtol, atol=atol))
        for a, b in zip(actual_leaves, expected_leaves)
    )


def _tree_any_changed(actual, expected):
    return any(
        not bool(jnp.array_equal(a, b))
        for a, b in zip(
            jax.tree_util.tree_leaves(actual),
            jax.tree_util.tree_leaves(expected),
        )
    )


@pytest.mark.parametrize(
    ("method_factory", "expected_lane", "n_classes"),
    [
        (fgn_ce, "row", 10),
        (sgn_ce, "param", 100),
    ],
    ids=["fgn-cifar10", "sgn-cifar100"],
)
def test_compiled_resnet20_cifar_step_and_batchnorm_state(
        method_factory,
        expected_lane,
        n_classes,
        key,
):
    params0, model_state0 = _init_resnet20(
        jax.random.fold_in(key, n_classes),
        n_classes=n_classes,
        base_width=1,
    )
    batch = {
        "x": jax.random.normal(
            jax.random.fold_in(key, 100 + n_classes),
            (2, 8, 8, 3),
            jnp.float32,
        ),
        "y": jnp.arange(2, dtype=jnp.int32) % n_classes,
    }

    method = method_factory(
        predict_fn=_resnet20_predict,
        has_aux=True,
        lam0=1.0,
        tol=0.0,
        maxiter=1,
        warm_start=False,
        stabilise_every=0,
        learning_rate=0.01,
        record_loss_every_k=1,
        record_cg_stats=True,
    )
    assert method.plan.lane == expected_lane
    state0 = method.init(params0, model_state=model_state0)

    curvature_state, grad = method.op.init(
        params0,
        batch,
        model_state=model_state0,
        with_grad=True,
    )
    tangent = jax.tree_util.tree_map(jnp.ones_like, params0)
    curvature_tangent = method.op.matvec(params0, curvature_state, tangent)
    loss0 = method.op.loss(params0, curvature_state, batch)
    assert _tree_all_finite((loss0, grad, curvature_tangent))

    _, expected_state1 = _resnet20_predict(
        params0,
        batch["x"],
        model_state0,
    )
    _, state_if_updated_twice = _resnet20_predict(
        params0,
        batch["x"],
        expected_state1,
    )
    assert _tree_allclose(curvature_state.model_state_new, expected_state1)

    compiled_step = jax.jit(method.step).lower(
        params0,
        batch,
        state0,
        key,
    ).compile()

    params1, state1, info1 = compiled_step(params0, batch, state0, key)
    params1_repeat, state1_repeat, info1_repeat = compiled_step(
        params0,
        batch,
        state0,
        key,
    )
    jax.block_until_ready(
        (params1, state1, info1, params1_repeat, state1_repeat, info1_repeat)
    )

    assert _tree_allclose(params1, params1_repeat)
    assert _tree_allclose(state1, state1_repeat)
    assert _tree_allclose(info1, info1_repeat)
    assert _tree_allclose(state1.model_state, expected_state1)
    assert not _tree_allclose(state1.model_state, state_if_updated_twice)
    assert int(state1.model_state["primal_calls"]) == 1

    assert _tree_all_finite((params1, state1, info1))
    assert _tree_any_changed(params1, params0)
    assert _tree_any_changed(
        state1.model_state["batch_stats"],
        model_state0["batch_stats"],
    )
    assert int(state1.step) == 1
    assert int(info1[mx.CG_ITERS]) == 1
    assert bool(jnp.isfinite(info1[mx.CG_RESID]))
    assert bool(jnp.isfinite(info1[mx.LOSS_BEFORE]))

    _, expected_state2 = _resnet20_predict(
        params1,
        batch["x"],
        state1.model_state,
    )
    params2, state2, info2 = compiled_step(
        params1,
        batch,
        state1,
        jax.random.fold_in(key, 1),
    )
    jax.block_until_ready((params2, state2, info2))

    assert _tree_allclose(state2.model_state, expected_state2)
    assert int(state2.model_state["primal_calls"]) == 2
    assert _tree_all_finite((params2, state2, info2))
    assert int(state2.step) == 2
    assert bool(jnp.isfinite(info2[mx.CG_RESID]))
    assert bool(jnp.isfinite(info2[mx.LOSS_BEFORE]))
