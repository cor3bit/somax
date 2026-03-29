from typing import Any, Optional, Callable

import jax
import optax

from .types import Params, Batch, PRNGKey, ScalarOrSchedule


def build_optax_tx(
        *,
        tx: optax.GradientTransformation | None = None,
        learning_rate: Optional[ScalarOrSchedule] = 1.0,
        clip_norm: float | None = None,
        weight_decay: float | None = None,
        weight_decay_mask: Any | None = None,
        direction_momentum: float | None = None,
        nesterov: bool = False,
) -> optax.GradientTransformation:
    """Build the owned post-direction transform.

    Contract:
    - Direction s solves (C + lam I) s = g (no sign).
    - Descent is enforced by optax.scale_by_learning_rate (negative sign).
    - If user provides tx, it is used as-is (expert mode).
    """
    if tx is not None:
        return tx

    transforms: list[optax.GradientTransformation] = []

    # Momentum on direction s (before safety transforms)
    if direction_momentum is not None:
        transforms.append(optax.trace(decay=float(direction_momentum), nesterov=nesterov))

    # Clip by global norm (on direction, after momentum)
    if clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(float(clip_norm)))

    # Add weight decay (not clipped; add after clipping)
    if weight_decay is not None:
        transforms.append(optax.add_decayed_weights(float(weight_decay), mask=weight_decay_mask))

    transforms.append(optax.scale_by_learning_rate(learning_rate))

    return optax.chain(*transforms)


def sophia_tx(
        learning_rate: float | optax.Schedule,
        *,
        weight_decay: float = 0.2,
        clip_value: float = 1.0,
        mask=None,
) -> optax.GradientTransformation:
    transforms: list[optax.GradientTransformation] = [
        optax.clip(clip_value),
        optax.add_decayed_weights(weight_decay, mask=mask),
        optax.scale_by_learning_rate(learning_rate),
    ]

    return optax.chain(*transforms)
