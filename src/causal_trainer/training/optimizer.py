from __future__ import annotations

from collections.abc import Callable

import jax
import optax


def build_learning_rate_schedule(
    name: str,
    start: float,
    end: float,
    total_steps: int,
    warmup_steps: int = 0,
) -> Callable[[jax.Array], jax.Array]:
    """Create a schedule whose ``end`` value is absolute, not a ratio."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps < total_steps:
        if warmup_steps == total_steps == 1:
            warmup_steps = 0
        else:
            raise ValueError("warmup_steps must be in [0, total_steps)")

    decay_steps = max(total_steps - warmup_steps, 1)
    if name == "cosine":
        decay = optax.cosine_decay_schedule(start, decay_steps, alpha=end / start)
    elif name == "linear":
        decay = optax.linear_schedule(start, end, decay_steps)
    elif name == "constant":
        decay = optax.constant_schedule(start)
    else:
        raise ValueError(f"unsupported scheduler: {name}")

    if warmup_steps == 0:
        return decay
    warmup = optax.linear_schedule(0.0, start, warmup_steps)
    return optax.join_schedules((warmup, decay), (warmup_steps,))


def build_optimizer(
    params,
    schedule: Callable[[jax.Array], jax.Array],
    *,
    weight_decay: float,
    max_grad_norm: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    external_learning_rate: bool = False,
) -> optax.GradientTransformation:
    decay_mask = jax.tree.map(lambda value: value.ndim > 1, params)
    transformations: list[optax.GradientTransformation] = []
    if max_grad_norm > 0:
        transformations.append(optax.clip_by_global_norm(max_grad_norm))
    transformations.append(
        optax.adamw(
            # Streaming packing cannot infer an exact optimizer-step horizon
            # from source-example metadata. In external mode the train step
            # supplies the schedule position and scales the complete AdamW
            # gradient + weight-decay update.
            learning_rate=1.0 if external_learning_rate else schedule,
            b1=beta1,
            b2=beta2,
            eps=epsilon,
            weight_decay=weight_decay,
            mask=decay_mask,
            # None keeps moment buffers in the parameter dtype, matching the
            # reference full-parameter trainer and avoiding a prohibitive FP32
            # first-moment allocation for the fixed 9B-scale checkpoint.
            mu_dtype=None,
        )
    )
    return optax.chain(*transformations)


__all__ = ["build_learning_rate_schedule", "build_optimizer"]
