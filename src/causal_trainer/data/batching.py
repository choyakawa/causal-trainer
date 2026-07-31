from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np


MODEL_FIELDS = ("input_ids", "attention_mask", "position_ids", "segment_ids")
_Source = TypeVar("_Source")
_Converted = TypeVar("_Converted")


@dataclass(frozen=True)
class BatchPlan:
    examples: int
    global_micro_batch_size: int
    accumulation_steps: int
    steps_per_epoch: int
    total_steps: int

    @classmethod
    def create(
        cls,
        examples: int,
        global_micro_batch_size: int,
        accumulation_steps: int,
        epochs: float,
        max_steps: int,
    ) -> "BatchPlan":
        effective = global_micro_batch_size * accumulation_steps
        if examples <= 0:
            raise ValueError("the prepared dataset must contain at least one row")
        if examples % effective:
            raise ValueError(
                f"prepared dataset rows ({examples}) must be divisible by the effective global batch "
                f"({effective}); append zero-loss rows before creating the batch plan"
            )
        steps_per_epoch = examples // effective
        if max_steps > 0:
            total_steps = max_steps
        else:
            total_steps = max(1, math.floor(steps_per_epoch * epochs))
        return cls(examples, global_micro_batch_size, accumulation_steps, steps_per_epoch, total_steps)

    @property
    def effective_batch_size(self) -> int:
        return self.global_micro_batch_size * self.accumulation_steps


def _collate(dataset, indices: np.ndarray, assistant_only_loss: bool) -> dict[str, np.ndarray]:
    records = [dataset[int(index)] for index in indices]
    for record_index, record in enumerate(records):
        missing = [field for field in MODEL_FIELDS if field not in record]
        if missing:
            raise ValueError(f"dataset record {record_index} is missing model fields: {missing}")
    batch = {
        field: np.asarray([record[field] for record in records], dtype=np.int32)
        for field in MODEL_FIELDS
    }
    if assistant_only_loss:
        missing_masks = [
            index for index, record in enumerate(records) if "assistant_masks" not in record
        ]
        if missing_masks:
            raise ValueError(
                "assistant-masked loss dataset is missing assistant_masks for records "
                f"{missing_masks}"
            )
        for record_index, record in enumerate(records):
            if len(record["assistant_masks"]) != len(record["input_ids"]):
                raise ValueError(
                    f"dataset record {record_index} has an assistant mask that is not aligned with input_ids"
                )
    weighted = [index for index, record in enumerate(records) if "loss_weights" in record]
    if weighted and len(weighted) != len(records):
        raise ValueError("a batch cannot mix weighted and unweighted training records")
    if weighted:
        for record_index, record in enumerate(records):
            if len(record["loss_weights"]) != len(record["input_ids"]):
                raise ValueError(
                    f"dataset record {record_index} has loss_weights that are not aligned with input_ids"
                )
        # EndPrompt+assistant-only preprocessing has already combined the
        # assistant span and terminal-span objectives. Keeping assistant_masks
        # in the record preserves role metadata without multiplying it twice.
        batch["loss_weights"] = np.asarray(
            [record["loss_weights"] for record in records],
            dtype=np.float32,
        )
    elif assistant_only_loss:
        batch["loss_weights"] = np.asarray(
            [record["assistant_masks"] for record in records],
            dtype=np.float32,
        )
    else:
        batch["loss_weights"] = batch["attention_mask"].astype(np.float32)
    if not np.isfinite(batch["loss_weights"]).all() or (batch["loss_weights"] < 0).any():
        raise ValueError("loss_weights must be finite and non-negative")
    return batch


def iter_global_batches(
    dataset,
    plan: BatchPlan,
    *,
    assistant_only_loss: bool,
    shuffle: bool,
    seed: int,
    start_step: int = 0,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield identical full global batches on every host.

    Device placement later keeps only each host's addressable slices. Duplicating
    this small integer batch on hosts makes multi-controller input semantics
    deterministic even when tensor-parallel axes replicate the batch.
    """

    if not 0 <= start_step <= plan.total_steps:
        raise ValueError(f"start_step must be in [0, {plan.total_steps}], got {start_step}")

    step = start_step
    epoch, first_step_in_epoch = divmod(start_step, plan.steps_per_epoch)
    while step < plan.total_steps:
        order = np.arange(plan.examples, dtype=np.int64)
        if shuffle:
            np.random.default_rng(seed + epoch).shuffle(order)
        usable = plan.steps_per_epoch * plan.effective_batch_size
        first_offset = first_step_in_epoch * plan.effective_batch_size
        for offset in range(first_offset, usable, plan.effective_batch_size):
            if step >= plan.total_steps:
                return
            yield _collate(
                dataset,
                order[offset : offset + plan.effective_batch_size],
                assistant_only_loss,
            )
            step += 1
        epoch += 1
        first_step_in_epoch = 0


def split_accumulation(batch: dict[str, Any], accumulation_steps: int) -> dict[str, Any]:
    if accumulation_steps == 1:
        return {key: value[None, ...] for key, value in batch.items()}
    result = {}
    for key, value in batch.items():
        if value.shape[0] % accumulation_steps:
            raise ValueError(f"batch field {key} cannot be split into {accumulation_steps} micro-batches")
        result[key] = value.reshape((accumulation_steps, -1, *value.shape[1:]))
    return result


def prefetch_map(
    values: Iterable[_Source],
    transform: Callable[[_Source], _Converted],
    *,
    prefetch_batches: int,
) -> Iterator[_Converted]:
    """Convert a finite number of values ahead of the value being consumed.

    A value returned to the caller is not counted in ``prefetch_batches``.  A
    setting of one therefore retains at most the active batch plus one already
    converted batch, while zero preserves strictly just-in-time conversion.
    The iterator is deliberately synchronous and bounded; when ``transform``
    initiates asynchronous JAX placement, the device transfer can overlap the
    caller's work without an unbounded producer thread or host queue.
    """

    if prefetch_batches < 0:
        raise ValueError("prefetch_batches cannot be negative")

    source = iter(values)
    pending: deque[_Converted] = deque()
    for _ in range(prefetch_batches + 1):
        try:
            value = next(source)
        except StopIteration:
            break
        pending.append(transform(value))

    while pending:
        yield pending.popleft()
        try:
            value = next(source)
        except StopIteration:
            continue
        pending.append(transform(value))


__all__ = ["BatchPlan", "iter_global_batches", "prefetch_map", "split_accumulation"]
