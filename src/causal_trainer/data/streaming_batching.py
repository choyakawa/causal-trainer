from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .batching import MODEL_FIELDS
from .packing import PackedExample


_UNKNOWN_TOTAL_STEPS = 2**31 - 1
_STREAM_DIGEST_DOMAIN = b"causal-trainer-stream-v1\0"


@dataclass(frozen=True, slots=True)
class StreamingBatchPlan:
    """Bounded streaming plan.

    ``total_steps`` is exact only when ``max_steps`` is set. Otherwise it is a
    large checkpoint sentinel: metadata row counts are estimates and cannot
    safely cap optimizer steps. ``steps_per_epoch`` remains an estimate for
    reporting, while epoch completion is driven only by real stream exhaustion.
    """

    source_examples_per_epoch: int | None
    global_micro_batch_size: int
    accumulation_steps: int
    epochs: int | None
    max_steps: int
    steps_per_epoch: int | None
    total_steps: int

    @classmethod
    def create(
        cls,
        source_examples_per_epoch: int | None,
        global_micro_batch_size: int,
        accumulation_steps: int,
        epochs: int,
        max_steps: int,
    ) -> StreamingBatchPlan:
        if source_examples_per_epoch is not None and source_examples_per_epoch < 0:
            raise ValueError("streaming source example count cannot be negative")
        if global_micro_batch_size <= 0 or accumulation_steps <= 0:
            raise ValueError("streaming batch sizes must be positive")
        if max_steps <= 0 and epochs <= 0:
            raise ValueError("streaming epochs must be positive unless max_steps is set")

        effective = global_micro_batch_size * accumulation_steps
        steps_per_epoch = (
            math.ceil(source_examples_per_epoch / effective)
            if source_examples_per_epoch is not None
            else None
        )
        if max_steps > 0:
            planned_epochs = None
            total_steps = max_steps
        else:
            planned_epochs = epochs
            total_steps = _UNKNOWN_TOTAL_STEPS
        return cls(
            source_examples_per_epoch,
            global_micro_batch_size,
            accumulation_steps,
            planned_epochs,
            max_steps,
            steps_per_epoch,
            total_steps,
        )

    @property
    def effective_batch_size(self) -> int:
        return self.global_micro_batch_size * self.accumulation_steps

    @property
    def total_source_examples(self) -> int | None:
        """Return the metadata-derived source-row estimate across planned epochs."""

        if self.source_examples_per_epoch is None or self.epochs is None:
            return None
        return self.source_examples_per_epoch * self.epochs

    @property
    def has_exact_step_limit(self) -> bool:
        return self.max_steps > 0


@dataclass(frozen=True, slots=True)
class StreamingGlobalBatch:
    data: Mapping[str, Any]
    source_examples: int
    source_examples_before: int
    source_examples_seen: int
    epoch_source_examples_before: int
    epoch_source_examples_seen: int
    epoch: int
    is_epoch_end: bool
    stream_digest_before: str
    stream_digest: str

    def with_data(self, data: Mapping[str, Any]) -> StreamingGlobalBatch:
        return replace(self, data=data)


def _streaming_global_batch_digest(
    batch: StreamingGlobalBatch,
    *,
    include_stream_state: bool,
) -> bytes:
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
        digest.update(value)

    metadata: dict[str, Any] = {
        "epoch": batch.epoch,
        "is_epoch_end": batch.is_epoch_end,
        "source_examples": batch.source_examples,
        "source_examples_before": batch.source_examples_before,
        "source_examples_seen": batch.source_examples_seen,
        "epoch_source_examples_before": batch.epoch_source_examples_before,
        "epoch_source_examples_seen": batch.epoch_source_examples_seen,
    }
    if include_stream_state:
        metadata["stream_digest_before"] = batch.stream_digest_before
        metadata["stream_digest"] = batch.stream_digest
    update(
        json.dumps(
            metadata,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    for name in sorted(batch.data):
        array = np.ascontiguousarray(batch.data[name])
        update(name.encode("utf-8"))
        update(array.dtype.str.encode("ascii"))
        update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        update(memoryview(array).cast("B"))
    return digest.digest()


def streaming_global_batch_digest(batch: StreamingGlobalBatch) -> bytes:
    """Hash one host-built global batch for cross-process consistency checks."""

    return _streaming_global_batch_digest(
        batch,
        include_stream_state=True,
    )


@dataclass(frozen=True, slots=True)
class _GroupedBatch:
    data: dict[str, np.ndarray]
    source_examples: int
    epoch: int
    is_epoch_end: bool = False


def _canonical_record(
    record: Mapping[str, Sequence[int | float]],
    *,
    assistant_only_loss: bool,
) -> PackedExample:
    missing = [name for name in MODEL_FIELDS if name not in record]
    if missing:
        raise ValueError(f"streaming packed record is missing model fields: {missing}")
    canonical: PackedExample = {
        name: [int(value) for value in record[name]]
        for name in MODEL_FIELDS
    }
    length = len(canonical["input_ids"])
    if any(len(canonical[name]) != length for name in MODEL_FIELDS):
        raise ValueError("streaming packed model fields are not aligned")

    if "assistant_masks" in record:
        assistant = [int(value) for value in record["assistant_masks"]]
        if len(assistant) != length:
            raise ValueError("streaming assistant mask is not aligned with input_ids")
        canonical["assistant_masks"] = assistant
    elif assistant_only_loss:
        raise ValueError("assistant_only_loss streaming record is missing assistant_masks")

    if "loss_weights" in record:
        weights = [float(value) for value in record["loss_weights"]]
    elif assistant_only_loss:
        weights = [float(value) for value in canonical["assistant_masks"]]
    else:
        weights = [float(value) for value in canonical["attention_mask"]]
    if len(weights) != length:
        raise ValueError("streaming loss_weights are not aligned with input_ids")
    weight_array = np.asarray(weights, dtype=np.float32)
    if not np.isfinite(weight_array).all() or (weight_array < 0).any():
        raise ValueError("streaming loss_weights must be finite and non-negative")
    canonical["loss_weights"] = weights

    attention = np.asarray(canonical["attention_mask"], dtype=np.int32)
    segments = np.asarray(canonical["segment_ids"], dtype=np.int32)
    has_target = np.any(
        (attention[:-1] != 0)
        & (attention[1:] != 0)
        & (segments[:-1] != 0)
        & (segments[:-1] == segments[1:])
        & (weight_array[1:] > 0)
    )
    if not has_target:
        raise ValueError(
            "a streaming packed row has no positive-weight causal target with a "
            "valid same-segment predecessor"
        )
    return canonical


def _dummy_record(record: PackedExample) -> PackedExample:
    dummy = {name: list(values) for name, values in record.items()}
    dummy["loss_weights"] = [0.0] * len(record["input_ids"])
    if "assistant_masks" in dummy:
        dummy["assistant_masks"] = [0] * len(record["input_ids"])
    return dummy


def _collate_records(records: Sequence[PackedExample]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(
            [record[name] for record in records],
            dtype=np.float32 if name == "loss_weights" else np.int32,
        )
        for name in (*MODEL_FIELDS, "loss_weights")
    }


def _envelope_record_counts(envelope: Any) -> list[int]:
    records = tuple(envelope.records)
    if not records:
        return []
    explicit = getattr(envelope, "record_source_examples", None)
    if explicit is None:
        counts = [0] * len(records)
        if counts:
            counts[-1] = int(envelope.source_examples)
        return counts
    counts = [int(value) for value in explicit]
    if len(counts) != len(records):
        raise ValueError("record_source_examples must align with streaming records")
    if any(value < 0 for value in counts):
        raise ValueError("record_source_examples must be non-negative")
    if records and sum(counts) != int(envelope.source_examples):
        raise ValueError("record_source_examples must sum to source_examples")
    return counts


def _iter_grouped_batches(
    envelopes: Iterable[Any],
    *,
    effective_batch_size: int,
    assistant_only_loss: bool,
) -> Iterator[_GroupedBatch]:
    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")

    buffer: list[tuple[PackedExample, int]] = []
    held: _GroupedBatch | None = None
    deferred_source_examples = 0
    active_epoch: int | None = None

    for envelope in envelopes:
        epoch = int(envelope.epoch)
        if active_epoch is None:
            active_epoch = epoch
        elif epoch != active_epoch:
            raise ValueError("streaming epoch changed without an epoch-end envelope")

        records = tuple(envelope.records)
        counts = _envelope_record_counts(envelope)
        if records:
            counts[0] += deferred_source_examples
            deferred_source_examples = 0
            buffer.extend(
                (
                    _canonical_record(record, assistant_only_loss=assistant_only_loss),
                    count,
                )
                for record, count in zip(records, counts, strict=True)
            )
        else:
            deferred_source_examples += int(envelope.source_examples)

        while len(buffer) >= effective_batch_size:
            batch_items = buffer[:effective_batch_size]
            del buffer[:effective_batch_size]
            candidate = _GroupedBatch(
                _collate_records([record for record, _ in batch_items]),
                sum(count for _, count in batch_items),
                epoch,
            )
            if held is not None:
                yield held
            held = candidate

        if not bool(envelope.is_epoch_end):
            continue

        if buffer:
            if held is not None:
                yield held
                held = None
            real_records = [record for record, _ in buffer]
            source_examples = sum(count for _, count in buffer) + deferred_source_examples
            deferred_source_examples = 0
            padded = [
                *real_records,
                *(
                    _dummy_record(real_records[-1])
                    for _ in range(effective_batch_size - len(real_records))
                ),
            ]
            held = _GroupedBatch(
                _collate_records(padded),
                source_examples,
                epoch,
            )
            buffer.clear()
        elif held is not None and deferred_source_examples:
            held = replace(
                held,
                source_examples=held.source_examples + deferred_source_examples,
            )
            deferred_source_examples = 0

        if held is None:
            raise ValueError(
                f"streaming epoch {epoch} produced no trainable records after preprocessing"
            )
        yield replace(held, is_epoch_end=True)
        held = None
        active_epoch = None

    if active_epoch is not None or buffer or held is not None or deferred_source_examples:
        raise ValueError("streaming source ended without an epoch-end envelope")


def iter_streaming_global_batches(
    envelopes: Iterable[Any],
    plan: StreamingBatchPlan,
    *,
    assistant_only_loss: bool,
    start_step: int = 0,
    expected_source_examples_before: int | None = None,
    expected_stream_digest_before: str | None = None,
) -> Iterator[StreamingGlobalBatch]:
    """Group lazy packing windows into fixed global optimizer batches.

    Resume is deterministic: batches before ``start_step`` are regenerated and
    discarded, so prefetched-but-uncommitted input is never skipped.
    """

    if start_step < 0 or start_step > plan.total_steps:
        raise ValueError(f"start_step must be in [0, {plan.total_steps}], got {start_step}")
    if expected_source_examples_before is not None and (
        type(expected_source_examples_before) is not int
        or expected_source_examples_before < 0
    ):
        raise ValueError(
            "expected_source_examples_before must be a non-negative integer or None"
        )
    if expected_stream_digest_before is not None and (
        len(expected_stream_digest_before) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_stream_digest_before
        )
    ):
        raise ValueError(
            "expected_stream_digest_before must be a lowercase SHA-256 digest or None"
        )

    source_examples_seen = 0
    epoch_source_examples_seen = 0
    active_epoch: int | None = None
    stream_digest = hashlib.sha256(_STREAM_DIGEST_DOMAIN)
    step = 0
    resume_boundary_validated = False

    def validate_resume_boundary() -> None:
        nonlocal resume_boundary_validated
        if (
            expected_source_examples_before is not None
            and source_examples_seen != expected_source_examples_before
        ):
            raise RuntimeError(
                "streaming resume replay reached source example "
                f"{source_examples_seen}, but checkpoint progress records "
                f"{expected_source_examples_before}"
            )
        replay_digest = stream_digest.hexdigest()
        if (
            expected_stream_digest_before is not None
            and replay_digest != expected_stream_digest_before
        ):
            raise RuntimeError(
                "streaming resume replay produced different training batches "
                "before the checkpoint; pin dataset_revision to the original "
                "immutable dataset commit"
            )
        resume_boundary_validated = True

    for grouped in _iter_grouped_batches(
        envelopes,
        effective_batch_size=plan.effective_batch_size,
        assistant_only_loss=assistant_only_loss,
    ):
        if step == start_step and not resume_boundary_validated:
            validate_resume_boundary()
        if active_epoch is None:
            active_epoch = grouped.epoch
            epoch_source_examples_seen = 0
        elif grouped.epoch != active_epoch:
            raise RuntimeError(
                "streaming global batches changed epoch without completing the prior epoch"
            )
        source_before = source_examples_seen
        source_examples_seen += grouped.source_examples
        epoch_source_before = epoch_source_examples_seen
        epoch_source_examples_seen += grouped.source_examples
        batch = StreamingGlobalBatch(
            data=grouped.data,
            source_examples=grouped.source_examples,
            source_examples_before=source_before,
            source_examples_seen=source_examples_seen,
            epoch_source_examples_before=epoch_source_before,
            epoch_source_examples_seen=epoch_source_examples_seen,
            epoch=grouped.epoch,
            is_epoch_end=grouped.is_epoch_end,
            stream_digest_before=stream_digest.hexdigest(),
            stream_digest="",
        )
        stream_digest.update(
            _streaming_global_batch_digest(
                batch,
                include_stream_state=False,
            )
        )
        batch = replace(
            batch,
            stream_digest=stream_digest.hexdigest(),
        )
        if grouped.is_epoch_end:
            active_epoch = None
            epoch_source_examples_seen = 0
        if step >= start_step:
            yield batch
        step += 1
        if plan.max_steps > 0 and step >= plan.max_steps:
            if step == start_step and not resume_boundary_validated:
                validate_resume_boundary()
            return

    if step < start_step:
        raise RuntimeError(
            "streaming resume requested optimizer step "
            f"{start_step}, but the reconstructed source produced only {step} batches"
        )
    if step == start_step and not resume_boundary_validated:
        validate_resume_boundary()


__all__ = [
    "StreamingBatchPlan",
    "StreamingGlobalBatch",
    "iter_streaming_global_batches",
    "streaming_global_batch_digest",
]
