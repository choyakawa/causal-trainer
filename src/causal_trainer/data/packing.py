"""Leakage-safe, fixed-length sequence packing utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import islice
from typing import Any


Scalar = int | float
PackedExample = dict[str, list[Scalar]]


def _as_int_list(value: Any, *, field_name: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list) or (value and isinstance(value[0], (list, tuple))):
        raise ValueError(f"Per-token field {field_name!r} must be a one-dimensional sequence.")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Per-token field {field_name!r} must contain integers.") from exc


def _as_float_list(value: Any, *, field_name: str) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list) or (value and isinstance(value[0], (list, tuple))):
        raise ValueError(f"Per-token field {field_name!r} must be a one-dimensional sequence.")
    try:
        output = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Per-token field {field_name!r} must contain numbers.") from exc
    if any(not math.isfinite(item) for item in output):
        raise ValueError(f"Per-token field {field_name!r} must contain only finite numbers.")
    return output


def _as_field_list(value: Any, *, field_name: str) -> list[Scalar]:
    if field_name == "loss_weights":
        return _as_float_list(value, field_name=field_name)
    return _as_int_list(value, field_name=field_name)


def _normalize_example(example: Mapping[str, Any], max_length: int) -> PackedExample:
    if "input_ids" not in example:
        raise ValueError("Every example must contain input_ids.")
    full_input_ids = _as_int_list(example["input_ids"], field_name="input_ids")
    input_ids = full_input_ids[:max_length]
    if not input_ids:
        raise ValueError("Cannot pack an empty input_ids sequence.")

    normalized: PackedExample = {"input_ids": input_ids}
    sequence_length = len(full_input_ids)
    for key, value in example.items():
        if key == "input_ids" or isinstance(value, (str, bytes, Mapping)):
            continue
        try:
            column = _as_field_list(value, field_name=key)
        except ValueError:
            if key in {"attention_mask", "position_ids", "segment_ids", "loss_weights", "labels"} or key.endswith(
                ("_mask", "_masks")
            ):
                raise
            continue
        if len(column) != sequence_length:
            raise ValueError(
                f"Per-token field {key!r} has length {len(column)}, expected {sequence_length}."
            )
        normalized[key] = column[:max_length]

    if "attention_mask" not in normalized:
        normalized["attention_mask"] = [1] * len(input_ids)
    if any(value not in (0, 1) for value in normalized["attention_mask"]):
        raise ValueError("attention_mask must contain only binary 0/1 values.")
    for mask_name in ("assistant_masks", "assistant_mask", "assistant_tokens_mask"):
        if mask_name in normalized and any(value not in (0, 1) for value in normalized[mask_name]):
            raise ValueError(f"{mask_name} must contain only binary 0/1 values.")
    if "position_ids" in normalized and any(value < 0 for value in normalized["position_ids"]):
        raise ValueError("position_ids must be non-negative.")
    if "loss_weights" in normalized and any(value < 0.0 for value in normalized["loss_weights"]):
        raise ValueError("loss_weights must be non-negative.")
    return normalized


def _validate_consistent_fields(examples: Sequence[PackedExample]) -> tuple[str, ...]:
    fields = tuple(examples[0].keys())
    field_set = set(fields)
    for index, example in enumerate(examples[1:], start=1):
        if set(example) != field_set:
            missing = sorted(field_set - set(example))
            extra = sorted(set(example) - field_set)
            raise ValueError(
                f"All examples in a packing batch must have the same per-token fields; "
                f"example {index} is missing {missing} and adds {extra}."
            )
    return fields


def best_fit_decreasing(
    examples: Sequence[PackedExample],
    *,
    max_length: int,
) -> list[list[PackedExample]]:
    """Pack one preprocessing batch using deterministic best-fit decreasing."""

    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if not examples:
        return []

    indexed = list(enumerate(examples))
    indexed.sort(key=lambda item: (-len(item[1]["input_ids"]), item[0]))
    bins: list[list[PackedExample]] = []
    used: list[int] = []

    for _, example in indexed:
        length = len(example["input_ids"])
        if length > max_length:
            raise ValueError(f"Sequence length {length} exceeds packing max_length {max_length}.")

        best_index: int | None = None
        best_remaining: int | None = None
        for index, occupied in enumerate(used):
            remaining_after = max_length - occupied - length
            if remaining_after < 0:
                continue
            if best_remaining is None or remaining_after < best_remaining:
                best_index = index
                best_remaining = remaining_after

        if best_index is None:
            bins.append([example])
            used.append(length)
        else:
            bins[best_index].append(example)
            used[best_index] += length

    return bins


def _padding_value(field_name: str, pad_token_id: int) -> Scalar:
    if field_name == "input_ids":
        return int(pad_token_id)
    if field_name == "loss_weights":
        return 0.0
    return 0


def _sequence_metadata(
    attention_mask: Sequence[int],
    *,
    segment_offset: int = 0,
) -> tuple[list[int], list[int], int]:
    """Build positions and unique segments for contiguous valid-token runs."""

    position_ids: list[int] = []
    segment_ids: list[int] = []
    current_segment = 0
    current_position = 0
    last_segment = segment_offset
    for mask_value in attention_mask:
        if int(mask_value) == 1:
            if current_segment == 0:
                last_segment += 1
                current_segment = last_segment
                current_position = 0
            position_ids.append(current_position)
            segment_ids.append(current_segment)
            current_position += 1
        else:
            position_ids.append(0)
            segment_ids.append(0)
            current_segment = 0
            current_position = 0
    return position_ids, segment_ids, last_segment


def pad_example(
    example: Mapping[str, Sequence[Scalar]],
    *,
    max_length: int,
    pad_token_id: int,
    padding_side: str = "left",
) -> PackedExample:
    """Pad an already aligned example to a fixed sequence length."""

    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be either 'left' or 'right'.")
    if "input_ids" not in example:
        raise ValueError("example must contain input_ids.")

    columns = {key: _as_field_list(value, field_name=key) for key, value in example.items()}
    length = len(columns["input_ids"])
    if length > max_length:
        raise ValueError(f"Packed sequence length {length} exceeds max_length {max_length}.")
    for key, column in columns.items():
        if len(column) != length:
            raise ValueError(f"Packed field {key!r} is not aligned with input_ids.")

    pad_size = max_length - length
    output: PackedExample = {}
    for key, column in columns.items():
        padding = [_padding_value(key, pad_token_id)] * pad_size
        output[key] = padding + column if padding_side == "left" else column + padding
    return output


def _materialize_bin(
    packed_bin: Sequence[PackedExample],
    *,
    fields: Sequence[str],
    max_length: int,
    pad_token_id: int,
    padding_side: str,
) -> PackedExample:
    result: PackedExample = {field: [] for field in fields}
    generated_position_ids: list[int] = []
    segment_ids: list[int] = []
    segment_offset = 0
    preserve_position_ids = "position_ids" in fields

    for example in packed_bin:
        for field in fields:
            result[field].extend(example[field])
        example_positions, example_segments, segment_offset = _sequence_metadata(
            example["attention_mask"],
            segment_offset=segment_offset,
        )
        if not preserve_position_ids:
            generated_position_ids.extend(example_positions)
        segment_ids.extend(example_segments)

    # Fresh segment IDs always win over source metadata. Explicit position IDs
    # are the one exception because terminal anchors intentionally use a
    # non-contiguous logical position range.
    if not preserve_position_ids:
        result["position_ids"] = generated_position_ids
    result["segment_ids"] = segment_ids
    return pad_example(
        result,
        max_length=max_length,
        pad_token_id=pad_token_id,
        padding_side=padding_side,
    )


def _batched(records: Iterable[Mapping[str, Any]], batch_size: int) -> Iterator[list[Mapping[str, Any]]]:
    iterator = iter(records)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def pack_examples(
    examples: Iterable[Mapping[str, Any]],
    *,
    max_length: int,
    pad_token_id: int,
    packing_batch_size: int = 1000,
    padding_side: str = "left",
) -> list[PackedExample]:
    """Pack examples in independent BFD batches and pad every result.

    No token is inserted between examples, and raw PT records intentionally do
    not receive EOS. Isolation never relies on a separator token: `segment_ids`
    provide the boundaries needed by block-diagonal causal attention, and the
    shifted loss rejects cross-segment prediction edges.
    """

    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if packing_batch_size <= 0:
        raise ValueError("packing_batch_size must be positive.")

    output: list[PackedExample] = []
    for batch in _batched(examples, packing_batch_size):
        normalized = [_normalize_example(example, max_length) for example in batch]
        fields = _validate_consistent_fields(normalized)
        # Segment IDs are always regenerated for leakage safety. Explicit
        # position IDs are preserved for terminal-anchor training; ordinary
        # examples receive zero-based positions generated per segment.
        source_fields = tuple(field for field in fields if field != "segment_ids")
        bins = best_fit_decreasing(normalized, max_length=max_length)
        output.extend(
            _materialize_bin(
                packed_bin,
                fields=source_fields,
                max_length=max_length,
                pad_token_id=pad_token_id,
                padding_side=padding_side,
            )
            for packed_bin in bins
        )
    return output


def pack_dataset(
    dataset: Any,
    *,
    max_length: int,
    pad_token_id: int,
    packing_batch_size: int = 1000,
    padding_side: str = "left",
    num_proc: int | None = None,
) -> Any:
    """Pack a Hugging Face Dataset-like object one map batch at a time.

    This wrapper remains dependency-free: objects without a ``map`` method are
    treated as ordinary iterables and return a list.
    """

    map_method = getattr(dataset, "map", None)
    if not callable(map_method):
        return pack_examples(
            dataset,
            max_length=max_length,
            pad_token_id=pad_token_id,
            packing_batch_size=packing_batch_size,
            padding_side=padding_side,
        )

    def pack_batch(batch: Mapping[str, Sequence[Any]]) -> dict[str, list[list[Scalar]]]:
        if "input_ids" not in batch:
            raise ValueError("Dataset batch must contain input_ids.")
        batch_length = len(batch["input_ids"])
        rows = [
            {key: values[index] for key, values in batch.items()}
            for index in range(batch_length)
        ]
        packed = pack_examples(
            rows,
            max_length=max_length,
            pad_token_id=pad_token_id,
            packing_batch_size=packing_batch_size,
            padding_side=padding_side,
        )
        if not packed:
            return {}
        return {key: [row[key] for row in packed] for key in packed[0]}

    map_kwargs: dict[str, Any] = {
        "batched": True,
        "batch_size": packing_batch_size,
    }
    column_names = getattr(dataset, "column_names", None)
    if column_names is not None:
        map_kwargs["remove_columns"] = list(column_names)
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc
    return map_method(pack_batch, **map_kwargs)


def pad_examples(
    examples: Iterable[Mapping[str, Any]],
    *,
    max_length: int,
    pad_token_id: int,
    padding_side: str = "left",
) -> list[PackedExample]:
    """Truncate and pad examples without combining independent samples."""

    output: list[PackedExample] = []
    for example in examples:
        normalized = _normalize_example(example, max_length)
        position_ids, segment_ids, _ = _sequence_metadata(normalized["attention_mask"])
        if "position_ids" not in normalized:
            normalized["position_ids"] = position_ids
        normalized["segment_ids"] = segment_ids
        output.append(
            pad_example(
                normalized,
                max_length=max_length,
                pad_token_id=pad_token_id,
                padding_side=padding_side,
            )
        )
    return output


__all__ = [
    "PackedExample",
    "Scalar",
    "best_fit_decreasing",
    "pack_dataset",
    "pack_examples",
    "pad_example",
    "pad_examples",
]
