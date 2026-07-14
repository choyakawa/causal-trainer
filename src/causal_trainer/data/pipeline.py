from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, Features, Value, load_dataset
from datasets import Sequence as DatasetSequence

from .packing import PackedExample, Scalar, pack_examples, pad_examples
from .preprocessing import EndPromptSettings, preprocess_dataset

MODEL_FIELDS = ("input_ids", "attention_mask", "position_ids", "segment_ids")
_CANONICAL_FIELD_ORDER = (*MODEL_FIELDS, "assistant_masks", "loss_weights")


class PackedArrayDataset:
    """A fixed-shape, in-memory prepared dataset with a stable fingerprint."""

    def __init__(
        self,
        columns: Mapping[str, np.ndarray],
        *,
        dummy_rows: int = 0,
    ) -> None:
        if not columns:
            raise ValueError("packed dataset must contain columns")
        if dummy_rows < 0:
            raise ValueError(f"dummy_rows cannot be negative, got {dummy_rows}")
        normalized: dict[str, np.ndarray] = {}
        rows: int | None = None
        sequence_length: int | None = None
        for name in _CANONICAL_FIELD_ORDER:
            if name not in columns:
                continue
            value = np.ascontiguousarray(columns[name])
            if value.ndim != 2:
                raise ValueError(f"packed field {name!r} must have rank 2, got {value.shape}")
            if rows is None:
                rows, sequence_length = value.shape
            elif value.shape != (rows, sequence_length):
                raise ValueError(
                    f"packed field {name!r} has shape {value.shape}, expected {(rows, sequence_length)}"
                )
            normalized[name] = value
        extra = sorted(set(columns) - set(normalized))
        if extra:
            raise ValueError(f"unsupported packed dataset fields: {extra}")
        missing = [name for name in (*MODEL_FIELDS, "loss_weights") if name not in normalized]
        if missing:
            raise ValueError(f"packed dataset is missing canonical fields: {missing}")
        real_rows = int(rows or 0)
        if dummy_rows and real_rows == 0:
            raise ValueError("dummy rows require at least one real packed row")
        self._columns = normalized
        self._real_rows = real_rows
        self._dummy_rows = int(dummy_rows)
        self._rows = real_rows + self._dummy_rows
        self._sequence_length = int(sequence_length or 0)
        self._dummy_record = (
            {
                name: (
                    np.zeros((self._sequence_length,), dtype=value.dtype)
                    if name in {"assistant_masks", "loss_weights"}
                    else value[-1]
                )
                for name, value in normalized.items()
            }
            if self._dummy_rows
            else None
        )
        self._fingerprint_value: str | None = None

    @property
    def _fingerprint(self) -> str:
        if self._fingerprint_value is None:
            self._fingerprint_value = _packed_arrays_fingerprint(
                self._columns,
                dummy_rows=self._dummy_rows,
            )
        return self._fingerprint_value

    @property
    def columns(self) -> Mapping[str, np.ndarray]:
        return self._columns

    @property
    def column_names(self) -> list[str]:
        return list(self._columns)

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def real_rows(self) -> int:
        return self._real_rows

    @property
    def dummy_rows(self) -> int:
        return self._dummy_rows

    def __len__(self) -> int:
        return self._rows

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if index < 0:
            index += self._rows
        if not 0 <= index < self._rows:
            raise IndexError(index)
        if index >= self._real_rows:
            if self._dummy_record is None:
                raise RuntimeError("dummy row metadata is unavailable")
            return dict(self._dummy_record)
        return {name: value[index] for name, value in self._columns.items()}


def load_training_split(name: str, split: str, *, token: str | None = None) -> Dataset:
    """Load either a Hub dataset or a common local tabular file."""

    path = Path(name)
    if path.is_file():
        suffix = path.suffix.lower()
        loader = {
            ".json": "json",
            ".jsonl": "json",
            ".parquet": "parquet",
            ".csv": "csv",
            ".tsv": "csv",
        }.get(suffix)
        if loader is None:
            raise ValueError(f"unsupported local dataset extension: {suffix}")
        kwargs: dict[str, Any] = {"data_files": str(path), "split": split}
        if suffix == ".tsv":
            kwargs["delimiter"] = "\t"
        return load_dataset(loader, **kwargs)
    return load_dataset(name, split=split, token=token)


def contiguous_shard_bounds(size: int, process_index: int, process_count: int) -> tuple[int, int]:
    """Return one balanced, gap-free interval of ``range(size)``."""

    if size < 0:
        raise ValueError(f"dataset size cannot be negative, got {size}")
    if process_count <= 0:
        raise ValueError(f"process_count must be positive, got {process_count}")
    if not 0 <= process_index < process_count:
        raise ValueError(
            f"process_index must be in [0, {process_count}), got {process_index}"
        )
    base, remainder = divmod(size, process_count)
    start = process_index * base + min(process_index, remainder)
    stop = start + base + int(process_index < remainder)
    return start, stop


def select_process_shard(
    dataset: Dataset,
    *,
    process_index: int,
    process_count: int,
) -> tuple[Dataset, int]:
    """Select a contiguous source shard without creating an indices cache file."""

    start, stop = contiguous_shard_bounds(len(dataset), process_index, process_count)
    if process_count == 1:
        return dataset, 0
    if start == stop:
        # ``datasets.Dataset.select(range(size, size))`` routes through the
        # contiguous fast path and rejects ``start == len(dataset)`` even
        # though the requested interval is empty.  Selecting an explicit
        # empty index list preserves the source schema and lets zero-row ranks
        # participate in the later fixed-schema collective merge.
        return dataset.select([], keep_in_memory=True), start
    return dataset.select(range(start, stop), keep_in_memory=True), start


def source_dataset_identity(dataset: Dataset) -> np.ndarray:
    """Return a fixed-size identity suitable for a cross-process equality check."""

    features = getattr(dataset, "features", None)
    if hasattr(features, "to_dict"):
        features = features.to_dict()
    payload = {
        "rows": len(dataset),
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "columns": list(getattr(dataset, "column_names", ())),
        "features": features,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return np.frombuffer(hashlib.sha256(encoded).digest(), dtype=np.uint8).copy()


def _records_from_batch(batch: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not batch:
        return []
    size = len(next(iter(batch.values())))
    return [{key: values[index] for key, values in batch.items()} for index in range(size)]


def _columns_from_records(
    records: Sequence[PackedExample],
) -> dict[str, list[list[Scalar]]]:
    if not records:
        return {}
    return {key: [record[key] for record in records] for key in records[0]}


def _prepared_features(
    *,
    max_sequence_length: int,
    assistant_only_loss: bool,
    has_explicit_loss_weights: bool,
) -> Features:
    names = list(MODEL_FIELDS)
    if assistant_only_loss:
        names.append("assistant_masks")
    if has_explicit_loss_weights:
        names.append("loss_weights")
    return Features(
        {
            name: DatasetSequence(
                Value("float32" if name == "loss_weights" else "int32"),
                length=max_sequence_length,
            )
            for name in names
        }
    )


def prepare_training_dataset(
    dataset: Dataset,
    tokenizer,
    *,
    dataset_text_field: str,
    max_sequence_length: int,
    assistant_only_loss: bool,
    endprompt: EndPromptSettings | None,
    packing: bool,
    packing_batch_size: int,
    preprocessing_num_workers: int | None,
    example_index_offset: int = 0,
    allow_empty: bool = False,
) -> Dataset:
    """Tokenize and finalize one source shard entirely in host memory."""

    tokenized = preprocess_dataset(
        dataset,
        tokenizer,
        dataset_text_field=dataset_text_field,
        max_sequence_length=max_sequence_length,
        assistant_only_loss=assistant_only_loss,
        endprompt=endprompt,
        num_proc=preprocessing_num_workers,
        example_index_offset=example_index_offset,
        allow_empty=allow_empty,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("the tokenizer must define pad_token_id or eos_token_id")

    final_features = _prepared_features(
        max_sequence_length=max_sequence_length,
        assistant_only_loss=assistant_only_loss,
        has_explicit_loss_weights=endprompt is not None,
    )
    if len(tokenized) == 0:
        if not allow_empty:
            raise ValueError("preprocessing produced no trainable records")
        return Dataset.from_dict(
            {name: [] for name in final_features},
            features=final_features,
        )

    def finalize_batch(
        batch: dict[str, list[Any]],
    ) -> dict[str, list[list[Scalar]]]:
        records = _records_from_batch(batch)
        if packing:
            finalized = pack_examples(
                records,
                max_length=max_sequence_length,
                pad_token_id=int(pad_token_id),
                packing_batch_size=len(records) or 1,
                padding_side="left",
            )
        else:
            finalized = pad_examples(
                records,
                max_length=max_sequence_length,
                pad_token_id=int(pad_token_id),
                padding_side="left",
            )
        return _columns_from_records(finalized)

    return tokenized.map(
        finalize_batch,
        batched=True,
        batch_size=packing_batch_size,
        remove_columns=tokenized.column_names,
        features=final_features,
        keep_in_memory=True,
        load_from_cache_file=False,
        desc="Packing records" if packing else "Padding records",
    )


def packed_dataset_to_arrays(
    dataset: Dataset,
    *,
    max_sequence_length: int,
    assistant_only_loss: bool,
) -> dict[str, np.ndarray]:
    """Convert a local Arrow result into the canonical typed merge payload."""

    rows = len(dataset)
    actual = set(dataset.column_names)
    required = set(MODEL_FIELDS)
    if assistant_only_loss:
        required.add("assistant_masks")
    missing = sorted(required - actual)
    extra = sorted(actual - required - {"loss_weights"})
    if missing or extra:
        raise ValueError(f"prepared dataset schema mismatch: missing={missing}, extra={extra}")

    columns: dict[str, np.ndarray] = {}
    names = [*MODEL_FIELDS]
    if assistant_only_loss:
        names.append("assistant_masks")
    for name in names:
        if rows:
            value = np.asarray(dataset[name], dtype=np.int32)
        else:
            value = np.empty((0, max_sequence_length), dtype=np.int32)
        if value.shape != (rows, max_sequence_length):
            raise ValueError(
                f"prepared field {name!r} has shape {value.shape}, expected {(rows, max_sequence_length)}"
            )
        columns[name] = np.ascontiguousarray(value)

    if "loss_weights" in actual:
        loss_weights = (
            np.asarray(dataset["loss_weights"], dtype=np.float32)
            if rows
            else np.empty((0, max_sequence_length), dtype=np.float32)
        )
    elif assistant_only_loss:
        loss_weights = columns["assistant_masks"].astype(np.float32, copy=True)
    else:
        loss_weights = columns["attention_mask"].astype(np.float32, copy=True)
    if loss_weights.shape != (rows, max_sequence_length):
        raise ValueError(
            f"prepared field 'loss_weights' has shape {loss_weights.shape}, "
            f"expected {(rows, max_sequence_length)}"
        )
    if not np.isfinite(loss_weights).all() or (loss_weights < 0).any():
        raise ValueError("prepared loss_weights must be finite and non-negative")
    columns["loss_weights"] = np.ascontiguousarray(loss_weights)
    return columns


def _packed_arrays_fingerprint(
    columns: Mapping[str, np.ndarray],
    *,
    dummy_rows: int = 0,
) -> str:
    hasher = hashlib.sha256()
    for name in _CANONICAL_FIELD_ORDER:
        if name not in columns:
            continue
        value = np.ascontiguousarray(columns[name])
        hasher.update(name.encode("utf-8"))
        hasher.update(value.dtype.str.encode("ascii"))
        hasher.update(json.dumps(value.shape).encode("ascii"))
        hasher.update(memoryview(value).cast("B"))
    hasher.update(b"virtual-zero-loss-dummy-rows-v1\0")
    hasher.update(int(dummy_rows).to_bytes(8, "little", signed=False))
    return hasher.hexdigest()


def _first_row_without_positive_shifted_target(
    columns: Mapping[str, np.ndarray],
    *,
    rows_per_chunk: int = 1024,
) -> int | None:
    attention_mask = columns["attention_mask"]
    segment_ids = columns["segment_ids"]
    loss_weights = columns["loss_weights"]
    for start in range(0, attention_mask.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, attention_mask.shape[0])
        attention = attention_mask[start:stop]
        segments = segment_ids[start:stop]
        weights = loss_weights[start:stop]
        valid_edge = (
            (attention[:, :-1] != 0)
            & (attention[:, 1:] != 0)
            & (segments[:, :-1] != 0)
            & (segments[:, :-1] == segments[:, 1:])
            & (weights[:, 1:] > 0)
        )
        row_has_target = np.any(valid_edge, axis=1)
        if not np.all(row_has_target):
            return start + int(np.flatnonzero(~row_has_target)[0])
    return None


def pad_packed_arrays_to_batch_multiple(
    columns: Mapping[str, np.ndarray],
    *,
    batch_multiple: int,
) -> tuple[PackedArrayDataset, int]:
    """Append finite, zero-loss duplicate rows so batching never drops a real row."""

    if batch_multiple <= 0:
        raise ValueError(f"batch_multiple must be positive, got {batch_multiple}")
    dataset = PackedArrayDataset(columns)
    if len(dataset) == 0:
        raise ValueError("global preprocessing produced no trainable packed rows")
    empty_target_row = _first_row_without_positive_shifted_target(dataset.columns)
    if empty_target_row is not None:
        raise ValueError(
            f"prepared packed row {empty_target_row} has no positive-weight causal target with a valid "
            "same-segment predecessor; the trainer will not silently drop it or execute an empty optimizer step"
        )
    dummy_rows = (-len(dataset)) % batch_multiple
    if dummy_rows == 0:
        return dataset, 0

    # Keep dummy rows virtual: copying every complete prepared column merely to
    # append a handful of rows can double host-memory use.  Indexed dummy rows
    # reuse the last finite attention context and substitute preallocated zero
    # assistant/loss rows.
    return PackedArrayDataset(columns, dummy_rows=dummy_rows), dummy_rows


__all__ = [
    "PackedArrayDataset",
    "contiguous_shard_bounds",
    "load_training_split",
    "packed_dataset_to_arrays",
    "pad_packed_arrays_to_batch_multiple",
    "prepare_training_dataset",
    "select_process_shard",
    "source_dataset_identity",
]
