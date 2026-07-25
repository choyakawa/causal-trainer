from itertools import pairwise

import numpy as np
import pytest
from datasets import Dataset, Features, Sequence, Value

from causal_trainer.data.pipeline import (
    contiguous_shard_bounds,
    load_training_split,
    packed_dataset_to_arrays,
    pad_packed_arrays_to_batch_multiple,
    prepare_training_dataset,
    select_process_shard,
)
from causal_trainer.data.preprocessing import EndPromptSettings, preprocess_dataset


def _canonical_columns(rows: int = 3, sequence_length: int = 4) -> dict[str, np.ndarray]:
    input_ids = np.arange(rows * sequence_length, dtype=np.int32).reshape(rows, sequence_length)
    return {
        "input_ids": input_ids,
        "attention_mask": np.ones_like(input_ids),
        "position_ids": np.broadcast_to(
            np.arange(sequence_length, dtype=np.int32),
            input_ids.shape,
        ).copy(),
        "segment_ids": np.ones_like(input_ids),
        "assistant_masks": np.ones_like(input_ids),
        "loss_weights": np.ones(input_ids.shape, dtype=np.float32),
    }


def test_hub_streaming_loader_forwards_config_revision_and_token(monkeypatch) -> None:
    calls = []
    sentinel = object()

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr("causal_trainer.data.pipeline.load_dataset", fake_load_dataset)

    result = load_training_split(
        "org/large-corpus",
        "train",
        token="secret",
        config_name="clean",
        revision="dataset-commit",
        streaming=True,
    )

    assert result is sentinel
    assert calls == [
        (
            ("org/large-corpus", "clean"),
            {
                "split": "train",
                "token": "secret",
                "revision": "dataset-commit",
                "streaming": True,
            },
        )
    ]


@pytest.mark.parametrize("size", [0, 1, 2, 7, 10])
@pytest.mark.parametrize("process_count", [1, 2, 3, 4, 12])
def test_contiguous_shards_cover_every_source_row_once(size: int, process_count: int) -> None:
    intervals = [
        contiguous_shard_bounds(size, process_index, process_count)
        for process_index in range(process_count)
    ]
    flattened = [index for start, stop in intervals for index in range(start, stop)]
    assert flattened == list(range(size))
    assert all(left[1] == right[0] for left, right in pairwise(intervals))


def test_real_dataset_empty_process_shard_keeps_schema_and_prepares_canonical_arrays() -> None:
    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = None

        def __call__(self, text, **kwargs):
            del text, kwargs
            return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    source = Dataset.from_dict({"text": ["a", "b", "c"]})
    shard, start = select_process_shard(source, process_index=7, process_count=8)

    assert start == len(source)
    assert len(shard) == 0
    assert shard.column_names == source.column_names
    assert shard.features == source.features
    assert shard.cache_files == []

    prepared = prepare_training_dataset(
        shard,
        TinyTokenizer(),
        dataset_text_field="text",
        max_sequence_length=8,
        assistant_only_loss=False,
        endprompt=None,
        packing=True,
        packing_batch_size=4,
        preprocessing_num_workers=None,
        example_index_offset=start,
        allow_empty=True,
    )
    columns = packed_dataset_to_arrays(
        prepared,
        max_sequence_length=8,
        assistant_only_loss=False,
    )

    assert prepared.cache_files == []
    assert all(value.shape == (0, 8) for value in columns.values())


def test_canonical_conversion_materializes_normal_loss_weights() -> None:
    features = Features(
        {
            name: Sequence(Value("int32"), length=3)
            for name in ("input_ids", "attention_mask", "position_ids", "segment_ids")
        }
    )
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[0, 1, 1]],
            "position_ids": [[0, 0, 1]],
            "segment_ids": [[0, 1, 1]],
        },
        features=features,
    )
    columns = packed_dataset_to_arrays(
        dataset,
        max_sequence_length=3,
        assistant_only_loss=False,
    )
    np.testing.assert_array_equal(columns["loss_weights"], [[0.0, 1.0, 1.0]])


def test_canonical_empty_assistant_shard_has_fixed_shapes() -> None:
    features = Features(
        {
            name: Sequence(Value("int32"), length=5)
            for name in (
                "input_ids",
                "attention_mask",
                "position_ids",
                "segment_ids",
                "assistant_masks",
            )
        }
    )
    dataset = Dataset.from_dict({name: [] for name in features}, features=features)
    columns = packed_dataset_to_arrays(
        dataset,
        max_sequence_length=5,
        assistant_only_loss=True,
    )
    assert set(columns) == {
        "input_ids",
        "attention_mask",
        "position_ids",
        "segment_ids",
        "assistant_masks",
        "loss_weights",
    }
    assert all(value.shape == (0, 5) for value in columns.values())
    assert columns["loss_weights"].dtype == np.float32


def test_tail_padding_keeps_every_real_row_and_adds_zero_loss_context() -> None:
    columns = _canonical_columns()
    dataset, dummy_rows = pad_packed_arrays_to_batch_multiple(columns, batch_multiple=4)
    assert dummy_rows == 1
    assert len(dataset) == 4
    assert dataset.real_rows == 3
    assert dataset.dummy_rows == 1
    for name, original in columns.items():
        assert dataset.columns[name] is original
    dummy = dataset[-1]
    np.testing.assert_array_equal(dummy["attention_mask"], columns["attention_mask"][-1])
    np.testing.assert_array_equal(
        dummy["loss_weights"],
        np.zeros_like(dummy["loss_weights"]),
    )
    np.testing.assert_array_equal(
        dummy["assistant_masks"],
        np.zeros_like(dummy["assistant_masks"]),
    )


def test_global_empty_prepared_dataset_is_rejected_before_dummy_padding() -> None:
    columns = {
        name: value[:0]
        for name, value in _canonical_columns(rows=1).items()
    }
    with pytest.raises(ValueError, match="no trainable packed rows"):
        pad_packed_arrays_to_batch_multiple(columns, batch_multiple=8)


def test_global_dataset_without_a_shifted_target_fails_without_dropping_rows() -> None:
    columns = _canonical_columns(rows=1, sequence_length=4)
    columns["attention_mask"][:] = [[0, 0, 0, 1]]
    columns["segment_ids"][:] = [[0, 0, 0, 1]]
    columns["loss_weights"][:] = columns["attention_mask"]
    with pytest.raises(ValueError, match="packed row 0 has no positive-weight causal target"):
        pad_packed_arrays_to_batch_multiple(columns, batch_multiple=8)


def test_zero_target_real_row_is_not_silently_dropped_from_mixed_data() -> None:
    columns = _canonical_columns(rows=2, sequence_length=4)
    columns["attention_mask"][0] = [0, 0, 0, 1]
    columns["segment_ids"][0] = [0, 0, 0, 1]
    columns["loss_weights"][0] = columns["attention_mask"][0]
    with pytest.raises(ValueError, match="will not silently drop it"):
        pad_packed_arrays_to_batch_multiple(columns, batch_multiple=2)


def test_tokenize_and_pack_transforms_do_not_create_arrow_cache_files() -> None:
    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = None

        def __call__(self, text, **kwargs):
            length = min(len(text), kwargs["max_length"])
            return {
                "input_ids": list(range(1, length + 1)),
                "attention_mask": [1] * length,
            }

    prepared = prepare_training_dataset(
        Dataset.from_dict({"text": ["abc", "de"]}),
        TinyTokenizer(),
        dataset_text_field="text",
        max_sequence_length=4,
        assistant_only_loss=False,
        endprompt=None,
        packing=True,
        packing_batch_size=2,
        preprocessing_num_workers=None,
    )
    assert prepared.cache_files == []


def test_real_selected_dataset_uses_global_endprompt_row_indices() -> None:
    class PromptTokenizer:
        def __call__(self, text, **kwargs):
            if text == "first":
                return {"input_ids": [90]}
            if text == "second":
                return {"input_ids": [91]}
            return {"input_ids": [1]}

    source = Dataset.from_dict({"text": ["x"] * 5})
    shard, start = select_process_shard(source, process_index=1, process_count=2)
    tokenized = preprocess_dataset(
        shard,
        PromptTokenizer(),
        dataset_text_field="text",
        max_sequence_length=4,
        endprompt=EndPromptSettings(4, 4, ("first", "second")),
        example_index_offset=start,
    )
    assert start == 3
    assert [row[-1] for row in tokenized["input_ids"]] == [91, 90]
