from dataclasses import replace

import numpy as np
import pytest

from causal_trainer.data.streaming import StreamingBatchEnvelope
from causal_trainer.data.streaming_batching import (
    StreamingBatchPlan,
    iter_streaming_global_batches,
    streaming_global_batch_digest,
)


def _record(value: int) -> dict[str, list[int] | list[float]]:
    return {
        "input_ids": [0, value, value + 1, value + 2],
        "attention_mask": [0, 1, 1, 1],
        "position_ids": [0, 0, 1, 2],
        "segment_ids": [0, 1, 1, 1],
        "loss_weights": [0.0, 1.0, 1.0, 1.0],
    }


def _end(epoch: int) -> StreamingBatchEnvelope:
    return StreamingBatchEnvelope((), 0, (), epoch, is_epoch_end=True)


def test_streaming_tail_is_zero_loss_padded_and_progress_uses_source_rows() -> None:
    envelopes = [
        StreamingBatchEnvelope(
            (_record(10), _record(20), _record(30)),
            3,
            (1, 1, 1),
            0,
        ),
        _end(0),
    ]
    plan = StreamingBatchPlan.create(3, 2, 1, 1, -1)

    batches = list(
        iter_streaming_global_batches(
            envelopes,
            plan,
            assistant_only_loss=False,
        )
    )

    assert [batch.source_examples_seen for batch in batches] == [2, 3]
    assert [batch.epoch_source_examples_before for batch in batches] == [0, 2]
    assert [batch.epoch_source_examples_seen for batch in batches] == [2, 3]
    assert batches[-1].is_epoch_end is True
    assert batches[-1].data["input_ids"].shape == (2, 4)
    np.testing.assert_array_equal(batches[-1].data["loss_weights"][1], 0.0)


def test_fully_filtered_tail_is_committed_with_last_trainable_batch() -> None:
    envelopes = [
        StreamingBatchEnvelope(
            (_record(10), _record(20)),
            2,
            (1, 1),
            0,
        ),
        StreamingBatchEnvelope((), 3, (), 0),
        _end(0),
    ]
    plan = StreamingBatchPlan.create(5, 2, 1, 1, -1)

    [batch] = list(
        iter_streaming_global_batches(
            envelopes,
            plan,
            assistant_only_loss=False,
        )
    )

    assert batch.source_examples == 5
    assert batch.source_examples_seen == 5
    assert batch.is_epoch_end is True


def test_fully_filtered_epoch_fails_with_an_explicit_error() -> None:
    plan = StreamingBatchPlan.create(3, 2, 1, 1, -1)

    with pytest.raises(ValueError, match="produced no trainable records"):
        list(
            iter_streaming_global_batches(
                [
                    StreamingBatchEnvelope((), 3, (), 0),
                    _end(0),
                ],
                plan,
                assistant_only_loss=True,
            )
        )


def test_streaming_epoch_boundary_flushes_without_cross_epoch_packing() -> None:
    envelopes = [
        StreamingBatchEnvelope((_record(10),), 1, (1,), 0),
        _end(0),
        StreamingBatchEnvelope((_record(20),), 1, (1,), 1),
        _end(1),
    ]
    plan = StreamingBatchPlan.create(1, 2, 1, 2, -1)

    batches = list(
        iter_streaming_global_batches(
            envelopes,
            plan,
            assistant_only_loss=False,
        )
    )

    assert [batch.epoch for batch in batches] == [0, 1]
    assert [batch.source_examples_seen for batch in batches] == [1, 2]
    assert [batch.epoch_source_examples_before for batch in batches] == [0, 0]
    assert [batch.epoch_source_examples_seen for batch in batches] == [1, 1]
    assert all(batch.is_epoch_end for batch in batches)
    np.testing.assert_array_equal(batches[0].data["loss_weights"][1], 0.0)
    np.testing.assert_array_equal(batches[1].data["loss_weights"][1], 0.0)


@pytest.mark.parametrize("estimate", [1, 10])
def test_metadata_estimate_does_not_cap_real_epoch_consumption(
    estimate: int,
) -> None:
    plan = StreamingBatchPlan.create(estimate, 2, 1, 1, -1)
    envelopes = [
        StreamingBatchEnvelope(
            (_record(10), _record(20), _record(30)),
            3,
            (1, 1, 1),
            0,
        ),
        _end(0),
    ]

    batches = list(
        iter_streaming_global_batches(
            envelopes,
            plan,
            assistant_only_loss=False,
        )
    )

    assert plan.total_source_examples == estimate
    assert plan.steps_per_epoch == (estimate + 1) // 2
    assert plan.has_exact_step_limit is False
    assert batches[-1].source_examples_seen == 3
    assert batches[-1].is_epoch_end is True


def test_streaming_resume_replays_batches_and_recovers_source_cursor() -> None:
    envelopes = [
        StreamingBatchEnvelope(
            tuple(_record(index * 10) for index in range(1, 6)),
            5,
            (1, 1, 1, 1, 1),
            0,
        ),
        _end(0),
    ]
    plan = StreamingBatchPlan.create(5, 2, 1, 1, -1)

    resumed = list(
        iter_streaming_global_batches(
            envelopes,
            plan,
            assistant_only_loss=False,
            start_step=1,
        )
    )

    assert resumed[0].source_examples_before == 2
    assert resumed[-1].source_examples_seen == 5


def test_streaming_resume_validates_cursor_and_digest_at_exact_eof() -> None:
    envelopes = [
        StreamingBatchEnvelope((_record(10),), 4, (4,), 0),
        _end(0),
    ]
    plan = StreamingBatchPlan.create(4, 1, 1, 1, -1)
    [completed] = list(
        iter_streaming_global_batches(
            envelopes,
            plan,
            assistant_only_loss=False,
        )
    )

    resumed = list(
        iter_streaming_global_batches(
            [
                StreamingBatchEnvelope((_record(10),), 4, (4,), 0),
                _end(0),
            ],
            plan,
            assistant_only_loss=False,
            start_step=1,
            expected_source_examples_before=completed.source_examples_seen,
            expected_stream_digest_before=completed.stream_digest,
        )
    )

    assert resumed == []


@pytest.mark.parametrize(
    ("expected_source_examples", "expected_digest", "match"),
    [
        (3, None, "checkpoint progress records 3"),
        (4, "f" * 64, "different training batches"),
    ],
)
def test_streaming_resume_rejects_mismatched_exact_eof_state(
    expected_source_examples: int,
    expected_digest: str | None,
    match: str,
) -> None:
    plan = StreamingBatchPlan.create(4, 1, 1, 1, -1)

    with pytest.raises(RuntimeError, match=match):
        list(
            iter_streaming_global_batches(
                [
                    StreamingBatchEnvelope((_record(10),), 4, (4,), 0),
                    _end(0),
                ],
                plan,
                assistant_only_loss=False,
                start_step=1,
                expected_source_examples_before=expected_source_examples,
                expected_stream_digest_before=expected_digest,
            )
        )


def test_streaming_resume_rejects_step_beyond_reconstructed_stream() -> None:
    plan = StreamingBatchPlan.create(4, 1, 1, 1, -1)

    with pytest.raises(RuntimeError, match="produced only 1 batches"):
        list(
            iter_streaming_global_batches(
                [
                    StreamingBatchEnvelope((_record(10),), 4, (4,), 0),
                    _end(0),
                ],
                plan,
                assistant_only_loss=False,
                start_step=2,
            )
        )


def test_packing_density_does_not_change_source_progress_total() -> None:
    plan = StreamingBatchPlan.create(4, 2, 1, 1, -1)
    unpacked = [
        StreamingBatchEnvelope(
            tuple(_record(index * 10) for index in range(1, 5)),
            4,
            (1, 1, 1, 1),
            0,
        ),
        _end(0),
    ]
    densely_packed = [
        StreamingBatchEnvelope(
            (_record(10), _record(20)),
            4,
            (2, 2),
            0,
        ),
        _end(0),
    ]

    unpacked_batches = list(
        iter_streaming_global_batches(
            unpacked,
            plan,
            assistant_only_loss=False,
        )
    )
    packed_batches = list(
        iter_streaming_global_batches(
            densely_packed,
            plan,
            assistant_only_loss=False,
        )
    )

    assert len(unpacked_batches) == 2
    assert len(packed_batches) == 1
    assert unpacked_batches[-1].source_examples_seen == 4
    assert packed_batches[-1].source_examples_seen == 4


def test_streaming_global_batch_digest_binds_data_and_progress() -> None:
    envelopes = [
        StreamingBatchEnvelope((_record(10),), 1, (1,), 0),
        _end(0),
    ]
    plan = StreamingBatchPlan.create(1, 1, 1, 1, -1)
    first = next(
        iter_streaming_global_batches(
            envelopes,
            plan,
            assistant_only_loss=False,
        )
    )
    same = replace(
        first,
        data={name: value.copy() for name, value in first.data.items()},
    )
    different_progress = replace(first, source_examples_seen=2)
    different_data = replace(
        first,
        data={
            **first.data,
            "input_ids": first.data["input_ids"] + 1,
        },
    )

    assert streaming_global_batch_digest(first) == streaming_global_batch_digest(same)
    assert streaming_global_batch_digest(first) != streaming_global_batch_digest(
        different_progress
    )
    assert streaming_global_batch_digest(first) != streaming_global_batch_digest(
        different_data
    )
