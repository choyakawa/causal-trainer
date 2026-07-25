from dataclasses import dataclass, replace
from types import SimpleNamespace

from causal_trainer.training.arguments import parse_args
from causal_trainer.training.runner import (
    _buffered_streaming_schedule_total,
    _distributed_arguments_digest,
    _project_streaming_progress_total,
    _resume_signature_payload,
    _streaming_schedule_position,
)


@dataclass(frozen=True)
class _Plan:
    total_steps: int = 1


class _Config:
    @staticmethod
    def to_dict() -> dict[str, int]:
        return {"num_hidden_layers": 40}


def test_streaming_schedule_position_survives_underestimated_metadata() -> None:
    batches = [
        SimpleNamespace(epoch=0, epoch_source_examples_before=0),
        SimpleNamespace(epoch=0, epoch_source_examples_before=2),
        SimpleNamespace(epoch=0, epoch_source_examples_before=3),
        SimpleNamespace(epoch=0, epoch_source_examples_before=100),
        SimpleNamespace(epoch=1, epoch_source_examples_before=0),
        SimpleNamespace(epoch=1, epoch_source_examples_before=100),
        SimpleNamespace(epoch=2, epoch_source_examples_before=100),
    ]

    positions = [
        _streaming_schedule_position(batch, 3)
        for batch in batches
    ]

    assert positions == [0, 2, 3, 3, 3, 6, 9]
    assert positions == sorted(positions)
    assert positions[-1] < _buffered_streaming_schedule_total(3 * 3)


def test_streaming_schedule_uses_fixed_five_percent_metadata_buffer() -> None:
    assert _buffered_streaming_schedule_total(0) == 1
    assert _buffered_streaming_schedule_total(1) == 2
    assert _buffered_streaming_schedule_total(19) == 20
    assert _buffered_streaming_schedule_total(20) == 21
    assert _buffered_streaming_schedule_total(100) == 105
    assert _buffered_streaming_schedule_total(101) == 107


def test_streaming_schedule_freezes_when_real_rows_exceed_metadata() -> None:
    batches = [
        SimpleNamespace(epoch=0, epoch_source_examples_before=before)
        for before in (3, 4, 100)
    ]

    assert [
        _streaming_schedule_position(batch, 3)
        for batch in batches
    ] == [3, 3, 3]
    assert 3 < _buffered_streaming_schedule_total(3)

    zero_estimate_batch = SimpleNamespace(
        epoch=10,
        epoch_source_examples_before=100,
    )
    assert _streaming_schedule_position(zero_estimate_batch, 0) == 0


def test_real_epoch_boundaries_correct_progress_total_estimate() -> None:
    assert _project_streaming_progress_total(5, 1, 3) == 15
    assert _project_streaming_progress_total(10, 2, 3) == 15
    assert _project_streaming_progress_total(14, 3, 3) == 14


def test_resolved_layer_scan_is_part_of_distributed_and_resume_identity() -> None:
    scanned = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--max-sequence-length",
            "524288",
        ]
    )
    unrolled = replace(scanned, scan_layers=False)

    assert _distributed_arguments_digest(scanned) != _distributed_arguments_digest(
        unrolled
    )

    payload = _resume_signature_payload(
        scanned,
        _Config(),
        _Plan(),
        SimpleNamespace(_fingerprint="synthetic"),
        SimpleNamespace(shape={"dp": 1, "fsdp": 1, "ep": 1, "tp": 8, "sp": 4}),
        preprocessing_process_count=8,
        prepared_real_rows=1,
        prepared_dummy_rows=0,
    )
    assert payload["arguments"]["scan_layers"] is True
    assert payload["streaming_scheduler_policy"] is None

    streaming_payload = _resume_signature_payload(
        replace(scanned, dataset_streaming=True),
        _Config(),
        _Plan(),
        SimpleNamespace(_fingerprint="synthetic"),
        SimpleNamespace(shape={"dp": 1, "fsdp": 1, "ep": 1, "tp": 8, "sp": 4}),
        preprocessing_process_count=8,
        prepared_real_rows=1,
        prepared_dummy_rows=0,
    )
    assert (
        streaming_payload["streaming_scheduler_policy"]
        == "metadata-105pct-horizon-cap-at-metadata-v1"
    )
    assert "hf_retry_initial_delay" not in payload["arguments"]
    assert "hf_retry_max_delay" not in payload["arguments"]


def test_full_parameter_freezing_is_part_of_distributed_and_resume_identity() -> None:
    unfrozen = parse_args(
        ["--repo-id", "local-model", "--dataset-name", "local-data"]
    )
    frozen = replace(unfrozen, frozen_parameters="lm_head|embed|norm")

    assert _distributed_arguments_digest(unfrozen) != _distributed_arguments_digest(frozen)

    payload = _resume_signature_payload(
        frozen,
        _Config(),
        _Plan(),
        SimpleNamespace(_fingerprint="synthetic"),
        SimpleNamespace(shape={"dp": 1, "fsdp": 1, "ep": 1, "tp": 8, "sp": 4}),
        preprocessing_process_count=8,
        prepared_real_rows=1,
        prepared_dummy_rows=0,
    )
    assert payload["arguments"]["frozen_parameters"] == "lm_head|embed|norm"
