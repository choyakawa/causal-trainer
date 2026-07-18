from dataclasses import dataclass, replace
from types import SimpleNamespace

from causal_trainer.training.arguments import parse_args
from causal_trainer.training.runner import (
    _distributed_arguments_digest,
    _resume_signature_payload,
)


@dataclass(frozen=True)
class _Plan:
    total_steps: int = 1


class _Config:
    @staticmethod
    def to_dict() -> dict[str, int]:
        return {"num_hidden_layers": 40}


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
