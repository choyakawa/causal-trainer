import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from causal_trainer.checkpointing.bundles import (
    ResumeCheckpoint,
    _checkpoint_artifact_digests,
    _checkpoint_artifact_sizes,
    _safetensors_contents,
    find_latest_checkpoint,
    prune_periodic_exports,
    save_adapter_checkpoint_bundle,
    save_checkpoint_bundle,
)
from causal_trainer.training.runner import _resume_action, _resume_identity_values

SIGNATURE = "a" * 64
CHECKPOINT_ID = "d" * 32
MANIFEST_DIGEST = "e" * 64


def _write_model_artifacts(path: Path, checkpoint_id: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model_path = path / "model.safetensors"
    model_path.unlink(missing_ok=True)
    save_file(
        {"weight": np.asarray([1, 2], dtype=np.int32)},
        model_path,
        metadata={"checkpoint_id": checkpoint_id},
    )
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def _write_adapter_artifacts(
    path: Path,
    checkpoint_id: str,
    adapter_config: dict,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.ones(
                (8, 2), dtype=np.float32
            )
        },
        path / "adapter_model.safetensors",
        metadata={"checkpoint_id": checkpoint_id},
    )
    (path / "adapter_config.json").write_text(
        json.dumps(adapter_config),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def _write_optimizer_artifacts(path: Path, step: int, checkpoint_id: str) -> None:
    optimizer_dir = path / "optimizer"
    optimizer_dir.mkdir(exist_ok=True)
    save_file(
        {"state/000000": np.asarray(step, dtype=np.int32)},
        optimizer_dir / "state.safetensors",
        metadata={"checkpoint_id": checkpoint_id},
    )
    tensors = [
        {
            "dtype": "int32",
            "file": "state.safetensors",
            "key": "state/000000",
            "ordinal": 0,
            "path": "count",
            "shape": [],
        }
    ]
    signature_payload = json.dumps(tensors, separators=(",", ":"), sort_keys=True).encode("utf-8")
    (optimizer_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_id": checkpoint_id,
                "format": "optimizer-state-safetensors",
                "format_version": 1,
                "global_step": step,
                "jax_version": "test",
                "layout_fingerprint": hashlib.sha256(signature_payload).hexdigest(),
                "optax_version": "test",
                "tensor_count": 1,
                "tensors": tensors,
                "total_size": 4,
            }
        ),
        encoding="utf-8",
    )


def test_checkpoint_inventory_tracks_named_chat_templates(tmp_path) -> None:
    _write_model_artifacts(tmp_path, CHECKPOINT_ID)
    template_directory = tmp_path / "chat_templates"
    template_directory.mkdir()
    (template_directory / "tool_use.jinja").write_text("tool template", encoding="utf-8")

    sizes = _checkpoint_artifact_sizes(tmp_path, False)
    digests = _checkpoint_artifact_digests(tmp_path, sizes)

    assert sizes["chat_templates/tool_use.jinja"] == len("tool template")
    assert len(digests["chat_templates/tool_use.jinja"]) == 64


def _write_completed_bundle(path: Path, step: int, signature: str, *, optimizer: bool = False) -> None:
    checkpoint_id = hashlib.sha256(f"{path}:{step}".encode()).hexdigest()[:32]
    _write_model_artifacts(path, checkpoint_id)
    if optimizer:
        _write_optimizer_artifacts(path, step, checkpoint_id)
    artifact_sizes = _checkpoint_artifact_sizes(path, optimizer)
    artifact_digests = _checkpoint_artifact_digests(path, artifact_sizes)
    (path / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "artifact_digests": artifact_digests,
                "artifact_sizes": artifact_sizes,
                "checkpoint_id": checkpoint_id,
                "format": "training-checkpoint-bundle",
                "artifact_kind": "merged",
                "format_version": 4,
                "global_step": step,
                "has_optimizer_state": optimizer,
                "total_steps": 10,
                "training_signature": signature,
            }
        ),
        encoding="utf-8",
    )


def _refresh_bundle_inventory(path: Path, optimizer: bool) -> None:
    manifest_path = path / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_sizes = _checkpoint_artifact_sizes(path, optimizer)
    manifest["artifact_sizes"] = artifact_sizes
    manifest["artifact_digests"] = _checkpoint_artifact_digests(path, artifact_sizes)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_checkpoint_bundle_omits_optimizer_when_disabled(monkeypatch, tmp_path) -> None:
    calls: list[str] = []
    export_options = {}
    bundle_plans = []

    def export_model(*args, **kwargs):
        calls.append("model")
        export_options.update(kwargs)
        destination = Path(args[1])
        _write_model_artifacts(destination, kwargs["checkpoint_id"])
        return destination

    def remove_optimizer(output_dir):
        calls.append("remove")

    monkeypatch.setattr("causal_trainer.checkpointing.huggingface.export_hf_checkpoint", export_model)
    monkeypatch.setattr("causal_trainer.checkpointing.huggingface.remove_optimizer_checkpoint", remove_optimizer)
    monkeypatch.setattr(
        "causal_trainer.checkpointing.huggingface._assert_same_export_plan",
        lambda kind, plan: bundle_plans.append((kind, list(plan))),
    )
    monkeypatch.setattr(
        "causal_trainer.checkpointing.huggingface.export_optimizer_checkpoint",
        lambda *args, **kwargs: calls.append("optimizer"),
    )

    def leaf_transform(path, leaf):
        del path
        return leaf

    transform_plan = {"kind": "identity"}
    save_checkpoint_bundle(
        object(),
        object(),
        tmp_path / "checkpoint",
        step=4,
        save_optimizer_state=False,
        model_source=tmp_path,
        tokenizer_source=tmp_path,
        config={},
        training_signature=SIGNATURE,
        total_steps=10,
        leaf_transform=leaf_transform,
        transform_plan=transform_plan,
    )
    assert calls == ["remove", "model"]
    assert export_options["leaf_transform"] is leaf_transform
    assert export_options["transform_plan"] is transform_plan
    assert bundle_plans == [
        (
            "checkpoint bundle",
            [
                {
                    "artifact_kind": "merged",
                    "leaf_transform": True,
                    "save_optimizer_state": False,
                    "step": 4,
                    "total_steps": 10,
                    "training_signature": SIGNATURE,
                    "transform_plan": transform_plan,
                }
            ],
        )
    ]
    manifest = tmp_path / "checkpoint" / "checkpoint_manifest.json"
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["format_version"] == 4
    assert manifest_data["artifact_kind"] == "merged"
    assert len(manifest_data["checkpoint_id"]) == 32
    assert manifest_data["has_optimizer_state"] is False
    assert set(manifest_data["artifact_sizes"]) == {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    }


def test_checkpoint_bundle_saves_optimizer_after_model(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, int | None]] = []
    optimizer_state = object()

    def export_model(*args, **kwargs):
        calls.append(("model", None))
        destination = Path(args[1])
        _write_model_artifacts(destination, kwargs["checkpoint_id"])
        return destination

    def export_optimizer(state, output_dir, *, step, **kwargs):
        assert state is optimizer_state
        calls.append(("optimizer", step))
        destination = Path(output_dir)
        _write_optimizer_artifacts(destination, step, kwargs["checkpoint_id"])
        return destination / "optimizer"

    monkeypatch.setattr("causal_trainer.checkpointing.huggingface.export_hf_checkpoint", export_model)
    monkeypatch.setattr("causal_trainer.checkpointing.huggingface.export_optimizer_checkpoint", export_optimizer)
    monkeypatch.setattr(
        "causal_trainer.checkpointing.huggingface.remove_optimizer_checkpoint",
        lambda output_dir: calls.append(("remove", None)),
    )

    save_checkpoint_bundle(
        object(),
        optimizer_state,
        tmp_path / "checkpoint",
        step=9,
        save_optimizer_state=True,
        model_source=tmp_path,
        tokenizer_source=tmp_path,
        config={},
        training_signature=SIGNATURE,
        total_steps=10,
    )
    assert calls == [("remove", None), ("model", None), ("optimizer", 9)]
    manifest = tmp_path / "checkpoint" / "checkpoint_manifest.json"
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["has_optimizer_state"] is True
    assert "optimizer/manifest.json" in manifest_data["artifact_sizes"]
    checkpoint_id = manifest_data["checkpoint_id"]
    model_contents = _safetensors_contents(tmp_path / "checkpoint" / "model.safetensors")
    optimizer_contents = _safetensors_contents(
        tmp_path / "checkpoint" / "optimizer" / "state.safetensors"
    )
    optimizer_manifest = json.loads(
        (tmp_path / "checkpoint" / "optimizer" / "manifest.json").read_text(encoding="utf-8")
    )
    assert model_contents is not None and model_contents[1]["checkpoint_id"] == checkpoint_id
    assert optimizer_contents is not None and optimizer_contents[1]["checkpoint_id"] == checkpoint_id
    assert optimizer_manifest["checkpoint_id"] == checkpoint_id


def test_checkpoint_overwrite_rotates_bundle_identity(monkeypatch, tmp_path) -> None:
    def export_model(*args, **kwargs):
        destination = Path(args[1])
        _write_model_artifacts(destination, kwargs["checkpoint_id"])
        return destination

    monkeypatch.setattr("causal_trainer.checkpointing.huggingface.export_hf_checkpoint", export_model)
    monkeypatch.setattr(
        "causal_trainer.checkpointing.huggingface.remove_optimizer_checkpoint", lambda output_dir: None
    )
    destination = tmp_path / "checkpoint"
    checkpoint_ids = []
    for step in (4, 5):
        save_checkpoint_bundle(
            object(),
            object(),
            destination,
            step=step,
            save_optimizer_state=False,
            model_source=tmp_path,
            tokenizer_source=tmp_path,
            config={},
            training_signature=SIGNATURE,
            total_steps=10,
        )
        manifest = json.loads((destination / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        checkpoint_ids.append(manifest["checkpoint_id"])

    assert checkpoint_ids[0] != checkpoint_ids[1]


def test_adapter_checkpoint_bundle_saves_and_is_discoverable(monkeypatch, tmp_path) -> None:
    adapter_config = {
        "base_model_name_or_path": "example/base-model",
        "lora_alpha": 8,
        "peft_type": "LORA",
        "r": 8,
        "target_modules": ["q_proj"],
        "task_type": "CAUSAL_LM",
    }
    adapter_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    expected_layout = {adapter_key: ("F32", (8, 2))}

    def mapping_fn(path):
        return path

    calls = []

    def export_adapter(params, output_dir, **kwargs):
        calls.append((params, kwargs["mapping_fn"]))
        destination = Path(output_dir)
        _write_adapter_artifacts(
            destination,
            kwargs["checkpoint_id"],
            dict(kwargs["adapter_config"]),
        )
        return destination

    monkeypatch.setattr("causal_trainer.checkpointing.huggingface.export_adapter_checkpoint", export_adapter)
    monkeypatch.setattr(
        "causal_trainer.checkpointing.huggingface.remove_optimizer_checkpoint", lambda output_dir: None
    )
    destination = tmp_path / "adapter"
    params = object()

    save_adapter_checkpoint_bundle(
        params,
        destination,
        step=10,
        tokenizer_source=tmp_path,
        adapter_config=adapter_config,
        mapping_fn=mapping_fn,
        training_signature=SIGNATURE,
        total_steps=10,
        expected_adapter_layout=expected_layout,
    )

    assert calls == [(params, mapping_fn)]
    manifest = json.loads((destination / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 4
    assert manifest["artifact_kind"] == "peft-adapter"
    assert manifest["has_optimizer_state"] is False
    assert set(manifest["artifact_sizes"]) == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
    }

    checkpoint = find_latest_checkpoint(
        destination,
        SIGNATURE,
        expected_model_layout=expected_layout,
        expected_artifact_kind="peft-adapter",
    )
    assert checkpoint is not None
    assert checkpoint.path == destination
    assert checkpoint.global_step == 10
    assert checkpoint.artifact_kind == "peft-adapter"


def test_adapter_checkpoint_bundle_saves_matching_optimizer_state(monkeypatch, tmp_path) -> None:
    adapter_config = {
        "base_model_name_or_path": "example/base-model",
        "lora_alpha": 8,
        "peft_type": "LORA",
        "r": 8,
        "target_modules": ["q_proj"],
        "task_type": "CAUSAL_LM",
    }
    adapter_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    expected_layout = {adapter_key: ("F32", (8, 2))}
    optimizer_state = object()

    def mapping_fn(path):
        return path

    def export_adapter(params, output_dir, **kwargs):
        del params
        destination = Path(output_dir)
        _write_adapter_artifacts(
            destination,
            kwargs["checkpoint_id"],
            dict(kwargs["adapter_config"]),
        )
        return destination

    def export_optimizer(state, output_dir, *, step, **kwargs):
        assert state is optimizer_state
        destination = Path(output_dir)
        _write_optimizer_artifacts(destination, step, kwargs["checkpoint_id"])
        return destination / "optimizer"

    monkeypatch.setattr(
        "causal_trainer.checkpointing.huggingface.export_adapter_checkpoint",
        export_adapter,
    )
    monkeypatch.setattr(
        "causal_trainer.checkpointing.huggingface.export_optimizer_checkpoint",
        export_optimizer,
    )
    monkeypatch.setattr(
        "causal_trainer.checkpointing.huggingface.remove_optimizer_checkpoint",
        lambda output_dir: None,
    )
    destination = tmp_path / "adapter"

    save_adapter_checkpoint_bundle(
        object(),
        destination,
        optimizer_state=optimizer_state,
        step=6,
        save_optimizer_state=True,
        tokenizer_source=tmp_path,
        adapter_config=adapter_config,
        mapping_fn=mapping_fn,
        training_signature=SIGNATURE,
        total_steps=10,
        expected_adapter_layout=expected_layout,
    )

    manifest = json.loads((destination / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert manifest["has_optimizer_state"] is True
    assert "optimizer/manifest.json" in manifest["artifact_sizes"]
    checkpoint = find_latest_checkpoint(
        destination,
        SIGNATURE,
        expected_model_layout=expected_layout,
        expected_artifact_kind="peft-adapter",
    )
    assert checkpoint is not None
    assert checkpoint.global_step == 6
    assert checkpoint.has_optimizer_state is True


def test_pruning_removes_nested_optimizer_state(tmp_path) -> None:
    for step in (1, 2):
        _write_completed_bundle(tmp_path / f"checkpoint-{step}", step, SIGNATURE, optimizer=True)

    incomplete = tmp_path / "checkpoint-100"
    incomplete.mkdir()

    prune_periodic_exports(tmp_path, save_total_limit=1)

    assert not (tmp_path / "checkpoint-1").exists()
    assert (tmp_path / "checkpoint-2" / "optimizer" / "manifest.json").is_file()
    assert incomplete.is_dir()


def test_find_latest_checkpoint_prefers_highest_completed_step(tmp_path) -> None:
    _write_completed_bundle(tmp_path / "checkpoint-2", 2, SIGNATURE)
    _write_completed_bundle(tmp_path / "checkpoint-6", 6, SIGNATURE, optimizer=True)
    incomplete = tmp_path / "checkpoint-9"
    incomplete.mkdir()
    (incomplete / "model.safetensors").write_bytes(b"partial")

    checkpoint = find_latest_checkpoint(tmp_path, SIGNATURE)

    assert checkpoint is not None
    assert checkpoint.path == tmp_path / "checkpoint-6"
    assert checkpoint.global_step == 6
    assert checkpoint.has_optimizer_state is True


def test_find_latest_checkpoint_prefers_completed_root_at_same_step(tmp_path) -> None:
    _write_completed_bundle(tmp_path / "checkpoint-10", 10, SIGNATURE, optimizer=True)
    _write_completed_bundle(tmp_path, 10, SIGNATURE, optimizer=True)

    checkpoint = find_latest_checkpoint(tmp_path, SIGNATURE)

    assert checkpoint is not None
    assert checkpoint.path == tmp_path


def test_find_latest_checkpoint_prefers_newer_periodic_over_older_root(tmp_path) -> None:
    _write_completed_bundle(tmp_path, 4, SIGNATURE, optimizer=True)
    _write_completed_bundle(tmp_path / "checkpoint-7", 7, SIGNATURE, optimizer=True)

    checkpoint = find_latest_checkpoint(tmp_path, SIGNATURE)

    assert checkpoint is not None
    assert checkpoint.path == tmp_path / "checkpoint-7"


def test_find_latest_checkpoint_rejects_incompatible_run(tmp_path) -> None:
    _write_completed_bundle(tmp_path / "checkpoint-4", 4, "b" * 64)
    with pytest.raises(ValueError, match="incompatible training configuration"):
        find_latest_checkpoint(tmp_path, "c" * 64)


def test_find_latest_checkpoint_rejects_malformed_completion_manifest(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint-4"
    checkpoint_path.mkdir()
    (checkpoint_path / "checkpoint_manifest.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt completed checkpoints"):
        find_latest_checkpoint(tmp_path, SIGNATURE)


def test_find_latest_checkpoint_rejects_wrong_completion_format(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint-4"
    checkpoint_path.mkdir()
    (checkpoint_path / "checkpoint_manifest.json").write_text(
        json.dumps({"format": "unexpected"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="corrupt completed checkpoints"):
        find_latest_checkpoint(tmp_path, SIGNATURE)


def test_find_latest_checkpoint_rejects_malformed_optimizer_manifest(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint-4"
    _write_completed_bundle(checkpoint_path, 4, SIGNATURE, optimizer=True)
    manifest_path = checkpoint_path / "optimizer" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["layout_fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_bundle_inventory(checkpoint_path, True)

    with pytest.raises(ValueError, match="corrupt completed checkpoints"):
        find_latest_checkpoint(tmp_path, SIGNATURE)


def test_find_latest_checkpoint_rejects_truncated_model_and_missing_config(tmp_path) -> None:
    truncated = tmp_path / "checkpoint-4"
    _write_completed_bundle(truncated, 4, SIGNATURE)
    model_path = truncated / "model.safetensors"
    model_path.write_bytes(model_path.read_bytes()[:-1])
    _refresh_bundle_inventory(truncated, False)
    with pytest.raises(ValueError, match="corrupt completed checkpoints"):
        find_latest_checkpoint(tmp_path, SIGNATURE)

    for child in tuple(tmp_path.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
    missing_config = tmp_path / "checkpoint-5"
    _write_completed_bundle(missing_config, 5, SIGNATURE)
    (missing_config / "config.json").unlink()
    with pytest.raises(ValueError, match="corrupt completed checkpoints"):
        find_latest_checkpoint(tmp_path, SIGNATURE)


def test_find_latest_checkpoint_rejects_same_size_model_from_another_bundle(tmp_path) -> None:
    first = tmp_path / "checkpoint-4"
    second = tmp_path / "other"
    _write_completed_bundle(first, 4, SIGNATURE)
    _write_completed_bundle(second, 4, SIGNATURE)
    first_model = first / "model.safetensors"
    second_model = second / "model.safetensors"
    assert first_model.stat().st_size == second_model.stat().st_size
    first_model.write_bytes(second_model.read_bytes())
    _refresh_bundle_inventory(first, False)

    with pytest.raises(ValueError, match="corrupt completed checkpoints"):
        find_latest_checkpoint(tmp_path, SIGNATURE)


def test_find_latest_checkpoint_rejects_same_size_safetensors_payload_corruption(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint-4"
    _write_completed_bundle(checkpoint_path, 4, SIGNATURE)
    model_path = checkpoint_path / "model.safetensors"
    original_size = model_path.stat().st_size
    payload = bytearray(model_path.read_bytes())
    header_size = int.from_bytes(payload[:8], "little")
    data_start = 8 + header_size
    assert data_start < len(payload)

    # Preserve both the Safetensors header and file size while corrupting only
    # tensor data. Structural validation alone cannot detect this mutation.
    payload[data_start] ^= 0x01
    model_path.write_bytes(payload)
    assert model_path.stat().st_size == original_size
    assert _safetensors_contents(model_path) is not None

    with pytest.raises(ValueError, match="corrupt completed checkpoints"):
        find_latest_checkpoint(tmp_path, SIGNATURE)


def test_find_latest_checkpoint_validates_expected_model_layout(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint-4"
    _write_completed_bundle(checkpoint_path, 4, SIGNATURE)
    checkpoint = find_latest_checkpoint(
        tmp_path,
        SIGNATURE,
        expected_model_layout={"weight": ("I32", (2,))},
    )
    assert checkpoint is not None
    with pytest.raises(ValueError, match="corrupt completed checkpoints"):
        find_latest_checkpoint(
            tmp_path,
            SIGNATURE,
            expected_model_layout={"weight": ("I32", (3,))},
        )


def test_find_latest_checkpoint_rejects_unhashable_safetensors_dtype(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint-4"
    _write_completed_bundle(checkpoint_path, 4, SIGNATURE)
    model_path = checkpoint_path / "model.safetensors"
    payload = model_path.read_bytes()
    header_size = int.from_bytes(payload[:8], "little")
    header = json.loads(payload[8 : 8 + header_size].decode("utf-8").rstrip(" "))
    header["weight"]["dtype"] = []
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    assert len(encoded) <= header_size
    encoded += b" " * (header_size - len(encoded))
    model_path.write_bytes(payload[:8] + encoded + payload[8 + header_size :])
    _refresh_bundle_inventory(checkpoint_path, False)

    with pytest.raises(ValueError, match="corrupt completed checkpoints"):
        find_latest_checkpoint(tmp_path, SIGNATURE)


def test_find_latest_checkpoint_rejects_optimizer_header_schema_mismatch(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint-4"
    _write_completed_bundle(checkpoint_path, 4, SIGNATURE, optimizer=True)
    manifest = json.loads((checkpoint_path / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    state_path = checkpoint_path / "optimizer" / "state.safetensors"
    save_file(
        {"state/000000": np.asarray(4.0, dtype=np.float32)},
        state_path,
        metadata={"checkpoint_id": manifest["checkpoint_id"]},
    )
    _refresh_bundle_inventory(checkpoint_path, True)

    with pytest.raises(ValueError, match="corrupt completed checkpoints"):
        find_latest_checkpoint(tmp_path, SIGNATURE)


def test_find_latest_checkpoint_rejects_incomplete_or_unsafe_artifact_inventory(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint-4"
    _write_completed_bundle(checkpoint_path, 4, SIGNATURE)
    manifest_path = checkpoint_path / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer_size = manifest["artifact_sizes"].pop("tokenizer.json")
    tokenizer_digest = manifest["artifact_digests"].pop("tokenizer.json")
    unsafe_name = "tokenizer.json\\..\\..\\outside"
    manifest["artifact_sizes"][unsafe_name] = tokenizer_size
    manifest["artifact_digests"][unsafe_name] = tokenizer_digest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt completed checkpoints"):
        find_latest_checkpoint(tmp_path, SIGNATURE)


def test_resume_action_requires_optimizer_for_partial_checkpoint(tmp_path) -> None:
    checkpoint = ResumeCheckpoint(
        tmp_path / "checkpoint-4",
        4,
        10,
        False,
        CHECKPOINT_ID,
        MANIFEST_DIGEST,
    )
    with pytest.raises(ValueError, match="exact continuation is impossible"):
        _resume_action(checkpoint, output_dir=tmp_path, total_steps=10)


def test_resume_action_distinguishes_continue_finalize_and_complete(tmp_path) -> None:
    assert _resume_action(None, output_dir=tmp_path, total_steps=10) == "start"
    assert (
        _resume_action(
            ResumeCheckpoint(
                tmp_path / "checkpoint-4",
                4,
                10,
                True,
                CHECKPOINT_ID,
                MANIFEST_DIGEST,
            ),
            output_dir=tmp_path,
            total_steps=10,
        )
        == "continue"
    )
    assert (
        _resume_action(
            ResumeCheckpoint(
                tmp_path / "checkpoint-10",
                10,
                10,
                False,
                CHECKPOINT_ID,
                MANIFEST_DIGEST,
            ),
            output_dir=tmp_path,
            total_steps=10,
        )
        == "finalize"
    )
    assert (
        _resume_action(
            ResumeCheckpoint(tmp_path, 10, 10, False, CHECKPOINT_ID, MANIFEST_DIGEST),
            output_dir=tmp_path,
            total_steps=10,
        )
        == "complete"
    )


def test_resume_action_rejects_different_training_plan(tmp_path) -> None:
    checkpoint = ResumeCheckpoint(
        tmp_path / "checkpoint-4",
        4,
        12,
        True,
        CHECKPOINT_ID,
        MANIFEST_DIGEST,
    )
    with pytest.raises(ValueError, match="does not match current plan"):
        _resume_action(checkpoint, output_dir=tmp_path, total_steps=10)


def test_resume_identity_binds_checkpoint_and_manifest_identity(tmp_path) -> None:
    first = ResumeCheckpoint(
        tmp_path / "checkpoint-4",
        4,
        10,
        True,
        "1" * 32,
        "2" * 64,
    )
    second = ResumeCheckpoint(
        tmp_path / "checkpoint-4",
        4,
        10,
        True,
        "3" * 32,
        "2" * 64,
    )
    first_identity = _resume_identity_values(
        first,
        output_dir=tmp_path,
        training_signature=SIGNATURE,
    )
    assert first_identity == _resume_identity_values(
        first,
        output_dir=tmp_path,
        training_signature=SIGNATURE,
    )
    assert first_identity != _resume_identity_values(
        second,
        output_dir=tmp_path,
        training_signature=SIGNATURE,
    )
    assert first_identity != _resume_identity_values(
        None,
        output_dir=tmp_path,
        training_signature=SIGNATURE,
    )
    assert first_identity != _resume_identity_values(
        ResumeCheckpoint(
            first.path,
            first.global_step,
            first.total_steps,
            first.has_optimizer_state,
            first.checkpoint_id,
            "4" * 64,
        ),
        output_dir=tmp_path,
        training_signature=SIGNATURE,
    )
    assert first_identity != _resume_identity_values(
        first,
        output_dir=tmp_path,
        training_signature="f" * 64,
    )
    assert first_identity != _resume_identity_values(
        ResumeCheckpoint(
            tmp_path,
            first.global_step,
            first.total_steps,
            first.has_optimizer_state,
            first.checkpoint_id,
            first.manifest_digest,
        ),
        output_dir=tmp_path,
        training_signature=SIGNATURE,
    )
