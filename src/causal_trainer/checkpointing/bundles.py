from __future__ import annotations

import hashlib
import json
import shutil
import struct
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_BUNDLE_FORMAT_VERSION = 4
_ARTIFACT_KINDS = {"merged", "peft-adapter"}
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
_SAFETENSORS_ITEM_SIZES = {
    "BOOL": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
}
_LOGICAL_ITEM_SIZES = {
    "bool": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
}
_ASSET_NAMES = {
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}
_ASSET_SUFFIXES = (".jinja", ".model", ".tiktoken")
_TOKENIZER_DATA_NAMES = {"tokenizer.json", "tokenizer.model", "vocab.json"}
_LOGICAL_TO_STORAGE_DTYPE = {
    "bool": "BOOL",
    "float16": "F16",
    "bfloat16": "BF16",
    "float32": "F32",
    "float64": "F64",
    "int8": "I8",
    "uint8": "U8",
    "int16": "I16",
    "uint16": "U16",
    "int32": "I32",
    "uint32": "U32",
    "int64": "I64",
    "uint64": "U64",
}


@dataclass(frozen=True, slots=True)
class ResumeCheckpoint:
    path: Path
    global_step: int
    total_steps: int
    has_optimizer_state: bool
    checkpoint_id: str
    manifest_digest: str
    artifact_kind: str = "merged"
    source_examples_seen: int = 0
    training_complete: bool = False
    streaming_data_digest: str | None = None


def training_signature(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_bundle_manifest(path: Path) -> dict[str, Any] | None:
    manifest_path = path / "checkpoint_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _shape_size(shape: list[Any]) -> int | None:
    result = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            return None
        result *= dimension
    return result


def _safetensors_contents(
    path: Path,
) -> tuple[dict[str, tuple[str, tuple[int, ...]]], dict[str, str]] | None:
    """Validate a Safetensors header without reading tensor payloads."""

    if path.is_symlink() or not path.is_file():
        return None
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                return None
            header_size = struct.unpack("<Q", prefix)[0]
            if (
                header_size <= 0
                or header_size > _MAX_SAFETENSORS_HEADER_BYTES
                or 8 + header_size > file_size
            ):
                return None
            raw_header = handle.read(header_size)
        header = json.loads(raw_header.decode("utf-8").rstrip(" "))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(header, dict):
        return None

    data_size = file_size - 8 - header_size
    intervals: list[tuple[int, int]] = []
    tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
    file_metadata: dict[str, str] = {}
    for key, metadata in header.items():
        if key == "__metadata__":
            if not isinstance(metadata, dict) or any(
                not isinstance(name, str) or not isinstance(value, str)
                for name, value in metadata.items()
            ):
                return None
            file_metadata = dict(metadata)
            continue
        if not isinstance(key, str) or not isinstance(metadata, dict):
            return None
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or dtype not in _SAFETENSORS_ITEM_SIZES
            or not isinstance(shape, list)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
        ):
            return None
        elements = _shape_size(shape)
        start, end = offsets
        if (
            elements is None
            or start < 0
            or end < start
            or end > data_size
            or end - start != elements * _SAFETENSORS_ITEM_SIZES[dtype]
        ):
            return None
        tensors[key] = (dtype, tuple(shape))
        intervals.append((start, end))
    if not tensors:
        return None
    intervals.sort()
    expected_start = 0
    for start, end in intervals:
        if start != expected_start:
            return None
        expected_start = end
    if expected_start != data_size:
        return None
    return tensors, file_metadata


def _contains_config(path: Path) -> bool:
    config_path = path / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(config, dict)


def _contains_model_weights(
    path: Path,
    checkpoint_id: str,
    expected_layout: dict[str, tuple[str, tuple[int, ...]]] | None = None,
) -> bool:
    single = path / "model.safetensors"
    index_path = path / "model.safetensors.index.json"
    model_shards = list(path.glob("model*.safetensors"))
    if any(shard.is_symlink() or not shard.is_file() for shard in model_shards):
        return False
    actual_shards = {shard.name for shard in model_shards}
    if single.is_symlink() or index_path.is_symlink():
        return False
    if single.is_file() and not single.is_symlink():
        if index_path.exists() or actual_shards != {single.name}:
            return False
        contents = _safetensors_contents(single)
        return (
            contents is not None
            and contents[1].get("checkpoint_id") == checkpoint_id
            and (expected_layout is None or contents[0] == expected_layout)
        )
    if index_path.is_symlink() or not index_path.is_file():
        return False
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    index_metadata = index.get("metadata") if isinstance(index, dict) else None
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or not isinstance(index_metadata, dict)
        or index_metadata.get("checkpoint_id") != checkpoint_id
    ):
        return False
    expected_by_file: dict[str, set[str]] = {}
    for key, filename in weight_map.items():
        if (
            not isinstance(key, str)
            or not isinstance(filename, str)
            or Path(filename).name != filename
        ):
            return False
        expected_by_file.setdefault(filename, set()).add(key)
    actual_layout: dict[str, tuple[str, tuple[int, ...]]] = {}
    for filename, expected_keys in expected_by_file.items():
        shard = path / filename
        contents = _safetensors_contents(shard)
        if (
            contents is None
            or set(contents[0]) != expected_keys
            or contents[1].get("checkpoint_id") != checkpoint_id
        ):
            return False
        actual_layout.update(contents[0])
    if actual_shards != set(expected_by_file):
        return False
    if expected_layout is not None and actual_layout != expected_layout:
        return False
    return True


def _contains_adapter_weights(
    path: Path,
    checkpoint_id: str,
    expected_layout: dict[str, tuple[str, tuple[int, ...]]] | None = None,
) -> bool:
    weights = path / "adapter_model.safetensors"
    config_path = path / "adapter_config.json"
    if weights.is_symlink() or config_path.is_symlink() or not config_path.is_file():
        return False
    if any(path.glob("model*.safetensors")) or (path / "model.safetensors.index.json").exists():
        return False
    contents = _safetensors_contents(weights)
    if (
        contents is None
        or contents[1].get("checkpoint_id") != checkpoint_id
        or (expected_layout is not None and contents[0] != expected_layout)
    ):
        return False
    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(adapter_config, dict):
        return False
    rank = adapter_config.get("r")
    return (
        adapter_config.get("peft_type") == "LORA"
        and adapter_config.get("task_type") == "CAUSAL_LM"
        and type(rank) is int
        and rank > 0
        and adapter_config.get("lora_alpha") == rank
        and isinstance(adapter_config.get("target_modules"), list)
    )


def _is_direct_artifact(name: str) -> bool:
    return (
        name in _ASSET_NAMES
        or name == "adapter_config.json"
        or name == "adapter_model.safetensors"
        or name.startswith("tokenizer.")
        or name.endswith(_ASSET_SUFFIXES)
        or name == "model.safetensors.index.json"
        or name == "model.safetensors"
        or (name.startswith("model-") and name.endswith(".safetensors"))
    )


def _checkpoint_artifact_sizes(
    path: Path,
    has_optimizer_state: bool,
    artifact_kind: str = "merged",
) -> dict[str, int]:
    if artifact_kind not in _ARTIFACT_KINDS:
        raise ValueError(f"unknown checkpoint artifact kind {artifact_kind!r}")
    artifacts: dict[str, int] = {}
    for artifact in path.iterdir():
        if not _is_direct_artifact(artifact.name):
            continue
        if "\\" in artifact.name or artifact.is_symlink() or not artifact.is_file():
            raise RuntimeError(f"checkpoint artifact must be a real file: {artifact}")
        artifacts[artifact.name] = artifact.stat().st_size
    template_dir = path / "chat_templates"
    if template_dir.exists():
        if template_dir.is_symlink() or not template_dir.is_dir():
            raise RuntimeError(f"checkpoint chat_templates must be a real directory: {template_dir}")
        for artifact in template_dir.iterdir():
            if (
                "\\" in artifact.name
                or artifact.is_symlink()
                or not artifact.is_file()
                or artifact.suffix != ".jinja"
            ):
                raise RuntimeError(f"checkpoint chat template must be a real .jinja file: {artifact}")
            artifacts[f"chat_templates/{artifact.name}"] = artifact.stat().st_size
    if artifact_kind == "merged":
        if "adapter_model.safetensors" in artifacts or "adapter_config.json" in artifacts:
            raise RuntimeError(f"merged checkpoint contains stale adapter artifacts: {path}")
        if "config.json" not in artifacts or "model.safetensors" not in artifacts:
            raise RuntimeError(f"checkpoint is missing model.safetensors or config.json: {path}")
        if not any(
            name in _TOKENIZER_DATA_NAMES or name.endswith((".model", ".tiktoken"))
            for name in artifacts
        ):
            raise RuntimeError(f"checkpoint is missing tokenizer data: {path}")
    else:
        if any(name.startswith("model") and name.endswith(".safetensors") for name in artifacts):
            raise RuntimeError(f"adapter checkpoint contains stale merged weights: {path}")
        if "adapter_model.safetensors" not in artifacts or "adapter_config.json" not in artifacts:
            raise RuntimeError(f"checkpoint is missing adapter weights or adapter_config.json: {path}")

    optimizer_dir = path / "optimizer"
    if has_optimizer_state:
        if optimizer_dir.is_symlink() or not optimizer_dir.is_dir():
            raise RuntimeError(f"checkpoint is missing optimizer state: {optimizer_dir}")
        for artifact in optimizer_dir.iterdir():
            if "\\" in artifact.name or artifact.is_symlink() or not artifact.is_file():
                raise RuntimeError(f"optimizer artifact must be a real file: {artifact}")
            artifacts[f"optimizer/{artifact.name}"] = artifact.stat().st_size
        if "optimizer/manifest.json" not in artifacts or not any(
            name.startswith("optimizer/") and name.endswith(".safetensors") for name in artifacts
        ):
            raise RuntimeError(f"checkpoint optimizer state is incomplete: {optimizer_dir}")
    elif optimizer_dir.exists():
        raise RuntimeError(f"checkpoint contains stale optimizer state: {optimizer_dir}")
    return dict(sorted(artifacts.items()))


def _artifact_digest(path: Path) -> str | None:
    """Hash an artifact in bounded chunks, including Safetensors payload bytes."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _checkpoint_artifact_digests(path: Path, artifact_sizes: dict[str, int]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for relative_name in artifact_sizes:
        artifact = path.joinpath(*PurePosixPath(relative_name).parts)
        digest = _artifact_digest(artifact)
        if digest is None:
            raise RuntimeError(f"cannot digest checkpoint artifact: {artifact}")
        digests[relative_name] = digest
    return digests


def _artifact_sizes_match(
    path: Path,
    manifest: dict[str, Any],
    has_optimizer_state: bool,
    *,
    artifact_kind: str = "merged",
    verify_digests: bool = True,
) -> bool:
    artifact_sizes = manifest.get("artifact_sizes")
    artifact_digests = manifest.get("artifact_digests")
    if (
        not isinstance(artifact_sizes, dict)
        or not artifact_sizes
        or not isinstance(artifact_digests, dict)
        or set(artifact_digests) != set(artifact_sizes)
    ):
        return False
    if artifact_kind == "merged":
        if "config.json" not in artifact_sizes or "model.safetensors" not in artifact_sizes:
            return False
    elif artifact_kind == "peft-adapter":
        if "adapter_config.json" not in artifact_sizes or "adapter_model.safetensors" not in artifact_sizes:
            return False
    else:
        return False
    if has_optimizer_state and (
        "optimizer/manifest.json" not in artifact_sizes
        or not any(
            name.startswith("optimizer/") and name.endswith(".safetensors")
            for name in artifact_sizes
        )
    ):
        return False
    if not has_optimizer_state and any(name.startswith("optimizer/") for name in artifact_sizes):
        return False
    try:
        if _checkpoint_artifact_sizes(path, has_optimizer_state, artifact_kind) != artifact_sizes:
            return False
    except (OSError, RuntimeError):
        return False

    for relative_name, expected_size in artifact_sizes.items():
        if (
            not isinstance(relative_name, str)
            or "\\" in relative_name
            or type(expected_size) is not int
            or expected_size < 0
        ):
            return False
        relative = PurePosixPath(relative_name)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or (len(relative.parts) == 1 and not _is_direct_artifact(relative.parts[0]))
            or (
                len(relative.parts) == 2
                and not (
                    (
                        relative.parts[0] == "optimizer"
                        and Path(relative.parts[1]).name == relative.parts[1]
                    )
                    or (
                        relative.parts[0] == "chat_templates"
                        and Path(relative.parts[1]).name == relative.parts[1]
                        and relative.parts[1].endswith(".jinja")
                    )
                )
            )
            or len(relative.parts) > 2
        ):
            return False
        artifact = path.joinpath(*relative.parts)
        if artifact.is_symlink() or not artifact.is_file():
            return False
        try:
            if artifact.stat().st_size != expected_size:
                return False
        except OSError:
            return False
        expected_digest = artifact_digests.get(relative_name)
        if not _is_lower_hex(expected_digest, 64):
            return False
        if verify_digests and _artifact_digest(artifact) != expected_digest:
            return False
    return True


def _contains_optimizer_state(path: Path, global_step: int, checkpoint_id: str) -> bool:
    optimizer_dir = path / "optimizer"
    if optimizer_dir.is_symlink() or not optimizer_dir.is_dir():
        return False
    optimizer_manifest = optimizer_dir / "manifest.json"
    if optimizer_manifest.is_symlink() or not optimizer_manifest.is_file():
        return False
    try:
        optimizer_metadata = json.loads(optimizer_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if (
        not isinstance(optimizer_metadata, dict)
        or optimizer_metadata.get("format") != "optimizer-state-safetensors"
        or optimizer_metadata.get("format_version") != 1
        or optimizer_metadata.get("global_step") != global_step
        or optimizer_metadata.get("checkpoint_id") != checkpoint_id
    ):
        return False
    tensors = optimizer_metadata.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        return False
    tensor_count = optimizer_metadata.get("tensor_count")
    total_size = optimizer_metadata.get("total_size")
    layout_fingerprint = optimizer_metadata.get("layout_fingerprint")
    if (
        type(tensor_count) is not int
        or tensor_count != len(tensors)
        or type(total_size) is not int
        or total_size < 0
        or not isinstance(layout_fingerprint, str)
        or len(layout_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in layout_fingerprint)
        or not optimizer_metadata.get("jax_version")
        or not isinstance(optimizer_metadata.get("jax_version"), str)
        or not optimizer_metadata.get("optax_version")
        or not isinstance(optimizer_metadata.get("optax_version"), str)
    ):
        return False
    signature_payload = json.dumps(tensors, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if hashlib.sha256(signature_payload).hexdigest() != layout_fingerprint:
        return False
    seen_paths: set[str] = set()
    expected_by_file: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {}
    computed_total_size = 0
    for tensor in tensors:
        if not isinstance(tensor, dict):
            return False
        ordinal = tensor.get("ordinal")
        key = tensor.get("key")
        tensor_path = tensor.get("path")
        shape = tensor.get("shape")
        dtype = tensor.get("dtype")
        filename = tensor.get("file")
        if (
            type(ordinal) is not int
            or ordinal < 0
            or key != f"state/{ordinal:06d}"
            or not isinstance(tensor_path, str)
            or not tensor_path
            or tensor_path in seen_paths
            or not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or not isinstance(dtype, str)
            or dtype not in _LOGICAL_ITEM_SIZES
            or not isinstance(filename, str)
            or Path(filename).name != filename
        ):
            return False
        seen_paths.add(tensor_path)
        elements = _shape_size(shape)
        if elements is None:
            return False
        expected_by_file.setdefault(filename, {})[key] = (
            _LOGICAL_TO_STORAGE_DTYPE[dtype],
            tuple(shape),
        )
        computed_total_size += elements * _LOGICAL_ITEM_SIZES[dtype]
    if computed_total_size != total_size:
        return False
    if {tensor["ordinal"] for tensor in tensors} != set(range(len(tensors))):
        return False
    expected_weight_map = {
        tensor["key"]: tensor["file"]
        for tensor in tensors
    }
    index_path = optimizer_dir / "state.safetensors.index.json"
    if len(expected_by_file) > 1:
        if index_path.is_symlink() or not index_path.is_file():
            return False
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if (
            not isinstance(index, dict)
            or index.get("weight_map") != expected_weight_map
            or not isinstance(index.get("metadata"), dict)
            or index["metadata"].get("checkpoint_id") != checkpoint_id
        ):
            return False
    elif index_path.exists():
        return False
    optimizer_shards = list(optimizer_dir.glob("state*.safetensors"))
    if any(shard.is_symlink() or not shard.is_file() for shard in optimizer_shards):
        return False
    actual_shards = {shard.name for shard in optimizer_shards}
    if actual_shards != set(expected_by_file):
        return False
    for filename, expected_tensors in expected_by_file.items():
        contents = _safetensors_contents(optimizer_dir / filename)
        if (
            contents is None
            or contents[0] != expected_tensors
            or contents[1].get("checkpoint_id") != checkpoint_id
        ):
            return False
    return True


def find_latest_checkpoint(
    output_dir: str | Path,
    expected_signature: str,
    *,
    expected_model_layout: dict[str, tuple[str, tuple[int, ...]]] | None = None,
    expected_artifact_kind: str = "merged",
    verify_artifact_digests: bool = True,
) -> ResumeCheckpoint | None:
    """Return the newest complete checkpoint compatible with this training run.

    Multi-process callers may set ``verify_artifact_digests`` only on process
    zero to avoid rereading every large tensor payload once per worker. All
    processes still perform the structural, size and checkpoint-ID checks.
    """

    if expected_artifact_kind not in _ARTIFACT_KINDS:
        raise ValueError(f"unknown expected artifact kind {expected_artifact_kind!r}")

    root = Path(output_dir)
    paths = [root]
    if root.is_dir():
        paths.extend(
            sorted(path for path in root.glob("checkpoint-*") if path.is_dir() and not path.is_symlink())
        )

    candidates: list[ResumeCheckpoint] = []
    incompatible: list[Path] = []
    corrupt: list[Path] = []
    for path in paths:
        manifest_path = path / "checkpoint_manifest.json"
        manifest = _read_bundle_manifest(path)
        if manifest is None:
            if manifest_path.exists() or manifest_path.is_symlink():
                corrupt.append(path)
            continue
        if manifest.get("format") != "training-checkpoint-bundle":
            corrupt.append(path)
            continue
        artifact_kind = manifest.get("artifact_kind")
        if (
            manifest.get("format_version") != _BUNDLE_FORMAT_VERSION
            or manifest.get("training_signature") != expected_signature
            or artifact_kind != expected_artifact_kind
            or artifact_kind not in _ARTIFACT_KINDS
        ):
            incompatible.append(path)
            continue
        global_step = manifest.get("global_step")
        total_steps = manifest.get("total_steps")
        if type(global_step) is not int or type(total_steps) is not int:
            corrupt.append(path)
            continue
        checkpoint_id = manifest.get("checkpoint_id")
        if (
            global_step < 0
            or total_steps <= 0
            or global_step > total_steps
            or not _is_lower_hex(checkpoint_id, 32)
        ):
            corrupt.append(path)
            continue
        if path != root:
            try:
                directory_step = int(path.name.removeprefix("checkpoint-"))
            except ValueError:
                corrupt.append(path)
                continue
            if directory_step != global_step:
                corrupt.append(path)
                continue
        if not isinstance(manifest.get("has_optimizer_state"), bool):
            corrupt.append(path)
            continue
        has_optimizer_state = manifest["has_optimizer_state"]
        source_examples_seen = manifest.get("source_examples_seen", 0)
        training_complete = manifest.get("training_complete", False)
        streaming_data_digest = manifest.get("streaming_data_digest")
        if (
            type(source_examples_seen) is not int
            or source_examples_seen < 0
            or not isinstance(training_complete, bool)
            or (
                streaming_data_digest is not None
                and not _is_lower_hex(streaming_data_digest, 64)
            )
            or (source_examples_seen > 0 and streaming_data_digest is None)
            or (
                training_complete
                and (
                    path != root
                    or global_step != total_steps
                    or streaming_data_digest is None
                )
            )
        ):
            corrupt.append(path)
            continue
        if (
            not _artifact_sizes_match(
                path,
                manifest,
                has_optimizer_state,
                artifact_kind=artifact_kind,
                verify_digests=verify_artifact_digests,
            )
            or (
                artifact_kind == "merged"
                and (
                    not _contains_config(path)
                    or not _contains_model_weights(path, checkpoint_id, expected_model_layout)
                )
            )
            or (
                artifact_kind == "peft-adapter"
                and not _contains_adapter_weights(path, checkpoint_id, expected_model_layout)
            )
            or (
                has_optimizer_state
                and not _contains_optimizer_state(path, global_step, checkpoint_id)
            )
        ):
            corrupt.append(path)
            continue
        candidates.append(
            ResumeCheckpoint(
                path,
                global_step,
                total_steps,
                has_optimizer_state,
                checkpoint_id,
                training_signature(manifest),
                artifact_kind,
                source_examples_seen,
                training_complete,
                streaming_data_digest,
            )
        )

    if incompatible:
        locations = ", ".join(str(path) for path in incompatible[:3])
        raise ValueError(
            "output_dir contains checkpoints from an incompatible training configuration: " + locations
        )
    if corrupt:
        locations = ", ".join(str(path) for path in corrupt[:3])
        raise ValueError("output_dir contains corrupt completed checkpoints: " + locations)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda checkpoint: (
            checkpoint.training_complete and checkpoint.path == root,
            checkpoint.global_step,
            checkpoint.path == root,
        ),
    )


def save_checkpoint_bundle(
    params: Any,
    optimizer_state: Any,
    output_dir: str | Path,
    *,
    step: int,
    save_optimizer_state: bool,
    model_source: str | Path,
    tokenizer_source: str | Path,
    config: Any,
    training_signature: str,
    total_steps: int,
    expected_model_layout: dict[str, tuple[str, tuple[int, ...]]] | None = None,
    leaf_transform: Callable[[str, Any], Any] | None = None,
    transform_plan: Mapping[str, Any] | None = None,
    source_examples_seen: int = 0,
    training_complete: bool = False,
    streaming_data_digest: str | None = None,
) -> Path:
    """Export a model and, when requested, its matching optimizer state."""

    import jax
    import numpy as np

    from ..distributed.runtime import sync_processes
    from .huggingface import (
        _assert_same_export_plan,
        _raise_if_process_zero_error,
        _write_json_atomic,
        export_hf_checkpoint,
        export_optimizer_checkpoint,
        remove_optimizer_checkpoint,
    )

    destination = Path(output_dir)
    completion_manifest = destination / "checkpoint_manifest.json"
    _assert_same_export_plan(
        "checkpoint bundle",
        [
            {
                "artifact_kind": "merged",
                "leaf_transform": leaf_transform is not None,
                "save_optimizer_state": save_optimizer_state,
                "step": step,
                "source_examples_seen": source_examples_seen,
                "streaming_data_digest": streaming_data_digest,
                "total_steps": total_steps,
                "training_complete": training_complete,
                "training_signature": training_signature,
                "transform_plan": transform_plan,
            }
        ],
    )
    # Perform local validation only after the plan digest agrees on every
    # worker.  A rank must never raise while its peers proceed to the next
    # checkpoint collective.
    if total_steps <= 0 or not 0 <= step <= total_steps:
        raise ValueError(f"checkpoint step {step} must be in [0, {total_steps}]")
    if type(source_examples_seen) is not int or source_examples_seen < 0:
        raise ValueError("source_examples_seen must be a non-negative integer")
    if not isinstance(training_complete, bool):
        raise TypeError("training_complete must be a boolean")
    if streaming_data_digest is not None and not _is_lower_hex(
        streaming_data_digest,
        64,
    ):
        raise ValueError("streaming_data_digest must be a lowercase SHA-256 digest or None")
    if source_examples_seen > 0 and streaming_data_digest is None:
        raise ValueError("streaming progress requires streaming_data_digest")
    if training_complete and (
        step != total_steps or streaming_data_digest is None
    ):
        raise ValueError(
            "a completed streaming checkpoint requires step == total_steps "
            "and streaming_data_digest"
        )
    if not isinstance(training_signature, str) or len(training_signature) != 64 or any(
        character not in "0123456789abcdef" for character in training_signature
    ):
        raise ValueError("training_signature must be a lowercase SHA-256 hex digest")
    checkpoint_id_bytes = np.zeros(16, dtype=np.uint8)
    checkpoint_id_error = None
    if jax.process_index() == 0:
        try:
            checkpoint_id_bytes[:] = np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8)
        except Exception as exc:
            checkpoint_id_error = exc
    _raise_if_process_zero_error("creating checkpoint identity", checkpoint_id_error)
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        checkpoint_id_bytes = np.asarray(
            multihost_utils.broadcast_one_to_all(checkpoint_id_bytes),
            dtype=np.uint8,
        )
    checkpoint_id = checkpoint_id_bytes.tobytes().hex()
    # The process-zero nonce is identical on every host even if the same
    # shared output directory is mounted under different absolute paths.
    barrier_id = checkpoint_id
    invalidate_error = None
    if jax.process_index() == 0:
        try:
            if completion_manifest.exists():
                if not completion_manifest.is_file() or completion_manifest.is_symlink():
                    raise RuntimeError(f"checkpoint manifest must be a real file: {completion_manifest}")
                completion_manifest.unlink()
        except Exception as exc:
            invalidate_error = exc
    _raise_if_process_zero_error("invalidating checkpoint bundle", invalidate_error)
    sync_processes(f"checkpoint-invalidate-{barrier_id}")

    # Invalidate old optimizer state before replacing model weights. If the
    # export is interrupted, no stale optimizer can be mistaken for a match.
    remove_optimizer_checkpoint(destination)

    destination = export_hf_checkpoint(
        params,
        output_dir,
        source=model_source,
        tokenizer_source=tokenizer_source,
        config=config,
        overwrite=True,
        checkpoint_id=checkpoint_id,
        leaf_transform=leaf_transform,
        transform_plan=transform_plan,
    )
    if save_optimizer_state:
        export_optimizer_checkpoint(
            optimizer_state,
            destination,
            step=step,
            overwrite=True,
            checkpoint_id=checkpoint_id,
        )
    completion_error = None
    if jax.process_index() == 0:
        try:
            artifact_sizes = _checkpoint_artifact_sizes(destination, save_optimizer_state, "merged")
            artifact_digests = _checkpoint_artifact_digests(destination, artifact_sizes)
            if (
                not _contains_config(destination)
                or not _contains_model_weights(destination, checkpoint_id, expected_model_layout)
                or (
                    save_optimizer_state
                    and not _contains_optimizer_state(destination, step, checkpoint_id)
                )
            ):
                raise RuntimeError(f"checkpoint artifacts failed structural validation: {destination}")
            _write_json_atomic(
                completion_manifest,
                {
                    "artifact_digests": artifact_digests,
                    "artifact_sizes": artifact_sizes,
                    "checkpoint_id": checkpoint_id,
                    "artifact_kind": "merged",
                    "format": "training-checkpoint-bundle",
                    "format_version": _BUNDLE_FORMAT_VERSION,
                    "global_step": int(step),
                    "has_optimizer_state": save_optimizer_state,
                    "source_examples_seen": int(source_examples_seen),
                    "streaming_data_digest": streaming_data_digest,
                    "total_steps": int(total_steps),
                    "training_complete": training_complete,
                    "training_signature": training_signature,
                },
            )
        except Exception as exc:
            completion_error = exc
    _raise_if_process_zero_error("completing checkpoint bundle", completion_error)
    sync_processes(f"checkpoint-complete-{barrier_id}")
    return destination


def save_adapter_checkpoint_bundle(
    params: Any,
    output_dir: str | Path,
    *,
    optimizer_state: Any = None,
    step: int,
    save_optimizer_state: bool = False,
    tokenizer_source: str | Path,
    adapter_config: Mapping[str, Any],
    mapping_fn: Callable[[str], Any],
    training_signature: str,
    total_steps: int,
    expected_adapter_layout: dict[str, tuple[str, tuple[int, ...]]] | None = None,
    source_examples_seen: int = 0,
    training_complete: bool = False,
    streaming_data_digest: str | None = None,
) -> Path:
    """Export a PEFT adapter and its optional matching optimizer state."""

    import jax
    import numpy as np

    from ..distributed.runtime import sync_processes
    from .huggingface import (
        _assert_same_export_plan,
        _raise_if_process_zero_error,
        _write_json_atomic,
        export_adapter_checkpoint,
        export_optimizer_checkpoint,
        remove_optimizer_checkpoint,
    )

    destination = Path(output_dir)
    completion_manifest = destination / "checkpoint_manifest.json"
    _assert_same_export_plan(
        "adapter checkpoint bundle",
        [
            {
                "adapter_config": dict(adapter_config),
                "artifact_kind": "peft-adapter",
                "save_optimizer_state": save_optimizer_state,
                "step": step,
                "source_examples_seen": source_examples_seen,
                "streaming_data_digest": streaming_data_digest,
                "total_steps": total_steps,
                "training_complete": training_complete,
                "training_signature": training_signature,
            }
        ],
    )
    if total_steps <= 0 or not 0 <= step <= total_steps:
        raise ValueError(f"checkpoint step {step} must be in [0, {total_steps}]")
    if type(source_examples_seen) is not int or source_examples_seen < 0:
        raise ValueError("source_examples_seen must be a non-negative integer")
    if not isinstance(training_complete, bool):
        raise TypeError("training_complete must be a boolean")
    if streaming_data_digest is not None and not _is_lower_hex(
        streaming_data_digest,
        64,
    ):
        raise ValueError("streaming_data_digest must be a lowercase SHA-256 digest or None")
    if source_examples_seen > 0 and streaming_data_digest is None:
        raise ValueError("streaming progress requires streaming_data_digest")
    if training_complete and (
        step != total_steps or streaming_data_digest is None
    ):
        raise ValueError(
            "a completed streaming checkpoint requires step == total_steps "
            "and streaming_data_digest"
        )
    if not _is_lower_hex(training_signature, 64):
        raise ValueError("training_signature must be a lowercase SHA-256 hex digest")
    if save_optimizer_state and optimizer_state is None:
        raise ValueError("optimizer_state is required when save_optimizer_state is True")

    checkpoint_id_bytes = np.zeros(16, dtype=np.uint8)
    checkpoint_id_error = None
    if jax.process_index() == 0:
        try:
            checkpoint_id_bytes[:] = np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8)
        except Exception as exc:
            checkpoint_id_error = exc
    _raise_if_process_zero_error("creating adapter checkpoint identity", checkpoint_id_error)
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        checkpoint_id_bytes = np.asarray(
            multihost_utils.broadcast_one_to_all(checkpoint_id_bytes),
            dtype=np.uint8,
        )
    checkpoint_id = checkpoint_id_bytes.tobytes().hex()

    invalidate_error = None
    if jax.process_index() == 0:
        try:
            if completion_manifest.exists() or completion_manifest.is_symlink():
                if not completion_manifest.is_file() or completion_manifest.is_symlink():
                    raise RuntimeError(f"checkpoint manifest must be a real file: {completion_manifest}")
                completion_manifest.unlink()
        except Exception as exc:
            invalidate_error = exc
    _raise_if_process_zero_error("invalidating adapter checkpoint bundle", invalidate_error)
    sync_processes(f"adapter-checkpoint-invalidate-{checkpoint_id}")
    remove_optimizer_checkpoint(destination)

    destination = export_adapter_checkpoint(
        params,
        destination,
        adapter_config=adapter_config,
        mapping_fn=mapping_fn,
        tokenizer_source=tokenizer_source,
        overwrite=True,
        checkpoint_id=checkpoint_id,
    )
    if save_optimizer_state:
        export_optimizer_checkpoint(
            optimizer_state,
            destination,
            step=step,
            overwrite=True,
            checkpoint_id=checkpoint_id,
        )

    completion_error = None
    if jax.process_index() == 0:
        try:
            artifact_sizes = _checkpoint_artifact_sizes(
                destination,
                save_optimizer_state,
                "peft-adapter",
            )
            artifact_digests = _checkpoint_artifact_digests(destination, artifact_sizes)
            if (
                not _contains_adapter_weights(destination, checkpoint_id, expected_adapter_layout)
                or (
                    save_optimizer_state
                    and not _contains_optimizer_state(destination, step, checkpoint_id)
                )
            ):
                raise RuntimeError(f"adapter checkpoint artifacts failed validation: {destination}")
            _write_json_atomic(
                completion_manifest,
                {
                    "artifact_digests": artifact_digests,
                    "artifact_kind": "peft-adapter",
                    "artifact_sizes": artifact_sizes,
                    "checkpoint_id": checkpoint_id,
                    "format": "training-checkpoint-bundle",
                    "format_version": _BUNDLE_FORMAT_VERSION,
                    "global_step": int(step),
                    "has_optimizer_state": save_optimizer_state,
                    "source_examples_seen": int(source_examples_seen),
                    "streaming_data_digest": streaming_data_digest,
                    "total_steps": int(total_steps),
                    "training_complete": training_complete,
                    "training_signature": training_signature,
                },
            )
        except Exception as exc:
            completion_error = exc
    _raise_if_process_zero_error("completing adapter checkpoint bundle", completion_error)
    sync_processes(f"adapter-checkpoint-complete-{checkpoint_id}")
    return destination


def prune_periodic_exports(output_dir: Path, save_total_limit: int) -> None:
    if save_total_limit <= 0:
        return
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if path.is_symlink() or not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        manifest_path = path / "checkpoint_manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            manifest.get("format") != "training-checkpoint-bundle"
            or manifest.get("format_version") != _BUNDLE_FORMAT_VERSION
            or manifest.get("global_step") != step
            or not _is_lower_hex(manifest.get("checkpoint_id"), 32)
            or not _is_lower_hex(manifest.get("training_signature"), 64)
        ):
            continue
        has_optimizer_state = manifest.get("has_optimizer_state")
        artifact_kind = manifest.get("artifact_kind")
        if not isinstance(has_optimizer_state, bool) or artifact_kind not in _ARTIFACT_KINDS:
            continue
        if (
            not _artifact_sizes_match(
                path,
                manifest,
                has_optimizer_state,
                artifact_kind=artifact_kind,
            )
            or (
                artifact_kind == "merged"
                and (
                    not _contains_config(path)
                    or not _contains_model_weights(path, manifest["checkpoint_id"])
                )
            )
            or (
                artifact_kind == "peft-adapter"
                and not _contains_adapter_weights(path, manifest["checkpoint_id"])
            )
            or (
                has_optimizer_state
                and not _contains_optimizer_state(path, step, manifest["checkpoint_id"])
            )
        ):
            continue
        checkpoints.append((step, path))
    checkpoints.sort()
    for _, path in checkpoints[:-save_total_limit]:
        resolved = path.resolve()
        root = output_dir.resolve()
        if root not in resolved.parents:
            raise RuntimeError(f"refusing to remove checkpoint outside output directory: {resolved}")
        shutil.rmtree(resolved)


__all__ = [
    "ResumeCheckpoint",
    "find_latest_checkpoint",
    "prune_periodic_exports",
    "save_adapter_checkpoint_bundle",
    "save_checkpoint_bundle",
    "training_signature",
]
