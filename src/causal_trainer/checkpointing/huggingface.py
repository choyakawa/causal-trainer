"""Hugging Face Safetensors import/export without a PyTorch dependency.

The in-memory parameter tree uses neutral module names and JAX linear kernels
in ``[input, output]`` order.  This module is the only place that knows about
the standard Hugging Face key names and their ``[output, input]`` layout.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import os
import shutil
import struct
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from ..distributed.runtime import parameter_partition_specs, path_to_string, sync_processes

_MAX_HEADER_BYTES = 100 * 1024 * 1024
_TOKENIZER_NAMES = {
    "added_tokens.json",
    "chat_template.json",
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
_TOKENIZER_SUFFIXES = (".jinja", ".model", ".tiktoken")


@dataclass(frozen=True, slots=True)
class TensorMapping:
    parameter_path: str
    hf_key: str
    transpose: bool = False


@dataclass(frozen=True, slots=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


def resolve_hf_source(
    repo_id_or_path: str | os.PathLike[str],
    *,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local directory or download the needed files from the Hub."""

    candidate = Path(repo_id_or_path).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError(f"model source must be a directory, got {candidate}")
        return candidate.resolve()

    from huggingface_hub import snapshot_download

    downloaded = snapshot_download(
        repo_id=str(repo_id_or_path),
        revision=revision,
        token=token,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_files_only=local_files_only,
        allow_patterns=[
            "*.json",
            "*.jinja",
            "chat_templates/*.jinja",
            "*.model",
            "*.safetensors",
            "*.tiktoken",
            "*.txt",
        ],
    )
    return Path(downloaded)


def load_hf_config(source: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(source) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing model config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _parameter_to_hf(path: str) -> TensorMapping:
    path = path.strip("/")
    if path.startswith("model/"):
        path = path.removeprefix("model/")

    if path == "embed_tokens/embedding":
        return TensorMapping(path, "model.embed_tokens.weight")
    if path == "norm/scale":
        return TensorMapping(path, "model.norm.weight")
    if path == "lm_head/kernel":
        return TensorMapping(path, "lm_head.weight", transpose=True)

    parts = path.split("/")
    if len(parts) >= 4 and parts[0] == "layers" and parts[1].isdigit():
        layer = parts[1]
        block = parts[2]
        tail = parts[3:]
        prefix = f"model.layers.{layer}"
        if block == "attention" and len(tail) == 2:
            projection, leaf = tail
            if projection not in {"q_proj", "k_proj", "v_proj", "o_proj"}:
                raise KeyError(f"unknown attention parameter path: {path}")
            if leaf == "kernel":
                return TensorMapping(path, f"{prefix}.self_attn.{projection}.weight", transpose=True)
            if leaf == "bias":
                return TensorMapping(path, f"{prefix}.self_attn.{projection}.bias")
        if block == "mlp" and len(tail) == 2:
            projection, leaf = tail
            if projection in {"gate_proj", "up_proj", "down_proj"} and leaf == "kernel":
                return TensorMapping(path, f"{prefix}.mlp.{projection}.weight", transpose=True)
        if block in {"input_layernorm", "post_attention_layernorm"} and tail == ["scale"]:
            return TensorMapping(path, f"{prefix}.{block}.weight")
    raise KeyError(f"no Hugging Face tensor mapping for parameter path {path!r}")


def parameter_to_hf_mapping(path: str) -> TensorMapping:
    """Public mapping hook for the fixed, neutral parameter tree."""

    return _parameter_to_hf(path)


def parameter_hf_layout(
    template: Any,
    mapping_fn: Callable[[str], TensorMapping] = parameter_to_hf_mapping,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Return the exact Safetensors dtype and shape expected for every parameter."""

    layout: dict[str, tuple[str, tuple[int, ...]]] = {}
    for key_path, leaf in jax.tree_util.tree_flatten_with_path(template)[0]:
        mapping = mapping_fn(path_to_string(key_path))
        shape, dtype = _shape_dtype(leaf)
        storage_shape = tuple(reversed(shape)) if mapping.transpose else shape
        if mapping.hf_key in layout:
            raise ValueError(f"parameter mapping produced duplicate key {mapping.hf_key!r}")
        layout[mapping.hf_key] = (_dtype_code(np.empty((), dtype=dtype)), storage_shape)
    return layout


_STORAGE_DTYPES: dict[str, tuple[np.dtype[Any], int]] = {
    "BOOL": (np.dtype("u1"), 1),
    "F16": (np.dtype("<f2"), 2),
    "BF16": (np.dtype("<u2"), 2),
    "F32": (np.dtype("<f4"), 4),
    "F64": (np.dtype("<f8"), 8),
    "I8": (np.dtype("i1"), 1),
    "U8": (np.dtype("u1"), 1),
    "I16": (np.dtype("<i2"), 2),
    "U16": (np.dtype("<u2"), 2),
    "I32": (np.dtype("<i4"), 4),
    "U32": (np.dtype("<u4"), 4),
    "I64": (np.dtype("<i8"), 8),
    "U64": (np.dtype("<u8"), 8),
}


def _bfloat16_dtype() -> np.dtype[Any]:
    import ml_dtypes

    return np.dtype(ml_dtypes.bfloat16)


def _logical_dtype(storage_dtype: str) -> np.dtype[Any]:
    if storage_dtype == "BF16":
        return _bfloat16_dtype()
    if storage_dtype not in _STORAGE_DTYPES:
        raise TypeError(f"unsupported Safetensors dtype {storage_dtype!r}")
    return _STORAGE_DTYPES[storage_dtype][0]


class _SafeTensorFile:
    """Minimal mmap reader, including BF16 which the NumPy adapter omits."""

    def __init__(self, path: Path):
        self.path = path
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated Safetensors file: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size <= 0 or header_size > _MAX_HEADER_BYTES or 8 + header_size > file_size:
                raise ValueError(f"invalid Safetensors header size {header_size} in {path}")
            raw_header = handle.read(header_size)
        header = json.loads(raw_header.decode("utf-8").rstrip(" "))
        if not isinstance(header, dict):
            raise ValueError(f"invalid Safetensors header in {path}")
        metadata = header.get("__metadata__", {})
        if not isinstance(metadata, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
        ):
            raise ValueError(f"invalid Safetensors file metadata in {path}")
        self.metadata = dict(metadata)
        self.data_start = 8 + header_size
        self.file_size = file_size
        self.tensors: dict[str, TensorInfo] = {}
        for key, raw in header.items():
            if key == "__metadata__":
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"invalid tensor metadata for {key!r} in {path}")
            dtype = str(raw["dtype"])
            shape = tuple(int(dim) for dim in raw["shape"])
            offsets = tuple(int(offset) for offset in raw["data_offsets"])
            if len(offsets) != 2 or offsets[0] < 0 or offsets[1] < offsets[0]:
                raise ValueError(f"invalid data offsets for {key!r} in {path}")
            if dtype not in _STORAGE_DTYPES:
                raise TypeError(f"unsupported Safetensors dtype {dtype!r} for {key!r}")
            expected_bytes = math_prod(shape) * _STORAGE_DTYPES[dtype][1]
            if offsets[1] - offsets[0] != expected_bytes or self.data_start + offsets[1] > file_size:
                raise ValueError(f"invalid byte length for {key!r} in {path}")
            self.tensors[key] = TensorInfo(dtype=dtype, shape=shape, data_offsets=(offsets[0], offsets[1]))

    def read_slice(self, key: str, index: tuple[slice | int, ...]) -> np.ndarray[Any, Any]:
        info = self.tensors[key]
        storage_dtype = _STORAGE_DTYPES[info.dtype][0]
        view = np.memmap(
            self.path,
            dtype=storage_dtype,
            mode="r",
            offset=self.data_start + info.data_offsets[0],
            shape=info.shape,
            order="C",
        )
        if info.dtype == "BF16":
            view = view.view(_bfloat16_dtype())
        # A scalar callback may arrive with ``(slice(None),)``; indexing a
        # zero-rank memmap that way produces a length-one buffer.
        # A scalar SafeTensor has only one valid selection, so normalize it.
        if not info.shape:
            index = ()
        # Copy detaches the callback result from the mmap before JAX transfers it.
        return np.array(view[index], copy=True, order="C")


def math_prod(shape: Sequence[int]) -> int:
    result = 1
    for dim in shape:
        if dim < 0:
            raise ValueError(f"negative tensor dimension in {tuple(shape)}")
        result *= dim
    return result


class SafeTensorIndex:
    """Map tensor keys to local Safetensors shards."""

    def __init__(
        self,
        source: str | os.PathLike[str],
        *,
        index_filename: str = "model.safetensors.index.json",
        file_pattern: str = "*.safetensors",
        kind: str = "model",
        allow_symlinks: bool = True,
        expected_checkpoint_id: str | None = None,
    ):
        self.source = Path(source)
        self.allow_symlinks = allow_symlinks
        self.expected_checkpoint_id = expected_checkpoint_id
        index_path = self.source / index_filename
        if index_path.is_symlink() and not self.allow_symlinks:
            raise ValueError(f"checkpoint index must not be a symbolic link: {index_path}")
        if index_path.is_file():
            with index_path.open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict):
                raise ValueError(f"invalid weight_map in {index_path}")
            index_metadata = index.get("metadata", {})
            if not isinstance(index_metadata, dict):
                raise ValueError(f"invalid index metadata in {index_path}")
            if (
                self.expected_checkpoint_id is not None
                and index_metadata.get("checkpoint_id") != self.expected_checkpoint_id
            ):
                raise ValueError(f"checkpoint identity mismatch in {index_path}")
            self.index_metadata = dict(index_metadata)
            self.weight_map = {str(key): str(value) for key, value in weight_map.items()}
            if any(Path(filename).name != filename for filename in self.weight_map.values()):
                raise ValueError(f"weight_map contains an unsafe shard path in {index_path}")
        else:
            self.index_metadata: dict[str, Any] = {}
            files = sorted(self.source.glob(file_pattern))
            if not files:
                raise FileNotFoundError(f"no {kind} Safetensors files found under {self.source}")
            self.weight_map: dict[str, str] = {}
            for file in files:
                if file.is_symlink() and not self.allow_symlinks:
                    raise ValueError(f"checkpoint shard must not be a symbolic link: {file}")
                reader = _SafeTensorFile(file)
                for key in reader.tensors:
                    if key in self.weight_map:
                        raise ValueError(f"duplicate tensor {key!r} in {file} and {self.weight_map[key]}")
                    self.weight_map[key] = file.name
        self._readers: dict[str, _SafeTensorFile] = {}

    def reader_for(self, key: str) -> _SafeTensorFile:
        try:
            filename = self.weight_map[key]
        except KeyError as exc:
            raise KeyError(f"checkpoint does not contain required tensor {key!r}") from exc
        if filename not in self._readers:
            path = self.source / filename
            if (path.is_symlink() and not self.allow_symlinks) or not path.is_file():
                raise FileNotFoundError(f"weight map refers to missing shard {path}")
            reader = _SafeTensorFile(path)
            if (
                self.expected_checkpoint_id is not None
                and reader.metadata.get("checkpoint_id") != self.expected_checkpoint_id
            ):
                raise ValueError(f"checkpoint identity mismatch in {path}")
            self._readers[filename] = reader
        return self._readers[filename]

    def info(self, key: str) -> TensorInfo:
        return self.reader_for(key).tensors[key]

    def read_slice(self, key: str, index: tuple[slice | int, ...]) -> np.ndarray[Any, Any]:
        return self.reader_for(key).read_slice(key, index)


def _shape_dtype(leaf: Any) -> tuple[tuple[int, ...], np.dtype[Any]]:
    if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
        raise TypeError(f"parameter template leaves must expose shape and dtype, got {type(leaf).__name__}")
    return tuple(int(dim) for dim in leaf.shape), np.dtype(leaf.dtype)


def _source_index(index: tuple[slice | int, ...], transpose: bool) -> tuple[slice | int, ...]:
    if not transpose:
        return index
    if len(index) != 2:
        raise ValueError("transpose mapping is only valid for rank-two tensors")
    return index[1], index[0]


def _make_slice_callback(
    checkpoint: SafeTensorIndex,
    mapping: TensorMapping,
    target_dtype: np.dtype[Any],
) -> Callable[[tuple[slice | int, ...]], np.ndarray[Any, Any]]:
    cache: dict[tuple[tuple[int | None, int | None, int | None] | int, ...], np.ndarray[Any, Any]] = {}

    def callback(index: tuple[slice | int, ...]) -> np.ndarray[Any, Any]:
        cache_key = tuple(
            (item.start, item.stop, item.step) if isinstance(item, slice) else int(item) for item in index
        )
        if cache_key not in cache:
            chunk = checkpoint.read_slice(mapping.hf_key, _source_index(index, mapping.transpose))
            if mapping.transpose:
                chunk = chunk.T
            if chunk.dtype != target_dtype:
                chunk = chunk.astype(target_dtype, copy=False)
            # ``np.ascontiguousarray`` promotes a zero-rank array to shape
            # ``(1,)``. Optimizer counters are scalar Safetensors, and JAX
            # 0.10 validates callback shard shapes exactly, so preserve their
            # rank while still making non-scalar slices C-contiguous.
            cache[cache_key] = chunk if chunk.ndim == 0 else np.ascontiguousarray(chunk)
        return cache[cache_key]

    return callback


def load_sharded_parameters(
    source: str | os.PathLike[str],
    template: Any,
    mesh: Mesh,
    *,
    specs: Any | None = None,
    mapping_fn: Callable[[str], TensorMapping] = parameter_to_hf_mapping,
    dtype: Any | None = None,
    expected_checkpoint_id: str | None = None,
) -> Any:
    """Load only this process's slices directly into global JAX arrays.

    ``template`` may contain concrete arrays or ``jax.ShapeDtypeStruct`` leaves.
    Every process calls this function with the same tree and mesh.  No process
    materializes a complete model tensor on its host.
    """

    checkpoint = SafeTensorIndex(source, expected_checkpoint_id=expected_checkpoint_id)
    if specs is None:
        specs = parameter_partition_specs(template)

    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    spec_leaves, spec_treedef = jax.tree_util.tree_flatten(
        specs, is_leaf=lambda value: isinstance(value, PartitionSpec)
    )
    if treedef != spec_treedef:
        raise ValueError("partition spec tree does not match parameter template")

    expected_hf_keys: set[str] = set()
    for key_path, _ in path_leaves:
        mapping = mapping_fn(path_to_string(key_path))
        if mapping.hf_key in expected_hf_keys:
            raise ValueError(f"multiple parameters map to {mapping.hf_key!r}")
        expected_hf_keys.add(mapping.hf_key)
    actual_hf_keys = set(checkpoint.weight_map)
    if actual_hf_keys != expected_hf_keys:
        missing = sorted(expected_hf_keys - actual_hf_keys)
        unexpected = sorted(actual_hf_keys - expected_hf_keys)
        raise ValueError(
            "checkpoint tensors do not exactly match the supported architecture; "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    loaded: list[jax.Array] = []
    for (key_path, leaf), spec in zip(path_leaves, spec_leaves, strict=True):
        parameter_path = path_to_string(key_path)
        mapping = mapping_fn(parameter_path)

        target_shape, template_dtype = _shape_dtype(leaf)
        target_dtype = np.dtype(dtype) if dtype is not None else template_dtype
        source_info = checkpoint.info(mapping.hf_key)
        mapped_shape = tuple(reversed(source_info.shape)) if mapping.transpose else source_info.shape
        if mapped_shape != target_shape:
            raise ValueError(
                f"shape mismatch for {parameter_path}: checkpoint {mapping.hf_key} has {source_info.shape}, "
                f"mapped shape {mapped_shape}, expected {target_shape}"
            )

        sharding = NamedSharding(mesh, spec)
        # Produces an early, readable divisibility error instead of an XLA error.
        sharding.shard_shape(target_shape)
        loaded.append(
            jax.make_array_from_callback(
                target_shape,
                sharding,
                _make_slice_callback(checkpoint, mapping, target_dtype),
                dtype=target_dtype,
            )
        )
    return jax.tree_util.tree_unflatten(treedef, loaded)


def load_optimizer_checkpoint(
    source: str | os.PathLike[str],
    template: Any,
    shardings: Any,
    *,
    expected_step: int,
    expected_checkpoint_id: str | None = None,
) -> Any:
    """Restore saved optimizer arrays into the initialized Optax state tree."""

    optimizer_dir = Path(source) / "optimizer"
    if optimizer_dir.is_symlink() or not optimizer_dir.is_dir():
        raise FileNotFoundError(f"optimizer checkpoint directory is missing or unsafe: {optimizer_dir}")
    manifest_path = optimizer_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"optimizer completion manifest is missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("format") != "optimizer-state-safetensors":
        raise ValueError(f"invalid optimizer manifest: {manifest_path}")
    if manifest.get("format_version") != 1 or manifest.get("global_step") != expected_step:
        raise ValueError(
            f"optimizer checkpoint step/version mismatch in {manifest_path}: "
            f"step={manifest.get('global_step')!r}, version={manifest.get('format_version')!r}"
        )
    if (
        expected_checkpoint_id is not None
        and manifest.get("checkpoint_id") != expected_checkpoint_id
    ):
        raise ValueError(f"optimizer checkpoint identity mismatch in {manifest_path}")
    try:
        current_optax_version = version("optax")
    except PackageNotFoundError:
        current_optax_version = "unknown"
    saved_versions = (manifest.get("jax_version"), manifest.get("optax_version"))
    current_versions = (jax.__version__, current_optax_version)
    if saved_versions != current_versions:
        raise ValueError(
            "optimizer dependency version mismatch: "
            f"checkpoint has JAX/Optax {saved_versions}, current environment has {current_versions}"
        )
    tensor_metadata = manifest.get("tensors")
    if not isinstance(tensor_metadata, list) or not tensor_metadata:
        raise ValueError(f"optimizer manifest does not contain a tensor list: {manifest_path}")
    if manifest.get("tensor_count") != len(tensor_metadata):
        raise ValueError(f"optimizer tensor_count does not match the tensor list: {manifest_path}")
    signature_payload = json.dumps(tensor_metadata, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if manifest.get("layout_fingerprint") != hashlib.sha256(signature_payload).hexdigest():
        raise ValueError(f"optimizer layout fingerprint mismatch: {manifest_path}")

    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    sharding_leaves, sharding_treedef = jax.tree_util.tree_flatten(
        shardings,
        is_leaf=lambda value: isinstance(value, NamedSharding),
    )
    if treedef != sharding_treedef:
        raise ValueError("optimizer sharding tree does not match the initialized optimizer state")
    if len(tensor_metadata) != len(path_leaves):
        raise ValueError(
            f"optimizer tensor count mismatch: checkpoint has {len(tensor_metadata)}, "
            f"initialized state has {len(path_leaves)}"
        )

    checkpoint = SafeTensorIndex(
        optimizer_dir,
        index_filename="state.safetensors.index.json",
        file_pattern="state*.safetensors",
        kind="optimizer",
        allow_symlinks=False,
        expected_checkpoint_id=expected_checkpoint_id,
    )
    expected_keys = {f"state/{ordinal:06d}" for ordinal in range(len(path_leaves))}
    if set(checkpoint.weight_map) != expected_keys:
        raise ValueError("optimizer Safetensors keys do not match the completion manifest")

    loaded: list[jax.Array] = []
    computed_total_size = 0
    for ordinal, ((key_path, leaf), sharding, metadata) in enumerate(
        zip(path_leaves, sharding_leaves, tensor_metadata, strict=True)
    ):
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid optimizer tensor metadata at ordinal {ordinal}")
        key = f"state/{ordinal:06d}"
        target_shape, target_dtype = _shape_dtype(leaf)
        expected_path = path_to_string(key_path)
        if (
            metadata.get("ordinal") != ordinal
            or metadata.get("key") != key
            or metadata.get("path") != expected_path
            or metadata.get("shape") != list(target_shape)
            or metadata.get("dtype") != target_dtype.name
            or metadata.get("file") != checkpoint.weight_map[key]
        ):
            raise ValueError(f"optimizer layout mismatch for {expected_path!r}")
        source_info = checkpoint.info(key)
        if source_info.shape != target_shape:
            raise ValueError(
                f"optimizer shape mismatch for {expected_path}: checkpoint {source_info.shape}, "
                f"expected {target_shape}"
            )
        expected_dtype_code = _dtype_code(np.empty((), dtype=target_dtype))
        if source_info.dtype != expected_dtype_code:
            raise ValueError(
                f"optimizer dtype mismatch for {expected_path}: checkpoint {source_info.dtype}, "
                f"expected {expected_dtype_code}"
            )
        computed_total_size += math_prod(target_shape) * target_dtype.itemsize
        if expected_path == "count" or expected_path.endswith("/count"):
            saved_count = int(np.asarray(checkpoint.read_slice(key, ())).item())
            if saved_count != expected_step:
                raise ValueError(
                    f"optimizer counter mismatch for {expected_path}: checkpoint {saved_count}, "
                    f"expected {expected_step}"
                )
        if not isinstance(sharding, NamedSharding):
            raise TypeError(f"optimizer sharding leaf for {expected_path} is not NamedSharding")
        sharding.shard_shape(target_shape)
        loaded.append(
            jax.make_array_from_callback(
                target_shape,
                sharding,
                _make_slice_callback(checkpoint, TensorMapping(expected_path, key), target_dtype),
                dtype=target_dtype,
            )
        )
    if manifest.get("total_size") != computed_total_size:
        raise ValueError(
            f"optimizer total_size mismatch: checkpoint {manifest.get('total_size')!r}, "
            f"expected {computed_total_size}"
        )
    return jax.tree_util.tree_unflatten(treedef, loaded)


def _dtype_code(array: np.ndarray[Any, Any]) -> str:
    name = array.dtype.name
    codes = {
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
    try:
        return codes[name]
    except KeyError as exc:
        raise TypeError(f"cannot export dtype {array.dtype}") from exc


def _tensor_bytes(array: np.ndarray[Any, Any], dtype_code: str) -> memoryview:
    contiguous = np.ascontiguousarray(array)
    if dtype_code == "BF16":
        contiguous = contiguous.view(np.uint16)
    if contiguous.dtype.byteorder == ">" or (contiguous.dtype.byteorder == "=" and os.sys.byteorder == "big"):
        contiguous = contiguous.byteswap()
    return memoryview(contiguous).cast("B")


def _encode_safetensors_header(
    tensors: Sequence[tuple[str, Sequence[int], np.dtype[Any]]],
    *,
    metadata: Mapping[str, str] | None = None,
) -> tuple[bytes, dict[str, int]]:
    """Precompute a Safetensors header for streaming payload writes."""

    file_metadata = {"format": "pt"}
    if metadata is not None:
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()):
            raise TypeError("Safetensors metadata keys and values must be strings")
        file_metadata.update(metadata)
    header: dict[str, Any] = {"__metadata__": file_metadata}
    byte_lengths: dict[str, int] = {}
    offset = 0
    for key, shape_value, dtype_value in tensors:
        shape = tuple(int(dimension) for dimension in shape_value)
        dtype = np.dtype(dtype_value)
        code = _dtype_code(np.empty((), dtype=dtype))
        byte_length = math_prod(shape) * dtype.itemsize
        header[key] = {
            "dtype": code,
            "shape": list(shape),
            "data_offsets": [offset, offset + byte_length],
        }
        byte_lengths[key] = byte_length
        offset += byte_length
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    return encoded, byte_lengths


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _copy_hf_assets(
    source: Path | None,
    destination: Path,
    config: Mapping[str, Any] | Any | None,
    *,
    write_config: bool = True,
    model_dtype: str | None = None,
) -> None:
    source_config: dict[str, Any] = {}
    emit_fixed_source_config = False
    if write_config and source is not None and (source / "config.json").is_file():
        source_config = load_hf_config(source)
    if write_config and config is not None:
        if isinstance(config, Mapping):
            updates = dict(config)
        elif hasattr(config, "to_dict"):
            updates = dict(config.to_dict())
        elif dataclasses.is_dataclass(config):
            updates = dataclasses.asdict(config)
        else:
            raise TypeError("config must be a mapping, dataclass, or expose to_dict()")
        source_config.update(updates)
        emit_fixed_source_config = {
            "partial_rotary_factor",
            "rope_scaling",
            "rope_theta",
            "torch_dtype",
        }.issubset(updates)
    if write_config and emit_fixed_source_config:
        # Reading accepts the Transformers 5.x canonical spellings, but
        # full/merged checkpoints always return to the user's fixed source
        # config format.
        source_config.pop("rope_parameters", None)
        source_config.pop("dtype", None)
    if write_config and source_config and model_dtype is not None:
        # Hugging Face's save_pretrained records the dtype of the actual model
        # parameters.  Do the same even when the source config was created for
        # a different training dtype.
        source_config["torch_dtype"] = model_dtype
        if not emit_fixed_source_config and "dtype" in source_config:
            source_config["dtype"] = model_dtype
    if write_config and source_config:
        _write_json_atomic(destination / "config.json", source_config)

    if source is None:
        return
    for path in source.iterdir():
        if not path.is_file() or path.name == "config.json":
            continue
        if path.name in _TOKENIZER_NAMES or path.name.startswith("tokenizer.") or path.suffix in _TOKENIZER_SUFFIXES:
            target = destination / path.name
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)

    template_source = source / "chat_templates"
    if template_source.is_dir():
        template_destination = destination / "chat_templates"
        template_destination.mkdir(parents=True, exist_ok=True)
        for path in sorted(template_source.glob("*.jinja")):
            if not path.is_file():
                continue
            target = template_destination / path.name
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)


def _normalized_shard_index(index: Any, shape: Sequence[int]) -> tuple[tuple[int, int], ...]:
    if not isinstance(index, tuple) or len(index) != len(shape):
        raise ValueError(f"unsupported shard index {index!r} for shape {tuple(shape)}")
    normalized: list[tuple[int, int]] = []
    for entry, dimension in zip(index, shape, strict=True):
        if not isinstance(entry, slice):
            raise ValueError(f"checkpoint export requires slice-based shard indices, got {index!r}")
        start, stop, step = entry.indices(int(dimension))
        if step != 1:
            raise ValueError(f"checkpoint export requires unit-stride shard indices, got {index!r}")
        normalized.append((start, stop))
    return tuple(normalized)


def _bounded_chunk_slices(
    shape: Sequence[int],
    *,
    itemsize: int,
    max_chunk_bytes: int,
) -> Iterator[tuple[slice, ...]]:
    """Tile a C-order shard into rectangular chunks bounded by host bytes."""

    normalized_shape = tuple(int(dimension) for dimension in shape)
    if math_prod(normalized_shape) == 0:
        return
    max_elements = max(1, max_chunk_bytes // itemsize)
    chunk_shape = list(normalized_shape)
    trailing_elements = 1
    for axis in range(len(normalized_shape) - 1, -1, -1):
        dimension = normalized_shape[axis]
        if trailing_elements * dimension <= max_elements:
            trailing_elements *= dimension
            continue
        chunk_shape[:axis] = [1] * axis
        chunk_shape[axis] = max(1, max_elements // trailing_elements)
        break

    starts = (range(0, dimension, chunk) for dimension, chunk in zip(normalized_shape, chunk_shape, strict=True))
    for offsets in itertools.product(*starts):
        yield tuple(
            slice(start, min(start + chunk, dimension))
            for start, chunk, dimension in zip(offsets, chunk_shape, normalized_shape, strict=True)
        )


def _gather_to_process_zero(
    array: Any,
    *,
    max_chunk_bytes: int = 64 * 1024**2,
    transpose: bool = False,
) -> np.ndarray[Any, Any] | None:
    """Gather one global JAX array through bounded host all-gathers.

    A direct reshard from a global multi-host mesh to one process-zero device
    is not a supported JAX device assignment. Instead, chunks of each unique
    global shard are balanced across processes that address a replica. Owners
    transfer one bounded chunk in parallel per round, every process participates
    in ``process_allgather``, and only process zero assembles the full tensor.
    """

    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")
    if not isinstance(array, jax.Array):
        host = np.asarray(array)
        if transpose:
            if host.ndim != 2:
                raise ValueError("checkpoint export can transpose only rank-two tensors")
            host = np.ascontiguousarray(host.T)
        return host if jax.process_index() == 0 else None
    if transpose and array.ndim != 2:
        raise ValueError("checkpoint export can transpose only rank-two tensors")
    if jax.process_count() == 1 and getattr(array, "is_fully_addressable", False):
        host = np.asarray(jax.device_get(array))
        return np.ascontiguousarray(host.T) if transpose else host

    from jax.experimental import multihost_utils

    shape: tuple[int, ...] = ()
    dtype = np.dtype(np.float32)
    shard_processes: dict[tuple[tuple[int, int], ...], set[int]] = {}
    local_shards: dict[tuple[tuple[int, int], ...], Any] = {}
    preparation_error = None
    try:
        participating_processes = {int(device.process_index) for device in array.sharding.device_set}
        if jax.process_count() > 1 and participating_processes != set(range(jax.process_count())):
            raise ValueError("multi-process checkpoint arrays must use a sharding that spans every process")

        shape, dtype = _shape_dtype(array)
        for shard in array.global_shards:
            key = _normalized_shard_index(shard.index, shape)
            shard_processes.setdefault(key, set()).add(int(shard.device.process_index))
        if not shard_processes:
            raise RuntimeError("global array does not expose any shards")

        covered_elements = sum(
            math_prod(tuple(stop - start for start, stop in key))
            for key in shard_processes
        )
        expected_elements = math_prod(shape)
        if covered_elements != expected_elements:
            raise RuntimeError(
                f"unique shard coverage has {covered_elements} elements, expected {expected_elements} for {shape}"
            )

        for shard in array.addressable_shards:
            key = _normalized_shard_index(shard.index, shape)
            local_shards.setdefault(key, shard)
    except Exception as exc:
        preparation_error = exc
    _raise_if_any_process_error("planning checkpoint tensor transfer", preparation_error)

    process_count = jax.process_count()
    chunk_elements = max(1, max_chunk_bytes // max(1, process_count * dtype.itemsize))
    per_process_chunk_bytes = chunk_elements * dtype.itemsize
    transfer_queues: dict[
        int,
        list[tuple[tuple[tuple[int, int], ...], tuple[slice, ...], tuple[int, ...]]],
    ] = {process: [] for process in range(process_count)}
    schedule_plan: list[dict[str, Any]] = []
    assigned_elements = [0] * process_count
    for key in sorted(shard_processes):
        candidates = sorted(shard_processes[key])
        shard_shape = tuple(stop - start for start, stop in key)
        for local_index in _bounded_chunk_slices(
            shard_shape,
            itemsize=dtype.itemsize,
            max_chunk_bytes=per_process_chunk_bytes,
        ):
            piece_shape = tuple(index.stop - index.start for index in local_index)
            piece_elements = math_prod(piece_shape)
            owner = min(candidates, key=lambda process: (assigned_elements[process], process))
            assigned_elements[owner] += piece_elements
            transfer_queues[owner].append((key, local_index, piece_shape))
            schedule_plan.append(
                {
                    "global_index": [list(bounds) for bounds in key],
                    "local_index": [[index.start, index.stop] for index in local_index],
                    "owner": owner,
                    "piece_shape": list(piece_shape),
                }
            )
    ownership_error = None
    missing_local_shards = {
        key
        for key, _, _ in transfer_queues[jax.process_index()]
        if key not in local_shards
    }
    if missing_local_shards:
        ownership_error = RuntimeError(
            f"process {jax.process_index()} does not address its elected checkpoint shards: "
            f"{sorted(missing_local_shards)[:3]}"
        )
    _raise_if_any_process_error("locating checkpoint shards", ownership_error)
    _assert_same_export_plan(
        "checkpoint tensor transfer",
        [
            {
                "chunk_elements": chunk_elements,
                "dtype": dtype.name,
                "shape": list(shape),
                "transpose": transpose,
                "work": schedule_plan,
            }
        ],
    )

    is_primary = jax.process_index() == 0
    output_shape = (shape[1], shape[0]) if transpose else shape
    allocation_error = None
    host_value = None
    if is_primary:
        try:
            host_value = np.empty(output_shape, dtype=dtype)
        except Exception as exc:
            allocation_error = exc
    _raise_if_process_zero_error("allocating checkpoint tensor", allocation_error)
    round_count = max((len(queue) for queue in transfer_queues.values()), default=0)
    for round_index in range(round_count):
        round_elements = max(
            math_prod(queue[round_index][2])
            for queue in transfer_queues.values()
            if round_index < len(queue)
        )
        local_item = (
            transfer_queues[jax.process_index()][round_index]
            if round_index < len(transfer_queues[jax.process_index()])
            else None
        )
        payload = None
        read_error = None
        try:
            payload = np.zeros((round_elements,), dtype=dtype)
            if local_item is not None:
                key, local_index, piece_shape = local_item
                source_shard = local_shards[key]
                piece = np.asarray(
                    jax.device_get(source_shard.data[local_index]),
                    dtype=dtype,
                )
                piece_elements = math_prod(piece_shape)
                if piece.shape != piece_shape or piece.size != piece_elements:
                    raise RuntimeError(
                        f"checkpoint shard chunk has shape {piece.shape}, expected {piece_shape}"
                    )
                payload[:piece_elements] = piece.reshape(-1)
                del piece
        except Exception as exc:
            read_error = exc
        _raise_if_any_process_error("reading checkpoint shard chunk", read_error)
        if payload is None:
            raise RuntimeError("checkpoint transfer payload allocation failed without a reported error")

        gathered = None
        gather_error = None
        try:
            gathered = np.asarray(
                multihost_utils.process_allgather(payload, tiled=False),
                dtype=dtype,
            ).reshape(process_count, round_elements)
        except Exception as exc:
            gather_error = exc
        _raise_if_any_process_error("receiving checkpoint shard chunks", gather_error)
        if gathered is None:
            raise RuntimeError("checkpoint all-gather failed without a reported error")
        write_error = None
        if is_primary:
            try:
                for owner, queue in transfer_queues.items():
                    if round_index >= len(queue):
                        continue
                    key, local_index, piece_shape = queue[round_index]
                    piece_elements = math_prod(piece_shape)
                    piece = gathered[owner, :piece_elements].reshape(piece_shape)
                    target = tuple(
                        slice(global_start + index.start, global_start + index.stop)
                        for (global_start, _), index in zip(key, local_index, strict=True)
                    )
                    if transpose:
                        host_value[target[1], target[0]] = piece.T
                    else:
                        host_value[target] = piece
            except Exception as exc:
                write_error = exc
        piece = None
        _raise_if_process_zero_error("assembling checkpoint tensor chunks", write_error)
        del gathered, payload
    return host_value


def _raise_if_process_zero_error(stage: str, error: Exception | None) -> None:
    """Propagate a process-zero filesystem failure before another collective."""

    if jax.process_count() == 1:
        if error is not None:
            raise RuntimeError(f"{stage} failed: {error}") from error
        return

    from jax.experimental import multihost_utils

    payload = np.zeros(4096, dtype=np.int32)
    if jax.process_index() == 0 and error is not None:
        encoded = f"{type(error).__name__}: {error}".encode("utf-8", errors="replace")[:4095]
        payload[0] = len(encoded)
        payload[1 : 1 + len(encoded)] = np.frombuffer(encoded, dtype=np.uint8).astype(np.int32)
    payload = np.asarray(multihost_utils.broadcast_one_to_all(payload))
    message_size = int(payload[0])
    if message_size:
        message = bytes(payload[1 : 1 + message_size].astype(np.uint8)).decode("utf-8", errors="replace")
        propagated = RuntimeError(f"{stage} failed on process zero: {message}")
        if jax.process_index() == 0 and error is not None:
            raise propagated from error
        raise propagated


def _raise_if_any_process_error(stage: str, error: Exception | None) -> None:
    """Collectively propagate an error raised by any worker."""

    if jax.process_count() == 1:
        if error is not None:
            raise RuntimeError(f"{stage} failed: {error}") from error
        return

    from jax.experimental import multihost_utils

    payload = np.zeros(4096, dtype=np.int32)
    if error is not None:
        encoded = f"process {jax.process_index()} {type(error).__name__}: {error}".encode(
            "utf-8", errors="replace"
        )[:4095]
        payload[0] = len(encoded)
        payload[1 : 1 + len(encoded)] = np.frombuffer(encoded, dtype=np.uint8).astype(np.int32)
    gathered = np.asarray(multihost_utils.process_allgather(payload, tiled=False)).reshape(-1, 4096)
    messages = []
    for process_payload in gathered:
        message_size = int(process_payload[0])
        if message_size:
            messages.append(
                bytes(process_payload[1 : 1 + message_size].astype(np.uint8)).decode(
                    "utf-8", errors="replace"
                )
            )
    if messages:
        propagated = RuntimeError(f"{stage} failed: {'; '.join(messages)}")
        if error is not None:
            raise propagated from error
        raise propagated


def _collective_operation_id() -> str:
    """Return a process-zero nonce shared by every JAX process.

    Barrier tags must be byte-for-byte identical on every host.  Deriving them
    from ``Path.absolute()`` is unsafe when workers mount the same shared
    directory at different local paths, so process zero supplies the nonce.
    """

    nonce = np.zeros(16, dtype=np.uint8)
    nonce_error = None
    if jax.process_index() == 0:
        try:
            nonce[:] = np.frombuffer(os.urandom(16), dtype=np.uint8)
        except Exception as exc:
            nonce_error = exc
    _raise_if_process_zero_error("creating checkpoint operation ID", nonce_error)
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        nonce = np.asarray(multihost_utils.broadcast_one_to_all(nonce), dtype=np.uint8)
    return nonce.tobytes().hex()


def _assert_same_export_plan(kind: str, plan: Sequence[Mapping[str, Any]]) -> None:
    """Fail collectively before leaf collectives when workers planned differently."""

    if jax.process_count() <= 1:
        return
    from jax.experimental import multihost_utils

    local_digest = np.zeros(32, dtype=np.uint8)
    serialization_error = None
    try:
        encoded = json.dumps(
            list(plan),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        local_digest[:] = np.frombuffer(hashlib.sha256(encoded).digest(), dtype=np.uint8)
    except Exception as exc:
        serialization_error = exc
    _raise_if_any_process_error(f"serializing {kind} export plan", serialization_error)
    gathered = np.asarray(multihost_utils.process_allgather(local_digest, tiled=False), dtype=np.uint8).reshape(-1, 32)
    if np.any(gathered != gathered[0]):
        fingerprints = ", ".join(bytes(row).hex()[:12] for row in gathered)
        raise RuntimeError(f"{kind} export plans differ across processes: {fingerprints}")


def _stream_tree_checkpoint(
    params: Any,
    output_file: Path,
    *,
    mapping_fn: Callable[[str], TensorMapping],
    overwrite: bool,
    checkpoint_id: str | None,
    leaf_transform: Callable[[str, jax.Array], jax.Array] | None,
    transform_plan: Mapping[str, Any] | None,
    kind: str,
    require_single_dtype: bool = True,
) -> str:
    """Collectively gather and immediately append one tensor at a time."""

    barrier_id = _collective_operation_id()
    _assert_same_export_plan(
        f"{kind} checkpoint options",
        [
            {
                "filename": output_file.name,
                "leaf_transform": leaf_transform is not None,
                "overwrite": overwrite,
                "transform_plan": transform_plan,
            }
        ],
    )
    output_exists = False
    setup_error = None
    if jax.process_index() == 0:
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if output_file.is_symlink() or (output_file.exists() and not output_file.is_file()):
                raise RuntimeError(f"checkpoint output must be a real file: {output_file}")
            output_exists = output_file.exists()
            if output_exists and overwrite:
                output_file.unlink()
        except Exception as exc:
            setup_error = exc
    _raise_if_process_zero_error(f"{kind} checkpoint setup", setup_error)
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        output_exists = bool(
            np.asarray(multihost_utils.broadcast_one_to_all(np.asarray(output_exists, dtype=np.bool_))).item()
        )
    if output_exists and not overwrite:
        raise FileExistsError(f"checkpoint output already exists: {output_file}")
    sync_processes(f"{kind}-export-create-{barrier_id}")

    records: list[tuple[str, jax.Array]] = []
    mappings: dict[str, TensorMapping] = {}
    parameter_paths: dict[str, str] = {}
    export_plan: list[dict[str, Any]] = []
    storage_layout: list[tuple[str, tuple[int, ...], np.dtype[Any]]] = []
    model_dtype: str | None = None
    planning_error = None
    try:
        path_leaves, _ = jax.tree_util.tree_flatten_with_path(params)
        if not path_leaves:
            raise ValueError(f"{kind} parameter tree does not contain any array leaves")
        model_dtypes: set[str] = set()
        for key_path, leaf in path_leaves:
            parameter_path = path_to_string(key_path)
            if not isinstance(leaf, jax.Array):
                raise TypeError(
                    f"{kind} checkpoint leaf {parameter_path!r} must be a concrete jax.Array, "
                    f"got {type(leaf).__name__}"
                )
            mapping = mapping_fn(parameter_path)
            shape, dtype = _shape_dtype(leaf)
            if mapping.transpose and len(shape) != 2:
                raise ValueError(f"transpose mapping for {mapping.hf_key!r} requires a rank-two tensor")
            if mapping.hf_key in mappings:
                raise ValueError(f"parameter mapping produced duplicate key {mapping.hf_key!r}")
            storage_shape = (shape[1], shape[0]) if mapping.transpose else shape
            records.append((mapping.hf_key, leaf))
            mappings[mapping.hf_key] = mapping
            parameter_paths[mapping.hf_key] = parameter_path
            storage_layout.append((mapping.hf_key, storage_shape, dtype))
            model_dtypes.add(dtype.name)
            export_plan.append(
                {
                    "dtype": dtype.name,
                    "key": mapping.hf_key,
                    "parameter_path": parameter_path,
                    "shape": list(shape),
                    "storage_shape": list(storage_shape),
                    "transpose": mapping.transpose,
                }
            )
        if require_single_dtype and len(model_dtypes) != 1:
            raise ValueError(f"checkpoint parameters must have one storage dtype, got {sorted(model_dtypes)}")
        for _, _, dtype in storage_layout:
            _dtype_code(np.empty((), dtype=dtype))
        model_dtype = next(iter(model_dtypes)) if len(model_dtypes) == 1 else "mixed"
        if require_single_dtype and model_dtype not in {"bfloat16", "float16", "float32"}:
            raise TypeError(f"unsupported checkpoint parameter dtype {model_dtype!r}")
    except Exception as exc:
        planning_error = exc
    _raise_if_any_process_error(f"planning {kind} checkpoint", planning_error)
    if model_dtype is None:
        raise RuntimeError(f"{kind} checkpoint planning failed without a reported error")
    _assert_same_export_plan(
        f"{kind} checkpoint",
        [
            {
                "filename": output_file.name,
                "leaves": export_plan,
                "leaf_transform": leaf_transform is not None,
                "transform_plan": transform_plan,
            }
        ],
    )

    metadata = {"checkpoint_id": checkpoint_id} if checkpoint_id is not None else None
    encoded_header, byte_lengths = _encode_safetensors_header(storage_layout, metadata=metadata)
    storage_shapes = {key: shape for key, shape, _ in storage_layout}
    storage_dtypes = {key: dtype for key, _, dtype in storage_layout}
    temporary = output_file.with_name(f".{output_file.name}.tmp-{barrier_id}")
    writer = None
    open_error = None
    if jax.process_index() == 0:
        try:
            temporary.unlink(missing_ok=True)
            writer = temporary.open("xb")
            prefix = struct.pack("<Q", len(encoded_header))
            if writer.write(prefix) != len(prefix) or writer.write(encoded_header) != len(encoded_header):
                raise OSError(f"short write while opening {kind} checkpoint")
        except Exception as exc:
            open_error = exc
            try:
                if writer is not None:
                    writer.close()
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    _raise_if_process_zero_error(f"opening {kind} checkpoint", open_error)

    try:
        for output_key, leaf in records:
            export_leaf = leaf
            transform_error = None
            if leaf_transform is not None:
                try:
                    export_leaf = leaf_transform(parameter_paths[output_key], leaf)
                    if not isinstance(export_leaf, jax.Array):
                        raise TypeError(
                            f"leaf transform for {parameter_paths[output_key]!r} must return a concrete "
                            f"jax.Array, got {type(export_leaf).__name__}"
                        )
                    base_shape, base_dtype = _shape_dtype(leaf)
                    transformed_shape, transformed_dtype = _shape_dtype(export_leaf)
                    if transformed_shape != base_shape:
                        raise ValueError(
                            f"leaf transform for {parameter_paths[output_key]!r} returned shape "
                            f"{transformed_shape}, expected {base_shape}"
                        )
                    if transformed_dtype != base_dtype:
                        raise TypeError(
                            f"leaf transform for {parameter_paths[output_key]!r} returned dtype "
                            f"{transformed_dtype.name}, expected {base_dtype.name}"
                        )
                    export_leaf.block_until_ready()
                except Exception as exc:
                    transform_error = exc
                _raise_if_any_process_error(f"transforming checkpoint tensor {output_key}", transform_error)
            try:
                host_value = _gather_to_process_zero(
                    export_leaf,
                    transpose=mappings[output_key].transpose,
                )
            finally:
                if export_leaf is not leaf:
                    del export_leaf

            write_error = None
            if jax.process_index() == 0:
                try:
                    if writer is None or host_value is None:
                        raise RuntimeError(f"failed to gather {output_key}")
                    expected_shape = storage_shapes[output_key]
                    if tuple(host_value.shape) != expected_shape:
                        raise RuntimeError(
                            f"gathered tensor {output_key} has shape {host_value.shape}, expected {expected_shape}"
                        )
                    if np.dtype(host_value.dtype) != storage_dtypes[output_key]:
                        raise TypeError(
                            f"gathered tensor {output_key} has dtype {host_value.dtype}, "
                            f"expected {storage_dtypes[output_key]}"
                        )
                    dtype_code = _dtype_code(host_value)
                    payload = _tensor_bytes(host_value, dtype_code)
                    if len(payload) != byte_lengths[output_key]:
                        raise RuntimeError(
                            f"gathered tensor {output_key} has {len(payload)} bytes, "
                            f"expected {byte_lengths[output_key]}"
                        )
                    try:
                        written = writer.write(payload)
                        if written != len(payload):
                            raise OSError(
                                f"short write for checkpoint tensor {output_key}: "
                                f"wrote {written} of {len(payload)} bytes"
                            )
                    finally:
                        payload.release()
                except Exception as exc:
                    write_error = exc
            _raise_if_process_zero_error(f"writing checkpoint tensor {output_key}", write_error)
            del host_value
    except BaseException:
        if jax.process_index() == 0:
            try:
                if writer is not None:
                    writer.close()
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    finalize_error = None
    if jax.process_index() == 0:
        try:
            if writer is None:
                raise RuntimeError("checkpoint writer was not opened")
            writer.flush()
            os.fsync(writer.fileno())
            writer.close()
            os.replace(temporary, output_file)
        except Exception as exc:
            finalize_error = exc
            try:
                if writer is not None and not writer.closed:
                    writer.close()
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    _raise_if_process_zero_error(f"finalizing {kind} checkpoint", finalize_error)
    sync_processes(f"{kind}-export-finished-{barrier_id}")
    return model_dtype


def export_hf_checkpoint(
    params: Any,
    output_dir: str | os.PathLike[str],
    *,
    source: str | os.PathLike[str] | None = None,
    config: Mapping[str, Any] | Any | None = None,
    tokenizer_source: str | os.PathLike[str] | None = None,
    mapping_fn: Callable[[str], TensorMapping] = parameter_to_hf_mapping,
    overwrite: bool = False,
    checkpoint_id: str | None = None,
    leaf_transform: Callable[[str, jax.Array], jax.Array] | None = None,
    transform_plan: Mapping[str, Any] | None = None,
) -> Path:
    """Write one standard ``model.safetensors`` via bounded per-leaf gathers.

    Every process participates in each parameter transfer. Only process zero
    materializes a complete parameter, writes it immediately, and then releases
    it; no host ever builds a complete gathered model tree.
    """

    destination = Path(output_dir)
    source_path = Path(source) if source is not None else None
    tokenizer_path = Path(tokenizer_source) if tokenizer_source is not None else source_path
    cleanup_error = None
    existing_weights = False
    if jax.process_index() == 0:
        try:
            destination.mkdir(parents=True, exist_ok=True)
            existing = (
                list(destination.glob("model*.safetensors"))
                + list(destination.glob("model.safetensors.index.json"))
                + [
                    path
                    for path in (destination / "adapter_model.safetensors", destination / "adapter_config.json")
                    if path.exists() or path.is_symlink()
                ]
            )
            existing_weights = bool(existing)
            if overwrite:
                for path in existing:
                    if path.is_symlink() or not path.is_file():
                        raise RuntimeError(f"checkpoint artifact must be a real file: {path}")
                    path.unlink()
        except Exception as exc:
            cleanup_error = exc
    _raise_if_process_zero_error("preparing model checkpoint output", cleanup_error)
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        existing_weights = bool(
            np.asarray(multihost_utils.broadcast_one_to_all(np.asarray(existing_weights, dtype=np.bool_))).item()
        )
    if existing_weights and not overwrite:
        raise FileExistsError(f"checkpoint output already contains model weights: {destination}")

    model_dtype = _stream_tree_checkpoint(
        params,
        destination / "model.safetensors",
        mapping_fn=mapping_fn,
        overwrite=True,
        checkpoint_id=checkpoint_id,
        leaf_transform=leaf_transform,
        transform_plan=transform_plan,
        kind="model",
        require_single_dtype=True,
    )
    assets_error = None
    if jax.process_index() == 0:
        try:
            _copy_hf_assets(source_path, destination, config, model_dtype=model_dtype)
            if tokenizer_path is not None and tokenizer_path != source_path:
                _copy_hf_assets(tokenizer_path, destination, None, write_config=False)
        except Exception as exc:
            assets_error = exc
    _raise_if_process_zero_error("writing model checkpoint assets", assets_error)
    sync_processes(f"model-assets-finished-{_collective_operation_id()}")
    return destination


def export_adapter_checkpoint(
    params: Any,
    output_dir: str | os.PathLike[str],
    *,
    adapter_config: Mapping[str, Any],
    mapping_fn: Callable[[str], TensorMapping],
    tokenizer_source: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
    checkpoint_id: str | None = None,
) -> Path:
    """Write a single PEFT-compatible adapter file and its configuration."""

    destination = Path(output_dir)
    config_path = destination / "adapter_config.json"
    cleanup_error = None
    existing_adapter = False
    if jax.process_index() == 0:
        try:
            destination.mkdir(parents=True, exist_ok=True)
            existing = [destination / "adapter_model.safetensors", config_path]
            existing.extend(destination.glob("model*.safetensors"))
            existing.extend(destination.glob("model.safetensors.index.json"))
            model_config = destination / "config.json"
            if model_config.exists() or model_config.is_symlink():
                existing.append(model_config)
            existing_adapter = any(path.exists() or path.is_symlink() for path in existing)
            if overwrite:
                for path in existing:
                    if path.is_symlink() or (path.exists() and not path.is_file()):
                        raise RuntimeError(f"adapter artifact must be a real file: {path}")
                    path.unlink(missing_ok=True)
        except Exception as exc:
            cleanup_error = exc
    _raise_if_process_zero_error("preparing adapter checkpoint output", cleanup_error)
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        existing_adapter = bool(
            np.asarray(multihost_utils.broadcast_one_to_all(np.asarray(existing_adapter, dtype=np.bool_))).item()
        )
    if existing_adapter and not overwrite:
        raise FileExistsError(f"checkpoint output already contains an adapter: {destination}")

    _stream_tree_checkpoint(
        params,
        destination / "adapter_model.safetensors",
        mapping_fn=mapping_fn,
        overwrite=True,
        checkpoint_id=checkpoint_id,
        leaf_transform=None,
        transform_plan=None,
        kind="adapter",
        require_single_dtype=True,
    )
    assets_error = None
    if jax.process_index() == 0:
        try:
            _write_json_atomic(config_path, dict(adapter_config))
            if tokenizer_source is not None:
                _copy_hf_assets(Path(tokenizer_source), destination, None, write_config=False)
        except Exception as exc:
            assets_error = exc
    _raise_if_process_zero_error("writing adapter checkpoint assets", assets_error)
    sync_processes(f"adapter-assets-finished-{_collective_operation_id()}")
    return destination


def _clear_optimizer_directory(checkpoint_dir: Path, destination: Path) -> None:
    if destination.is_symlink():
        raise RuntimeError(f"optimizer checkpoint path must not be a symbolic link: {destination}")
    if not destination.exists():
        return
    root = checkpoint_dir.resolve()
    resolved = destination.resolve()
    if destination.name != "optimizer" or root not in resolved.parents:
        raise RuntimeError(f"refusing to remove optimizer state outside checkpoint directory: {resolved}")
    if not destination.is_dir():
        raise RuntimeError(f"optimizer checkpoint path must be a real directory: {destination}")
    shutil.rmtree(destination)


def remove_optimizer_checkpoint(output_dir: str | os.PathLike[str]) -> None:
    """Remove stale optimizer state from a model checkpoint on process zero."""

    checkpoint_dir = Path(output_dir)
    destination = checkpoint_dir / "optimizer"
    barrier_id = _collective_operation_id()
    remove_error = None
    if jax.process_index() == 0:
        try:
            _clear_optimizer_directory(checkpoint_dir, destination)
        except Exception as exc:
            remove_error = exc
    _raise_if_process_zero_error("removing stale optimizer checkpoint", remove_error)
    sync_processes(f"optimizer-remove-{barrier_id}")


def export_optimizer_checkpoint(
    optimizer_state: Any,
    output_dir: str | os.PathLike[str],
    *,
    step: int,
    overwrite: bool = False,
    checkpoint_id: str | None = None,
) -> Path:
    """Stream all Optax leaves into one ``optimizer/state.safetensors``."""

    _assert_same_export_plan("optimizer checkpoint options", [{"overwrite": overwrite, "step": step}])
    if step < 0:
        raise ValueError("optimizer checkpoint step cannot be negative")
    checkpoint_dir = Path(output_dir)
    destination = checkpoint_dir / "optimizer"
    output_exists = False
    setup_error = None
    if jax.process_index() == 0:
        try:
            if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
                raise RuntimeError(f"optimizer checkpoint path must be a real directory: {destination}")
            output_exists = destination.exists() and any(destination.iterdir())
            if output_exists and overwrite:
                _clear_optimizer_directory(checkpoint_dir, destination)
            destination.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            setup_error = exc
    _raise_if_process_zero_error("optimizer checkpoint setup", setup_error)
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        output_exists = bool(
            np.asarray(multihost_utils.broadcast_one_to_all(np.asarray(output_exists, dtype=np.bool_))).item()
        )
    if output_exists and not overwrite:
        raise FileExistsError(f"checkpoint output already contains optimizer state: {destination}")

    path_to_key: dict[str, str] = {}
    tensor_metadata: dict[str, dict[str, Any]] = {}
    export_plan: list[dict[str, Any]] = []
    total_size = 0
    planning_error = None
    try:
        path_leaves, _ = jax.tree_util.tree_flatten_with_path(optimizer_state)
        if not path_leaves:
            raise ValueError("optimizer state does not contain any array leaves")
        for ordinal, (key_path, leaf) in enumerate(path_leaves):
            optimizer_path = path_to_string(key_path)
            if not isinstance(leaf, jax.Array):
                raise TypeError(
                    f"optimizer checkpoint leaf {optimizer_path!r} must be a concrete jax.Array, "
                    f"got {type(leaf).__name__}"
                )
            shape, dtype = _shape_dtype(leaf)
            key = f"state/{ordinal:06d}"
            if optimizer_path in path_to_key:
                raise ValueError(f"duplicate optimizer path {optimizer_path!r}")
            path_to_key[optimizer_path] = key
            tensor_metadata[key] = {
                "dtype": dtype.name,
                "file": "state.safetensors",
                "key": key,
                "ordinal": ordinal,
                "path": optimizer_path,
                "shape": list(shape),
            }
            total_size += math_prod(shape) * dtype.itemsize
            export_plan.append(
                {
                    "dtype": dtype.name,
                    "key": key,
                    "path": optimizer_path,
                    "shape": list(shape),
                }
            )
    except Exception as exc:
        planning_error = exc
    _raise_if_any_process_error("planning optimizer checkpoint", planning_error)
    _assert_same_export_plan(
        "optimizer checkpoint",
        [
            {
                "filename": "state.safetensors",
                "leaves": export_plan,
                "step": step,
            }
        ],
    )

    def optimizer_mapping(path: str) -> TensorMapping:
        try:
            return TensorMapping(path, path_to_key[path])
        except KeyError as exc:
            raise KeyError(f"unknown optimizer state path {path!r}") from exc

    _stream_tree_checkpoint(
        optimizer_state,
        destination / "state.safetensors",
        mapping_fn=optimizer_mapping,
        overwrite=True,
        checkpoint_id=checkpoint_id,
        leaf_transform=None,
        transform_plan={"kind": "optimizer-state", "step": int(step)},
        kind="optimizer",
        require_single_dtype=False,
    )

    finalize_error = None
    if jax.process_index() == 0:
        try:
            try:
                optax_version = version("optax")
            except PackageNotFoundError:
                optax_version = "unknown"
            tensors = [tensor_metadata[f"state/{ordinal:06d}"] for ordinal in range(len(tensor_metadata))]
            signature_payload = json.dumps(tensors, separators=(",", ":"), sort_keys=True).encode("utf-8")
            manifest = {
                "format": "optimizer-state-safetensors",
                "format_version": 1,
                "global_step": int(step),
                "jax_version": jax.__version__,
                "optax_version": optax_version,
                "layout_fingerprint": hashlib.sha256(signature_payload).hexdigest(),
                "tensor_count": len(tensors),
                "tensors": tensors,
                "total_size": total_size,
            }
            if checkpoint_id is not None:
                manifest["checkpoint_id"] = checkpoint_id
            _write_json_atomic(destination / "manifest.json", manifest)
        except Exception as exc:
            finalize_error = exc
    _raise_if_process_zero_error("finalizing optimizer checkpoint", finalize_error)
    sync_processes(f"optimizer-manifest-finished-{_collective_operation_id()}")
    return destination


__all__ = [
    "SafeTensorIndex",
    "TensorInfo",
    "TensorMapping",
    "export_adapter_checkpoint",
    "export_hf_checkpoint",
    "export_optimizer_checkpoint",
    "load_hf_config",
    "load_optimizer_checkpoint",
    "load_sharded_parameters",
    "parameter_hf_layout",
    "parameter_to_hf_mapping",
    "remove_optimizer_checkpoint",
    "resolve_hf_source",
]
