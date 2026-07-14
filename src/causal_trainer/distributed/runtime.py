"""Small, explicit helpers for multi-host JAX and five-axis sharding.

Importing this module does not discover devices.  Call
``initialize_distributed`` before ``create_mesh`` (and before any other code
calls :func:`jax.devices`) when running with more than one process.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from glob import glob
from typing import Any

import jax
import numpy as np
from jax.experimental.mesh_utils import create_device_mesh, create_hybrid_device_mesh
from jax.sharding import Mesh, NamedSharding, PartitionSpec

MESH_AXIS_NAMES = ("dp", "fsdp", "ep", "tp", "sp")
"""Mesh axes in CLI order: data, parameter, expert, tensor, sequence."""


@dataclass(frozen=True, slots=True)
class DistributedOptions:
    """Arguments accepted by :func:`jax.distributed.initialize`."""

    enabled: bool = True
    coordinator_address: str | None = None
    num_processes: int | None = None
    process_id: int | None = None
    local_device_ids: tuple[int, ...] | None = None
    initialization_timeout: int = 1800


def parse_local_device_ids(value: str | Sequence[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        values = tuple(int(part.strip()) for part in stripped.split(","))
    else:
        values = tuple(int(item) for item in value)
    if not values or any(item < 0 for item in values) or len(set(values)) != len(values):
        raise ValueError(f"local device IDs must be distinct non-negative integers, got {values!r}")
    return values


def _distributed_environment_present() -> bool:
    """Detect environments for which JAX supports argument-free bootstrap.

    This intentionally reads environment variables and local sysfs only:
    querying a backend to decide whether to initialize would itself be too late.
    """

    multi_process_counts = (
        "SLURM_NTASKS",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
    )
    for name in multi_process_counts:
        value = os.environ.get(name)
        if value:
            try:
                if int(value) > 1:
                    return True
            except ValueError:
                return True

    # Cloud TPU variables have changed names across runtime generations.  None
    # of these checks initializes a backend.
    if any(
        os.environ.get(name)
        for name in (
            "CLOUD_TPU_TASK_ID",
            "TPU_WORKER_ID",
            "TPU_PROCESS_ADDRESSES",
            "TPU_HOST_BOUNDS",
            "TPU_CHIPS_PER_HOST_BOUNDS",
            "MEGASCALE_NUM_SLICES",
        )
    ):
        return True

    # Current TPU VM images can omit every environment variable above. JAX's
    # argument-free bootstrap then discovers the slice through the VM metadata
    # service, but the runner still needs a backend-free signal telling it to
    # make that call. TPU PCI drivers expose one accelerator class entry per
    # local chip; reading these symlinks does not initialize a JAX client.
    platforms = os.environ.get("JAX_PLATFORMS") or os.environ.get("JAX_PLATFORM_NAME")
    if platforms and "tpu" not in {item.strip().lower() for item in platforms.split(",")}:
        return False
    return any(
        os.path.basename(os.path.realpath(path)).startswith("tpu")
        for path in glob("/sys/class/accel/accel*/device/driver")
    )


def initialize_distributed(options: DistributedOptions | None = None, **overrides: Any) -> bool:
    """Initialize JAX distributed without first touching the device backend.

    Returns ``True`` when a distributed client is initialized and ``False`` for
    an ordinary single-process run.  Explicit coordinator/process arguments
    always request initialization.  Otherwise supported TPU, Slurm and OpenMPI
    environments are detected without backend discovery and initialized with
    JAX's native auto-detection.
    """

    if options is None:
        options = DistributedOptions(**overrides)
    elif overrides:
        raise TypeError("pass either DistributedOptions or keyword overrides, not both")

    if jax.distributed.is_initialized():
        return True
    if not options.enabled:
        return False

    explicit = any(
        value is not None
        for value in (
            options.coordinator_address,
            options.num_processes,
            options.process_id,
            options.local_device_ids,
        )
    )
    torchrun_values = {
        "coordinator_address": (
            f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
            if os.environ.get("MASTER_ADDR") and os.environ.get("MASTER_PORT")
            else None
        ),
        "num_processes": int(os.environ["WORLD_SIZE"]) if os.environ.get("WORLD_SIZE") else None,
        "process_id": int(os.environ["RANK"]) if os.environ.get("RANK") else None,
        "local_device_ids": (
            (int(os.environ["LOCAL_RANK"]),) if os.environ.get("LOCAL_RANK") is not None else None
        ),
    }
    torchrun_complete = all(value is not None for value in torchrun_values.values())
    if not explicit and not _distributed_environment_present():
        if not torchrun_complete:
            return False

    coordinator_address = options.coordinator_address
    num_processes = options.num_processes
    process_id = options.process_id
    local_device_ids = options.local_device_ids
    if not explicit and torchrun_complete:
        coordinator_address = torchrun_values["coordinator_address"]
        num_processes = torchrun_values["num_processes"]
        process_id = torchrun_values["process_id"]
        local_device_ids = torchrun_values["local_device_ids"]

    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=local_device_ids,
        initialization_timeout=options.initialization_timeout,
    )
    return True


def _parse_dims(value: str | Sequence[int], *, name: str) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            dims = tuple(int(part.strip()) for part in value.split(","))
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated integer list") from exc
    else:
        dims = tuple(int(item) for item in value)
    if len(dims) != len(MESH_AXIS_NAMES):
        raise ValueError(f"{name} must contain exactly five dimensions in {MESH_AXIS_NAMES} order")
    if sum(dim == -1 for dim in dims) > 1:
        raise ValueError(f"{name} may contain at most one -1, got {dims!r}")
    if any(dim == 0 or dim < -1 for dim in dims):
        raise ValueError(f"{name} entries must be positive or -1, got {dims!r}")
    return dims


def resolve_axis_dims(value: str | Sequence[int], device_count: int) -> tuple[int, ...]:
    """Resolve a five-axis shape, replacing its optional ``-1``."""

    if device_count <= 0:
        raise ValueError(f"device_count must be positive, got {device_count}")
    dims = list(_parse_dims(value, name="sharding_axis"))
    known_product = math.prod(dim for dim in dims if dim != -1)
    if -1 in dims:
        if device_count % known_product:
            raise ValueError(
                f"known mesh dimensions have product {known_product}, which does not divide {device_count} devices"
            )
        dims[dims.index(-1)] = device_count // known_product
    if math.prod(dims) != device_count:
        raise ValueError(f"mesh shape {tuple(dims)} uses {math.prod(dims)} devices, but JAX found {device_count}")
    return tuple(dims)


def _local_mesh_shape(global_shape: Sequence[int], local_device_count: int, process_count: int) -> tuple[int, ...]:
    """Remove process factors from early, low-communication mesh axes."""

    local = list(global_shape)
    remaining = process_count
    for index, dim in enumerate(global_shape):
        factor = math.gcd(dim, remaining)
        local[index] //= factor
        remaining //= factor
        if remaining == 1:
            break
    if remaining != 1 or math.prod(local) != local_device_count:
        raise ValueError(
            f"cannot form a uniform {tuple(global_shape)} mesh from {process_count} processes "
            f"with {local_device_count} local devices each"
        )
    return tuple(local)


def _granule_dims(
    value: str | Sequence[int] | None,
    granule_count: int,
    global_shape: Sequence[int],
) -> tuple[int, ...]:
    if value is not None:
        return resolve_axis_dims(value, granule_count)

    # Match the 0.0.62 topology intent: keep the high-communication TP/SP
    # dimensions inside a host/slice whenever the requested shape permits it.
    result = [1] * len(global_shape)
    remaining = granule_count
    for index, dim in enumerate(global_shape):
        factor = math.gcd(dim, remaining)
        result[index] = factor
        remaining //= factor
        if remaining == 1:
            break
    if remaining != 1:
        raise ValueError(f"cannot map {granule_count} host/slice granules onto mesh shape {tuple(global_shape)}")
    return tuple(result)


def create_mesh(
    axis_dims: str | Sequence[int],
    *,
    dcn_axis_dims: str | Sequence[int] | None = None,
    axis_names: Sequence[str] = MESH_AXIS_NAMES,
    backend: str | None = None,
    allow_split_physical_axes: bool = True,
) -> Mesh:
    """Create a topology-aware global mesh.

    ``jax.distributed.initialize`` must already have run in a multi-process
    job.  For a single slice, process boundaries are treated as the outer
    network granules, matching the topology used by the reference trainer.
    Multi-slice devices use their native ``slice_index`` metadata.
    """

    names = tuple(axis_names)
    if len(names) != len(MESH_AXIS_NAMES) or len(set(names)) != len(names):
        raise ValueError("axis_names must contain five distinct names")

    devices = jax.devices(backend)
    global_shape = resolve_axis_dims(axis_dims, len(devices))
    slices = {getattr(device, "slice_index", 0) for device in devices}
    num_slices = len(slices)

    if num_slices > 1:
        outer = _granule_dims(dcn_axis_dims, num_slices, global_shape)
        inner = tuple(global_dim // outer_dim for global_dim, outer_dim in zip(global_shape, outer, strict=True))
        if any(global_dim % outer_dim for global_dim, outer_dim in zip(global_shape, outer, strict=True)):
            raise ValueError(f"DCN shape {outer} does not divide global mesh {global_shape}")
        device_array = create_hybrid_device_mesh(
            mesh_shape=inner,
            dcn_mesh_shape=outer,
            devices=devices,
            process_is_granule=False,
            should_sort_granules_by_key=True,
            allow_split_physical_axes=allow_split_physical_axes,
        )
    elif jax.process_count() > 1:
        local_shape = _local_mesh_shape(global_shape, jax.local_device_count(backend), jax.process_count())
        inferred_outer = tuple(
            global_dim // local_dim for global_dim, local_dim in zip(global_shape, local_shape, strict=True)
        )
        outer = (
            _granule_dims(dcn_axis_dims, jax.process_count(), global_shape)
            if dcn_axis_dims is not None
            else inferred_outer
        )
        inner = tuple(global_dim // outer_dim for global_dim, outer_dim in zip(global_shape, outer, strict=True))
        if math.prod(outer) != jax.process_count() or math.prod(inner) != jax.local_device_count(backend):
            raise ValueError(
                f"DCN mesh {outer} is incompatible with global mesh {global_shape}, "
                f"{jax.process_count()} processes and {jax.local_device_count(backend)} local devices"
            )
        device_array = create_hybrid_device_mesh(
            mesh_shape=inner,
            dcn_mesh_shape=outer,
            devices=devices,
            process_is_granule=True,
            should_sort_granules_by_key=True,
            allow_split_physical_axes=allow_split_physical_axes,
        )
    else:
        device_array = create_device_mesh(
            mesh_shape=global_shape,
            devices=devices,
            allow_split_physical_axes=allow_split_physical_axes,
        )
    return Mesh(device_array, names)


def path_to_string(path: Sequence[Any]) -> str:
    """Convert a JAX key path to the slash form used by parameter rules."""

    parts: list[str] = []
    for entry in path:
        if hasattr(entry, "key"):
            parts.append(str(entry.key))
        elif hasattr(entry, "idx"):
            parts.append(str(entry.idx))
        elif hasattr(entry, "name"):
            parts.append(str(entry.name))
        else:
            parts.append(str(entry))
    return "/".join(parts)


_COLUMN_KERNEL = re.compile(
    r"(?:^|/)(?:q_proj|k_proj|v_proj|gate_proj|up_proj)/kernel$|"
    r"^(?:model/)?embed_tokens/embedding$|^(?:model/)?lm_head/kernel$"
)
_ROW_KERNEL = re.compile(r"(?:^|/)(?:o_proj|down_proj)/kernel$")


def parameter_partition_spec(
    path: str,
    shape: Sequence[int] | None = None,
) -> PartitionSpec:
    """Return the reference-compatible spec for a neutral parameter path.

    Linear kernels use the JAX ``[input, output]`` layout. Biases and norm
    scales are replicated.
    """

    normalized = path.strip("/")
    if _COLUMN_KERNEL.search(normalized):
        spec = PartitionSpec("fsdp", "tp")
    elif _ROW_KERNEL.search(normalized):
        spec = PartitionSpec("tp", "fsdp")
    else:
        spec = PartitionSpec()
    if shape is not None and len(spec) > len(tuple(shape)):
        raise ValueError(f"partition spec {spec} has higher rank than {normalized} with shape {tuple(shape)}")
    return spec


def parameter_partition_specs(params: Any) -> Any:
    """Build a PartitionSpec PyTree matching an abstract or concrete parameter tree."""

    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: parameter_partition_spec(path_to_string(path), getattr(leaf, "shape", None)),
        params,
    )


def named_shardings(specs: Any, mesh: Mesh) -> Any:
    return jax.tree_util.tree_map(
        lambda spec: NamedSharding(mesh, spec),
        specs,
        is_leaf=lambda value: isinstance(value, PartitionSpec),
    )


def batch_partition_spec() -> PartitionSpec:
    return PartitionSpec(("dp", "fsdp"), "sp")


def batch_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, batch_partition_spec())


def replicated_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, PartitionSpec())


def process_local_to_global(
    local_data: Any,
    sharding: NamedSharding,
    *,
    global_shape: Sequence[int] | None = None,
) -> jax.Array:
    """Stitch a NumPy process-local value into a global JAX array."""

    return jax.make_array_from_process_local_data(sharding, local_data, global_shape=global_shape)


def host_global_to_array(value: np.ndarray, sharding: NamedSharding) -> jax.Array:
    """Place an identical full global host value from every process.

    The callback is invoked only for locally addressable device slices. This is
    intentionally used for small integer training batches: it is unambiguous
    when TP replicates a batch shard across devices or processes.
    """

    host_value = np.asarray(value)
    return jax.make_array_from_callback(
        host_value.shape,
        sharding,
        lambda index: np.asarray(host_value[index]),
    )


_PACKED_ARRAY_CONTROL_BYTES = 25
_MAX_PACKED_ARRAY_SCHEMA_BYTES = 1024 * 1024
_DEFAULT_PACKED_ARRAY_CHUNK_BYTES = 64 * 1024**2


def _packed_array_schema(columns: Mapping[str, np.ndarray], names: Sequence[str]) -> bytes:
    """Encode the small, non-payload part of a packed-column layout."""

    schema = bytearray(b"packed-host-arrays-v1\0")
    schema.extend(len(names).to_bytes(4, "little", signed=False))
    for name in names:
        value = columns[name]
        encoded_name = name.encode("utf-8")
        encoded_dtype = value.dtype.str.encode("ascii")
        if len(encoded_name) > 0xFFFFFFFF or len(encoded_dtype) > 0xFFFF:
            raise ValueError(f"packed column metadata is too large for {name!r}")
        schema.extend(len(encoded_name).to_bytes(4, "little", signed=False))
        schema.extend(encoded_name)
        schema.extend(len(encoded_dtype).to_bytes(2, "little", signed=False))
        schema.extend(encoded_dtype)
        schema.extend(int(value.shape[1]).to_bytes(8, "little", signed=False))
    if len(schema) > _MAX_PACKED_ARRAY_SCHEMA_BYTES:
        raise ValueError(
            f"packed column schema uses {len(schema)} bytes, maximum is "
            f"{_MAX_PACKED_ARRAY_SCHEMA_BYTES}"
        )
    return bytes(schema)


def _write_control_uint64(control: np.ndarray, offset: int, value: int) -> None:
    encoded = int(value).to_bytes(8, "little", signed=False)
    control[offset : offset + 8] = np.frombuffer(encoded, dtype=np.uint8)


def _read_control_uint64(control: np.ndarray, offset: int) -> int:
    return int.from_bytes(np.asarray(control[offset : offset + 8], dtype=np.uint8).tobytes(), "little")


def merge_packed_host_arrays(
    local_columns: Mapping[str, np.ndarray],
    *,
    max_chunk_bytes: int = _DEFAULT_PACKED_ARRAY_CHUNK_BYTES,
) -> dict[str, np.ndarray]:
    """Merge process-local packed rows in source-process order without disk I/O.

    Every column must be a C-contiguous numeric NumPy array shaped
    ``[local_rows, sequence_length]``.  Column names, dtypes, and sequence
    lengths must be identical on every process, while ``local_rows`` may differ
    or be zero.

    Multi-process calls first exchange a fixed-size control record and the
    exact binary schema.  They then visit source processes in rank order and
    broadcast fixed-row chunks.  All processes therefore execute precisely the
    same collective schedule and return identical arrays ordered as process 0
    rows, process 1 rows, and so on.  The result is preallocated, so additional
    host memory is bounded by a constant number of chunks rather than a second
    concatenated copy of the entire result.

    Payload leaves are broadcast as bytes.  JAX implements this host broadcast
    with a cross-process sum, so byte transport preserves floating-point bit
    patterns instead of applying floating-point arithmetic to loss weights.

    ``max_chunk_bytes`` bounds the sum of all column payloads in a normal
    chunk.  If a single packed row is larger than the limit, one complete row
    is still transferred so no real record is discarded.  A single-process
    call returns the validated input arrays directly and performs no
    collective.
    """

    process_count = jax.process_count()
    local_error: Exception | None = None
    columns: dict[str, np.ndarray] = {}
    names: tuple[str, ...] = ()
    local_rows = 0
    schema = b""
    try:
        if (
            not isinstance(max_chunk_bytes, int)
            or isinstance(max_chunk_bytes, bool)
            or max_chunk_bytes <= 0
        ):
            raise ValueError(f"max_chunk_bytes must be a positive integer, got {max_chunk_bytes!r}")
        if max_chunk_bytes > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("max_chunk_bytes exceeds the uint64 control-record limit")
        if not isinstance(local_columns, Mapping) or not local_columns:
            raise ValueError("local_columns must be a non-empty mapping of packed arrays")
        if any(not isinstance(name, str) or not name for name in local_columns):
            raise TypeError("packed column names must be non-empty strings")

        names = tuple(sorted(local_columns))
        expected_rows: int | None = None
        expected_sequence_length: int | None = None
        for name in names:
            value = local_columns[name]
            if not isinstance(value, np.ndarray):
                raise TypeError(f"packed column {name!r} must be a numpy.ndarray")
            if value.ndim != 2 or value.shape[1] <= 0:
                raise ValueError(
                    f"packed column {name!r} must have shape [rows, positive sequence length], "
                    f"got {value.shape}"
                )
            if not value.flags.c_contiguous:
                raise ValueError(f"packed column {name!r} must be C-contiguous")
            if value.dtype.kind not in "biuf":
                raise TypeError(f"packed column {name!r} has unsupported dtype {value.dtype}")
            canonical_dtype = np.dtype(jax.dtypes.canonicalize_dtype(value.dtype))
            if canonical_dtype != value.dtype:
                raise TypeError(
                    f"packed column {name!r} dtype {value.dtype} is not enabled by this JAX runtime; "
                    f"cast it explicitly to {canonical_dtype} before merging"
                )
            if expected_rows is None:
                expected_rows = int(value.shape[0])
            elif value.shape[0] != expected_rows:
                raise ValueError(
                    f"packed columns have different row counts: {name!r} has {value.shape[0]}, "
                    f"expected {expected_rows}"
                )
            if expected_sequence_length is None:
                expected_sequence_length = int(value.shape[1])
            elif value.shape[1] != expected_sequence_length:
                raise ValueError(
                    f"packed columns have different sequence lengths: {name!r} has {value.shape[1]}, "
                    f"expected {expected_sequence_length}"
                )
            columns[name] = value
        local_rows = expected_rows if expected_rows is not None else 0
        schema = _packed_array_schema(columns, names)
    except Exception as exc:
        local_error = exc

    if process_count == 1:
        if local_error is not None:
            raise local_error
        return {name: columns[name] for name in names}

    from jax.experimental import multihost_utils

    # The first collective always has a fixed shape, including when local
    # validation failed.  This lets every rank report the failure together
    # instead of one rank raising while its peers enter a data collective.
    control = np.zeros((_PACKED_ARRAY_CONTROL_BYTES,), dtype=np.uint8)
    control[0] = np.uint8(local_error is not None)
    if local_error is None:
        _write_control_uint64(control, 1, max_chunk_bytes)
        _write_control_uint64(control, 9, local_rows)
        _write_control_uint64(control, 17, len(schema))
    gathered_control = np.asarray(
        multihost_utils.process_allgather(control, tiled=False),
        dtype=np.uint8,
    )
    if gathered_control.size != process_count * _PACKED_ARRAY_CONTROL_BYTES:
        raise RuntimeError(
            "packed-array control all-gather returned an unexpected shape "
            f"{gathered_control.shape}"
        )
    gathered_control = gathered_control.reshape(process_count, _PACKED_ARRAY_CONTROL_BYTES)
    failed_processes = np.flatnonzero(gathered_control[:, 0]).tolist()
    if failed_processes:
        message = f"invalid packed arrays on source processes {failed_processes}"
        if local_error is not None:
            raise ValueError(message) from local_error
        raise ValueError(message)

    chunk_limits = [_read_control_uint64(row, 1) for row in gathered_control]
    if any(limit != chunk_limits[0] for limit in chunk_limits[1:]):
        raise ValueError(f"max_chunk_bytes differs across processes: {chunk_limits}")
    row_counts = [_read_control_uint64(row, 9) for row in gathered_control]
    schema_lengths = [_read_control_uint64(row, 17) for row in gathered_control]
    maximum_schema_length = max(schema_lengths)
    if maximum_schema_length <= 0 or maximum_schema_length > _MAX_PACKED_ARRAY_SCHEMA_BYTES:
        raise ValueError(f"invalid packed column schema lengths: {schema_lengths}")

    schema_payload = np.zeros((maximum_schema_length,), dtype=np.uint8)
    schema_payload[: len(schema)] = np.frombuffer(schema, dtype=np.uint8)
    gathered_schemas = np.asarray(
        multihost_utils.process_allgather(schema_payload, tiled=False),
        dtype=np.uint8,
    )
    if gathered_schemas.size != process_count * maximum_schema_length:
        raise RuntimeError(
            "packed-array schema all-gather returned an unexpected shape "
            f"{gathered_schemas.shape}"
        )
    gathered_schemas = gathered_schemas.reshape(process_count, maximum_schema_length)
    reference_schema = bytes(gathered_schemas[0, : schema_lengths[0]])
    mismatched_schemas = [
        source
        for source, length in enumerate(schema_lengths)
        if bytes(gathered_schemas[source, :length]) != reference_schema
    ]
    if mismatched_schemas:
        raise ValueError(f"packed column schemas differ on source processes {mismatched_schemas}")

    bytes_per_row = sum(int(columns[name].shape[1]) * columns[name].dtype.itemsize for name in names)
    if bytes_per_row <= 0:
        raise ValueError("packed column schema has no per-row payload")
    total_rows = sum(row_counts)
    if total_rows > np.iinfo(np.intp).max:
        raise OverflowError(f"merged packed row count {total_rows} exceeds the host index limit")
    if total_rows == 0:
        return {
            name: np.empty((0, columns[name].shape[1]), dtype=columns[name].dtype)
            for name in names
        }
    rows_per_chunk = min(max(row_counts), max(1, chunk_limits[0] // bytes_per_row))
    merged: dict[str, np.ndarray] = {}
    outgoing: dict[str, np.ndarray] = {}
    allocation_error: Exception | None = None
    try:
        merged = {
            name: np.empty((total_rows, columns[name].shape[1]), dtype=columns[name].dtype)
            for name in names
        }
        outgoing = {
            name: np.zeros(
                (rows_per_chunk, columns[name].shape[1] * columns[name].dtype.itemsize),
                dtype=np.uint8,
            )
            for name in names
        }
    except Exception as exc:
        allocation_error = exc
    allocation_status = np.asarray([allocation_error is not None], dtype=np.uint8)
    gathered_allocation_status = np.asarray(
        multihost_utils.process_allgather(allocation_status, tiled=False),
        dtype=np.uint8,
    ).reshape(-1)
    if gathered_allocation_status.size != process_count:
        raise RuntimeError(
            "packed-array allocation status all-gather returned an unexpected shape "
            f"{gathered_allocation_status.shape}"
        )
    failed_allocations = np.flatnonzero(gathered_allocation_status).tolist()
    if failed_allocations:
        message = f"could not allocate packed merge buffers on processes {failed_allocations}"
        if allocation_error is not None:
            raise MemoryError(message) from allocation_error
        raise MemoryError(message)

    process_index = jax.process_index()
    output_start = 0
    for source, source_rows in enumerate(row_counts):
        for source_start in range(0, source_rows, rows_per_chunk):
            valid_rows = min(rows_per_chunk, source_rows - source_start)
            for name in names:
                outgoing[name].fill(0)
            if process_index == source:
                source_stop = source_start + valid_rows
                for name in names:
                    source_block = np.ascontiguousarray(columns[name][source_start:source_stop])
                    outgoing[name][:valid_rows] = source_block.view(np.uint8).reshape(valid_rows, -1)
            incoming = multihost_utils.broadcast_one_to_all(
                outgoing,
                is_source=process_index == source,
            )
            output_stop = output_start + valid_rows
            for name in names:
                block = np.asarray(incoming[name])
                expected_shape = (
                    rows_per_chunk,
                    columns[name].shape[1] * columns[name].dtype.itemsize,
                )
                if block.shape != expected_shape or block.dtype != np.dtype(np.uint8):
                    raise RuntimeError(
                        f"broadcast for packed column {name!r} returned {block.shape}/{block.dtype}, "
                        f"expected integer bytes with shape {expected_shape}"
                    )
                packed_bytes = np.ascontiguousarray(block[:valid_rows], dtype=np.uint8)
                merged[name][output_start:output_stop] = packed_bytes.view(
                    columns[name].dtype
                ).reshape(valid_rows, columns[name].shape[1])
            output_start = output_stop
            del incoming

    if output_start != total_rows:
        raise RuntimeError(f"merged {output_start} packed rows, expected {total_rows}")
    return merged


def sync_processes(tag: str) -> None:
    """Barrier all processes using a caller-supplied, globally unique tag."""

    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices(tag)


def shard_tree_from_host(params: Any, shardings: Any) -> Any:
    """Place a host PyTree with a matching NamedSharding PyTree."""

    return jax.tree_util.tree_map(
        lambda value, sharding: jax.device_put(value, sharding),
        params,
        shardings,
        is_leaf=lambda value: isinstance(value, NamedSharding),
    )


PartitionRule = Callable[[str, Sequence[int] | None], PartitionSpec]
PartitionRuleMap = Mapping[str, PartitionSpec]


__all__ = [
    "MESH_AXIS_NAMES",
    "DistributedOptions",
    "PartitionRule",
    "PartitionRuleMap",
    "batch_partition_spec",
    "batch_sharding",
    "create_mesh",
    "host_global_to_array",
    "initialize_distributed",
    "merge_packed_host_arrays",
    "named_shardings",
    "parameter_partition_spec",
    "parameter_partition_specs",
    "parse_local_device_ids",
    "path_to_string",
    "process_local_to_global",
    "replicated_sharding",
    "resolve_axis_dims",
    "shard_tree_from_host",
    "sync_processes",
]
