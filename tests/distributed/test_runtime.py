import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec

import causal_trainer.distributed.runtime as distributed
from causal_trainer.distributed.runtime import (
    merge_packed_host_arrays,
    parameter_partition_spec,
    resolve_axis_dims,
)


def test_distributed_environment_detects_tpu_sysfs_without_cloud_variables(monkeypatch) -> None:
    for name in (
        "SLURM_NTASKS",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "CLOUD_TPU_TASK_ID",
        "TPU_WORKER_ID",
        "TPU_PROCESS_ADDRESSES",
        "TPU_HOST_BOUNDS",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "MEGASCALE_NUM_SLICES",
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        distributed,
        "glob",
        lambda _pattern: ["/sys/class/accel/accel0/device/driver"],
    )
    monkeypatch.setattr(distributed.os.path, "realpath", lambda _path: "/sys/bus/pci/drivers/tpu_driver")

    assert distributed._distributed_environment_present() is True


def test_explicit_cpu_platform_ignores_tpu_sysfs(monkeypatch) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(
        distributed,
        "glob",
        lambda _pattern: ["/sys/class/accel/accel0/device/driver"],
    )
    monkeypatch.setattr(distributed.os.path, "realpath", lambda _path: "/sys/bus/pci/drivers/tpu_driver")

    assert distributed._distributed_environment_present() is False


def test_inferred_data_parallel_axis() -> None:
    assert resolve_axis_dims("-1, 1, 1, 4, 1", 32) == (8, 1, 1, 4, 1)


def test_reference_parameter_rules() -> None:
    assert parameter_partition_spec("layers/0/attention/q_proj/kernel") == PartitionSpec("fsdp", "tp")
    assert parameter_partition_spec("layers/0/attention/k_proj/kernel") == PartitionSpec("fsdp", "tp")
    assert parameter_partition_spec("layers/0/attention/v_proj/kernel") == PartitionSpec("fsdp", "tp")
    assert parameter_partition_spec("layers/0/attention/o_proj/kernel") == PartitionSpec("tp", "fsdp")
    assert parameter_partition_spec("layers/0/attention/q_proj/bias") == PartitionSpec()


def test_single_process_packed_merge_preserves_typed_rows(monkeypatch) -> None:
    monkeypatch.setattr(distributed.jax, "process_count", lambda: 1)
    columns = {
        "input_ids": np.asarray([[1, 2], [3, 4]], dtype=np.int32),
        "loss_weights": np.asarray([[1.0, 0.1], [0.5, 0.0]], dtype=np.float32),
    }
    merged = merge_packed_host_arrays(columns, max_chunk_bytes=1)
    assert merged.keys() == columns.keys()
    for name in columns:
        assert merged[name] is columns[name]


def test_single_process_packed_merge_accepts_canonical_empty_shard(monkeypatch) -> None:
    monkeypatch.setattr(distributed.jax, "process_count", lambda: 1)
    columns = {
        "input_ids": np.empty((0, 8), dtype=np.int32),
        "loss_weights": np.empty((0, 8), dtype=np.float32),
    }
    merged = merge_packed_host_arrays(columns)
    assert all(value.shape == (0, 8) for value in merged.values())


def test_multi_process_packed_merge_is_rank_major_and_keeps_fractional_weights(monkeypatch) -> None:
    local = {
        "input_ids": np.asarray([[10, 11], [12, 13]], dtype=np.int32),
        "loss_weights": np.asarray([[1.0, 0.1], [0.5, 0.0]], dtype=np.float32),
    }
    remote = {
        "input_ids": np.asarray([[20, 21], [22, 23], [24, 25]], dtype=np.int32),
        "loss_weights": np.asarray([[0.25, 1.0], [0.0, 0.75], [0.1, 0.2]], dtype=np.float32),
    }
    names = tuple(sorted(local))
    schema = distributed._packed_array_schema(local, names)

    monkeypatch.setattr(distributed.jax, "process_count", lambda: 3)
    monkeypatch.setattr(distributed.jax, "process_index", lambda: 0)

    def write_uint64(control: np.ndarray, offset: int, value: int) -> None:
        control[offset : offset + 8] = np.frombuffer(
            int(value).to_bytes(8, "little", signed=False),
            dtype=np.uint8,
        )

    def control_for(rows: int) -> np.ndarray:
        control = np.zeros(25, dtype=np.uint8)
        write_uint64(control, 1, 1)
        write_uint64(control, 9, rows)
        write_uint64(control, 17, len(schema))
        return control

    def fake_allgather(value, *, tiled):
        value = np.asarray(value)
        if value.shape == (25,):
            return np.stack((control_for(2), control_for(3), control_for(0)))
        if value.shape == (len(schema),):
            return np.stack((value, value, value))
        if value.shape == (1,):
            return np.zeros((3, 1), dtype=np.uint8)
        raise AssertionError(f"unexpected all-gather payload {value.shape}")

    broadcast_index = 0

    def fake_broadcast(value, *, is_source):
        nonlocal broadcast_index
        current = broadcast_index
        broadcast_index += 1
        if current < 2:
            assert is_source is True
            return value
        assert is_source is False
        remote_index = current - 2
        return {
            name: np.ascontiguousarray(
                remote[name][remote_index : remote_index + 1]
            ).view(np.uint8).reshape(1, -1)
            for name in names
        }

    monkeypatch.setattr(multihost_utils, "process_allgather", fake_allgather)
    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", fake_broadcast)

    merged = merge_packed_host_arrays(local, max_chunk_bytes=1)
    np.testing.assert_array_equal(
        merged["input_ids"],
        np.concatenate((local["input_ids"], remote["input_ids"])),
    )
    np.testing.assert_array_equal(
        merged["loss_weights"].view(np.uint32),
        np.concatenate((local["loss_weights"], remote["loss_weights"])).view(np.uint32),
    )
    assert broadcast_index == 5
