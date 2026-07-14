import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from safetensors.numpy import load_file, save_file

import causal_trainer.checkpointing.huggingface as hf_io
from causal_trainer.checkpointing.huggingface import (
    SafeTensorIndex,
    export_hf_checkpoint,
    export_optimizer_checkpoint,
    load_optimizer_checkpoint,
    load_sharded_parameters,
    parameter_hf_layout,
    parameter_to_hf_mapping,
)
from causal_trainer.distributed.runtime import MESH_AXIS_NAMES, parameter_partition_specs, path_to_string
from causal_trainer.modeling.architecture import parameter_shapes
from causal_trainer.modeling.config import ModelConfig
from causal_trainer.training.optimizer import build_optimizer


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        partial_rotary_factor=0.5,
        max_position_embeddings=16,
        pad_token_id=0,
        eos_token_id=2,
    )


def test_safetensors_load_and_export_round_trip(tmp_path) -> None:
    config = _config()
    template = parameter_shapes(config, jnp.float32)
    layout = parameter_hf_layout(template)
    assert layout["model.layers.0.self_attn.k_proj.weight"] == ("F32", (4, 8))
    assert layout["model.layers.0.self_attn.q_proj.bias"] == ("F32", (8,))
    assert "model.layers.0.self_attn.o_proj.bias" not in layout
    assert layout["model.layers.0.mlp.gate_proj.weight"] == ("F32", (12, 8))
    assert layout["model.layers.0.mlp.up_proj.weight"] == ("F32", (12, 8))
    tensors = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(template)[0]:
        mapping = parameter_to_hf_mapping(path_to_string(path))
        target = np.arange(np.prod(leaf.shape), dtype=np.float32).reshape(leaf.shape)
        # Safetensors' NumPy writer expects C-contiguous inputs and does not
        # serialize a transposed view's strides.
        tensors[mapping.hf_key] = np.ascontiguousarray(target.T) if mapping.transpose else target
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")

    devices = np.asarray(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1, 1))
    mesh = Mesh(devices, MESH_AXIS_NAMES)
    params = load_sharded_parameters(
        tmp_path,
        template,
        mesh,
        specs=parameter_partition_specs(template),
    )
    expected_q = tensors["model.layers.0.self_attn.q_proj.weight"].T
    np.testing.assert_array_equal(
        np.asarray(params["layers"][0]["attention"]["q_proj"]["kernel"]),
        expected_q,
    )

    output = tmp_path / "export"
    checkpoint_id = "c" * 32
    export_hf_checkpoint(
        params,
        output,
        source=tmp_path,
        config=config,
        checkpoint_id=checkpoint_id,
    )
    assert SafeTensorIndex(output, expected_checkpoint_id=checkpoint_id).info(
        "model.embed_tokens.weight"
    ).shape == (16, 8)
    exported = load_file(output / "model.safetensors")
    np.testing.assert_array_equal(
        exported["model.layers.0.self_attn.q_proj.weight"],
        tensors["model.layers.0.self_attn.q_proj.weight"],
    )
    np.testing.assert_array_equal(
        exported["model.layers.0.mlp.gate_proj.weight"],
        tensors["model.layers.0.mlp.gate_proj.weight"],
    )
    np.testing.assert_array_equal(
        exported["model.layers.0.mlp.up_proj.weight"],
        tensors["model.layers.0.mlp.up_proj.weight"],
    )


def test_hf_asset_copy_preserves_named_chat_templates(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    template_directory = source / "chat_templates"
    template_directory.mkdir()
    (template_directory / "tool_use.jinja").write_text("tool template", encoding="utf-8")

    hf_io._copy_hf_assets(source, destination, None, write_config=False)

    assert (destination / "tokenizer.json").is_file()
    assert (
        destination / "chat_templates" / "tool_use.jinja"
    ).read_text(encoding="utf-8") == "tool template"


def test_hf_asset_copy_normalizes_v5_config_to_fixed_source_format(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    source_config = {
        "dtype": "bfloat16",
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 1_000_000_000.0,
            "partial_rotary_factor": 0.5,
        },
    }
    (source / "config.json").write_text(json.dumps(source_config), encoding="utf-8")

    hf_io._copy_hf_assets(source, destination, _config())

    written = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    assert "dtype" not in written
    assert "rope_parameters" not in written
    assert written["torch_dtype"] == "bfloat16"
    assert written["rope_scaling"] is None
    assert written["rope_theta"] == _config().rope_theta
    assert written["partial_rotary_factor"] == _config().partial_rotary_factor


def test_model_load_rejects_an_output_projection_bias(tmp_path) -> None:
    config = _config()
    template = parameter_shapes(config, jnp.float32)
    tensors = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(template)[0]:
        mapping = parameter_to_hf_mapping(path_to_string(path))
        value = np.zeros(leaf.shape, dtype=np.float32)
        tensors[mapping.hf_key] = np.ascontiguousarray(value.T) if mapping.transpose else value
    tensors["model.layers.0.self_attn.o_proj.bias"] = np.zeros(config.hidden_size, dtype=np.float32)
    save_file(tensors, tmp_path / "model.safetensors")

    devices = np.asarray(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1, 1))
    mesh = Mesh(devices, MESH_AXIS_NAMES)
    with pytest.raises(ValueError, match="do not exactly match"):
        load_sharded_parameters(
            tmp_path,
            template,
            mesh,
            specs=parameter_partition_specs(template),
        )


def test_model_index_accepts_snapshot_style_safetensors_symlink(tmp_path) -> None:
    blob_dir = tmp_path / "blobs"
    snapshot_dir = tmp_path / "snapshot"
    blob_dir.mkdir()
    snapshot_dir.mkdir()
    blob = blob_dir / "weights.safetensors"
    save_file({"weight": np.asarray([1], dtype=np.int32)}, blob)
    link = snapshot_dir / "model.safetensors"
    try:
        link.symlink_to(blob)
    except OSError:
        pytest.skip("symbolic links are unavailable on this host")

    index = SafeTensorIndex(snapshot_dir)
    assert index.info("weight").shape == (1,)


def test_checkpoint_gather_propagates_owner_read_error_before_data_allgather(
    monkeypatch,
) -> None:
    from jax.experimental import multihost_utils

    class FakeDevice:
        def __init__(self, process_index: int) -> None:
            self.process_index = process_index

    class FakeShard:
        def __init__(self, start: int, process_index: int) -> None:
            self.index = (slice(start, start + 1),)
            self.device = FakeDevice(process_index)
            self.data = np.ones((1,), dtype=np.float32)

    shards = [FakeShard(0, 0), FakeShard(1, 1)]

    class FakeSharding:
        device_set = tuple(shard.device for shard in shards)

    class FakeArray:
        shape = (2,)
        ndim = 1
        dtype = np.dtype(np.float32)
        sharding = FakeSharding()
        global_shards = shards
        addressable_shards = [shards[1]]

    broadcasts: list[dict[str, object]] = []

    def process_allgather(payload, *, tiled):
        if np.asarray(payload).shape == (32,):
            return np.stack([payload, payload])
        rows = [np.zeros_like(payload), np.asarray(payload)]
        return np.stack(rows)

    def broadcast_one_to_all(value, **kwargs):
        broadcasts.append(kwargs)
        return value

    def fail_device_get(value):
        raise RuntimeError("simulated read failure")

    monkeypatch.setattr(hf_io.jax, "Array", FakeArray)
    monkeypatch.setattr(hf_io.jax, "process_count", lambda: 2)
    monkeypatch.setattr(hf_io.jax, "process_index", lambda: 1)
    monkeypatch.setattr(hf_io.jax, "device_get", fail_device_get)
    monkeypatch.setattr(multihost_utils, "process_allgather", process_allgather)
    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", broadcast_one_to_all)

    with pytest.raises(RuntimeError, match="reading checkpoint shard chunk failed"):
        hf_io._gather_to_process_zero(FakeArray())

    assert broadcasts
    assert all("is_source" not in kwargs for kwargs in broadcasts)


def test_checkpoint_gather_nonprimary_returns_none_and_holds_only_bounded_chunks(
    monkeypatch,
) -> None:
    from jax.experimental import multihost_utils

    class FakeDevice:
        def __init__(self, process_index: int) -> None:
            self.process_index = process_index

    class FakeShard:
        def __init__(self, start: int, process_index: int, values) -> None:
            self.index = (slice(start, start + 2),)
            self.device = FakeDevice(process_index)
            self.data = np.asarray(values, dtype=np.float32)

    shards = [FakeShard(0, 0, [10, 11]), FakeShard(2, 1, [20, 21])]

    class FakeSharding:
        device_set = tuple(shard.device for shard in shards)

    class FakeArray:
        shape = (4,)
        ndim = 1
        dtype = np.dtype(np.float32)
        sharding = FakeSharding()
        global_shards = shards
        addressable_shards = [shards[1]]

    data_allgathers: list[np.ndarray] = []
    device_reads: list[np.ndarray] = []

    def process_allgather(payload, *, tiled):
        assert tiled is False
        if np.asarray(payload).shape == (32,):
            return np.stack([payload, payload])
        if np.asarray(payload).shape != (4096,):
            gathered = np.stack([np.zeros_like(payload), np.asarray(payload)])
            data_allgathers.append(gathered.copy())
            return gathered
        return np.stack([np.zeros_like(payload), np.asarray(payload)])

    def broadcast_one_to_all(value, **kwargs):
        assert "is_source" not in kwargs
        return np.asarray(value)

    def device_get(value):
        value = np.asarray(value)
        device_reads.append(value.copy())
        return value

    monkeypatch.setattr(hf_io.jax, "Array", FakeArray)
    monkeypatch.setattr(hf_io.jax, "process_count", lambda: 2)
    monkeypatch.setattr(hf_io.jax, "process_index", lambda: 1)
    monkeypatch.setattr(hf_io.jax, "device_get", device_get)
    monkeypatch.setattr(multihost_utils, "process_allgather", process_allgather)
    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", broadcast_one_to_all)

    result = hf_io._gather_to_process_zero(FakeArray(), max_chunk_bytes=4)

    assert result is None
    assert [value.tolist() for value in device_reads] == [[20.0], [21.0]]
    assert [value[1].tolist() for value in data_allgathers] == [[20.0], [21.0]]
    assert all(value.nbytes <= 8 for value in data_allgathers)


def test_nonprimary_model_export_transforms_and_participates_in_every_leaf_without_writing(
    monkeypatch,
    tmp_path,
) -> None:
    from jax.experimental import multihost_utils

    params = {
        "lm_head": {"kernel": jnp.arange(6, dtype=jnp.float32).reshape(2, 3)},
        "norm": {"scale": jnp.ones((2,), dtype=jnp.float32)},
    }
    gathered: list[tuple[tuple[int, ...], bool]] = []
    transformed: list[str] = []
    plans: list[list[dict[str, object]]] = []
    barriers: list[str] = []

    def gather(leaf, *, transpose=False, **kwargs):
        gathered.append((tuple(leaf.shape), transpose))
        return None

    def transform(path, leaf):
        transformed.append(path)
        return leaf

    monkeypatch.setattr(hf_io.jax, "process_count", lambda: 2)
    monkeypatch.setattr(hf_io.jax, "process_index", lambda: 1)
    monkeypatch.setattr(hf_io, "_collective_operation_id", lambda: "shared-operation")
    monkeypatch.setattr(hf_io, "_raise_if_process_zero_error", lambda stage, error: None)
    monkeypatch.setattr(hf_io, "_raise_if_any_process_error", lambda stage, error: None)
    monkeypatch.setattr(hf_io, "_assert_same_export_plan", lambda kind, plan: plans.append(list(plan)))
    monkeypatch.setattr(hf_io, "_gather_to_process_zero", gather)
    monkeypatch.setattr(hf_io, "sync_processes", barriers.append)
    monkeypatch.setattr(
        multihost_utils,
        "broadcast_one_to_all",
        lambda value: np.asarray(False, dtype=np.bool_),
    )

    destination = tmp_path / "rank-zero-only"
    export_hf_checkpoint(
        params,
        destination,
        config={"model_type": "test"},
        leaf_transform=transform,
        transform_plan={"kind": "identity"},
    )

    assert transformed == ["lm_head/kernel", "norm/scale"]
    assert gathered == [((2, 3), True), ((2,), False)]
    assert len(plans) == 2
    assert plans[0] == [
        {
            "filename": "model.safetensors",
            "leaf_transform": True,
            "overwrite": True,
            "transform_plan": {"kind": "identity"},
        }
    ]
    leaves = plans[1][0]["leaves"]
    assert isinstance(leaves, list) and len(leaves) == 2
    assert plans[1][0]["leaf_transform"] is True
    assert plans[1][0]["transform_plan"] == {"kind": "identity"}
    assert barriers == [
        "model-export-create-shared-operation",
        "model-export-finished-shared-operation",
        "model-assets-finished-shared-operation",
    ]
    assert not destination.exists()


def test_model_export_applies_leaf_transform_immediately_before_each_gather(
    monkeypatch,
    tmp_path,
) -> None:
    params = {
        "first": jnp.asarray([[1.0, 2.0]], dtype=jnp.float32),
        "second": jnp.asarray([[3.0, 4.0]], dtype=jnp.float32),
    }
    events: list[tuple[str, str]] = []
    gathered_values: list[np.ndarray] = []
    plans: list[list[dict[str, object]]] = []

    def mapping(path: str) -> hf_io.TensorMapping:
        return hf_io.TensorMapping(path, f"custom.{path}")

    def transform(path, leaf):
        events.append(("transform", path))
        return leaf + jnp.asarray(10.0, dtype=leaf.dtype)

    def gather(leaf, **kwargs):
        del kwargs
        events.append(("gather", ""))
        value = np.asarray(jax.device_get(leaf))
        gathered_values.append(value)
        return value

    monkeypatch.setattr(hf_io, "_assert_same_export_plan", lambda kind, plan: plans.append(list(plan)))
    monkeypatch.setattr(hf_io, "_gather_to_process_zero", gather)

    export_hf_checkpoint(
        params,
        tmp_path / "transformed",
        config={"model_type": "test"},
        mapping_fn=mapping,
        leaf_transform=transform,
        transform_plan={"kind": "add", "value": 10},
    )

    assert events == [
        ("transform", "first"),
        ("gather", ""),
        ("transform", "second"),
        ("gather", ""),
    ]
    np.testing.assert_array_equal(gathered_values[0], np.asarray([[11.0, 12.0]], dtype=np.float32))
    np.testing.assert_array_equal(gathered_values[1], np.asarray([[13.0, 14.0]], dtype=np.float32))
    assert plans[0][0]["transform_plan"] == {"kind": "add", "value": 10}
    assert plans[1][0]["transform_plan"] == {"kind": "add", "value": 10}


@pytest.mark.parametrize(
    ("leaf_transform", "message"),
    [
        (lambda path, leaf: np.asarray(leaf), "must return a concrete jax.Array"),
        (lambda path, leaf: leaf[:1], "returned shape"),
        (lambda path, leaf: leaf.astype(jnp.float16), "returned dtype"),
    ],
)
def test_model_export_collectively_rejects_invalid_leaf_transform(
    tmp_path,
    leaf_transform,
    message,
) -> None:
    params = {"weight": jnp.ones((2, 2), dtype=jnp.float32)}

    with pytest.raises(RuntimeError, match=message):
        export_hf_checkpoint(
            params,
            tmp_path / "invalid-transform",
            config={"model_type": "test"},
            mapping_fn=lambda path: hf_io.TensorMapping(path, f"custom.{path}"),
            leaf_transform=leaf_transform,
            transform_plan={"kind": "invalid-test"},
        )


def test_export_plan_mismatch_is_reported_collectively(monkeypatch) -> None:
    from jax.experimental import multihost_utils

    def mismatched_allgather(payload, *, tiled):
        assert tiled is False
        if np.asarray(payload).shape == (4096,):
            return np.stack([payload, payload])
        other = np.asarray(payload).copy()
        other[0] ^= np.uint8(0xFF)
        return np.stack([payload, other])

    monkeypatch.setattr(hf_io.jax, "process_count", lambda: 2)
    monkeypatch.setattr(multihost_utils, "process_allgather", mismatched_allgather)

    with pytest.raises(RuntimeError, match="export plans differ across processes"):
        hf_io._assert_same_export_plan(
            "model checkpoint",
            [{"hf_key": "weight", "shape": [2, 2], "dtype": "bfloat16"}],
        )


def test_model_export_uses_explicit_mapping_and_writes_one_file(tmp_path) -> None:
    params = {
        name: (jnp.arange(2, dtype=jnp.float32) + offset).reshape(1, 2)
        for offset, name in enumerate(("first", "second", "third"))
    }

    def mapping(path: str) -> hf_io.TensorMapping:
        return hf_io.TensorMapping(path, f"custom.{path}", transpose=path == "first")

    destination = export_hf_checkpoint(
        params,
        tmp_path / "export",
        config={"model_type": "test", "torch_dtype": "bfloat16"},
        mapping_fn=mapping,
    )

    exported = load_file(destination / "model.safetensors")
    assert set(exported) == {"custom.first", "custom.second", "custom.third"}
    assert exported["custom.first"].shape == (2, 1)
    assert exported["custom.second"].shape == (1, 2)
    assert exported["custom.third"].shape == (1, 2)
    assert not list(destination.glob("model-*-of-*.safetensors"))
    assert not (destination / "model.safetensors.index.json").exists()
    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    assert config["torch_dtype"] == "float32"


def test_optimizer_checkpoint_writes_one_file_and_completion_manifest(tmp_path) -> None:
    checkpoint_id = "a" * 32
    params = {"kernel": jnp.arange(8, dtype=jnp.bfloat16).reshape(2, 4)}
    optimizer = build_optimizer(
        params,
        lambda _: jnp.asarray(1e-3, dtype=jnp.float32),
        weight_decay=0.0,
        max_grad_norm=1.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    state = optimizer.init(params)
    _, state = optimizer.update(jax.tree.map(jnp.ones_like, params), state, params)
    destination = export_optimizer_checkpoint(
        state,
        tmp_path / "checkpoint",
        step=1,
        checkpoint_id=checkpoint_id,
    )

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_id"] == checkpoint_id
    assert manifest["format_version"] == 1
    assert manifest["global_step"] == 1
    assert manifest["tensor_count"] == len(jax.tree_util.tree_leaves(state))
    assert any(metadata["dtype"] == "bfloat16" for metadata in manifest["tensors"])
    assert (destination / "state.safetensors").is_file()
    assert not list(destination.glob("state-*-of-*.safetensors"))
    assert not (destination / "state.safetensors.index.json").exists()

    expected_leaves = jax.tree_util.tree_flatten(state)[0]
    for metadata, expected in zip(manifest["tensors"], expected_leaves, strict=True):
        stored = load_file(destination / metadata["file"])[metadata["key"]]
        np.testing.assert_array_equal(stored, np.asarray(expected))
        assert stored.shape == expected.shape
        assert metadata["shape"] == list(expected.shape)
        assert metadata["dtype"] == np.dtype(expected.dtype).name

    devices = np.asarray(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1, 1))
    mesh = Mesh(devices, MESH_AXIS_NAMES)
    replicated = NamedSharding(mesh, PartitionSpec())
    state_template = jax.eval_shape(optimizer.init, params)
    state_shardings = jax.tree.map(lambda _: replicated, state_template)
    with pytest.raises(ValueError, match="step/version mismatch"):
        load_optimizer_checkpoint(
            tmp_path / "checkpoint",
            state_template,
            state_shardings,
            expected_step=2,
            expected_checkpoint_id=checkpoint_id,
        )
    restored = load_optimizer_checkpoint(
        tmp_path / "checkpoint",
        state_template,
        state_shardings,
        expected_step=1,
        expected_checkpoint_id=checkpoint_id,
    )
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(state)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(restored),
        jax.tree_util.tree_leaves(state),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    gradients = jax.tree.map(jnp.ones_like, params)
    expected_updates, expected_next_state = optimizer.update(gradients, state, params)
    resumed_updates, resumed_next_state = optimizer.update(gradients, restored, params)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(resumed_updates),
        jax.tree_util.tree_leaves(expected_updates),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    for actual, expected in zip(
        jax.tree_util.tree_leaves(resumed_next_state),
        jax.tree_util.tree_leaves(expected_next_state),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    manifest["jax_version"] = "incompatible-test-version"
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dependency version mismatch"):
        load_optimizer_checkpoint(
            tmp_path / "checkpoint",
            state_template,
            state_shardings,
            expected_step=1,
            expected_checkpoint_id=checkpoint_id,
        )

    with pytest.raises(ValueError, match="identity mismatch"):
        load_optimizer_checkpoint(
            tmp_path / "checkpoint",
            state_template,
            state_shardings,
            expected_step=1,
            expected_checkpoint_id="b" * 32,
        )

    replacement = {"count": jnp.asarray(8, dtype=jnp.int32)}
    export_optimizer_checkpoint(
        replacement,
        tmp_path / "checkpoint",
        step=8,
        overwrite=True,
    )
    assert (destination / "state.safetensors").is_file()
    assert not list(destination.glob("state-*-of-*.safetensors"))
    assert not (destination / "state.safetensors.index.json").exists()
    replacement_manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert replacement_manifest["global_step"] == 8
    assert replacement_manifest["tensor_count"] == 1
