from __future__ import annotations

import ast
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from causal_trainer.kernels.efficient_attention import (
    AttentionKernelProfile,
    AttentionMetadata,
    attention_partition_specs,
    default_attention_scale,
    efficient_attention,
    prepare_attention_metadata,
    resolve_attention_profile,
    validate_attention_profile,
)
from causal_trainer.modeling.architecture import decoder_forward, forward
from causal_trainer.modeling.attention import _attention_precanonicalized, attention


def _mesh_shape(*, dp: int = 1, fsdp: int = 1, tp: int = 1, sp: int = 1) -> dict[str, int]:
    return {"dp": dp, "fsdp": fsdp, "ep": 1, "tp": tp, "sp": sp}


def _single_device_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1, 1, 1, 1)
    return Mesh(devices, ("dp", "fsdp", "ep", "tp", "sp"))


def test_efficient_is_the_model_api_default() -> None:
    for function in (
        _attention_precanonicalized,
        attention,
        decoder_forward,
        forward,
    ):
        assert inspect.signature(function).parameters["implementation"].default == "efficient"


@pytest.mark.parametrize(
    ("tp", "expected_block_h", "expected_block_q"),
    [
        (4, 8, 256),
        (8, 4, 512),
        (16, 2, 1024),
    ],
)
def test_profiles_derive_query_tiles_from_local_head_count(
    tp: int,
    expected_block_h: int,
    expected_block_q: int,
) -> None:
    profile = resolve_attention_profile(
        (1, 4096, 32, 128),
        (1, 4096, 1, 128),
        _mesh_shape(tp=tp),
    )

    assert profile.block_h == expected_block_h
    assert profile.block_q == expected_block_q
    assert profile.block_h * profile.block_q == 2048
    assert profile.block_kv_outer == 512
    assert profile.block_kv_compute == 256


def test_synthetic_head_count_uses_the_same_profile_formula() -> None:
    profile = resolve_attention_profile(
        (2, 2048, 16, 128),
        (2, 2048, 1, 128),
        _mesh_shape(tp=4),
    )

    assert profile == AttentionKernelProfile(
        block_h=4,
        block_q=512,
        block_kv_outer=512,
        block_kv_compute=256,
    )


def test_profile_resolver_uses_aligned_divisors_below_preferred_tiles() -> None:
    profile = resolve_attention_profile(
        (1, 384, 32, 128),
        (1, 384, 1, 128),
        _mesh_shape(tp=4),
    )

    assert profile == AttentionKernelProfile(
        block_h=8,
        block_q=128,
        block_kv_outer=384,
        block_kv_compute=128,
    )


def test_profile_resolver_rejects_sp_local_length_without_an_aligned_divisor() -> None:
    with pytest.raises(ValueError, match="local_query=192"):
        resolve_attention_profile(
            (1, 384, 32, 128),
            (1, 384, 1, 128),
            _mesh_shape(tp=4, sp=2),
        )


@pytest.mark.parametrize(
    ("sequence", "expected_outer", "expected_compute"),
    [
        (16_384, 512, 256),
        (32_768, 1024, 256),
        (262_144, 1024, 256),
        (524_288, 2048, 512),
    ],
)
def test_kv_profile_tiers_follow_global_sequence_length(
    sequence: int,
    expected_outer: int,
    expected_compute: int,
) -> None:
    profile = resolve_attention_profile(
        (1, sequence, 32, 128),
        (1, sequence, 1, 128),
        _mesh_shape(tp=4),
    )

    assert profile.block_kv_outer == expected_outer
    assert profile.block_kv_compute == expected_compute
    assert sequence % profile.block_kv_outer == 0
    assert profile.block_kv_outer % profile.block_kv_compute == 0


def test_profile_resolver_uses_the_sp_local_query_length() -> None:
    profile = resolve_attention_profile(
        (1, 1024, 32, 128),
        (1, 1024, 1, 128),
        _mesh_shape(tp=4, sp=8),
    )

    assert profile.block_h == 8
    assert profile.block_q == 128


@pytest.mark.parametrize(
    ("query_shape", "key_value_shape", "mesh_shape"),
    [
        ((1, 4096, 32), (1, 4096, 1, 128), _mesh_shape(tp=4)),
        ((1, 4096, 32, 128), (2, 4096, 1, 128), _mesh_shape(tp=4)),
        ((1, 4096, 32, 128), (1, 2048, 1, 128), _mesh_shape(tp=4)),
        ((1, 4096, 32, 128), (1, 4096, 1, 64), _mesh_shape(tp=4)),
        ((1, 4096, 30, 128), (1, 4096, 4, 128), _mesh_shape(tp=2)),
        ((1, 4096, 32, 128), (1, 4096, 1, 128), _mesh_shape(tp=3)),
        ((1, 4096, 32, 128), (1, 4096, 1, 128), _mesh_shape(tp=4, sp=3)),
    ],
)
def test_profile_resolver_rejects_invalid_shapes_and_meshes(
    query_shape: tuple[int, ...],
    key_value_shape: tuple[int, ...],
    mesh_shape: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        resolve_attention_profile(query_shape, key_value_shape, mesh_shape)


@pytest.mark.parametrize(
    "profile",
    [
        AttentionKernelProfile(block_h=3, block_q=128, block_kv_outer=256, block_kv_compute=128),
        AttentionKernelProfile(block_h=8, block_q=192, block_kv_outer=256, block_kv_compute=128),
        AttentionKernelProfile(block_h=8, block_q=128, block_kv_outer=384, block_kv_compute=128),
        AttentionKernelProfile(block_h=8, block_q=128, block_kv_outer=256, block_kv_compute=96),
        AttentionKernelProfile(block_h=8, block_q=128, block_kv_outer=128, block_kv_compute=256),
    ],
)
def test_profile_validation_rejects_illegal_tiles(profile: AttentionKernelProfile) -> None:
    with pytest.raises(ValueError):
        validate_attention_profile(
            profile,
            (1, 4096, 32, 128),
            (1, 4096, 1, 128),
            _mesh_shape(tp=4),
        )


def test_profile_validation_accepts_the_resolved_profile() -> None:
    query_shape = (1, 4096, 32, 128)
    key_value_shape = (1, 4096, 1, 128)
    mesh_shape = _mesh_shape(tp=4)
    profile = resolve_attention_profile(query_shape, key_value_shape, mesh_shape)

    validate_attention_profile(profile, query_shape, key_value_shape, mesh_shape)


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_default_scale_is_derived_from_head_dimension(head_dim: int) -> None:
    actual = jnp.asarray(default_attention_scale(head_dim), dtype=jnp.bfloat16)
    expected = jnp.asarray(head_dim**-0.5, dtype=jnp.bfloat16)

    assert actual == expected


@pytest.mark.parametrize("head_dim", [0, -128])
def test_default_scale_rejects_non_positive_head_dimension(head_dim: int) -> None:
    with pytest.raises(ValueError):
        default_attention_scale(head_dim)


def test_public_kernel_rejects_unsupported_head_dimension_before_tpu_lowering() -> None:
    mesh = _single_device_mesh()
    query = jnp.zeros((1, 256, 32, 64), dtype=jnp.bfloat16)
    key = jnp.zeros((1, 256, 1, 64), dtype=jnp.bfloat16)
    profile = resolve_attention_profile(query.shape, key.shape, mesh)
    metadata = prepare_attention_metadata(
        jnp.ones((1, 256), dtype=jnp.int32),
        mesh=mesh,
        profile=profile,
    )

    with pytest.raises(ValueError, match="requires head_dim=128"):
        efficient_attention(
            query,
            key,
            key,
            metadata,
            mesh=mesh,
            profile=profile,
        )


def test_public_kernel_rejects_noncausal_mode_before_tpu_lowering() -> None:
    mesh = _single_device_mesh()
    query = jnp.zeros((1, 256, 32, 128), dtype=jnp.bfloat16)
    key = jnp.zeros((1, 256, 1, 128), dtype=jnp.bfloat16)
    profile = resolve_attention_profile(query.shape, key.shape, mesh)
    metadata = prepare_attention_metadata(
        jnp.ones((1, 256), dtype=jnp.int32),
        mesh=mesh,
        profile=profile,
    )

    with pytest.raises(ValueError, match="implements causal attention only"):
        efficient_attention(
            query,
            key,
            key,
            metadata,
            mesh=mesh,
            profile=profile,
            causal=False,
        )


def test_metadata_summarizes_left_padding_and_unaligned_segment_boundaries() -> None:
    # The first segment begins after a complete padding block. Segment changes
    # at 179 and 301 are unaligned with both metadata tiles.
    segment_ids = jnp.concatenate(
        (
            jnp.zeros(130, dtype=jnp.int32),
            jnp.ones(49, dtype=jnp.int32),
            jnp.full(122, 2, dtype=jnp.int32),
            jnp.full(211, 3, dtype=jnp.int32),
        )
    )[None, :]
    profile = AttentionKernelProfile(
        block_h=1,
        block_q=128,
        block_kv_outer=256,
        block_kv_compute=128,
    )

    metadata = prepare_attention_metadata(
        segment_ids,
        mesh=_single_device_mesh(),
        profile=profile,
    )

    assert isinstance(metadata, AttentionMetadata)
    np.testing.assert_array_equal(np.asarray(metadata.query_segment_ids), np.asarray(segment_ids))
    np.testing.assert_array_equal(np.asarray(metadata.kv_segment_ids), np.asarray(segment_ids))
    np.testing.assert_array_equal(
        np.asarray(metadata.q_block_segments),
        np.asarray([[[0, 0], [1, 2], [2, 3], [3, 3]]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.kv_block_segments),
        np.asarray([[[1, 2], [2, 3]]], dtype=np.int32),
    )


def test_metadata_uses_zero_summary_for_fully_padded_blocks() -> None:
    segment_ids = jnp.zeros((2, 512), dtype=jnp.int32)
    profile = AttentionKernelProfile(
        block_h=1,
        block_q=128,
        block_kv_outer=256,
        block_kv_compute=128,
    )

    metadata = prepare_attention_metadata(
        segment_ids,
        mesh=_single_device_mesh(),
        profile=profile,
    )

    assert metadata.q_block_segments.shape == (2, 4, 2)
    assert metadata.kv_block_segments.shape == (2, 2, 2)
    assert metadata.q_block_segments.dtype == jnp.int32
    assert metadata.kv_block_segments.dtype == jnp.int32
    assert not bool(jnp.any(metadata.q_block_segments))
    assert not bool(jnp.any(metadata.kv_block_segments))


def test_metadata_rejects_invalid_rank_and_profile_divisibility() -> None:
    mesh = _single_device_mesh()
    profile = AttentionKernelProfile(
        block_h=1,
        block_q=128,
        block_kv_outer=256,
        block_kv_compute=128,
    )
    with pytest.raises(ValueError):
        prepare_attention_metadata(jnp.zeros((512,), dtype=jnp.int32), mesh=mesh, profile=profile)

    with pytest.raises(ValueError):
        prepare_attention_metadata(
            jnp.zeros((1, 384), dtype=jnp.int32),
            mesh=mesh,
            profile=profile,
        )


def test_partition_specs_keep_query_sharded_and_kv_replicated() -> None:
    class FakeMesh:
        axis_names = ("dp", "fsdp", "ep", "tp", "sp")
        shape: ClassVar = {"dp": 2, "fsdp": 2, "ep": 1, "tp": 4, "sp": 2}

    specs = attention_partition_specs(FakeMesh())
    values = specs.values() if isinstance(specs, Mapping) else specs

    expected = {
        PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
        PartitionSpec(("dp", "fsdp"), "sp", "tp"),
        PartitionSpec(("dp", "fsdp"), None, None, None),
        PartitionSpec(("dp", "fsdp"), "sp"),
        PartitionSpec(("dp", "fsdp"), None),
        PartitionSpec(("dp", "fsdp"), "sp", None),
        PartitionSpec(("dp", "fsdp"), None, None),
    }
    assert set(values) == expected


def test_efficient_kernel_has_no_splash_import_dependency() -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "causal_trainer"
        / "kernels"
        / "efficient_attention.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imported_modules = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or ""
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
    )

    assert not any("splash" in module_name.lower() for module_name in imported_modules)


def test_decoder_prepares_attention_metadata_once_outside_the_layer_body() -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "causal_trainer"
        / "modeling"
        / "architecture.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def preparation_calls(function_name: str) -> int:
        return sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "prepare_attention_metadata"
            for node in ast.walk(functions[function_name])
        )

    assert preparation_calls("decoder_forward") == 1
    assert preparation_calls("_decoder_layer") == 0


def test_efficient_backward_owns_only_the_tp_sp_dkv_reductions() -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "causal_trainer"
        / "kernels"
        / "efficient_attention.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    backward = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_distributed_backward"
    )
    psum_calls = [
        node
        for node in ast.walk(backward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "lax"
        and node.func.attr == "psum"
    ]
    reduction_axis_literals = {
        tuple(element.value for element in node.elts)
        for node in ast.walk(backward)
        if isinstance(node, ast.Tuple)
        and node.elts
        and all(isinstance(element, ast.Constant) for element in node.elts)
        and all(isinstance(element.value, str) for element in node.elts)
    }

    assert len(psum_calls) == 2  # One each for dK and dV; no outer transpose reduction.
    assert ("tp", "sp") in reduction_axis_literals
    assert not any("dp" in axes for axes in reduction_axis_literals)
