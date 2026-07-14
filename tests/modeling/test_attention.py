from __future__ import annotations

from typing import ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import pytest

import causal_trainer.modeling.attention as attention_module
from causal_trainer.modeling.attention import (
    attention,
    make_causal_segment_mask,
    normalize_segment_ids,
    splash_partition_specs,
)
from causal_trainer.modeling.config import validate_splash_block_size


def test_packed_vanilla_attention_isolates_segments_in_both_directions() -> None:
    key = jax.random.PRNGKey(0)
    query_key, key_key, value_key = jax.random.split(key, 3)
    query = jax.random.normal(query_key, (1, 6, 4, 2))
    keys = jax.random.normal(key_key, (1, 6, 1, 2))
    values = jax.random.normal(value_key, (1, 6, 1, 2))
    valid = jnp.ones((1, 6), dtype=jnp.bool_)
    segments = jnp.asarray([[1, 1, 1, 2, 2, 2]], dtype=jnp.int32)

    baseline = attention(
        query,
        keys,
        values,
        valid,
        segments,
        implementation="vanilla",
    )

    # A later causal segment cannot affect an earlier one.
    later_changed = attention(
        query.at[:, 3:].set(100.0),
        keys.at[:, 3:].set(-100.0),
        values.at[:, 3:].set(50.0),
        valid,
        segments,
        implementation="vanilla",
    )
    assert jnp.allclose(later_changed[:, :3], baseline[:, :3])

    # This is the direction that catches cross-sample packed leakage: without
    # the segment mask, the later sample could attend to the earlier sample.
    earlier_changed = attention(
        query.at[:, :3].set(-70.0),
        keys.at[:, :3].set(80.0),
        values.at[:, :3].set(-90.0),
        valid,
        segments,
        implementation="vanilla",
    )
    assert jnp.allclose(earlier_changed[:, 3:], baseline[:, 3:])


def test_segment_ids_are_canonicalized_by_contiguous_run() -> None:
    valid = jnp.asarray([[0, 1, 1, 1, 1, 1, 1]], dtype=jnp.bool_)
    # Valid zero labels and a non-contiguous reuse of zero must not collapse
    # distinct packed samples into one Splash/vanilla segment.
    labels = jnp.asarray([[0, 0, 0, 8, 8, 0, 0]], dtype=jnp.int32)
    normalized = normalize_segment_ids(valid, labels)
    assert jnp.array_equal(normalized, jnp.asarray([[0, 1, 1, 2, 2, 3, 3]], jnp.int32))

    allowed = make_causal_segment_mask(valid, labels)[0, 0]
    assert not bool(allowed[5, 1])
    assert not bool(allowed[5, 3])
    assert bool(allowed[5, 5])
    assert bool(allowed[6, 5])
    assert not bool(jnp.any(allowed[0]))


def test_precanonicalized_attention_path_reuses_ids_without_run_detection(monkeypatch) -> None:
    query = jnp.arange(1 * 5 * 2 * 2, dtype=jnp.float32).reshape(1, 5, 2, 2) / 10.0
    key = jnp.arange(1 * 5 * 1 * 2, dtype=jnp.float32).reshape(1, 5, 1, 2) / 20.0
    value = key + 1.0
    valid = jnp.asarray([[1, 1, 0, 1, 1]], dtype=jnp.bool_)
    raw_segments = jnp.asarray([[0, 0, 0, 0, 0]], dtype=jnp.int32)
    canonical = normalize_segment_ids(valid, raw_segments)

    expected = attention(
        query,
        key,
        value,
        valid,
        raw_segments,
        implementation="vanilla",
    )

    def unexpected_normalization(*_args, **_kwargs):
        raise AssertionError("precanonicalized decoder path must not normalize again")

    monkeypatch.setattr(attention_module, "normalize_segment_ids", unexpected_normalization)
    actual = attention_module._attention_precanonicalized(
        query,
        key,
        value,
        valid,
        canonical,
        implementation="vanilla",
    )
    assert jnp.allclose(actual, expected)


def test_splash_block_sizes_match_jax_tpu_lane_and_divisibility_constraints() -> None:
    assert validate_splash_block_size(4096, 128, "block_q") == 128
    assert validate_splash_block_size(4096, 512, "block_q") == 512
    assert validate_splash_block_size(4096, 8192, "block_q") == 4096
    with pytest.raises(ValueError, match="multiple of 128"):
        validate_splash_block_size(4096, 64, "block_q")
    with pytest.raises(ValueError, match="must be divisible"):
        validate_splash_block_size(4096, 384, "block_q")
    with pytest.raises(ValueError, match="must be divisible"):
        validate_splash_block_size(4100, 128, "block_q")


def test_splash_shards_query_heads_but_replicates_single_kv_head() -> None:
    class FakeMesh:
        axis_names = ("dp", "fsdp", "ep", "tp", "sp")
        shape: ClassVar = {"dp": 2, "fsdp": 2, "ep": 1, "tp": 4, "sp": 1}

    query, key_value, query_segments, key_value_segments = splash_partition_specs(FakeMesh())
    assert query == jax.sharding.PartitionSpec(("dp", "fsdp"), None, "tp", None)
    assert key_value == jax.sharding.PartitionSpec(("dp", "fsdp"), None, None, None)
    assert query_segments == jax.sharding.PartitionSpec(("dp", "fsdp"), None)
    assert key_value_segments == jax.sharding.PartitionSpec(("dp", "fsdp"), None)


def test_splash_sequence_parallel_shards_q_and_replicates_complete_kv() -> None:
    class FakeMesh:
        axis_names = ("dp", "fsdp", "ep", "tp", "sp")
        shape: ClassVar = {"dp": 2, "fsdp": 1, "ep": 1, "tp": 4, "sp": 2}

    query, key_value, query_segments, key_value_segments = splash_partition_specs(FakeMesh())
    assert query == jax.sharding.PartitionSpec("dp", "sp", "tp", None)
    assert key_value == jax.sharding.PartitionSpec("dp", None, None, None)
    assert query_segments == jax.sharding.PartitionSpec("dp", "sp")
    assert key_value_segments == jax.sharding.PartitionSpec("dp", None)


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_splash_passes_segments_scales_q_and_preserves_gqa_layout(monkeypatch, kv_heads: int) -> None:
    captured = {}

    class FakeBlockSizes:
        def __init__(self, **values):
            self.__dict__.update(values)

    class FakeCausalMask:
        def __init__(self, shape):
            self.shape = shape

    class FakeMultiHeadMask:
        def __init__(self, masks):
            self.masks = tuple(masks)

    class FakeSegmentIds(NamedTuple):
        q: jax.Array
        kv: jax.Array

    def fake_factory(*, mask, block_sizes):
        captured["mask"] = mask
        captured["block_sizes"] = block_sizes

        def fake_kernel(q, k, v, segment_ids):
            del v
            # q: [query heads in this KV group, sequence, dim]
            # k: [sequence, dim]. Encoding all inputs in the result verifies
            # both nested vmaps and the SegmentIds pytree axes.
            segment_term = segment_ids.q[None, :, None].astype(q.dtype)
            return q + k[None, :, :] + segment_term

        return fake_kernel

    monkeypatch.setattr(attention_module, "BlockSizes", FakeBlockSizes)
    monkeypatch.setattr(attention_module, "CausalMask", FakeCausalMask)
    monkeypatch.setattr(attention_module, "MultiHeadMask", FakeMultiHeadMask)
    monkeypatch.setattr(attention_module, "SegmentIds", FakeSegmentIds)
    monkeypatch.setattr(attention_module, "make_splash_attention_single_device", fake_factory)
    monkeypatch.setattr(attention_module.jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(attention_module.jax, "device_count", lambda: 1)

    batch, sequence, query_heads, dim = 2, 128, 4, 128
    query_head_values = jnp.arange(query_heads, dtype=jnp.float32)
    query = jnp.broadcast_to(
        query_head_values[None, None, :, None],
        (batch, sequence, query_heads, dim),
    )
    key_head_values = jnp.arange(batch * kv_heads, dtype=jnp.float32).reshape(batch, kv_heads) + 10.0
    key = jnp.broadcast_to(
        key_head_values[:, None, :, None],
        (batch, sequence, kv_heads, dim),
    )
    value = jnp.zeros_like(key)
    # A valid zero-labelled first sample must remain distinct from the second.
    raw_segments = jnp.broadcast_to(
        jnp.concatenate((jnp.zeros(64, jnp.int32), jnp.full(64, 7, jnp.int32)))[None, :],
        (batch, sequence),
    )
    valid = jnp.ones((batch, sequence), dtype=jnp.bool_)

    output = attention(
        query,
        key,
        value,
        valid,
        raw_segments,
        implementation="splash",
        scale=0.25,
        block_q=128,
        block_k=128,
    )

    canonical_segments = jnp.broadcast_to(
        jnp.concatenate((jnp.ones(64, jnp.int32), jnp.full(64, 2, jnp.int32)))[None, :],
        (batch, sequence),
    )
    repeated_key = jnp.repeat(key, query_heads // kv_heads, axis=2)
    expected = query * 0.25 + repeated_key + canonical_segments[:, :, None, None]
    assert output.shape == (batch, sequence, query_heads, dim)
    assert jnp.array_equal(output, expected)

    mask = captured["mask"]
    assert len(mask.masks) == query_heads // kv_heads
    assert all(item.shape == (sequence, sequence) for item in mask.masks)
    blocks = captured["block_sizes"]
    assert blocks.block_q == blocks.block_q_dkv == blocks.block_q_dq == 128
    assert blocks.block_kv == blocks.block_kv_compute == 128
    assert blocks.block_kv_dkv == blocks.block_kv_dkv_compute == blocks.block_kv_dq == 128


@pytest.mark.parametrize(
    ("sequence", "requested_k", "expected_outer", "expected_compute"),
    [
        (1024, 1024, 1024, 1024),
        (1024, 2048, 1024, 1024),
        (1152, 1152, 1152, 128),
        (1536, 1536, 1536, 512),
        (1920, 1920, 1920, 128),
        (2048, 2048, 2048, 512),
        (4096, 4096, 4096, 512),
    ],
)
def test_splash_large_kv_outer_block_selects_a_divisible_compute_tile(
    sequence: int,
    requested_k: int,
    expected_outer: int,
    expected_compute: int,
) -> None:
    blocks = attention_module._splash_block_sizes(sequence, sequence, 128, requested_k)

    assert blocks.block_q == blocks.block_q_dkv == blocks.block_q_dq == 128
    assert blocks.block_kv == blocks.block_kv_dkv == blocks.block_kv_dq == expected_outer
    assert blocks.block_kv_compute == blocks.block_kv_dkv_compute == expected_compute
    assert expected_outer % expected_compute == 0
