import jax.numpy as jnp
import pytest

from causal_trainer.modeling.rotary import (
    apply_rotary_cos_sin,
    apply_rotary_embedding,
    canonicalize_segment_ids,
    make_position_ids,
    rotary_cos_sin,
)


def test_rotary_defaults_to_gpt_j_adjacent_pairs_and_leaves_tail_unchanged() -> None:
    query = jnp.asarray([[[[1.0, 2.0, 3.0, 4.0]]]])
    key = query.copy()
    positions = jnp.asarray([[1]])
    rotated_q, rotated_k = apply_rotary_embedding(query, key, positions, jnp.asarray([1.0]))
    expected_pair = jnp.asarray(
        [1.0 * jnp.cos(1.0) - 2.0 * jnp.sin(1.0), 2.0 * jnp.cos(1.0) + 1.0 * jnp.sin(1.0)]
    )
    assert jnp.allclose(rotated_q[0, 0, 0, :2], expected_pair)
    assert jnp.array_equal(rotated_q[..., 2:], query[..., 2:])
    assert jnp.array_equal(rotated_q, rotated_k)


def test_rotary_supports_gpt_neox_half_pairs_and_leaves_tail_unchanged() -> None:
    query = jnp.asarray([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]])
    key = query.copy()
    positions = jnp.asarray([[1]])
    rotated_q, rotated_k = apply_rotary_embedding(
        query,
        key,
        positions,
        jnp.asarray([1.0, 0.5]),
        rope_style="gpt-neox",
    )
    cosine = jnp.cos(jnp.asarray([1.0, 0.5]))
    sine = jnp.sin(jnp.asarray([1.0, 0.5]))
    expected = jnp.concatenate(
        (
            query[0, 0, 0, :2] * cosine - query[0, 0, 0, 2:4] * sine,
            query[0, 0, 0, 2:4] * cosine + query[0, 0, 0, :2] * sine,
        )
    )
    assert jnp.allclose(rotated_q[0, 0, 0, :4], expected)
    assert jnp.array_equal(rotated_q[..., 4:], query[..., 4:])
    assert jnp.array_equal(rotated_q, rotated_k)


def test_rotary_rejects_unknown_style() -> None:
    query = jnp.ones((1, 1, 1, 2))
    with pytest.raises(ValueError, match="rope_style"):
        apply_rotary_embedding(
            query,
            query,
            jnp.asarray([[0]]),
            jnp.asarray([1.0]),
            rope_style="unknown",
        )


def test_positions_reset_for_zero_label_and_reused_segment_runs() -> None:
    valid = jnp.asarray([[0, 1, 1, 1, 1, 1, 1]], dtype=jnp.bool_)
    labels = jnp.asarray([[0, 0, 0, 7, 7, 0, 0]], dtype=jnp.int32)

    canonical = canonicalize_segment_ids(valid, labels)
    positions = make_position_ids(valid, labels)

    assert jnp.array_equal(canonical, jnp.asarray([[0, 1, 1, 2, 2, 3, 3]], jnp.int32))
    assert jnp.array_equal(positions, jnp.asarray([[0, 0, 1, 0, 1, 0, 1]], jnp.int32))


def test_positions_restart_after_an_internal_masked_gap_without_explicit_segments() -> None:
    valid = jnp.asarray([[1, 1, 0, 1, 1]], dtype=jnp.bool_)
    assert jnp.array_equal(make_position_ids(valid), jnp.asarray([[0, 1, 0, 0, 1]], jnp.int32))


def test_precomputed_rotary_uses_actual_reset_and_jump_positions_without_a_cache() -> None:
    positions = jnp.asarray([[0, 1, 0, 2_000_000]], dtype=jnp.int32)
    inv_freq = jnp.asarray([1.0, 0.01], dtype=jnp.float32)
    cosine, sine = rotary_cos_sin(positions, inv_freq, jnp.bfloat16)

    assert cosine.shape == sine.shape == (1, 4, 2)
    assert cosine.dtype == sine.dtype == jnp.dtype(jnp.bfloat16)
    expected_angles = positions.astype(jnp.float32)[..., None] * inv_freq[None, None, :]
    assert jnp.array_equal(cosine, jnp.cos(expected_angles).astype(jnp.bfloat16))
    assert jnp.array_equal(sine, jnp.sin(expected_angles).astype(jnp.bfloat16))

    query = jnp.arange(1 * 4 * 2 * 8, dtype=jnp.float32).reshape(1, 4, 2, 8).astype(jnp.bfloat16)
    key = (query + jnp.asarray(0.5, jnp.bfloat16)).astype(jnp.bfloat16)
    precomputed_q, precomputed_k = apply_rotary_cos_sin(query, key, cosine, sine)
    direct_q, direct_k = apply_rotary_embedding(query, key, positions, inv_freq)

    assert jnp.array_equal(precomputed_q, direct_q)
    assert jnp.array_equal(precomputed_k, direct_k)
    # rotary_dim is four, so partial RoPE leaves the remaining channels exact.
    assert jnp.array_equal(precomputed_q[..., 4:], query[..., 4:])
    assert jnp.array_equal(precomputed_k[..., 4:], key[..., 4:])
