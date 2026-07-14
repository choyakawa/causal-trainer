"""Partial rotary position embeddings with selectable channel pairing."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import ModelConfig, RopeStyle

Array = jax.Array


def canonicalize_segment_ids(
    attention_mask: Array,
    segment_ids: Array | None = None,
) -> Array:
    """Return one-based IDs for contiguous valid runs and zero for padding.

    Packed attention only needs to know whether two tokens belong to the same
    contiguous sample.  Canonicalizing runs instead of trusting the numeric ID
    itself makes two otherwise troublesome inputs safe:

    * some tokenizers use ``0`` for the first *valid* segment, even though
      Splash conventionally reserves zero for padding;
    * a malformed/rebatched input can reuse an ID after another segment.  If
      passed through verbatim, the later run could attend to the earlier run.

    The packer already emits monotonically increasing one-based IDs, so this is
    an identity operation for normal training batches.
    """

    valid = jnp.asarray(attention_mask, dtype=jnp.bool_)
    if valid.ndim != 2:
        raise ValueError(f"attention_mask must have shape [batch, sequence], got {valid.shape}")

    if segment_ids is None:
        labels = jnp.zeros(valid.shape, dtype=jnp.int32)
    else:
        labels = jnp.asarray(segment_ids, dtype=jnp.int32)
        if labels.shape != valid.shape:
            raise ValueError(f"segment_ids must have shape {valid.shape}, got {labels.shape}")

    previous_valid = jnp.concatenate(
        (jnp.zeros((valid.shape[0], 1), dtype=jnp.bool_), valid[:, :-1]),
        axis=1,
    )
    previous_labels = jnp.concatenate((labels[:, :1], labels[:, :-1]), axis=1)
    starts = valid & (~previous_valid | (labels != previous_labels))
    runs = jnp.cumsum(starts, axis=1, dtype=jnp.int32)
    return jnp.where(valid, runs, 0).astype(jnp.int32)


def inverse_frequencies(config: ModelConfig) -> Array:
    """Return the float32 inverse-frequency vector for the rotated channels."""

    channels = jnp.arange(0, config.rotary_dim, 2, dtype=jnp.float32)
    # Keep the same operation ordering as the Transformers
    # implementation: 1 / (base ** (index / dim)).
    base = jnp.asarray(config.rope_theta, jnp.float32)
    return jnp.asarray(1.0, jnp.float32) / jnp.power(base, channels / config.rotary_dim)


def make_position_ids(
    attention_mask: Array,
    segment_ids: Array | None = None,
) -> Array:
    """Create zero-based positions, resetting at every packed segment.

    Segment labels are interpreted as contiguous runs.  Positions therefore
    reset safely even when a valid run is labelled zero or a label is reused
    later in the row.
    """

    valid = jnp.asarray(attention_mask, dtype=jnp.bool_)
    if valid.ndim != 2:
        raise ValueError(f"attention_mask must have shape [batch, sequence], got {valid.shape}")

    segments = canonicalize_segment_ids(valid, segment_ids)
    return _make_position_ids_precanonicalized(valid, segments)


def _make_position_ids_precanonicalized(
    attention_mask: Array,
    segment_ids: Array,
) -> Array:
    """Create reset positions from already-canonical contiguous-run IDs.

    This private entry point lets the decoder canonicalize packed segments once
    and reuse the result for both positions and every attention layer. Public
    callers should use :func:`make_position_ids`, which keeps the defensive
    canonicalization boundary.
    """

    valid = jnp.asarray(attention_mask, dtype=jnp.bool_)
    if valid.ndim != 2:
        raise ValueError(f"attention_mask must have shape [batch, sequence], got {valid.shape}")
    segments = jnp.asarray(segment_ids, dtype=jnp.int32)
    if segments.shape != valid.shape:
        raise ValueError(f"segment_ids must have shape {valid.shape}, got {segments.shape}")

    batch, sequence = valid.shape
    indices = jnp.broadcast_to(jnp.arange(sequence, dtype=jnp.int32), (batch, sequence))
    previous = jnp.concatenate(
        (jnp.zeros((batch, 1), dtype=segments.dtype), segments[:, :-1]),
        axis=1,
    )
    starts = valid & ((segments != previous) | (previous == 0))
    start_indices = jnp.where(starts, indices, 0)
    segment_starts = jax.lax.associative_scan(jnp.maximum, start_indices, axis=1)
    return jnp.where(valid, indices - segment_starts, 0).astype(jnp.int32)


def _rotate_adjacent_pairs(x: Array, cosine: Array, sine: Array) -> Array:
    """Rotate (0,1), (2,3), ... channel pairs."""

    even = x[..., ::2]
    odd = x[..., 1::2]
    out_even = even * cosine - odd * sine
    out_odd = odd * cosine + even * sine
    return jnp.stack((out_even, out_odd), axis=-1).reshape(x.shape)


def _rotate_half_pairs(x: Array, cosine: Array, sine: Array) -> Array:
    """Rotate (0,d/2), (1,d/2+1), ... channel pairs."""

    first, second = jnp.split(x, 2, axis=-1)
    out_first = first * cosine - second * sine
    out_second = second * cosine + first * sine
    return jnp.concatenate((out_first, out_second), axis=-1)


def rotary_cos_sin(
    position_ids: Array,
    inv_freq: Array,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Compute only the requested tokens' RoPE values, with FP32 angles.

    No maximum-position cache is constructed. This is important for sparse or
    discontinuous logical positions (including packed resets and EndPrompt
    jumps): memory is proportional to ``[batch, sequence, rotary_dim / 2]``
    rather than ``max(position_ids)`` or ``max_position_embeddings``.
    """

    positions = jnp.asarray(position_ids, dtype=jnp.float32)
    if positions.ndim != 2:
        raise ValueError(f"position_ids must have shape [batch, sequence], got {positions.shape}")
    frequencies = jnp.asarray(inv_freq, dtype=jnp.float32)
    if frequencies.ndim != 1 or frequencies.shape[0] == 0:
        raise ValueError(f"inv_freq must be a non-empty vector, got {frequencies.shape}")

    # Multiplication and transcendental evaluation deliberately happen in
    # FP32. The compact per-token trigonometric values are cast once before the
    # decoder layer/remat loop and then reused by every layer.
    angles = positions[..., None] * frequencies[None, None, :]
    output_dtype = jnp.dtype(dtype)
    return jnp.cos(angles).astype(output_dtype), jnp.sin(angles).astype(output_dtype)


def apply_rotary_cos_sin(
    query: Array,
    key: Array,
    cosine: Array,
    sine: Array,
    *,
    rope_style: RopeStyle = "gpt-j",
) -> tuple[Array, Array]:
    """Apply precomputed partial RoPE to Q and K using the selected pairing."""

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must have shape [batch, sequence, heads, head_dim]")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError(f"incompatible query/key shapes: {query.shape} and {key.shape}")

    cosine = jnp.asarray(cosine)
    sine = jnp.asarray(sine)
    if cosine.shape != sine.shape:
        raise ValueError(f"cosine and sine shapes differ: {cosine.shape} and {sine.shape}")
    if cosine.ndim != 3 or cosine.shape[:2] != query.shape[:2]:
        expected_prefix = query.shape[:2]
        raise ValueError(
            "cosine and sine must have shape "
            f"[{expected_prefix[0]}, {expected_prefix[1]}, rotary_dim / 2], got {cosine.shape}"
        )

    rotary_dim = cosine.shape[-1] * 2
    if rotary_dim <= 0 or rotary_dim > query.shape[-1] or rotary_dim % 2:
        raise ValueError(f"invalid rotary dimension {rotary_dim} for head_dim={query.shape[-1]}")
    if rope_style not in {"gpt-j", "gpt-neox"}:
        raise ValueError("rope_style must be 'gpt-j' or 'gpt-neox'")
    cosine = cosine[:, :, None, :]
    sine = sine[:, :, None, :]

    def rotate(x: Array) -> Array:
        rotary_channels = x[..., :rotary_dim]
        if rope_style == "gpt-j":
            rotated = _rotate_adjacent_pairs(rotary_channels, cosine, sine)
        else:
            rotated = _rotate_half_pairs(rotary_channels, cosine, sine)
        if rotary_dim == x.shape[-1]:
            return rotated
        return jnp.concatenate((rotated, x[..., rotary_dim:]), axis=-1)

    return rotate(query), rotate(key)


def apply_rotary_embedding(
    query: Array,
    key: Array,
    position_ids: Array,
    inv_freq: Array,
    *,
    rope_style: RopeStyle = "gpt-j",
) -> tuple[Array, Array]:
    """Apply partial RoPE to Q and K in ``[B, S, H, D]`` layout."""

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must have shape [batch, sequence, heads, head_dim]")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError(f"incompatible query/key shapes: {query.shape} and {key.shape}")

    positions = jnp.asarray(position_ids, dtype=jnp.float32)
    if positions.ndim == 1:
        positions = jnp.broadcast_to(positions[None, :], query.shape[:2])
    if positions.shape != query.shape[:2]:
        raise ValueError(f"position_ids must have shape {query.shape[:2]}, got {positions.shape}")

    inv_freq = jnp.asarray(inv_freq, dtype=jnp.float32)
    rotary_dim = inv_freq.shape[0] * 2
    if rotary_dim <= 0 or rotary_dim > query.shape[-1] or rotary_dim % 2:
        raise ValueError(f"invalid rotary dimension {rotary_dim} for head_dim={query.shape[-1]}")

    cosine, sine = rotary_cos_sin(positions, inv_freq, query.dtype)
    return apply_rotary_cos_sin(query, key, cosine, sine, rope_style=rope_style)
