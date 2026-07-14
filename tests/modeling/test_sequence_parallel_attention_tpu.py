"""TPU correctness gate for query-sequence-sharded Splash attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from causal_trainer.modeling.attention import (
    make_causal_segment_mask,
    splash_attention,
    vanilla_attention,
)


def _has_four_tpu_devices() -> bool:
    try:
        return jax.default_backend() == "tpu" and jax.local_device_count() >= 4
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_four_tpu_devices(),
    reason="requires at least four TPU devices with JAX 0.10",
)


def _sp_tp_mesh() -> Mesh:
    devices = np.asarray(jax.local_devices()[:4], dtype=object).reshape(1, 1, 1, 2, 2)
    return Mesh(devices, ("dp", "fsdp", "ep", "tp", "sp"))


def test_query_sequence_sharded_splash_forward_and_vjp_match_vanilla() -> None:
    batch, sequence, query_heads, head_dim = 1, 256, 4, 128
    query = (
        jax.random.normal(
            jax.random.PRNGKey(40),
            (batch, sequence, query_heads, head_dim),
        )
        * 0.05
    ).astype(jnp.bfloat16)
    key = (
        jax.random.normal(
            jax.random.PRNGKey(41),
            (batch, sequence, 1, head_dim),
        )
        * 0.05
    ).astype(jnp.bfloat16)
    value = (
        jax.random.normal(
            jax.random.PRNGKey(42),
            (batch, sequence, 1, head_dim),
        )
        * 0.05
    ).astype(jnp.bfloat16)
    positions = jnp.arange(sequence)[None, :]
    cotangent = (
        jax.random.normal(
            jax.random.PRNGKey(43),
            query.shape,
        )
        * 0.02
    ).astype(jnp.bfloat16)
    # Only queries on the second SP shard contribute to the VJP.  Segment 2
    # begins on shard 0 and ends on shard 1, so its dK/dV on positions 40:128
    # can only be correct when local Q sees the replicated global K/V sequence
    # and the query/key SegmentIds use their distinct shardings.
    cotangent = jnp.where(
        ((positions >= 160) & (positions < 193))[:, :, None, None],
        cotangent,
        jnp.asarray(0.0, cotangent.dtype),
    )
    segments = jnp.where(
        positions < 40,
        jnp.asarray(1, jnp.int32),
        jnp.where(
            positions < 193,
            jnp.asarray(2, jnp.int32),
            jnp.asarray(3, jnp.int32),
        ),
    )
    valid = jnp.ones((batch, sequence), dtype=jnp.bool_)
    allowed = make_causal_segment_mask(valid, segments)
    scale = head_dim**-0.5
    mesh = _sp_tp_mesh()

    def reference(q, k, v):
        return vanilla_attention(q, k, v, allowed, scale=scale)

    def actual(q, k, v):
        return splash_attention(
            q,
            k,
            v,
            segments,
            scale=scale,
            block_q=128,
            block_k=128,
            mesh=mesh,
        )

    def forward_and_vjp(function, q, k, v):
        output, pullback = jax.vjp(function, q, k, v)
        return output, pullback(cotangent)

    expected_output, (expected_dq, expected_dk, expected_dv) = jax.jit(
        lambda q, k, v: forward_and_vjp(reference, q, k, v)
    )(query, key, value)
    with jax.set_mesh(mesh):
        actual_output, (actual_dq, actual_dk, actual_dv) = jax.jit(
            lambda q, k, v: forward_and_vjp(actual, q, k, v)
        )(query, key, value)

    assert actual_output.sharding.spec == PartitionSpec(None, "sp", "tp")
    np.testing.assert_allclose(
        np.asarray(actual_output),
        np.asarray(expected_output),
        rtol=3e-3,
        atol=3e-3,
    )
    for actual_gradient, expected_gradient in (
        (actual_dq, expected_dq),
        (actual_dk, expected_dk),
        (actual_dv, expected_dv),
    ):
        np.testing.assert_allclose(
            np.asarray(actual_gradient),
            np.asarray(expected_gradient),
            rtol=0.0,
            atol=5e-2,
        )

    # These checks make the comparison non-vacuous: the remote half of the
    # cross-shard segment receives dK/dV, while packed neighbours do not leak.
    for expected_gradient in (expected_dk, expected_dv):
        assert np.any(np.asarray(expected_gradient[:, 64:96]) != 0.0)
        assert np.all(np.asarray(expected_gradient[:, :40]) == 0.0)
        assert np.all(np.asarray(expected_gradient[:, 193:]) == 0.0)
    assert np.any(np.asarray(expected_dq[:, 160:193]) != 0.0)
    assert np.all(np.asarray(expected_dq[:, :160]) == 0.0)
    assert np.all(np.asarray(expected_dq[:, 193:]) == 0.0)
