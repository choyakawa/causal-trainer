from __future__ import annotations

from types import SimpleNamespace

import pytest

from causal_trainer.kernels.fused_cross_entropy import _per_device_sequence_chunk_size


def _mesh_shape(*, dp: int = 1, fsdp: int = 1, sp: int = 1):
    return SimpleNamespace(shape={"dp": dp, "fsdp": fsdp, "sp": sp})


def test_token_budget_counts_local_rows_without_flattening_global_dp() -> None:
    # Global [32, 4096] remains a rectangular global slice.  With four data
    # shards, [8, 16] rows are projected locally: exactly the 128-row budget.
    assert _per_device_sequence_chunk_size(
        batch_size=32,
        sequence_length=4096,
        token_budget=128,
        mesh=_mesh_shape(dp=4),
    ) == 16


def test_token_budget_accounts_for_sequence_parallel_shards() -> None:
    # The global slice is SP-divisible.  Each shard sees [8, 16], while the
    # global sequence slice is 32 rather than a cross-shard flattened 128.
    assert _per_device_sequence_chunk_size(
        batch_size=32,
        sequence_length=4096,
        token_budget=128,
        mesh=_mesh_shape(dp=4, sp=2),
    ) == 32


def test_token_budget_rejects_less_than_one_local_sequence_position() -> None:
    with pytest.raises(ValueError, match="one sequence position"):
        _per_device_sequence_chunk_size(
            batch_size=32,
            sequence_length=4096,
            token_budget=7,
            mesh=_mesh_shape(dp=4),
        )


def test_token_budget_does_not_expand_a_short_sequence() -> None:
    assert _per_device_sequence_chunk_size(
        batch_size=8,
        sequence_length=8,
        token_budget=128,
        mesh=_mesh_shape(dp=2),
    ) == 8
