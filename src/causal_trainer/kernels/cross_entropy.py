"""Memory-bounded TPU Pallas sparse cross-entropy.

This is a focused, dependency-free adaptation of ejkernel's TPU kernel.  Both
the replicated- and vocab-parallel forward paths use an online softmax, so each
local logits slab is read from HBM once.  The analytic backward reads one tile
at a time and writes the corresponding logits gradient without materializing a
probability tensor in HBM.

Only the training contract used by this project is implemented: integer
targets, non-negative per-token weights, no label smoothing, and no z-loss.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

Array = jax.Array
_BLOCK_M = 256
_BLOCK_V = 4096
_TPU_LANES = 128
# An out-of-shard target can legitimately become -100 after subtracting the
# vocab offset, so TP uses an unreachable local-only masking sentinel.
_LOCAL_IGNORE_INDEX = -(2**30)


def _pallas_out_shape(
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    like: Array | None = None,
) -> jax.ShapeDtypeStruct:
    """Describe a Pallas output correctly inside a manual ``shard_map``."""

    if like is not None:
        manual_axis_type = getattr(like, "manual_axis_type", None)
        if manual_axis_type is None:
            manual_axis_type = getattr(getattr(like, "aval", None), "manual_axis_type", None)
        if manual_axis_type is not None:
            return jax.ShapeDtypeStruct(shape, dtype, manual_axis_type=manual_axis_type)

    abstract_mesh = jax.sharding.get_abstract_mesh()
    axis_sizes = getattr(abstract_mesh, "axis_sizes", None) or getattr(abstract_mesh, "shape", {})
    axis_size = getattr(axis_sizes, "get", lambda _axis, default: default)
    varying = frozenset(
        axis for axis in abstract_mesh.manual_axes if int(axis_size(axis, 1)) > 1
    )
    return jax.ShapeDtypeStruct(
        shape,
        dtype,
        manual_axis_type=jax.sharding.ManualAxisType(varying=varying),
    )


def _pad_rows_2d(x: Array, pad_rows: int, pad_value: float = 0.0) -> Array:
    if pad_rows == 0:
        return x
    return jnp.pad(x, ((0, pad_rows), (0, 0)), constant_values=pad_value)


def _pad_rows_1d(x: Array, pad_rows: int, pad_value: float = 0.0) -> Array:
    if pad_rows == 0:
        return x
    return jnp.pad(x, ((0, pad_rows),), constant_values=pad_value)


def _repeat_row_metadata(x: Array) -> Array:
    """Store row scalars in a TPU-native 128-lane replicated layout."""

    return jnp.broadcast_to(x[:, None], (x.shape[0], _TPU_LANES))


def _copy_tile(src_ref, dst_ref, sem_ref, row_start, col_start, block_m: int, size: int) -> None:
    copy = pltpu.make_async_copy(
        src_ref=src_ref.at[pl.ds(row_start, block_m), pl.ds(col_start, size)],
        dst_ref=dst_ref.at[pl.ds(0, block_m), pl.ds(0, size)],
        sem=sem_ref.at[0],
    )
    copy.start()
    copy.wait()


def _copy_rows(src_ref, dst_ref, sem_ref, row_start, block_m: int) -> None:
    copy = pltpu.make_async_copy(
        src_ref=src_ref.at[pl.ds(row_start, block_m)],
        dst_ref=dst_ref.at[pl.ds(0, block_m)],
        sem=sem_ref.at[0],
    )
    copy.start()
    copy.wait()


def _ce_stats_kernel(
    logits_ref,
    targets_ref,
    weights_ref,
    max_ref,
    sum_exp_ref,
    target_logit_ref,
    logits_tile_ref,
    target_tile_ref,
    weight_tile_ref,
    dma_sem_ref,
    *,
    ignore_index: int,
    vocab_size: int,
    block_v: int,
    block_m: int,
) -> None:
    """Stream one row block once and emit online-softmax statistics."""

    row_start = pl.program_id(0) * block_m
    _, padded_vocab_size = logits_ref.shape
    offsets = jax.lax.broadcasted_iota(jnp.int32, (block_m, block_v), 1)
    row_offsets = jax.lax.broadcasted_iota(jnp.int32, (block_m, _TPU_LANES), 0)
    row_active = (row_start + row_offsets) < logits_ref.shape[0]

    _copy_rows(targets_ref, target_tile_ref, dma_sem_ref, row_start, block_m)
    _copy_rows(weights_ref, weight_tile_ref, dma_sem_ref, row_start, block_m)
    targets = target_tile_ref[...].astype(jnp.int32)
    weights = weight_tile_ref[...].astype(jnp.float32)
    valid = row_active & (targets != ignore_index) & (weights != 0.0)

    max_ref[...] = jnp.full((block_m, _TPU_LANES), -jnp.inf, dtype=jnp.float32)
    sum_exp_ref[...] = jnp.zeros((block_m, _TPU_LANES), dtype=jnp.float32)
    target_logit_ref[...] = jnp.zeros((block_m, _TPU_LANES), dtype=jnp.float32)

    @pl.when(jnp.any(valid))
    def _compute_active_block() -> None:
        # Keep row statistics replicated over one physical 128-lane vector,
        # matching the layout used by JAX's TPU Flash Attention kernels. The wrapper
        # selects one copy after the Pallas call.
        running_max = jnp.full((block_m, _TPU_LANES), -jnp.inf, dtype=jnp.float32)
        running_sum_exp = jnp.zeros((block_m, _TPU_LANES), dtype=jnp.float32)
        target_logit = jnp.zeros((block_m, _TPU_LANES), dtype=jnp.float32)

        # ``logits_ref`` is padded to a complete VMEM/DMA tile by the caller.
        # Every transfer therefore has one fixed, statically aligned minor
        # dimension.  The original vocabulary width remains a separate mask
        # so padding never participates in softmax or target lookup.
        for block_index in range(padded_vocab_size // block_v):
            start = block_index * block_v
            _copy_tile(
                logits_ref,
                logits_tile_ref,
                dma_sem_ref,
                row_start,
                start,
                block_m,
                block_v,
            )
            tile = logits_tile_ref[...].astype(jnp.float32)
            vocab_index = start + offsets
            in_vocab = vocab_index < vocab_size
            masked = jnp.where(in_vocab, tile, -jnp.inf)
            tile_max = jnp.tile(jnp.max(masked, axis=1)[:, None], (1, _TPU_LANES))
            new_max = jnp.maximum(running_max, tile_max)
            rescale = jnp.exp(running_max - new_max)
            expanded_new_max = jnp.tile(new_max, (1, block_v // _TPU_LANES))
            tile_sum_exp = jnp.sum(
                jnp.where(in_vocab, jnp.exp(tile - expanded_new_max), 0.0),
                axis=1,
            )
            tile_sum_exp = jnp.tile(tile_sum_exp[:, None], (1, _TPU_LANES))
            running_sum_exp = running_sum_exp * rescale + tile_sum_exp
            running_max = new_max
            expanded_targets = jnp.tile(targets, (1, block_v // _TPU_LANES))
            tile_target_logit = jnp.sum(
                jnp.where(
                    in_vocab & (vocab_index == expanded_targets),
                    tile,
                    0.0,
                ),
                axis=1,
            )
            target_logit = target_logit + jnp.tile(
                tile_target_logit[:, None],
                (1, _TPU_LANES),
            )

        max_ref[...] = jnp.where(valid, running_max, -jnp.inf).astype(jnp.float32)
        sum_exp_ref[...] = jnp.where(valid, running_sum_exp, 0.0).astype(jnp.float32)
        target_logit_ref[...] = jnp.where(valid, target_logit, 0.0).astype(jnp.float32)


def _ce_stats_pallas(
    logits: Array,
    targets: Array,
    weights: Array,
    *,
    ignore_index: int,
    block_v: int,
    block_m: int,
) -> tuple[Array, Array, Array]:
    if block_v % _TPU_LANES:
        raise ValueError("Pallas CE vocab tile must be divisible by 128")
    rows, vocab_size = logits.shape
    padded_rows = pl.cdiv(rows, block_m) * block_m
    padded_vocab = pl.cdiv(vocab_size, block_v) * block_v
    pad_rows = padded_rows - rows
    pad_vocab = padded_vocab - vocab_size
    logits_pad = jnp.pad(logits, ((0, pad_rows), (0, pad_vocab)))
    targets_pad = _repeat_row_metadata(
        _pad_rows_1d(targets, pad_rows, ignore_index).astype(jnp.int32)
    )
    weights_pad = _repeat_row_metadata(
        _pad_rows_1d(weights, pad_rows).astype(jnp.float32)
    )

    local_max, local_sum_exp, local_target_logit = pl.pallas_call(
        functools.partial(
            _ce_stats_kernel,
            ignore_index=ignore_index,
            vocab_size=vocab_size,
            block_v=block_v,
            block_m=block_m,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=[
                pl.BlockSpec((block_m, _TPU_LANES), lambda row_block: (row_block, 0)),
                pl.BlockSpec((block_m, _TPU_LANES), lambda row_block: (row_block, 0)),
                pl.BlockSpec((block_m, _TPU_LANES), lambda row_block: (row_block, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((block_m, block_v), logits.dtype),
                pltpu.VMEM((block_m, _TPU_LANES), targets_pad.dtype),
                pltpu.VMEM((block_m, _TPU_LANES), weights_pad.dtype),
                pltpu.SemaphoreType.DMA((1,)),
            ],
            grid=(padded_rows // block_m,),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        out_shape=[
            _pallas_out_shape((padded_rows, _TPU_LANES), jnp.float32),
            _pallas_out_shape((padded_rows, _TPU_LANES), jnp.float32),
            _pallas_out_shape((padded_rows, _TPU_LANES), jnp.float32),
        ],
    )(logits_pad, targets_pad, weights_pad)
    return local_max[:rows, 0], local_sum_exp[:rows, 0], local_target_logit[:rows, 0]


def _ce_bwd_kernel(
    logits_ref,
    lse_ref,
    targets_ref,
    weights_ref,
    dy_ref,
    dlogits_ref,
    logits_tile_ref,
    lse_tile_ref,
    target_tile_ref,
    weight_tile_ref,
    dy_tile_ref,
    dma_sem_ref,
    *,
    ignore_index: int,
    vocab_size: int,
    block_v: int,
    block_m: int,
) -> None:
    """Write one ``(rows, vocab)`` gradient tile without an HBM softmax."""

    row_start = pl.program_id(0) * block_m
    vocab_block = pl.program_id(1)
    _, padded_vocab_size = logits_ref.shape
    start = vocab_block * block_v
    offsets = jax.lax.broadcasted_iota(jnp.int32, (block_m, block_v), 1)
    row_offsets = jax.lax.broadcasted_iota(jnp.int32, (block_m, _TPU_LANES), 0)

    # The wrapper pads the minor dimension to a whole ``block_v``.  Avoid a
    # dynamic-size tail DMA: JAX 0.10 TPU only supports the aligned fixed-size
    # transfer used here. ``padded_vocab_size`` is retained as a defensive
    # static consistency check for future block-size changes.
    if padded_vocab_size % block_v:
        raise ValueError("padded logits vocabulary must be divisible by block_v")
    _copy_tile(logits_ref, logits_tile_ref, dma_sem_ref, row_start, start, block_m, block_v)
    _copy_rows(lse_ref, lse_tile_ref, dma_sem_ref, row_start, block_m)
    _copy_rows(targets_ref, target_tile_ref, dma_sem_ref, row_start, block_m)
    _copy_rows(weights_ref, weight_tile_ref, dma_sem_ref, row_start, block_m)
    _copy_rows(dy_ref, dy_tile_ref, dma_sem_ref, row_start, block_m)

    targets = target_tile_ref[...].astype(jnp.int32)
    weights = weight_tile_ref[...].astype(jnp.float32)
    row_active = (row_start + row_offsets) < logits_ref.shape[0]
    valid = row_active & (targets != ignore_index) & (weights != 0.0)
    factor = jnp.where(
        valid,
        weights * dy_tile_ref[...].astype(jnp.float32),
        0.0,
    )
    tile = logits_tile_ref[...].astype(jnp.float32)
    expanded_valid = jnp.tile(valid, (1, block_v // _TPU_LANES))
    expanded_lse = jnp.tile(
        lse_tile_ref[...].astype(jnp.float32),
        (1, block_v // _TPU_LANES),
    )
    probabilities = jnp.where(expanded_valid, jnp.exp(tile - expanded_lse), 0.0)
    vocab_index = start + offsets
    target_is_local = (targets >= 0) & (targets < vocab_size)
    comparable_target = jnp.where(target_is_local, targets, -1)
    expanded_target = jnp.tile(comparable_target, (1, block_v // _TPU_LANES))
    one_hot = (vocab_index == expanded_target).astype(jnp.float32)
    in_vocab = vocab_index < vocab_size
    expanded_factor = jnp.tile(factor, (1, block_v // _TPU_LANES))
    gradient = expanded_factor * jnp.where(
        in_vocab,
        probabilities - one_hot,
        0.0,
    )
    dlogits_ref[...] = gradient.astype(dlogits_ref.dtype)


def _ce_bwd_pallas(
    logits: Array,
    lse: Array,
    targets: Array,
    weights: Array,
    dy: Array,
    *,
    ignore_index: int,
    block_v: int,
    block_m: int,
) -> Array:
    if block_v % _TPU_LANES:
        raise ValueError("Pallas CE vocab tile must be divisible by 128")
    rows, vocab_size = logits.shape
    padded_rows = pl.cdiv(rows, block_m) * block_m
    padded_vocab = pl.cdiv(vocab_size, block_v) * block_v
    pad_rows = padded_rows - rows
    pad_vocab = padded_vocab - vocab_size
    logits_pad = jnp.pad(logits, ((0, pad_rows), (0, pad_vocab)))
    lse_pad = _repeat_row_metadata(
        _pad_rows_1d(lse, pad_rows).astype(jnp.float32)
    )
    targets_pad = _repeat_row_metadata(
        _pad_rows_1d(targets, pad_rows, ignore_index).astype(jnp.int32)
    )
    weights_pad = _repeat_row_metadata(
        _pad_rows_1d(weights, pad_rows).astype(jnp.float32)
    )
    dy_pad = _repeat_row_metadata(
        _pad_rows_1d(dy, pad_rows).astype(jnp.float32)
    )
    vocab_blocks = padded_vocab // block_v

    output = pl.pallas_call(
        functools.partial(
            _ce_bwd_kernel,
            ignore_index=ignore_index,
            vocab_size=vocab_size,
            block_v=block_v,
            block_m=block_m,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(
                (block_m, block_v),
                lambda row_block, vocab_block: (row_block, vocab_block),
            ),
            scratch_shapes=[
                pltpu.VMEM((block_m, block_v), logits.dtype),
                pltpu.VMEM((block_m, _TPU_LANES), lse_pad.dtype),
                pltpu.VMEM((block_m, _TPU_LANES), targets_pad.dtype),
                pltpu.VMEM((block_m, _TPU_LANES), weights_pad.dtype),
                pltpu.VMEM((block_m, _TPU_LANES), dy_pad.dtype),
                pltpu.SemaphoreType.DMA((1,)),
            ],
            grid=(padded_rows // block_m, vocab_blocks),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        out_shape=_pallas_out_shape(logits_pad.shape, logits.dtype, logits),
        # Backward is the final consumer of the saved logits slab.  Reusing
        # that HBM allocation for dlogits avoids having both buffers live at
        # once; XLA will insert a copy if an enclosing program has another
        # live use and therefore cannot honor the alias safely.
        input_output_aliases={0: 0},
    )(logits_pad, lse_pad, targets_pad, weights_pad, dy_pad)
    return output[:rows, :vocab_size]


def _replicated_loss_and_lse(
    logits: Array,
    targets: Array,
    weights: Array,
    *,
    ignore_index: int,
    block_v: int,
    block_m: int,
) -> tuple[Array, Array]:
    local_max, local_sum_exp, target_logit = _ce_stats_pallas(
        logits,
        targets,
        weights,
        ignore_index=ignore_index,
        block_v=block_v,
        block_m=block_m,
    )
    valid = (targets != ignore_index) & (weights != 0.0)
    lse = jnp.where(valid, jnp.log(local_sum_exp) + local_max, 0.0)
    loss = jnp.where(valid, weights * (lse - target_logit), 0.0)
    return loss.astype(jnp.float32), lse.astype(jnp.float32)


def _tp_loss_and_lse(
    logits: Array,
    targets: Array,
    weights: Array,
    *,
    ignore_index: int,
    block_v: int,
    block_m: int,
    vocab_axis: str,
) -> tuple[Array, Array, Array]:
    local_vocab = logits.shape[-1]
    vocab_start = jax.lax.axis_index(vocab_axis) * local_vocab
    local_targets = jnp.where(
        targets == ignore_index,
        jnp.asarray(_LOCAL_IGNORE_INDEX, jnp.int32),
        targets.astype(jnp.int32) - vocab_start,
    )
    local_max, local_sum_exp, local_target_logit = _ce_stats_pallas(
        logits,
        local_targets,
        weights,
        ignore_index=_LOCAL_IGNORE_INDEX,
        block_v=block_v,
        block_m=block_m,
    )
    global_max = jax.lax.pmax(local_max, vocab_axis)
    finite = jnp.isfinite(local_max) & jnp.isfinite(global_max)
    scaled_sum_exp = jnp.where(
        finite,
        local_sum_exp * jnp.exp(local_max - global_max),
        0.0,
    )
    global_sum_exp = jax.lax.psum(scaled_sum_exp, vocab_axis)
    target_logit = jax.lax.psum(local_target_logit, vocab_axis)
    valid = (targets != ignore_index) & (weights != 0.0)
    lse = jnp.where(valid, jnp.log(global_sum_exp) + global_max, 0.0)
    loss = jnp.where(valid, weights * (lse - target_logit), 0.0)
    return loss.astype(jnp.float32), lse.astype(jnp.float32), local_targets


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _replicated_ce(
    logits: Array,
    targets: Array,
    weights: Array,
    ignore_index: int,
    block_v: int,
    block_m: int,
) -> Array:
    loss, _ = _replicated_loss_and_lse(
        logits,
        targets,
        weights,
        ignore_index=ignore_index,
        block_v=block_v,
        block_m=block_m,
    )
    return loss


def _replicated_ce_fwd(logits, targets, weights, ignore_index, block_v, block_m):
    loss, lse = _replicated_loss_and_lse(
        logits,
        targets,
        weights,
        ignore_index=ignore_index,
        block_v=block_v,
        block_m=block_m,
    )
    return loss, (logits, lse, targets, weights)


def _replicated_ce_bwd(ignore_index, block_v, block_m, residual, dy):
    logits, lse, targets, weights = residual
    dlogits = _ce_bwd_pallas(
        logits,
        lse,
        targets,
        weights,
        dy,
        ignore_index=ignore_index,
        block_v=block_v,
        block_m=block_m,
    )
    return dlogits, None, None


_replicated_ce.defvjp(_replicated_ce_fwd, _replicated_ce_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _tp_ce(
    logits: Array,
    targets: Array,
    weights: Array,
    ignore_index: int,
    block_v: int,
    block_m: int,
    vocab_axis: str,
) -> Array:
    loss, _, _ = _tp_loss_and_lse(
        logits,
        targets,
        weights,
        ignore_index=ignore_index,
        block_v=block_v,
        block_m=block_m,
        vocab_axis=vocab_axis,
    )
    return loss


def _tp_ce_fwd(logits, targets, weights, ignore_index, block_v, block_m, vocab_axis):
    loss, lse, local_targets = _tp_loss_and_lse(
        logits,
        targets,
        weights,
        ignore_index=ignore_index,
        block_v=block_v,
        block_m=block_m,
        vocab_axis=vocab_axis,
    )
    return loss, (logits, lse, local_targets, weights)


def _tp_ce_bwd(ignore_index, block_v, block_m, vocab_axis, residual, dy):
    del ignore_index, vocab_axis
    logits, lse, local_targets, weights = residual
    dlogits = _ce_bwd_pallas(
        logits,
        lse,
        local_targets,
        weights,
        dy,
        ignore_index=_LOCAL_IGNORE_INDEX,
        block_v=block_v,
        block_m=block_m,
    )
    return dlogits, None, None


_tp_ce.defvjp(_tp_ce_fwd, _tp_ce_bwd)


def _local_sparse_cross_entropy(
    logits: Array,
    targets: Array,
    weights: Array,
    *,
    vocab_axis: str | None,
) -> Array:
    leading_shape = logits.shape[:-1]
    flat_logits = logits.reshape((-1, logits.shape[-1]))
    flat_targets = targets.reshape(-1).astype(jnp.int32)
    flat_weights = weights.reshape(-1).astype(jnp.float32)
    if vocab_axis is None:
        flat_loss = _replicated_ce(
            flat_logits,
            flat_targets,
            flat_weights,
            -100,
            _BLOCK_V,
            _BLOCK_M,
        )
    else:
        flat_loss = _tp_ce(
            flat_logits,
            flat_targets,
            flat_weights,
            -100,
            _BLOCK_V,
            _BLOCK_M,
            vocab_axis,
        )
    return flat_loss.reshape(leading_shape)


def pallas_fused_sparse_cross_entropy(
    logits: Array,
    targets: Array,
    weights: Array,
    *,
    mesh: Mesh,
) -> Array:
    """Return per-token TPU Pallas CE for globally sharded training arrays.

    ``logits`` must have shape ``[batch, sequence, vocab]`` and use the
    trainer's standard ``dp/fsdp, sp, tp`` layout.  ``targets`` and ``weights``
    use the matching leading layout and are replicated over ``tp``.  The
    vocab-parallel custom VJP relies on JAX's varying-manual-axis checks, hence
    the JAX 0.10 requirement.
    """

    if logits.ndim != 3 or targets.shape != logits.shape[:-1] or weights.shape != targets.shape:
        raise ValueError("Pallas CE expects logits [batch, sequence, vocab] and matching target/weight rows")

    batch_axis = tuple(axis for axis in ("dp", "fsdp") if axis in mesh.axis_names)
    if not batch_axis:
        batch_axis = None
    elif len(batch_axis) == 1:
        batch_axis = batch_axis[0]
    sequence_axis = "sp" if "sp" in mesh.axis_names else None
    vocab_axis = "tp" if "tp" in mesh.axis_names and mesh.shape["tp"] > 1 else None
    logits_spec = P(batch_axis, sequence_axis, vocab_axis)
    rows_spec = P(batch_axis, sequence_axis)

    def local(local_logits: Array, local_targets: Array, local_weights: Array) -> Array:
        return _local_sparse_cross_entropy(
            local_logits,
            local_targets,
            local_weights,
            vocab_axis=vocab_axis,
        )

    return shard_map(
        local,
        mesh=mesh,
        in_specs=(logits_spec, rows_spec, rows_spec),
        out_specs=rows_spec,
        check_vma=True,
    )(logits, targets, weights)


__all__ = ["pallas_fused_sparse_cross_entropy"]
