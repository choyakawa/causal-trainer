"""Low-rank adapters for the decoder's trainable projection kernels.

The adapter layout mirrors the corresponding layer subtree, but contains only
the seven projection adapters.  Kernels use the project's JAX ``[input,
output]`` layout, so an adapter is represented by ``lora_a`` in
``[input, rank]`` layout and ``lora_b`` in ``[rank, output]`` layout.  There is
deliberately no alpha or scaling term: the effective kernel is exactly
``kernel + lora_a @ lora_b``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .config import ModelConfig

if TYPE_CHECKING:
    from ..checkpointing.huggingface import TensorMapping

Array = jax.Array
Adapter = Mapping[str, Any]

LORA_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
"""Projection names adapted by the training-only LoRA mode, in stable order."""

_COLUMN_PROJECTIONS = frozenset(("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"))
_ROW_PROJECTIONS = frozenset(("o_proj", "down_proj"))
_ADAPTER_LEAVES = frozenset(("lora_a", "lora_b"))


def _validate_rank(rank: int) -> int:
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError(f"LoRA rank must be a positive integer, got {rank!r}")
    return rank


def _adapter_dtype(dtype: jnp.dtype) -> jnp.dtype:
    result = jnp.dtype(dtype)
    if not jnp.issubdtype(result, jnp.floating):
        raise TypeError(f"LoRA parameters require a floating dtype, got {result}")
    return result


def _projection_dimensions(config: ModelConfig) -> dict[str, tuple[int, int]]:
    return {
        "q_proj": (config.hidden_size, config.query_width),
        "k_proj": (config.hidden_size, config.key_value_width),
        "v_proj": (config.hidden_size, config.key_value_width),
        "o_proj": (config.query_width, config.hidden_size),
        "gate_proj": (config.hidden_size, config.intermediate_size),
        "up_proj": (config.hidden_size, config.intermediate_size),
        "down_proj": (config.intermediate_size, config.hidden_size),
    }


def _adapter_shape(
    in_features: int,
    out_features: int,
    rank: int,
    dtype: jnp.dtype,
) -> dict[str, jax.ShapeDtypeStruct]:
    return {
        "lora_a": jax.ShapeDtypeStruct((in_features, rank), dtype),
        "lora_b": jax.ShapeDtypeStruct((rank, out_features), dtype),
    }


def lora_parameter_shapes(
    config: ModelConfig,
    rank: int,
    dtype: jnp.dtype,
) -> dict[str, Any]:
    """Return the abstract adapter PyTree for every decoder layer."""

    rank = _validate_rank(rank)
    dtype = _adapter_dtype(dtype)
    dimensions = _projection_dimensions(config)
    layers = []
    for _ in range(config.num_hidden_layers):
        layers.append(
            {
                "attention": {
                    name: _adapter_shape(*dimensions[name], rank, dtype)
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                },
                "mlp": {
                    name: _adapter_shape(*dimensions[name], rank, dtype)
                    for name in ("gate_proj", "up_proj", "down_proj")
                },
            }
        )
    return {"layers": tuple(layers)}


def init_lora_params(
    config: ModelConfig,
    rng: Array,
    rank: int,
    dtype: jnp.dtype,
) -> dict[str, Any]:
    """Initialize adapter A with He-uniform values and adapter B with zeros.

    For an A matrix in ``[input, rank]`` layout, He-uniform has the exact
    interval ``[-sqrt(6 / input), sqrt(6 / input)]``.  Zero-initialized B makes
    every adapter an identity-preserving addition at step zero.
    """

    rank = _validate_rank(rank)
    dtype = _adapter_dtype(dtype)
    dimensions = _projection_dimensions(config)
    keys = iter(jax.random.split(rng, config.num_hidden_layers * len(LORA_PROJECTIONS)))

    def initialize(name: str) -> dict[str, Array]:
        in_features, out_features = dimensions[name]
        limit = math.sqrt(6.0 / in_features)
        lora_a = jax.random.uniform(
            next(keys),
            (in_features, rank),
            dtype=dtype,
            minval=-limit,
            maxval=limit,
        )
        lora_b = jnp.zeros((rank, out_features), dtype=dtype)
        return {"lora_a": lora_a, "lora_b": lora_b}

    layers = []
    for _ in range(config.num_hidden_layers):
        layers.append(
            {
                "attention": {
                    name: initialize(name)
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                },
                "mlp": {
                    name: initialize(name)
                    for name in ("gate_proj", "up_proj", "down_proj")
                },
            }
        )
    return {"layers": tuple(layers)}


def _path_parts(path: str | Sequence[Any]) -> tuple[str, ...]:
    if isinstance(path, str):
        return tuple(part for part in path.replace("\\", "/").strip("/").split("/") if part)

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
    return tuple(parts)


def _path_string(path: str | Sequence[Any]) -> str:
    return "/".join(_path_parts(path))


def lora_partition_specs(tree: Any) -> Any:
    """Build the reference-compatible PartitionSpec tree for adapters.

    These rules intentionally match the architecture's stable reference:
    column-parallel projections replicate A and shard B like a column-wise
    kernel, while row-parallel projections shard A like a row-wise kernel and
    replicate B.  This also preserves the reference behavior when FSDP or
    FSDP is larger than one. Sequence parallelism shards activations only, so
    adapters are deliberately replicated over ``sp``.
    """

    def spec_for(path: Sequence[Any], _leaf: Any) -> P:
        parts = _path_parts(path)
        if len(parts) < 2 or parts[-2] not in LORA_PROJECTIONS or parts[-1] not in _ADAPTER_LEAVES:
            raise ValueError(f"unexpected leaf in LoRA parameter tree: {_path_string(path)!r}")
        projection, leaf_name = parts[-2:]
        if projection in _COLUMN_PROJECTIONS:
            return P() if leaf_name == "lora_a" else P("fsdp", "tp")
        if projection in _ROW_PROJECTIONS:
            return P("tp", "fsdp") if leaf_name == "lora_a" else P()
        raise AssertionError(f"unclassified LoRA projection: {projection}")

    return jax.tree_util.tree_map_with_path(spec_for, tree)


def _adapter_matrices(adapter: Adapter) -> tuple[Any, Any]:
    try:
        lora_a = adapter["lora_a"]
        lora_b = adapter["lora_b"]
    except (KeyError, TypeError) as exc:
        raise ValueError("adapter must contain lora_a and lora_b") from exc
    if getattr(lora_a, "ndim", None) != 2 or getattr(lora_b, "ndim", None) != 2:
        raise ValueError("lora_a and lora_b must both be rank-two arrays")
    if lora_a.shape[1] != lora_b.shape[0]:
        raise ValueError(
            f"adapter rank mismatch: lora_a has shape {lora_a.shape}, lora_b has shape {lora_b.shape}"
        )
    return lora_a, lora_b


def lora_linear_delta(
    x: Array,
    adapter: Adapter,
    compute_dtype: jnp.dtype,
) -> Array:
    """Apply the unscaled low-rank branch without materializing ``A @ B``."""

    dtype = jnp.dtype(compute_dtype)
    lora_a, lora_b = _adapter_matrices(adapter)
    if x.shape[-1] != lora_a.shape[0]:
        raise ValueError(
            f"input width {x.shape[-1]} does not match lora_a input width {lora_a.shape[0]}"
        )
    hidden = jnp.einsum(
        "...i,ir->...r",
        x.astype(dtype),
        lora_a.astype(dtype),
        precision=None,
    )
    return jnp.einsum(
        "...r,ro->...o",
        hidden,
        lora_b.astype(dtype),
        precision=None,
    )


def adapter_for_kernel_path(
    adapters: Mapping[str, Any],
    path: str | Sequence[Any],
) -> Adapter | None:
    """Return the adapter matching a neutral base-kernel path, if any."""

    parts = _path_parts(path)
    if parts[:1] == ("model",):
        parts = parts[1:]
    if len(parts) < 2 or parts[-1] != "kernel" or parts[-2] not in LORA_PROJECTIONS:
        return None

    node: Any = adapters
    for part in parts[:-1]:
        if isinstance(node, Mapping):
            if part not in node:
                return None
            node = node[part]
        elif isinstance(node, (tuple, list)):
            try:
                index = int(part)
            except ValueError:
                return None
            if index < 0 or index >= len(node):
                return None
            node = node[index]
        else:
            return None

    if not isinstance(node, Mapping) or not _ADAPTER_LEAVES.issubset(node):
        return None
    return node


def merge_lora_kernel(kernel: Array, adapter: Adapter) -> Array:
    """Return ``kernel + lora_a @ lora_b`` with the kernel's storage dtype."""

    lora_a, lora_b = _adapter_matrices(adapter)
    if kernel.ndim != 2:
        raise ValueError(f"a merged projection kernel must be rank two, got shape {kernel.shape}")
    if kernel.shape != (lora_a.shape[0], lora_b.shape[1]):
        raise ValueError(
            f"kernel shape {kernel.shape} is incompatible with adapter shapes "
            f"{lora_a.shape} and {lora_b.shape}"
        )
    with jax.default_matmul_precision("float32"):
        delta = jnp.matmul(
            lora_a.astype(kernel.dtype),
            lora_b.astype(kernel.dtype),
        )
    return kernel + delta


def _merge_components(kernel: Array, lora_a: Array, lora_b: Array) -> Array:
    return merge_lora_kernel(kernel, {"lora_a": lora_a, "lora_b": lora_b})


def _hashable_sharding(sharding: Any) -> Any:
    try:
        hash(sharding)
    except TypeError:
        return (type(sharding).__module__, type(sharding).__qualname__, repr(sharding))
    return sharding


def _compile_signature(kernel: Array, lora_a: Array, lora_b: Array) -> tuple[Any, ...]:
    def array_signature(value: Array) -> tuple[Any, ...]:
        return (
            tuple(value.shape),
            jnp.dtype(value.dtype).name,
            _hashable_sharding(value.sharding),
        )

    return (
        array_signature(kernel),
        array_signature(lora_a),
        array_signature(lora_b),
    )


def make_lora_export_transform(
    adapters: Mapping[str, Any],
) -> Callable[[str, Array], Array]:
    """Create a per-leaf export hook that never materializes a merged tree.

    Merge executables are shared by leaves with the same shapes, dtypes and
    three input shardings.  Their output is constrained to the base kernel's
    sharding, so all processes cooperatively form one temporary merged kernel
    before the checkpoint exporter gathers that single leaf.  Non-target
    leaves are returned by identity.
    """

    compiled_by_signature: dict[tuple[Any, ...], Callable[[Array, Array, Array], Array]] = {}

    def transform(path: str, base: Array) -> Array:
        adapter = adapter_for_kernel_path(adapters, path)
        if adapter is None:
            return base
        if not isinstance(base, jax.Array):
            raise TypeError(f"LoRA export base leaf must be a concrete jax.Array, got {type(base).__name__}")

        lora_a, lora_b = _adapter_matrices(adapter)
        if not isinstance(lora_a, jax.Array) or not isinstance(lora_b, jax.Array):
            raise TypeError("LoRA export adapters must be concrete jax.Array leaves")
        signature = _compile_signature(base, lora_a, lora_b)
        compiled = compiled_by_signature.get(signature)
        if compiled is None:
            compiled = jax.jit(_merge_components, out_shardings=base.sharding)
            compiled_by_signature[signature] = compiled
        return compiled(base, lora_a, lora_b)

    return transform


def split_lora_trainable_params(
    base_params: Mapping[str, Any],
    adapters: Mapping[str, Any],
    *,
    train_embed_and_lm_head: bool,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Move selected full-rank leaves out of the frozen LoRA base tree.

    The returned dictionaries only rearrange PyTree references; they do not
    copy accelerator arrays. Removing the two leaves from the frozen tree
    avoids retaining a stale multi-gigabyte embedding/head pair after the
    first optimizer update.
    """

    if not train_embed_and_lm_head:
        return dict(base_params), adapters
    missing = [name for name in ("embed_tokens", "lm_head") if name not in base_params]
    if missing:
        raise ValueError(f"base parameter tree is missing embedding-enabled LoRA leaves: {missing}")
    frozen_base = {
        key: value
        for key, value in base_params.items()
        if key not in {"embed_tokens", "lm_head"}
    }
    trainable = {
        "adapters": adapters,
        "embed_tokens": base_params["embed_tokens"],
        "lm_head": base_params["lm_head"],
    }
    return frozen_base, trainable


def lora_adapter_params(trainable_params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the adapter subtree from standard or embedding-enabled LoRA."""

    return trainable_params["adapters"] if "adapters" in trainable_params else trainable_params


def compose_lora_export_params(
    frozen_base: Mapping[str, Any],
    trainable_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Reassemble a complete base tree for one-leaf-at-a-time merged export."""

    if "adapters" not in trainable_params:
        return dict(frozen_base)
    if "embed_tokens" not in trainable_params or "lm_head" not in trainable_params:
        raise ValueError("embedding-enabled LoRA trainables are incomplete")
    output = dict(frozen_base)
    output["embed_tokens"] = trainable_params["embed_tokens"]
    output["lm_head"] = trainable_params["lm_head"]
    return output


def lora_export_plan(
    rank: int,
    *,
    train_embed_and_lm_head: bool = False,
) -> dict[str, Any]:
    """Return stable, JSON-serializable metadata for collective HF export."""

    return {
        "kind": "lora-merge",
        "rank": _validate_rank(rank),
        "targets": list(LORA_PROJECTIONS),
        "train_embed_and_lm_head": bool(train_embed_and_lm_head),
    }


def lora_parameter_to_peft_mapping(path: str) -> TensorMapping:
    """Map trainable adapter leaves to PEFT's serialized key layout."""

    from ..checkpointing.huggingface import TensorMapping

    parts = list(_path_parts(path))
    if parts[:1] == ["adapters"]:
        parts = parts[1:]
    if parts == ["embed_tokens", "embedding"]:
        return TensorMapping(path, "base_model.model.model.embed_tokens.weight")
    if parts == ["lm_head", "kernel"]:
        return TensorMapping(path, "base_model.model.lm_head.weight", transpose=True)
    if len(parts) == 5 and parts[0] == "layers" and parts[1].isdigit():
        _, layer, block, projection, leaf = parts
        if projection not in LORA_PROJECTIONS or leaf not in _ADAPTER_LEAVES:
            raise KeyError(f"unknown adapter parameter path {path!r}")
        if block == "attention" and projection in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            module = f"self_attn.{projection}"
        elif block == "mlp" and projection in {"gate_proj", "up_proj", "down_proj"}:
            module = f"mlp.{projection}"
        else:
            raise KeyError(f"adapter block/projection mismatch in path {path!r}")
        matrix_name = "lora_A" if leaf == "lora_a" else "lora_B"
        return TensorMapping(
            path,
            f"base_model.model.model.layers.{layer}.{module}.{matrix_name}.weight",
            transpose=True,
        )
    raise KeyError(f"no PEFT tensor mapping for adapter parameter path {path!r}")


def peft_adapter_config(
    base_model_name_or_path: str,
    rank: int,
    *,
    train_embed_and_lm_head: bool,
    revision: str | None = None,
) -> dict[str, Any]:
    """Return a PEFT config whose alpha/r scaling preserves this LoRA math."""

    rank = _validate_rank(rank)
    return {
        "base_model_name_or_path": str(base_model_name_or_path),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        # The training branch is x @ A @ B with no extra multiplier. PEFT
        # applies alpha/r, so alpha must equal r for an identical merged model.
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "modules_to_save": ["embed_tokens", "lm_head"] if train_embed_and_lm_head else None,
        "peft_type": "LORA",
        "r": rank,
        "revision": revision,
        "target_modules": list(LORA_PROJECTIONS),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }


__all__ = [
    "LORA_PROJECTIONS",
    "adapter_for_kernel_path",
    "compose_lora_export_params",
    "init_lora_params",
    "lora_adapter_params",
    "lora_export_plan",
    "lora_linear_delta",
    "lora_parameter_shapes",
    "lora_parameter_to_peft_mapping",
    "lora_partition_specs",
    "make_lora_export_transform",
    "merge_lora_kernel",
    "peft_adapter_config",
    "split_lora_trainable_params",
]
