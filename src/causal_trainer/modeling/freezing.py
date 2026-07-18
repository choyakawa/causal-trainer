from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_FULL_TOP_LEVEL_KEYS = frozenset({"embed_tokens", "layers", "norm", "lm_head"})
_FROZEN_LAYER_KEYS = frozenset({"input_layernorm", "post_attention_layernorm"})
_TRAINABLE_LAYER_KEYS = frozenset({"attention", "mlp"})
_FULL_LAYER_KEYS = _FROZEN_LAYER_KEYS | _TRAINABLE_LAYER_KEYS
_SUPPORTED_COMPONENTS = frozenset({"embed", "lm_head", "norm"})


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping, got {type(value).__name__}")
    return value


def _require_exact_keys(
    value: Any,
    expected: frozenset[str],
    path: str,
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, path)
    actual = set(mapping)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)!r}")
        if unexpected:
            details.append(f"unexpected {sorted(unexpected, key=repr)!r}")
        raise ValueError(f"{path} has invalid keys: " + ", ".join(details))
    return mapping


def _require_layers(value: Any, path: str) -> tuple[Any, ...] | list[Any]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{path} must be a tuple or list, got {type(value).__name__}")
    return value


def _copy_layer_sequence(
    source: tuple[Any, ...] | list[Any],
    layers: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...] | list[dict[str, Any]]:
    return tuple(layers) if isinstance(source, tuple) else layers


def _validate_frozen_module(module: Any, path: str, leaf_name: str) -> None:
    _require_exact_keys(module, frozenset({leaf_name}), path)


def _validate_full_params(params: Any) -> tuple[Mapping[str, Any], tuple[Any, ...] | list[Any]]:
    full = _require_exact_keys(params, _FULL_TOP_LEVEL_KEYS, "params")
    _validate_frozen_module(full["embed_tokens"], "params/embed_tokens", "embedding")
    _validate_frozen_module(full["lm_head"], "params/lm_head", "kernel")
    _validate_frozen_module(full["norm"], "params/norm", "scale")
    layers = _require_layers(full["layers"], "params/layers")
    for index, value in enumerate(layers):
        path = f"params/layers/{index}"
        layer = _require_exact_keys(value, _FULL_LAYER_KEYS, path)
        _require_mapping(layer["attention"], f"{path}/attention")
        _require_mapping(layer["mlp"], f"{path}/mlp")
        _validate_frozen_module(layer["input_layernorm"], f"{path}/input_layernorm", "scale")
        _validate_frozen_module(
            layer["post_attention_layernorm"],
            f"{path}/post_attention_layernorm",
            "scale",
        )
    return full, layers


def split_full_trainable_params(
    params: Mapping[str, Any],
    frozen_components: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split selected fixed-model components out of the trainable tree.

    Supported components are ``embed``, ``lm_head``, and ``norm``. The norm
    component covers the final normalization and both normalizations in every
    decoder layer. Attention and MLP modules always remain full-rank
    trainables. Only containers are rebuilt; every leaf is reused.
    """

    components = frozenset(frozen_components)
    if not components:
        raise ValueError("frozen_components must not be empty")
    unsupported = components - _SUPPORTED_COMPONENTS
    if unsupported:
        raise ValueError(f"unsupported frozen components: {sorted(unsupported)!r}")
    full, layers = _validate_full_params(params)
    frozen_layers = []
    trainable_layers = []
    for layer in layers:
        frozen_layer = {}
        trainable_layer = {
            "attention": layer["attention"],
            "mlp": layer["mlp"],
        }
        for name in _FROZEN_LAYER_KEYS:
            destination = frozen_layer if "norm" in components else trainable_layer
            destination[name] = layer[name]
        frozen_layers.append(frozen_layer)
        trainable_layers.append(trainable_layer)

    frozen_params = {}
    trainable_params = {"layers": _copy_layer_sequence(layers, trainable_layers)}
    component_by_key = {
        "embed_tokens": "embed",
        "norm": "norm",
        "lm_head": "lm_head",
    }
    for key, component in component_by_key.items():
        destination = frozen_params if component in components else trainable_params
        destination[key] = full[key]
    if "norm" in components:
        frozen_params["layers"] = _copy_layer_sequence(layers, frozen_layers)
    return frozen_params, trainable_params


def compose_full_trainable_params(
    frozen_params: Mapping[str, Any],
    trainable_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Reassemble a complete fixed-architecture tree from complementary partitions."""

    frozen = _require_mapping(frozen_params, "frozen_params")
    trainable = _require_mapping(trainable_params, "trainable_params")
    unexpected = (set(frozen) | set(trainable)) - _FULL_TOP_LEVEL_KEYS
    if unexpected:
        raise ValueError(f"split parameter trees have unexpected keys: {sorted(unexpected)!r}")
    top_level_keys = _FULL_TOP_LEVEL_KEYS - {"layers"}
    overlap = (set(frozen) & set(trainable)) & top_level_keys
    if overlap:
        raise ValueError(f"split parameter trees overlap at: {sorted(overlap)!r}")
    missing = top_level_keys - (set(frozen) | set(trainable))
    if missing:
        raise ValueError(f"split parameter trees are missing: {sorted(missing)!r}")

    top_level = {
        key: frozen[key] if key in frozen else trainable[key]
        for key in top_level_keys
    }
    _validate_frozen_module(top_level["embed_tokens"], "params/embed_tokens", "embedding")
    _validate_frozen_module(top_level["lm_head"], "params/lm_head", "kernel")
    _validate_frozen_module(top_level["norm"], "params/norm", "scale")

    frozen_layers = (
        _require_layers(frozen["layers"], "frozen_params/layers")
        if "layers" in frozen
        else None
    )
    trainable_layers = (
        _require_layers(trainable["layers"], "trainable_params/layers")
        if "layers" in trainable
        else None
    )
    if frozen_layers is None and trainable_layers is None:
        raise ValueError("split parameter trees are missing layers")
    if frozen_layers is not None and trainable_layers is not None:
        if isinstance(frozen_layers, tuple) != isinstance(trainable_layers, tuple):
            raise TypeError(
                "frozen_params/layers and trainable_params/layers must use the same sequence type"
            )
        if len(frozen_layers) != len(trainable_layers):
            raise ValueError(
                "frozen_params/layers and trainable_params/layers must contain the same number of layers"
            )
    layer_source = frozen_layers if trainable_layers is None else trainable_layers
    if layer_source is None:
        raise AssertionError("validated layer source is unavailable")

    full_layers = []
    for index in range(len(layer_source)):
        frozen_layer = (
            _require_mapping(frozen_layers[index], f"frozen_params/layers/{index}")
            if frozen_layers is not None
            else {}
        )
        trainable_layer = (
            _require_mapping(trainable_layers[index], f"trainable_params/layers/{index}")
            if trainable_layers is not None
            else {}
        )
        unexpected_layer = (set(frozen_layer) | set(trainable_layer)) - _FULL_LAYER_KEYS
        if unexpected_layer:
            raise ValueError(
                f"split layer {index} has unexpected keys: {sorted(unexpected_layer)!r}"
            )
        overlap_layer = set(frozen_layer) & set(trainable_layer)
        if overlap_layer:
            raise ValueError(f"split layer {index} overlaps at: {sorted(overlap_layer)!r}")
        full_layer = {**frozen_layer, **trainable_layer}
        _require_exact_keys(full_layer, _FULL_LAYER_KEYS, f"params/layers/{index}")
        _require_mapping(full_layer["attention"], f"params/layers/{index}/attention")
        _require_mapping(full_layer["mlp"], f"params/layers/{index}/mlp")
        _validate_frozen_module(
            full_layer["input_layernorm"],
            f"params/layers/{index}/input_layernorm",
            "scale",
        )
        _validate_frozen_module(
            full_layer["post_attention_layernorm"],
            f"params/layers/{index}/post_attention_layernorm",
            "scale",
        )
        full_layers.append(full_layer)

    return {
        "embed_tokens": top_level["embed_tokens"],
        "layers": _copy_layer_sequence(layer_source, full_layers),
        "norm": top_level["norm"],
        "lm_head": top_level["lm_head"],
    }


__all__ = ["compose_full_trainable_params", "split_full_trainable_params"]
