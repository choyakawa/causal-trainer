from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from causal_trainer.modeling.freezing import (
    compose_full_trainable_params,
    split_full_trainable_params,
)


def _projection(*, bias: bool = False) -> dict[str, object]:
    projection = {"kernel": object()}
    if bias:
        projection["bias"] = object()
    return projection


def _layer() -> dict[str, Any]:
    return {
        "attention": {
            "q_proj": _projection(bias=True),
            "k_proj": _projection(bias=True),
            "v_proj": _projection(bias=True),
            "o_proj": _projection(),
        },
        "mlp": {
            "gate_proj": _projection(),
            "up_proj": _projection(),
            "down_proj": _projection(),
        },
        "input_layernorm": {"scale": object()},
        "post_attention_layernorm": {"scale": object()},
    }


def _params(*, layer_container: type[tuple] | type[list] = tuple) -> dict[str, Any]:
    layers = [_layer(), _layer()]
    return {
        "embed_tokens": {"embedding": object()},
        "layers": layer_container(layers),
        "norm": {"scale": object()},
        "lm_head": {"kernel": object()},
    }


def _leaves_by_path(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            result.update(_leaves_by_path(child, (*prefix, str(key))))
        return result
    if isinstance(value, (tuple, list)):
        result = {}
        for index, child in enumerate(value):
            result.update(_leaves_by_path(child, (*prefix, str(index))))
        return result
    return {prefix: value}


@pytest.mark.parametrize("layer_container", [tuple, list])
def test_split_paths_are_disjoint_and_compose_restores_every_leaf_by_identity(
    layer_container: type[tuple] | type[list],
) -> None:
    params = _params(layer_container=layer_container)
    original = _leaves_by_path(params)

    frozen, trainable = split_full_trainable_params(params, ("lm_head", "embed", "norm"))

    frozen_leaves = _leaves_by_path(frozen)
    trainable_leaves = _leaves_by_path(trainable)
    expected_frozen = {
        ("embed_tokens", "embedding"),
        ("norm", "scale"),
        ("lm_head", "kernel"),
    }
    for index in range(2):
        expected_frozen.add(("layers", str(index), "input_layernorm", "scale"))
        expected_frozen.add(("layers", str(index), "post_attention_layernorm", "scale"))
    assert set(frozen_leaves) == expected_frozen
    assert set(frozen_leaves).isdisjoint(trainable_leaves)
    assert set(frozen_leaves) | set(trainable_leaves) == set(original)
    assert isinstance(frozen["layers"], layer_container)
    assert isinstance(trainable["layers"], layer_container)

    composed = compose_full_trainable_params(frozen, trainable)

    restored = _leaves_by_path(composed)
    assert type(composed["layers"]) is type(params["layers"])
    assert jax.tree_util.tree_structure(composed) == jax.tree_util.tree_structure(params)
    assert set(restored) == set(original)
    for path, leaf in original.items():
        assert restored[path] is leaf
    assert set(params["layers"][0]) == {
        "attention",
        "mlp",
        "input_layernorm",
        "post_attention_layernorm",
    }


@pytest.mark.parametrize(
    ("components", "expected_frozen"),
    [
        (("lm_head",), {("lm_head", "kernel")}),
        (("embed",), {("embed_tokens", "embedding")}),
        (
            ("norm",),
            {
                ("norm", "scale"),
                ("layers", "0", "input_layernorm", "scale"),
                ("layers", "0", "post_attention_layernorm", "scale"),
                ("layers", "1", "input_layernorm", "scale"),
                ("layers", "1", "post_attention_layernorm", "scale"),
            },
        ),
    ],
)
def test_split_freezes_only_requested_components(
    components: tuple[str, ...],
    expected_frozen: set[tuple[str, ...]],
) -> None:
    params = _params()
    frozen, trainable = split_full_trainable_params(params, components)

    assert set(_leaves_by_path(frozen)) == expected_frozen
    assert set(_leaves_by_path(frozen)).isdisjoint(_leaves_by_path(trainable))
    composed = compose_full_trainable_params(frozen, trainable)
    for path, leaf in _leaves_by_path(params).items():
        assert _leaves_by_path(composed)[path] is leaf


def test_split_and_compose_treat_jax_metadata_and_arrays_as_opaque_leaves() -> None:
    params = _params()
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("data",))
    values = {
        "embedding": jnp.ones((2, 3), dtype=jnp.bfloat16),
        "head": jax.ShapeDtypeStruct((3, 5), jnp.bfloat16),
        "final_norm": PartitionSpec(None),
        "layer_norm": NamedSharding(mesh, PartitionSpec()),
        "projection": PartitionSpec("data", None),
    }
    params["embed_tokens"]["embedding"] = values["embedding"]
    params["lm_head"]["kernel"] = values["head"]
    params["norm"]["scale"] = values["final_norm"]
    params["layers"][0]["input_layernorm"]["scale"] = values["layer_norm"]
    params["layers"][0]["attention"]["q_proj"]["kernel"] = values["projection"]

    frozen, trainable = split_full_trainable_params(params, ("lm_head", "embed", "norm"))
    composed = compose_full_trainable_params(frozen, trainable)

    assert frozen["embed_tokens"]["embedding"] is values["embedding"]
    assert frozen["lm_head"]["kernel"] is values["head"]
    assert frozen["norm"]["scale"] is values["final_norm"]
    assert frozen["layers"][0]["input_layernorm"]["scale"] is values["layer_norm"]
    assert trainable["layers"][0]["attention"]["q_proj"]["kernel"] is values["projection"]
    assert composed["embed_tokens"]["embedding"] is values["embedding"]
    assert composed["layers"][0]["attention"]["q_proj"]["kernel"] is values["projection"]


def test_split_rejects_missing_unexpected_and_malformed_structure() -> None:
    missing = _params()
    missing.pop("norm")
    with pytest.raises(ValueError, match=r"params has invalid keys: missing \['norm'\]"):
        split_full_trainable_params(missing, ("norm",))

    unexpected = _params()
    unexpected["extra"] = {"kernel": object()}
    with pytest.raises(ValueError, match="unexpected"):
        split_full_trainable_params(unexpected, ("norm",))

    malformed_layer = _params()
    malformed_layer["layers"][0].pop("post_attention_layernorm")
    with pytest.raises(ValueError, match="post_attention_layernorm"):
        split_full_trainable_params(malformed_layer, ("norm",))

    wrong_layers = _params()
    wrong_layers["layers"] = {"0": _layer()}
    with pytest.raises(TypeError, match="must be a tuple or list"):
        split_full_trainable_params(wrong_layers, ("norm",))

    with pytest.raises(ValueError, match="must not be empty"):
        split_full_trainable_params(_params(), ())
    with pytest.raises(ValueError, match="unsupported frozen components"):
        split_full_trainable_params(_params(), ("attention",))


def test_compose_rejects_overlapping_or_mismatched_partitions() -> None:
    frozen, trainable = split_full_trainable_params(
        _params(),
        ("lm_head", "embed", "norm"),
    )
    frozen["layers"][0]["attention"] = {"q_proj": _projection()}
    with pytest.raises(ValueError, match=r"overlaps.*attention"):
        compose_full_trainable_params(frozen, trainable)

    frozen, trainable = split_full_trainable_params(
        _params(),
        ("lm_head", "embed", "norm"),
    )
    trainable["layers"] = trainable["layers"][:-1]
    with pytest.raises(ValueError, match="same number of layers"):
        compose_full_trainable_params(frozen, trainable)

    frozen, trainable = split_full_trainable_params(
        _params(),
        ("lm_head", "embed", "norm"),
    )
    trainable["layers"] = list(trainable["layers"])
    with pytest.raises(TypeError, match="same sequence type"):
        compose_full_trainable_params(frozen, trainable)
