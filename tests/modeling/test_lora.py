import json
import math

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from safetensors.numpy import load_file

from causal_trainer.checkpointing.huggingface import export_adapter_checkpoint
from causal_trainer.modeling.config import ModelConfig
from causal_trainer.modeling.lora import (
    LORA_PROJECTIONS,
    adapter_for_kernel_path,
    compose_lora_export_params,
    init_lora_params,
    lora_adapter_params,
    lora_export_plan,
    lora_linear_delta,
    lora_parameter_shapes,
    lora_parameter_to_peft_mapping,
    lora_partition_specs,
    make_lora_export_transform,
    merge_lora_kernel,
    peft_adapter_config,
    split_lora_trainable_params,
)


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        partial_rotary_factor=0.5,
        max_position_embeddings=32,
        pad_token_id=0,
        eos_token_id=2,
        rope_theta=10_000.0,
    )


def test_lora_parameter_shapes_cover_exact_projection_set() -> None:
    shapes = lora_parameter_shapes(tiny_config(), rank=3, dtype=jnp.float32)
    assert LORA_PROJECTIONS == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    assert len(shapes["layers"]) == 2
    layer = shapes["layers"][0]
    assert tuple(layer["attention"]) == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert tuple(layer["mlp"]) == ("gate_proj", "up_proj", "down_proj")
    assert layer["attention"]["q_proj"]["lora_a"].shape == (8, 3)
    assert layer["attention"]["q_proj"]["lora_b"].shape == (3, 8)
    assert layer["attention"]["k_proj"]["lora_b"].shape == (3, 4)
    assert layer["mlp"]["down_proj"]["lora_a"].shape == (16, 3)
    assert layer["mlp"]["down_proj"]["lora_b"].shape == (3, 8)
    assert layer["mlp"]["down_proj"]["lora_b"].dtype == jnp.dtype(jnp.float32)


def test_lora_initialization_has_he_uniform_a_and_zero_b() -> None:
    config = tiny_config()
    params = init_lora_params(config, jax.random.PRNGKey(5), rank=3, dtype=jnp.float32)
    for layer in params["layers"]:
        adapters = (*layer["attention"].values(), *layer["mlp"].values())
        for adapter in adapters:
            assert jnp.all(adapter["lora_b"] == 0)

    query_a = params["layers"][0]["attention"]["q_proj"]["lora_a"]
    limit = math.sqrt(6.0 / config.hidden_size)
    assert jnp.all(query_a >= -limit)
    assert jnp.all(query_a <= limit)
    assert jnp.any(query_a != 0)


def test_lora_delta_and_kernel_merge_are_unscaled() -> None:
    adapter = {
        "lora_a": jnp.asarray([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]], jnp.float32),
        "lora_b": jnp.asarray([[2.0, -3.0], [0.25, 5.0]], jnp.float32),
    }
    x = jnp.asarray([[2.0, -1.0, 0.5], [-3.0, 4.0, 1.0]], jnp.float32)
    expected_delta = (x @ adapter["lora_a"]) @ adapter["lora_b"]
    assert jnp.allclose(lora_linear_delta(x, adapter, jnp.float32), expected_delta)

    kernel = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    expected_kernel = kernel + adapter["lora_a"] @ adapter["lora_b"]
    merged = merge_lora_kernel(kernel, adapter)
    assert merged.dtype == kernel.dtype
    assert jnp.allclose(merged, expected_kernel)


def test_adapter_lookup_uses_neutral_kernel_paths() -> None:
    adapters = init_lora_params(tiny_config(), jax.random.PRNGKey(3), rank=2, dtype=jnp.float32)
    expected = adapters["layers"][1]["mlp"]["down_proj"]
    assert adapter_for_kernel_path(adapters, "layers/1/mlp/down_proj/kernel") is expected
    assert adapter_for_kernel_path(
        adapters,
        (
            jax.tree_util.DictKey("layers"),
            jax.tree_util.SequenceKey(1),
            jax.tree_util.DictKey("mlp"),
            jax.tree_util.DictKey("down_proj"),
            jax.tree_util.DictKey("kernel"),
        ),
    ) is expected
    assert adapter_for_kernel_path(adapters, "model/layers/1/mlp/down_proj/kernel") is expected
    assert adapter_for_kernel_path(adapters, "layers/1/mlp/down_proj/bias") is None
    assert adapter_for_kernel_path(adapters, "layers/1/input_layernorm/kernel") is None
    assert adapter_for_kernel_path(adapters, "layers/9/mlp/down_proj/kernel") is None


def test_lora_partition_specs_match_projection_orientation() -> None:
    shapes = lora_parameter_shapes(tiny_config(), rank=3, dtype=jnp.bfloat16)
    specs = lora_partition_specs(shapes)
    layer = specs["layers"][0]
    for name in ("q_proj", "k_proj", "v_proj"):
        assert layer["attention"][name]["lora_a"] == P()
        assert layer["attention"][name]["lora_b"] == P("fsdp", "tp")
    for name in ("gate_proj", "up_proj"):
        assert layer["mlp"][name]["lora_a"] == P()
        assert layer["mlp"][name]["lora_b"] == P("fsdp", "tp")
    assert layer["attention"]["o_proj"]["lora_a"] == P("tp", "fsdp")
    assert layer["attention"]["o_proj"]["lora_b"] == P()
    assert layer["mlp"]["down_proj"]["lora_a"] == P("tp", "fsdp")
    assert layer["mlp"]["down_proj"]["lora_b"] == P()


def test_lora_export_plan_is_stable_and_has_no_scale() -> None:
    assert lora_export_plan(256) == {
        "kind": "lora-merge",
        "rank": 256,
        "targets": list(LORA_PROJECTIONS),
        "train_embed_and_lm_head": False,
    }


def test_peft_adapter_mapping_uses_standard_keys_and_transposes() -> None:
    query_a = lora_parameter_to_peft_mapping("layers/3/attention/q_proj/lora_a")
    gate_a = lora_parameter_to_peft_mapping("layers/3/mlp/gate_proj/lora_a")
    up_b = lora_parameter_to_peft_mapping("layers/3/mlp/up_proj/lora_b")
    down_b = lora_parameter_to_peft_mapping("adapters/layers/4/mlp/down_proj/lora_b")
    embedding = lora_parameter_to_peft_mapping("embed_tokens/embedding")
    lm_head = lora_parameter_to_peft_mapping("lm_head/kernel")

    assert query_a.hf_key == (
        "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight"
    )
    assert query_a.transpose is True
    assert gate_a.hf_key == (
        "base_model.model.model.layers.3.mlp.gate_proj.lora_A.weight"
    )
    assert gate_a.transpose is True
    assert up_b.hf_key == (
        "base_model.model.model.layers.3.mlp.up_proj.lora_B.weight"
    )
    assert up_b.transpose is True
    assert down_b.hf_key == (
        "base_model.model.model.layers.4.mlp.down_proj.lora_B.weight"
    )
    assert down_b.transpose is True
    assert embedding.hf_key == "base_model.model.model.embed_tokens.weight"
    assert embedding.transpose is False
    assert lm_head.hf_key == "base_model.model.lm_head.weight"
    assert lm_head.transpose is True


def test_peft_config_preserves_unscaled_adapter_math() -> None:
    config = peft_adapter_config(
        "owner/base",
        256,
        train_embed_and_lm_head=True,
        revision="stable",
    )
    assert config["r"] == config["lora_alpha"] == 256
    assert config["target_modules"] == list(LORA_PROJECTIONS)
    assert config["modules_to_save"] == ["embed_tokens", "lm_head"]
    assert config["base_model_name_or_path"] == "owner/base"
    assert config["revision"] == "stable"


def test_embedding_enabled_lora_moves_full_rank_leaves_without_copying_them() -> None:
    embedding = {"embedding": object()}
    lm_head = {"kernel": object()}
    layers = (object(),)
    norm = {"scale": object()}
    adapters = {"layers": (object(),)}
    base = {
        "embed_tokens": embedding,
        "layers": layers,
        "norm": norm,
        "lm_head": lm_head,
    }

    frozen, trainable = split_lora_trainable_params(
        base,
        adapters,
        train_embed_and_lm_head=True,
    )

    assert set(frozen) == {"layers", "norm"}
    assert trainable["embed_tokens"] is embedding
    assert trainable["lm_head"] is lm_head
    assert lora_adapter_params(trainable) is adapters
    composed = compose_lora_export_params(frozen, trainable)
    assert composed["embed_tokens"] is embedding
    assert composed["lm_head"] is lm_head
    assert composed["layers"] is layers
    assert composed["norm"] is norm
    assert "adapters" not in composed
    assert lora_export_plan(8, train_embed_and_lm_head=True)["train_embed_and_lm_head"] is True


def test_lora_export_transform_merges_targets_and_leaves_other_parameters_untouched() -> None:
    adapters = init_lora_params(tiny_config(), jax.random.PRNGKey(8), rank=2, dtype=jnp.float32)
    first_adapter = adapters["layers"][0]["attention"]["q_proj"]
    second_adapter = adapters["layers"][1]["attention"]["q_proj"]
    first_adapter["lora_b"] = jnp.ones_like(first_adapter["lora_b"])
    second_adapter["lora_b"] = jnp.full_like(second_adapter["lora_b"], 2.0)
    base = jnp.arange(64, dtype=jnp.float32).reshape(8, 8)

    transform = make_lora_export_transform(adapters)
    first = transform("layers/0/attention/q_proj/kernel", base)
    second = transform("model/layers/1/attention/q_proj/kernel", base)
    assert jnp.allclose(first, merge_lora_kernel(base, first_adapter))
    assert jnp.allclose(second, merge_lora_kernel(base, second_adapter))
    untouched = jnp.ones((8,), dtype=jnp.float32)
    assert transform("norm/scale", untouched) is untouched


def test_embedding_enabled_adapter_export_is_peft_compatible_and_single_file(tmp_path) -> None:
    config = tiny_config()
    adapters = init_lora_params(config, jax.random.PRNGKey(13), rank=2, dtype=jnp.float32)
    trainable = {
        "adapters": adapters,
        "embed_tokens": {
            "embedding": jnp.arange(config.vocab_size * config.hidden_size, dtype=jnp.float32).reshape(
                config.vocab_size,
                config.hidden_size,
            )
        },
        "lm_head": {
            "kernel": jnp.arange(config.hidden_size * config.vocab_size, dtype=jnp.float32).reshape(
                config.hidden_size,
                config.vocab_size,
            )
        },
    }
    adapter_config = peft_adapter_config(
        "owner/base",
        2,
        train_embed_and_lm_head=True,
        revision="stable",
    )

    destination = export_adapter_checkpoint(
        trainable,
        tmp_path / "adapter",
        adapter_config=adapter_config,
        mapping_fn=lora_parameter_to_peft_mapping,
        checkpoint_id="d" * 32,
    )

    weights = load_file(destination / "adapter_model.safetensors")
    assert len(weights) == config.num_hidden_layers * len(LORA_PROJECTIONS) * 2 + 2
    assert weights[
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    ].shape == (2, config.hidden_size)
    assert weights[
        "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight"
    ].shape == (config.key_value_width, 2)
    assert weights["base_model.model.model.embed_tokens.weight"].shape == (
        config.vocab_size,
        config.hidden_size,
    )
    assert weights["base_model.model.lm_head.weight"].shape == (
        config.vocab_size,
        config.hidden_size,
    )
    assert json.loads((destination / "adapter_config.json").read_text(encoding="utf-8")) == adapter_config
    assert not list(destination.glob("model*.safetensors"))
    assert not (destination / "model.safetensors.index.json").exists()
