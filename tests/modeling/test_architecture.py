from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

import causal_trainer.modeling.architecture as architecture_module
from causal_trainer.modeling.architecture import decoder_forward, forward, init_params, parameter_shapes
from causal_trainer.modeling.config import ModelConfig
from causal_trainer.modeling.lora import init_lora_params


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


def single_device_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1, 1))
    return Mesh(devices, ("dp", "fsdp", "ep", "tp", "sp"))


def test_parameter_tree_and_forward_shape() -> None:
    config = tiny_config()
    shapes = parameter_shapes(config, jnp.float32)
    assert shapes["layers"][0]["attention"]["k_proj"]["kernel"].shape == (8, 4)
    assert shapes["layers"][0]["attention"]["q_proj"]["bias"].shape == (8,)
    assert "bias" not in shapes["layers"][0]["attention"]["o_proj"]
    params = init_params(config, jax.random.PRNGKey(0), jnp.float32)
    assert "bias" not in params["layers"][0]["attention"]["o_proj"]
    input_ids = jnp.asarray([[1, 2, 3, 4], [5, 6, 0, 0]], jnp.int32)
    attention_mask = input_ids != 0
    logits = forward(
        params,
        config,
        input_ids,
        attention_mask=attention_mask,
        implementation="vanilla",
        compute_dtype=jnp.float32,
    )
    assert logits.shape == (2, 4, 32)
    assert jnp.all(jnp.isfinite(logits))


def test_embedding_padding_index_matches_transformers_initialization_and_gradient() -> None:
    config = tiny_config()
    params = init_params(config, jax.random.PRNGKey(3), jnp.float32)
    embedding = params["embed_tokens"]["embedding"]
    assert jnp.array_equal(embedding[config.pad_token_id], jnp.zeros(config.hidden_size))

    # A loaded checkpoint is allowed to contain a nonzero padding row, but
    # nn.Embedding(padding_idx=...) still suppresses its gradient.
    loaded_embedding = embedding.at[config.pad_token_id].set(jnp.arange(config.hidden_size, dtype=jnp.float32))

    def loss(table):
        replaced = {**params, "embed_tokens": {"embedding": table}}
        hidden = decoder_forward(
            replaced,
            config,
            jnp.asarray([[config.pad_token_id]], dtype=jnp.int32),
            attention_mask=jnp.ones((1, 1), dtype=jnp.bool_),
            implementation="vanilla",
            compute_dtype=jnp.float32,
        )
        return jnp.sum(hidden)

    gradient = jax.grad(loss)(loaded_embedding)
    assert jnp.array_equal(gradient[config.pad_token_id], jnp.zeros(config.hidden_size))


def test_forward_keeps_packed_samples_independent() -> None:
    config = tiny_config()
    params = init_params(config, jax.random.PRNGKey(7), jnp.float32)
    valid = jnp.ones((1, 4), dtype=jnp.bool_)
    segments = jnp.asarray([[1, 1, 2, 2]], dtype=jnp.int32)

    baseline = forward(
        params,
        config,
        jnp.asarray([[1, 2, 3, 4]], jnp.int32),
        attention_mask=valid,
        segment_ids=segments,
        implementation="vanilla",
        compute_dtype=jnp.float32,
    )
    first_sample_changed = forward(
        params,
        config,
        jnp.asarray([[8, 9, 3, 4]], jnp.int32),
        attention_mask=valid,
        segment_ids=segments,
        implementation="vanilla",
        compute_dtype=jnp.float32,
    )
    second_sample_changed = forward(
        params,
        config,
        jnp.asarray([[1, 2, 10, 11]], jnp.int32),
        attention_mask=valid,
        segment_ids=segments,
        implementation="vanilla",
        compute_dtype=jnp.float32,
    )

    # The second packed sample cannot see keys/values from the first, and the
    # first cannot see the second (also guaranteed causally).
    assert jnp.allclose(first_sample_changed[:, 2:], baseline[:, 2:])
    assert jnp.allclose(second_sample_changed[:, :2], baseline[:, :2])


def test_decoder_precomputes_segments_and_rotary_values_once(monkeypatch) -> None:
    config = replace(tiny_config(), rope_style="gpt-neox")
    params = init_params(config, jax.random.PRNGKey(11), jnp.float32)
    input_ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    valid = jnp.ones_like(input_ids, dtype=jnp.bool_)
    raw_segments = jnp.asarray([[0, 0, 7, 7]], dtype=jnp.int32)
    # Explicit reset + jump models the logical-position contract used by
    # packed examples and EndPrompt without creating a dense position cache.
    positions = jnp.asarray([[0, 1, 0, 31]], dtype=jnp.int32)
    calls = {"segments": 0, "rotary": 0}
    seen_positions = []
    seen_rope_styles = []

    original_canonicalize = architecture_module.canonicalize_segment_ids
    original_rotary = architecture_module.rotary_cos_sin
    original_apply_rotary = architecture_module.apply_rotary_cos_sin

    def counted_canonicalize(*args, **kwargs):
        calls["segments"] += 1
        return original_canonicalize(*args, **kwargs)

    def counted_rotary(position_ids, *args, **kwargs):
        calls["rotary"] += 1
        seen_positions.append(position_ids)
        return original_rotary(position_ids, *args, **kwargs)

    def counted_apply_rotary(*args, **kwargs):
        seen_rope_styles.append(kwargs.get("rope_style"))
        return original_apply_rotary(*args, **kwargs)

    monkeypatch.setattr(architecture_module, "canonicalize_segment_ids", counted_canonicalize)
    monkeypatch.setattr(architecture_module, "rotary_cos_sin", counted_rotary)
    monkeypatch.setattr(architecture_module, "apply_rotary_cos_sin", counted_apply_rotary)
    output = decoder_forward(
        params,
        config,
        input_ids,
        attention_mask=valid,
        position_ids=positions,
        segment_ids=raw_segments,
        implementation="vanilla",
        compute_dtype=jnp.float32,
        remat_policy="nothing_saveable",
    )

    assert output.shape == (1, 4, config.hidden_size)
    assert calls == {"segments": 1, "rotary": 1}
    assert len(seen_positions) == 1
    assert jnp.array_equal(seen_positions[0], positions)
    assert seen_rope_styles
    assert set(seen_rope_styles) == {"gpt-neox"}


def test_tiled_mlp_matches_dense_path_and_preserves_lora_gradients() -> None:
    config = tiny_config()
    params = init_params(config, jax.random.PRNGKey(19), jnp.float32)
    adapters = init_lora_params(config, jax.random.PRNGKey(20), rank=2, dtype=jnp.float32)
    # Make B nonzero so every adapter path, including separate gate/up paths,
    # contributes to both the value and gradient comparison.
    adapters = jax.tree.map(lambda value: value + jnp.asarray(0.02, value.dtype), adapters)
    input_ids = jnp.asarray([[1, 2, 3, 4, 5]], dtype=jnp.int32)
    valid = jnp.ones_like(input_ids, dtype=jnp.bool_)
    segments = jnp.asarray([[1, 1, 2, 2, 2]], dtype=jnp.int32)

    def hidden(adapter_params, chunk_size):
        return decoder_forward(
            params,
            config,
            input_ids,
            attention_mask=valid,
            segment_ids=segments,
            implementation="vanilla",
            lora_params=adapter_params,
            compute_dtype=jnp.float32,
            mlp_chunk_size=chunk_size,
        )

    dense = hidden(adapters, 0)
    tiled = hidden(adapters, 2)
    assert jnp.allclose(tiled, dense, rtol=1e-5, atol=1e-5)

    dense_grad = jax.grad(lambda value: jnp.sum(hidden(value, 0)))(adapters)
    tiled_grad = jax.grad(lambda value: jnp.sum(hidden(value, 2)))(adapters)
    for actual, expected in zip(
        jax.tree.leaves(tiled_grad),
        jax.tree.leaves(dense_grad),
        strict=True,
    ):
        assert jnp.allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_layer_scan_matches_unrolled_forward_and_external_parameter_gradients() -> None:
    config = tiny_config()
    params = init_params(config, jax.random.PRNGKey(27), jnp.float32)
    input_ids = jnp.asarray([[1, 2, 3, 4, 5]], dtype=jnp.int32)
    valid = jnp.ones_like(input_ids, dtype=jnp.bool_)
    segments = jnp.asarray([[1, 1, 2, 2, 2]], dtype=jnp.int32)
    coefficients = jnp.arange(40, dtype=jnp.float32).reshape(1, 5, 8) / 40.0

    def loss(parameter_tree, scan_layers):
        hidden = decoder_forward(
            parameter_tree,
            config,
            input_ids,
            attention_mask=valid,
            segment_ids=segments,
            implementation="vanilla",
            remat_policy="nothing_saveable",
            compute_dtype=jnp.float32,
            mlp_chunk_size=2,
            scan_layers=scan_layers,
        )
        return jnp.sum(hidden * coefficients), hidden

    (unrolled_loss, unrolled), unrolled_grad = jax.value_and_grad(
        lambda value: loss(value, False),
        has_aux=True,
    )(params)
    (scanned_loss, scanned), scanned_grad = jax.value_and_grad(
        lambda value: loss(value, True),
        has_aux=True,
    )(params)

    assert jnp.allclose(scanned, unrolled, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(scanned_loss, unrolled_loss, rtol=1e-6, atol=1e-6)
    assert jax.tree.structure(scanned_grad) == jax.tree.structure(params)
    assert isinstance(scanned_grad["layers"], tuple)
    for actual, expected in zip(
        jax.tree.leaves(scanned_grad),
        jax.tree.leaves(unrolled_grad),
        strict=True,
    ):
        assert jnp.allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_layer_scan_matches_lora_gradient_with_per_layer_dropout_rngs() -> None:
    config = replace(tiny_config(), attention_dropout=0.2)
    params = init_params(config, jax.random.PRNGKey(28), jnp.float32)
    adapters = init_lora_params(config, jax.random.PRNGKey(29), rank=2, dtype=jnp.float32)
    adapters = jax.tree.map(lambda value: value + jnp.asarray(0.02, value.dtype), adapters)
    input_ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    dropout_rng = jax.random.PRNGKey(30)

    def loss(adapter_tree, scan_layers):
        hidden = decoder_forward(
            params,
            config,
            input_ids,
            implementation="vanilla",
            deterministic=False,
            dropout_rng=dropout_rng,
            lora_params=adapter_tree,
            remat_policy="nothing_saveable",
            compute_dtype=jnp.float32,
            scan_layers=scan_layers,
        )
        return jnp.sum(hidden), hidden

    (unrolled_loss, unrolled), unrolled_grad = jax.value_and_grad(
        lambda value: loss(value, False),
        has_aux=True,
    )(adapters)
    (scanned_loss, scanned), scanned_grad = jax.value_and_grad(
        lambda value: loss(value, True),
        has_aux=True,
    )(adapters)

    assert jnp.allclose(scanned, unrolled, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(scanned_loss, unrolled_loss, rtol=1e-6, atol=1e-6)
    assert jax.tree.structure(scanned_grad) == jax.tree.structure(adapters)
    assert isinstance(scanned_grad["layers"], tuple)
    for actual, expected in zip(
        jax.tree.leaves(scanned_grad),
        jax.tree.leaves(unrolled_grad),
        strict=True,
    ):
        assert jnp.allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_gathered_kv_projection_weights_preserve_lora_values_and_gradients() -> None:
    config = tiny_config()
    params = init_params(config, jax.random.PRNGKey(21), jnp.float32)
    adapters = init_lora_params(config, jax.random.PRNGKey(22), rank=2, dtype=jnp.float32)
    adapters = jax.tree.map(lambda value: value + jnp.asarray(0.02, value.dtype), adapters)
    input_ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    mesh = single_device_mesh()

    def hidden(adapter_params, gather):
        return decoder_forward(
            params,
            config,
            input_ids,
            implementation="vanilla",
            mesh=mesh,
            lora_params=adapter_params,
            compute_dtype=jnp.float32,
            gather_kv_projection_weights=gather,
        )

    with jax.set_mesh(mesh):
        baseline = jax.jit(lambda value: hidden(value, False))(adapters)
        gathered = jax.jit(lambda value: hidden(value, True))(adapters)
        baseline_grad = jax.jit(jax.grad(lambda value: jnp.sum(hidden(value, False))))(adapters)
        gathered_grad = jax.jit(jax.grad(lambda value: jnp.sum(hidden(value, True))))(adapters)

    assert jnp.allclose(gathered, baseline, rtol=1e-5, atol=1e-5)
    for actual, expected in zip(
        jax.tree.leaves(gathered_grad),
        jax.tree.leaves(baseline_grad),
        strict=True,
    ):
        assert jnp.allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_gathered_kv_projection_constrains_only_small_output_weights(monkeypatch) -> None:
    config = tiny_config()
    params = init_params(config, jax.random.PRNGKey(24), jnp.float32)
    adapters = init_lora_params(config, jax.random.PRNGKey(25), rank=2, dtype=jnp.float32)
    input_ids = jnp.asarray([[1, 2]], dtype=jnp.int32)
    mesh = single_device_mesh()
    constrained: list[tuple[tuple[int, ...], PartitionSpec]] = []

    def record_constraint(value, sharding):
        constrained.append((tuple(value.shape), sharding.spec))
        return value

    monkeypatch.setattr(architecture_module.jax.lax, "with_sharding_constraint", record_constraint)
    decoder_forward(
        params,
        config,
        input_ids,
        implementation="vanilla",
        mesh=mesh,
        lora_params=adapters,
        compute_dtype=jnp.float32,
        gather_kv_projection_weights=True,
    )

    assert constrained == [
        item
        for _ in range(config.num_hidden_layers)
        for item in (
            ((config.hidden_size, config.key_value_width), PartitionSpec("fsdp", None)),
            ((2, config.key_value_width), PartitionSpec("fsdp", None)),
            ((config.hidden_size, config.key_value_width), PartitionSpec("fsdp", None)),
            ((2, config.key_value_width), PartitionSpec("fsdp", None)),
        )
    ]


def test_gathered_kv_projection_requires_a_mesh_and_static_bool() -> None:
    config = tiny_config()
    params = init_params(config, jax.random.PRNGKey(26), jnp.float32)
    input_ids = jnp.asarray([[1, 2]], dtype=jnp.int32)

    with pytest.raises(ValueError, match="requires a mesh"):
        decoder_forward(
            params,
            config,
            input_ids,
            implementation="vanilla",
            compute_dtype=jnp.float32,
            gather_kv_projection_weights=True,
        )
    with pytest.raises(TypeError, match="static bool"):
        decoder_forward(
            params,
            config,
            input_ids,
            implementation="vanilla",
            mesh=single_device_mesh(),
            compute_dtype=jnp.float32,
            gather_kv_projection_weights=1,
        )


def test_mlp_chunk_size_is_static_and_non_negative() -> None:
    config = tiny_config()
    params = init_params(config, jax.random.PRNGKey(23), jnp.float32)
    input_ids = jnp.asarray([[1, 2]], dtype=jnp.int32)
    with pytest.raises(ValueError, match="non-negative"):
        decoder_forward(
            params,
            config,
            input_ids,
            implementation="vanilla",
            compute_dtype=jnp.float32,
            mlp_chunk_size=-1,
        )
    with pytest.raises(TypeError, match="static non-negative integer"):
        decoder_forward(
            params,
            config,
            input_ids,
            implementation="vanilla",
            compute_dtype=jnp.float32,
            mlp_chunk_size=True,
        )
    with pytest.raises(TypeError, match="scan_layers must be a static bool"):
        decoder_forward(
            params,
            config,
            input_ids,
            implementation="vanilla",
            compute_dtype=jnp.float32,
            scan_layers=1,
        )
