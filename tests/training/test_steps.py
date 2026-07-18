import jax.numpy as jnp
import optax

from causal_trainer.training import steps


def test_frozen_full_step_uses_frozen_head_contract_and_updates_only_trainables(
    monkeypatch,
) -> None:
    compose_calls: list[bool] = []

    def fake_compose(frozen_params, trainable_params):
        compose_calls.append(True)
        return {
            "body": trainable_params["body"],
            "lm_head": frozen_params["lm_head"],
        }

    def fake_decoder(params, _config, input_ids, **_kwargs):
        return jnp.broadcast_to(params["body"], (*input_ids.shape, 1))

    def fake_loss(
        hidden_states,
        lm_head_kernel,
        _input_ids,
        _attention_mask,
        *,
        lm_head_trainable,
        **_kwargs,
    ):
        assert lm_head_trainable is False
        nll_sum = jnp.sum(hidden_states) * lm_head_kernel[0, 0]
        return nll_sum, jnp.asarray(hidden_states.shape[1], jnp.float32)

    monkeypatch.setattr(steps, "compose_full_trainable_params", fake_compose)
    monkeypatch.setattr(steps, "decoder_forward", fake_decoder)
    monkeypatch.setattr(steps, "chunked_causal_lm_loss", fake_loss)

    trainable_params = {"body": jnp.asarray([1.0], jnp.float32)}
    frozen_params = {"lm_head": {"kernel": jnp.asarray([[2.0]], jnp.float32)}}
    optimizer = optax.sgd(0.1)
    optimizer_state = optimizer.init(trainable_params)
    train_step = steps.make_frozen_full_train_step(
        object(),
        None,
        optimizer,
        lambda _step: jnp.asarray(0.1, jnp.float32),
        attention_implementation="vanilla",
        gradient_accumulation_steps=1,
        remat_policy="none",
        compute_dtype=jnp.float32,
        block_q=1,
        block_k=1,
        loss_token_budget=1,
        loss_implementation="xla",
        mlp_chunk_size=0,
        sparse_loss_skip=False,
        frozen_param_shardings=None,
        trainable_param_shardings=None,
        optimizer_state_shardings=None,
        batch_named_sharding=None,
        replicated_named_sharding=None,
        lm_head_trainable=False,
    )
    batch = {
        "input_ids": jnp.asarray([[1, 2]], jnp.int32),
        "attention_mask": jnp.ones((1, 2), jnp.bool_),
        "position_ids": jnp.asarray([[0, 1]], jnp.int32),
        "segment_ids": jnp.zeros((1, 2), jnp.int32),
        "loss_weights": jnp.ones((1, 2), jnp.float32),
    }

    updated, _, metrics = train_step(
        frozen_params,
        trainable_params,
        optimizer_state,
        batch,
        jnp.asarray(0, jnp.int32),
    )

    assert compose_calls
    assert jnp.allclose(updated["body"], jnp.asarray([0.8], jnp.float32))
    assert jnp.allclose(metrics["loss"], 2.0)
    assert jnp.allclose(metrics["grad_norm"], 2.0)
