import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from causal_trainer.distributed.runtime import MESH_AXIS_NAMES
from causal_trainer.training.optimizer import build_learning_rate_schedule, build_optimizer
from causal_trainer.training.steps import (
    infer_optimizer_state_shardings,
    initialize_optimizer,
    optimizer_state_template,
)


def test_cosine_end_is_an_absolute_learning_rate() -> None:
    schedule = build_learning_rate_schedule("cosine", 2e-5, 5e-6, 100)
    assert jnp.allclose(schedule(0), 2e-5)
    assert jnp.allclose(schedule(100), 5e-6)


def test_external_learning_rate_leaves_adamw_updates_unscaled() -> None:
    params = {"kernel": jnp.ones((1, 1), dtype=jnp.float32)}
    gradients = {"kernel": jnp.ones((1, 1), dtype=jnp.float32)}

    def schedule(_position):
        return jnp.asarray(0.25, dtype=jnp.float32)

    options = {
        "weight_decay": 0.0,
        "max_grad_norm": 0.0,
        "beta1": 0.0,
        "beta2": 0.0,
        "epsilon": 1e-8,
    }
    internal = build_optimizer(params, schedule, **options)
    external = build_optimizer(
        params,
        schedule,
        external_learning_rate=True,
        **options,
    )

    internal_updates, _ = internal.update(gradients, internal.init(params), params)
    external_updates, _ = external.update(gradients, external.init(params), params)

    np.testing.assert_allclose(
        external_updates["kernel"] * schedule(0),
        internal_updates["kernel"],
    )


def test_optimizer_template_and_shardings_do_not_require_concrete_moments() -> None:
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1, 1))
    mesh = Mesh(devices, MESH_AXIS_NAMES)
    replicated = NamedSharding(mesh, PartitionSpec())
    params = {"kernel": jax.device_put(jnp.ones((2, 3), dtype=jnp.bfloat16), replicated)}
    optimizer = build_optimizer(
        params,
        lambda _: jnp.asarray(1e-3, dtype=jnp.float32),
        weight_decay=0.0,
        max_grad_norm=1.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )

    template = optimizer_state_template(optimizer, params)
    shardings = infer_optimizer_state_shardings(template, params, replicated)
    state = initialize_optimizer(optimizer, params, shardings)

    assert jax.tree_util.tree_structure(template) == jax.tree_util.tree_structure(state)
    for abstract, concrete in zip(
        jax.tree_util.tree_leaves(template),
        jax.tree_util.tree_leaves(state),
        strict=True,
    ):
        assert abstract.shape == concrete.shape
        assert abstract.dtype == concrete.dtype
