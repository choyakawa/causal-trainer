from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from causal_trainer.kernels.fused_cross_entropy import fused_linear_cross_entropy


def _iter_jaxpr_equations(value: Any) -> Iterator[Any]:
    if hasattr(value, "jaxpr"):
        yield from _iter_jaxpr_equations(value.jaxpr)
        return
    if hasattr(value, "eqns"):
        for equation in value.eqns:
            yield equation
            for parameter in equation.params.values():
                yield from _iter_jaxpr_equations(parameter)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_jaxpr_equations(item)


def _loss_and_aux(hidden, head, targets, weights, *, sparse_skip):
    nll_sum, weight_sum = fused_linear_cross_entropy(
        hidden,
        head,
        targets,
        weights,
        token_budget=4,
        compute_dtype=jnp.float32,
        implementation="xla",
        sparse_skip=sparse_skip,
    )
    return nll_sum / jnp.maximum(weight_sum, 1.0), (nll_sum, weight_sum)


def test_sparse_skip_with_trailing_inactive_chunks_matches_dense_path() -> None:
    batch, sequence, hidden_size, vocab = 2, 8, 4, 7
    hidden = (
        jax.random.normal(jax.random.PRNGKey(0), (batch, sequence, hidden_size)) * 0.125
    ).astype(jnp.float32)
    head = (
        jax.random.normal(jax.random.PRNGKey(1), (hidden_size, vocab)) * 0.125
    ).astype(jnp.float32)
    targets = jax.random.randint(
        jax.random.PRNGKey(2),
        (batch, sequence),
        0,
        vocab,
        dtype=jnp.int32,
    )
    weights = jnp.zeros((batch, sequence), dtype=jnp.float32)
    weights = weights.at[:, :2].set(jnp.asarray([[1.0, 0.25], [0.5, 1.0]], jnp.float32))

    def evaluate(sparse_skip):
        return jax.jit(
            jax.value_and_grad(
                lambda x, w: _loss_and_aux(
                    x,
                    w,
                    targets,
                    weights,
                    sparse_skip=sparse_skip,
                ),
                argnums=(0, 1),
                has_aux=True,
            )
        )(hidden, head)

    (expected_loss, expected_aux), expected_gradients = evaluate(False)
    (actual_loss, actual_aux), actual_gradients = evaluate(True)

    np.testing.assert_allclose(actual_loss, expected_loss, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(actual_aux[0], expected_aux[0], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(actual_aux[1], expected_aux[1], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(actual_gradients[0], expected_gradients[0], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(actual_gradients[1], expected_gradients[1], rtol=0.0, atol=1e-6)


def test_sparse_skip_cond_has_only_one_scalar_result() -> None:
    batch, sequence, hidden_size, vocab = 2, 8, 4, 7
    hidden = jax.ShapeDtypeStruct((batch, sequence, hidden_size), jnp.float32)
    head = jax.ShapeDtypeStruct((hidden_size, vocab), jnp.float32)
    targets = jax.ShapeDtypeStruct((batch, sequence), jnp.int32)
    weights = jax.ShapeDtypeStruct((batch, sequence), jnp.float32)

    closed_jaxpr = jax.make_jaxpr(
        lambda x, w, y, m: fused_linear_cross_entropy(
            x,
            w,
            y,
            m,
            token_budget=4,
            compute_dtype=jnp.float32,
            implementation="xla",
            sparse_skip=True,
        )
    )(hidden, head, targets, weights)
    conditionals = [
        equation
        for equation in _iter_jaxpr_equations(closed_jaxpr)
        if equation.primitive.name == "cond"
    ]

    assert len(conditionals) == 1
    branches = conditionals[0].params["branches"]
    assert all(len(branch.jaxpr.outvars) == 1 for branch in branches)
