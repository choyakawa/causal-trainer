import numpy as np
import pytest

from causal_trainer.data.batching import (
    BatchPlan,
    iter_global_batches,
    prefetch_map,
    split_accumulation,
)


def _row(value: int) -> dict[str, list[int]]:
    return {
        "input_ids": [value, value],
        "attention_mask": [1, 1],
        "position_ids": [0, 1],
        "segment_ids": [1, 1],
    }


def test_plan_uses_micro_batch_times_accumulation() -> None:
    plan = BatchPlan.create(128, 16, 2, 1.0, -1)
    assert plan.effective_batch_size == 32
    assert plan.steps_per_epoch == 4


def test_max_steps_overrides_epoch_budget() -> None:
    plan = BatchPlan.create(128, 16, 2, 0.0, 7)
    assert plan.total_steps == 7


def test_plan_rejects_an_unpadded_remainder() -> None:
    with pytest.raises(ValueError, match="append zero-loss rows"):
        BatchPlan.create(10, 2, 2, 1.0, -1)


def test_batches_cover_every_prepared_row_and_split_accumulation() -> None:
    dataset = [_row(index) for index in range(12)]
    plan = BatchPlan.create(12, 2, 2, 1.0, -1)
    batches = list(iter_global_batches(dataset, plan, assistant_only_loss=False, shuffle=False, seed=0))
    assert len(batches) == 3
    np.testing.assert_array_equal(
        np.concatenate([batch["input_ids"][:, 0] for batch in batches]),
        np.arange(12),
    )
    split = split_accumulation(batches[0], 2)
    assert split["input_ids"].shape == (2, 2, 2)
    np.testing.assert_array_equal(split["loss_weights"], split["attention_mask"].astype(np.float32))


def test_batches_resume_at_exact_global_step() -> None:
    dataset = [_row(index) for index in range(12)]
    plan = BatchPlan.create(12, 2, 1, 2.0, -1)
    full = list(iter_global_batches(dataset, plan, assistant_only_loss=False, shuffle=True, seed=9))
    resumed = list(
        iter_global_batches(
            dataset,
            plan,
            assistant_only_loss=False,
            shuffle=True,
            seed=9,
            start_step=7,
        )
    )
    assert len(full) == 12
    assert len(resumed) == 5
    for expected, actual in zip(full[7:], resumed, strict=True):
        np.testing.assert_array_equal(actual["input_ids"], expected["input_ids"])
    assert not list(
        iter_global_batches(
            dataset,
            plan,
            assistant_only_loss=False,
            shuffle=True,
            seed=9,
            start_step=plan.total_steps,
        )
    )


def test_assistant_mask_is_collated_without_token_shift() -> None:
    dataset = [_row(1), _row(2)]
    dataset[0]["assistant_masks"] = [0, 1]
    dataset[1]["assistant_masks"] = [1, 0]
    plan = BatchPlan.create(2, 2, 1, 1.0, -1)

    batch = next(
        iter_global_batches(
            dataset,
            plan,
            assistant_only_loss=True,
            shuffle=False,
            seed=0,
        )
    )

    np.testing.assert_array_equal(batch["loss_weights"], [[0.0, 1.0], [1.0, 0.0]])


def test_fractional_loss_weights_are_collated_without_boolean_coercion() -> None:
    dataset = [_row(1), _row(2)]
    dataset[0]["loss_weights"] = [1.0, 0.1]
    dataset[1]["loss_weights"] = [0.25, 0.0]
    plan = BatchPlan.create(2, 2, 1, 1.0, -1)

    batch = next(
        iter_global_batches(
            dataset,
            plan,
            assistant_only_loss=False,
            shuffle=False,
            seed=0,
        )
    )

    assert batch["loss_weights"].dtype == np.float32
    np.testing.assert_allclose(batch["loss_weights"], [[1.0, 0.1], [0.25, 0.0]])


def test_endprompt_assistant_weights_are_already_combined() -> None:
    dataset = [_row(1), _row(2)]
    dataset[0]["assistant_masks"] = [0, 1]
    dataset[1]["assistant_masks"] = [1, 0]
    dataset[0]["loss_weights"] = [0.0, 0.25]
    dataset[1]["loss_weights"] = [0.5, 0.1]
    plan = BatchPlan.create(2, 2, 1, 1.0, -1)

    batch = next(
        iter_global_batches(
            dataset,
            plan,
            assistant_only_loss=True,
            shuffle=False,
            seed=0,
        )
    )

    # The terminal 0.1 target is intentionally supervised even though it is
    # not part of the tokenizer's assistant span.
    np.testing.assert_allclose(batch["loss_weights"], [[0.0, 0.25], [0.5, 0.1]])


def test_weighted_and_unweighted_records_cannot_share_a_batch() -> None:
    dataset = [_row(1), _row(2)]
    dataset[0]["loss_weights"] = [1.0, 0.1]
    plan = BatchPlan.create(2, 2, 1, 1.0, -1)

    with pytest.raises(ValueError, match="mix weighted and unweighted"):
        next(
            iter_global_batches(
                dataset,
                plan,
                assistant_only_loss=False,
                shuffle=False,
                seed=0,
            )
        )


def test_assistant_mask_is_required_and_aligned_for_every_record() -> None:
    dataset = [_row(1), _row(2)]
    dataset[0]["assistant_masks"] = [0, 1]
    plan = BatchPlan.create(2, 2, 1, 1.0, -1)

    with pytest.raises(ValueError, match="records \\[1\\]"):
        next(
            iter_global_batches(
                dataset,
                plan,
                assistant_only_loss=True,
                shuffle=False,
                seed=0,
            )
        )


def test_prefetch_map_keeps_exactly_one_converted_batch_ahead() -> None:
    converted = []

    def transform(value: int) -> int:
        converted.append(value)
        return value * 10

    values = prefetch_map(range(4), transform, prefetch_batches=1)
    assert next(values) == 0
    assert converted == [0, 1]
    assert next(values) == 10
    assert converted == [0, 1, 2]
    assert list(values) == [20, 30]
    assert converted == [0, 1, 2, 3]


def test_prefetch_map_can_disable_lookahead_and_rejects_negative_depth() -> None:
    converted = []
    values = prefetch_map(range(2), lambda value: converted.append(value) or value, prefetch_batches=0)
    assert next(values) == 0
    assert converted == [0]

    invalid = prefetch_map([], lambda value: value, prefetch_batches=-1)
    with pytest.raises(ValueError, match="cannot be negative"):
        next(invalid)
