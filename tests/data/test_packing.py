from __future__ import annotations

import pytest

from causal_trainer.data.packing import pack_examples, pad_examples


def row(token: int, length: int, assistant_masks=None):
    result = {
        "input_ids": [token] * length,
        "attention_mask": [1] * length,
    }
    if assistant_masks is not None:
        result["assistant_masks"] = assistant_masks
    return result


def test_bfd_packs_without_separator_and_resets_segment_positions():
    packed = pack_examples(
        [row(1, 6), row(2, 4), row(3, 4), row(4, 2)],
        max_length=8,
        pad_token_id=99,
    )

    assert len(packed) == 2
    assert packed[0]["input_ids"] == [1] * 6 + [4] * 2
    assert packed[0]["position_ids"] == [0, 1, 2, 3, 4, 5, 0, 1]
    assert packed[0]["segment_ids"] == [1] * 6 + [2] * 2
    assert packed[1]["input_ids"] == [2] * 4 + [3] * 4
    assert 99 not in packed[0]["input_ids"]


def test_bfd_is_scoped_to_each_packing_batch():
    packed = pack_examples(
        [row(1, 6), row(2, 6), row(3, 2), row(4, 2)],
        max_length=8,
        pad_token_id=0,
        packing_batch_size=2,
    )
    # Global BFD would make two 6+2 rows.  Per-batch BFD makes two padded
    # six-token rows followed by one 2+2 row.
    assert len(packed) == 3


def test_left_padding_is_fixed_length_and_all_masks_stay_aligned():
    packed = pack_examples(
        [
            row(5, 3, [0, 1, 1]),
            row(6, 2, [0, 1]),
        ],
        max_length=8,
        pad_token_id=99,
    )
    result = packed[0]

    assert result["input_ids"] == [99, 99, 99, 5, 5, 5, 6, 6]
    assert result["attention_mask"] == [0, 0, 0, 1, 1, 1, 1, 1]
    assert result["assistant_masks"] == [0, 0, 0, 0, 1, 1, 0, 1]
    assert result["position_ids"] == [0, 0, 0, 0, 1, 2, 0, 1]
    assert result["segment_ids"] == [0, 0, 0, 1, 1, 1, 2, 2]
    assert all(len(value) == 8 for value in result.values())


def test_metadata_resets_for_every_contiguous_valid_run():
    packed = pack_examples(
        [
            {"input_ids": [10, 11, 12, 13], "attention_mask": [1, 1, 0, 1]},
            {"input_ids": [20, 21, 22], "attention_mask": [0, 1, 1]},
        ],
        max_length=7,
        pad_token_id=99,
    )[0]

    assert packed["attention_mask"] == [1, 1, 0, 1, 0, 1, 1]
    assert packed["position_ids"] == [0, 1, 0, 0, 0, 0, 1]
    assert packed["segment_ids"] == [1, 1, 0, 2, 0, 3, 3]


def test_truncation_keeps_all_per_token_fields_synchronized():
    packed = pack_examples(
        [
            {
                "input_ids": [1, 2, 3, 4, 5, 6],
                "attention_mask": [1, 1, 1, 1, 1, 1],
                "assistant_masks": [0, 0, 1, 1, 0, 1],
            }
        ],
        max_length=4,
        pad_token_id=0,
    )[0]

    assert packed["input_ids"] == [1, 2, 3, 4]
    assert packed["assistant_masks"] == [0, 0, 1, 1]
    assert packed["position_ids"] == [0, 1, 2, 3]


def test_endprompt_packing_preserves_anchored_positions_and_float_weights():
    packed = pack_examples(
        [
            {
                "input_ids": [10, 11, 90],
                "attention_mask": [1, 1, 1],
                "position_ids": [0, 1, 15],
                "loss_weights": [1.0, 1.0, 0.1],
            },
            {
                "input_ids": [20, 91],
                "attention_mask": [1, 1],
                "position_ids": [0, 31],
                "loss_weights": [1.0, 0.25],
            },
        ],
        max_length=6,
        pad_token_id=0,
    )[0]

    assert packed["input_ids"] == [0, 10, 11, 90, 20, 91]
    assert packed["position_ids"] == [0, 0, 1, 15, 0, 31]
    assert packed["segment_ids"] == [0, 1, 1, 1, 2, 2]
    assert packed["loss_weights"] == [0.0, 1.0, 1.0, 0.1, 1.0, 0.25]


def test_endprompt_assistant_masks_and_combined_weights_survive_packing():
    packed = pack_examples(
        [
            {
                "input_ids": [10, 11, 90],
                "attention_mask": [1, 1, 1],
                "assistant_masks": [0, 1, 0],
                "position_ids": [0, 1, 15],
                "loss_weights": [0.0, 1.0, 0.1],
            },
            {
                "input_ids": [20, 21, 91],
                "attention_mask": [1, 1, 1],
                "assistant_masks": [0, 1, 0],
                "position_ids": [0, 1, 31],
                "loss_weights": [0.0, 0.5, 0.25],
            },
        ],
        max_length=6,
        pad_token_id=0,
    )[0]

    assert packed["segment_ids"] == [1, 1, 1, 2, 2, 2]
    assert packed["assistant_masks"] == [0, 1, 0, 0, 1, 0]
    assert packed["loss_weights"] == [0.0, 1.0, 0.1, 0.0, 0.5, 0.25]


def test_endprompt_nonpacking_padding_also_preserves_anchor_positions():
    padded = pad_examples(
        [
            {
                "input_ids": [10, 90],
                "attention_mask": [1, 1],
                "position_ids": [0, 63],
                "loss_weights": [1.0, 0.1],
            }
        ],
        max_length=4,
        pad_token_id=0,
    )[0]

    assert padded["position_ids"] == [0, 0, 0, 63]
    assert padded["segment_ids"] == [0, 0, 1, 1]
    assert padded["loss_weights"] == [0.0, 0.0, 1.0, 0.1]


def test_inconsistent_optional_masks_fail_instead_of_silently_dropping_them():
    with pytest.raises(ValueError, match="same per-token fields"):
        pack_examples(
            [row(1, 2, [0, 1]), row(2, 2)],
            max_length=4,
            pad_token_id=0,
        )


def test_pad_examples_does_not_combine_samples():
    padded = pad_examples(
        [row(7, 2), row(8, 3)],
        max_length=4,
        pad_token_id=99,
    )
    assert [item["input_ids"] for item in padded] == [
        [99, 99, 7, 7],
        [99, 8, 8, 8],
    ]
    assert padded[0]["segment_ids"] == [0, 0, 1, 1]


def test_pad_examples_metadata_respects_internal_padding():
    padded = pad_examples(
        [{"input_ids": [1, 2, 3, 4], "attention_mask": [0, 1, 1, 0]}],
        max_length=6,
        pad_token_id=99,
    )[0]

    assert padded["attention_mask"] == [0, 0, 0, 1, 1, 0]
    assert padded["position_ids"] == [0, 0, 0, 0, 1, 0]
    assert padded["segment_ids"] == [0, 0, 0, 1, 1, 0]


def test_misaligned_mask_is_rejected():
    with pytest.raises(ValueError, match="expected 3"):
        pack_examples(
            [{"input_ids": [1, 2, 3], "assistant_masks": [0, 1]}],
            max_length=4,
            pad_token_id=0,
        )

