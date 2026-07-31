from __future__ import annotations

from itertools import islice
from types import SimpleNamespace

import pytest

from causal_trainer.data.streaming import (
    StreamingBatchEnvelope,
    iter_retrying_records,
    iter_streaming_batches,
    streaming_dataset_metadata,
)
from causal_trainer.data.streaming_batching import (
    StreamingBatchPlan,
    iter_streaming_global_batches,
)
from causal_trainer.hf_retry import is_retryable_hf_error, retry_hf_call


class TextTokenizer:
    pad_token_id = 0
    eos_token_id = None

    def __call__(self, text, **kwargs):
        del kwargs
        tokens = [max(1, len(text))] * len(text)
        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
        }


class MessagesTokenizer:
    pad_token_id = 0
    eos_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        if not any(message["role"] == "assistant" for message in messages):
            return {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "assistant_masks": [0, 0],
            }
        return {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "assistant_masks": [0, 1, 1],
        }


def _text_source(size: int, *, consumed=None):
    def source(epoch: int):
        for index in range(size):
            if consumed is not None:
                consumed.append((epoch, index))
            yield {"text": "x" * (index % 3 + 2)}

    return source


def test_streaming_first_window_is_yielded_without_consuming_history() -> None:
    consumed = []
    batches = iter_streaming_batches(
        _text_source(100, consumed=consumed),
        TextTokenizer(),
        num_epochs=1,
        dataset_text_field="text",
        max_sequence_length=8,
        pad_token_id=0,
        packing=True,
        packing_batch_size=3,
        retry_initial_delay=0,
        retry_max_delay=0,
    )

    first = next(batches)

    assert consumed == [(0, 0), (0, 1), (0, 2)]
    assert first.source_examples == 3
    assert sum(first.record_source_examples) == 3
    assert all("loss_weights" in record for record in first.records)


def test_streaming_flushes_short_packing_window_before_epoch_marker() -> None:
    batches = list(
        iter_streaming_batches(
            _text_source(5),
            TextTokenizer(),
            num_epochs=1,
            dataset_text_field="text",
            max_sequence_length=8,
            pad_token_id=0,
            packing=True,
            packing_batch_size=2,
            retry_initial_delay=0,
            retry_max_delay=0,
        )
    )

    assert [batch.source_examples for batch in batches] == [2, 2, 1, 0]
    assert [batch.is_epoch_end for batch in batches] == [False, False, False, True]
    assert sum(batch.source_examples for batch in batches) == 5
    assert sum(sum(batch.record_source_examples) for batch in batches) == 5


def test_packing_envelope_assigns_all_source_rows_to_packed_bins() -> None:
    def source(epoch: int):
        del epoch
        yield {"text": "xxxx"}
        yield {"text": "xx"}

    first = next(
        iter_streaming_batches(
            source,
            TextTokenizer(),
            num_epochs=1,
            dataset_text_field="text",
            max_sequence_length=6,
            pad_token_id=0,
            packing=True,
            packing_batch_size=2,
            retry_initial_delay=0,
            retry_max_delay=0,
        )
    )

    assert len(first.records) == 1
    assert first.record_source_examples == (2,)
    assert first.records[0]["segment_ids"] == [1, 1, 1, 1, 2, 2]


def test_streaming_last_assistant_only_loss_survives_packing_per_record() -> None:
    class MultiTurnTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_assistant_tokens_mask"] is True
            offset = 10 if messages[0]["content"] == "first-a" else 20
            return {
                "input_ids": [offset, offset + 1, offset + 2, offset + 3],
                "attention_mask": [1, 1, 1, 1],
                "assistant_masks": [0, 1, 0, 1],
            }

    def source(epoch: int):
        del epoch
        for suffix in ("a", "b"):
            yield {
                "messages": [
                    {"role": "user", "content": f"first-{suffix}"},
                    {"role": "assistant", "content": f"earlier-{suffix}"},
                    {"role": "user", "content": f"last-{suffix}"},
                    {"role": "assistant", "content": f"final-{suffix}"},
                ]
            }

    first = next(
        iter_streaming_batches(
            source,
            MultiTurnTokenizer(),
            num_epochs=1,
            dataset_text_field="messages",
            max_sequence_length=8,
            pad_token_id=0,
            last_assistant_only_loss=True,
            packing=True,
            packing_batch_size=2,
            retry_initial_delay=0,
            retry_max_delay=0,
        )
    )

    assert len(first.records) == 1
    assert first.records[0]["segment_ids"] == [1, 1, 1, 1, 2, 2, 2, 2]
    assert first.records[0]["assistant_masks"] == [0, 0, 0, 1, 0, 0, 0, 1]
    assert first.records[0]["loss_weights"] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize("packing", [False, True], ids=["padding", "packing"])
@pytest.mark.parametrize("loss_mode", ["assistant", "last_assistant"])
def test_streaming_truncated_zero_loss_record_is_dropped_before_finalization(
    packing: bool,
    loss_mode: str,
) -> None:
    class TruncationTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["truncation"] is False
            if messages[0]["content"] == "late":
                assistant_masks = (
                    [0] * 8 + [1, 1]
                    if loss_mode == "assistant"
                    else [0, 1, 1, 0, 0, 0, 0, 0, 1, 1]
                )
                return {
                    "input_ids": list(range(90, 100)),
                    "attention_mask": [1] * 10,
                    "assistant_masks": assistant_masks,
                }
            offset = 10 if messages[0]["content"] == "kept-a" else 20
            return {
                "input_ids": [offset, offset + 1, offset + 2, offset + 3],
                "attention_mask": [1, 1, 1, 1],
                "assistant_masks": [0, 0, 1, 1],
            }

    def source(epoch: int):
        del epoch
        truncated_conversation = (
            [
                {"role": "user", "content": "late"},
                {"role": "assistant", "content": "outside truncation limit"},
            ]
            if loss_mode == "assistant"
            else [
                {"role": "user", "content": "late"},
                {"role": "assistant", "content": "earlier inside truncation limit"},
                {"role": "user", "content": "follow-up"},
                {"role": "assistant", "content": "final outside truncation limit"},
            ]
        )
        yield {
            "messages": truncated_conversation
        }
        yield {
            "messages": [
                {"role": "user", "content": "kept-a"},
                {"role": "assistant", "content": "inside truncation limit a"},
            ]
        }
        yield {
            "messages": [
                {"role": "user", "content": "kept-b"},
                {"role": "assistant", "content": "inside truncation limit b"},
            ]
        }

    first = next(
        iter_streaming_batches(
            source,
            TruncationTokenizer(),
            num_epochs=1,
            dataset_text_field="messages",
            max_sequence_length=8,
            pad_token_id=0,
            assistant_only_loss=loss_mode == "assistant",
            last_assistant_only_loss=loss_mode == "last_assistant",
            packing=packing,
            packing_batch_size=3,
            retry_initial_delay=0,
            retry_max_delay=0,
        )
    )

    assert first.source_examples == 3
    if packing:
        assert first.record_source_examples == (3,)
        assert len(first.records) == 1
        assert first.records[0]["input_ids"] == [10, 11, 12, 13, 20, 21, 22, 23]
        assert first.records[0]["segment_ids"] == [1, 1, 1, 1, 2, 2, 2, 2]
        assert first.records[0]["assistant_masks"] == [0, 0, 1, 1, 0, 0, 1, 1]
        assert first.records[0]["loss_weights"] == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    else:
        assert first.record_source_examples == (1, 2)
        assert len(first.records) == 2
        assert [record["input_ids"] for record in first.records] == [
            [0, 0, 0, 0, 10, 11, 12, 13],
            [0, 0, 0, 0, 20, 21, 22, 23],
        ]
        assert [record["assistant_masks"] for record in first.records] == [
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1],
        ]
        assert [record["loss_weights"] for record in first.records] == [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        ]


def test_filtered_rows_are_charged_to_the_last_prepared_record() -> None:
    def source(epoch: int):
        del epoch
        yield {"messages": [{"role": "user", "content": "no response"}]}
        yield {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }

    first = next(
        iter_streaming_batches(
            source,
            MessagesTokenizer(),
            num_epochs=1,
            dataset_text_field="messages",
            max_sequence_length=6,
            pad_token_id=0,
            assistant_only_loss=True,
            packing=True,
            packing_batch_size=2,
            retry_initial_delay=0,
            retry_max_delay=0,
        )
    )

    assert first.source_examples == 2
    assert first.record_source_examples == (2,)
    assert first.records[0]["loss_weights"][-3:] == [0.0, 1.0, 1.0]


def test_fully_filtered_window_still_reports_consumed_source_rows() -> None:
    def source(epoch: int):
        del epoch
        yield {"messages": [{"role": "user", "content": "no response"}]}

    batches = list(
        iter_streaming_batches(
            source,
            MessagesTokenizer(),
            num_epochs=1,
            dataset_text_field="messages",
            max_sequence_length=6,
            pad_token_id=0,
            assistant_only_loss=True,
            packing=True,
            packing_batch_size=2,
            retry_initial_delay=0,
            retry_max_delay=0,
        )
    )

    assert batches[0] == StreamingBatchEnvelope(
        records=(),
        source_examples=1,
        record_source_examples=(),
        epoch=0,
    )
    assert batches[1].is_epoch_end


def test_global_batching_defers_fully_filtered_source_progress() -> None:
    record = {
        "input_ids": [1, 2],
        "attention_mask": [1, 1],
        "position_ids": [0, 1],
        "segment_ids": [1, 1],
        "loss_weights": [1.0, 1.0],
    }
    envelopes = [
        StreamingBatchEnvelope((), 2, (), 0),
        StreamingBatchEnvelope((record,), 1, (1,), 0),
        StreamingBatchEnvelope((), 0, (), 0, is_epoch_end=True),
    ]
    plan = StreamingBatchPlan.create(
        source_examples_per_epoch=3,
        global_micro_batch_size=1,
        accumulation_steps=1,
        epochs=1,
        max_steps=0,
    )

    batches = list(
        iter_streaming_global_batches(
            envelopes,
            plan,
            assistant_only_loss=False,
        )
    )

    assert len(batches) == 1
    assert batches[0].source_examples == 3
    assert batches[0].source_examples_seen == 3
    assert batches[0].is_epoch_end


def test_nonpacking_records_have_canonical_loss_weights() -> None:
    first = next(
        iter_streaming_batches(
            _text_source(1),
            TextTokenizer(),
            num_epochs=1,
            dataset_text_field="text",
            max_sequence_length=4,
            pad_token_id=0,
            packing=False,
            packing_batch_size=2,
            retry_initial_delay=0,
            retry_max_delay=0,
        )
    )

    assert first.records[0]["attention_mask"] == [0, 0, 1, 1]
    assert first.records[0]["loss_weights"] == [0.0, 0.0, 1.0, 1.0]


def test_epochs_never_share_a_packing_window() -> None:
    calls = []
    batches = list(
        iter_streaming_batches(
            _text_source(3, consumed=calls),
            TextTokenizer(),
            num_epochs=2,
            dataset_text_field="text",
            max_sequence_length=8,
            pad_token_id=0,
            packing=True,
            packing_batch_size=4,
            retry_initial_delay=0,
            retry_max_delay=0,
        )
    )

    assert [(batch.epoch, batch.source_examples, batch.is_epoch_end) for batch in batches] == [
        (0, 3, False),
        (0, 0, True),
        (1, 3, False),
        (1, 0, True),
    ]
    assert calls == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def test_none_epochs_repeat_complete_epochs_until_caller_stops() -> None:
    batches = list(
        islice(
            iter_streaming_batches(
                _text_source(1),
                TextTokenizer(),
                num_epochs=None,
                dataset_text_field="text",
                max_sequence_length=4,
                pad_token_id=0,
                packing=False,
                packing_batch_size=2,
                retry_initial_delay=0,
                retry_max_delay=0,
            ),
            4,
        )
    )

    assert [(batch.epoch, batch.is_epoch_end) for batch in batches] == [
        (0, False),
        (0, True),
        (1, False),
        (1, True),
    ]


def test_metadata_reads_split_count_estimate_without_scanning() -> None:
    known = SimpleNamespace(
        info=SimpleNamespace(
            splits={"train": SimpleNamespace(num_examples=12)},
        ),
        split="train",
    )
    unknown = SimpleNamespace(info=SimpleNamespace(splits={"train": {}}), split="train")

    known_metadata = streaming_dataset_metadata(known)
    unknown_metadata = streaming_dataset_metadata(unknown)

    assert known_metadata.num_examples == 12
    assert known_metadata.total_examples(3) == 36
    assert known_metadata.total_examples(None) is None
    assert unknown_metadata.num_examples is None
    assert unknown_metadata.total_examples(3) is None


def test_retrying_records_rebuilds_and_skips_rows_already_yielded(monkeypatch) -> None:
    calls = 0
    sleeps = []

    def source(epoch: int):
        nonlocal calls
        assert epoch == 4
        calls += 1
        this_call = calls
        for index in range(5):
            if this_call == 1 and index == 2:
                raise ConnectionError("temporary disconnect")
            yield {"id": index}

    monkeypatch.setattr("causal_trainer.hf_retry.time.sleep", sleeps.append)

    records = list(
        iter_retrying_records(
            source,
            4,
            initial_delay=0.25,
            max_delay=1.0,
            operation="test stream",
        )
    )

    assert [record["id"] for record in records] == [0, 1, 2, 3, 4]
    assert calls == 2
    assert sleeps == [0.25]


def test_retrying_records_also_retries_a_disconnect_while_replaying(monkeypatch) -> None:
    calls = 0
    sleeps = []

    def source(epoch: int):
        nonlocal calls
        del epoch
        calls += 1
        this_call = calls
        for index in range(4):
            if this_call == 1 and index == 2:
                raise ConnectionError("initial disconnect")
            if this_call == 2 and index == 1:
                raise TimeoutError("replay disconnect")
            yield {"id": index}

    monkeypatch.setattr("causal_trainer.hf_retry.time.sleep", sleeps.append)

    records = list(
        iter_retrying_records(
            source,
            0,
            initial_delay=0.25,
            max_delay=1.0,
        )
    )

    assert [record["id"] for record in records] == [0, 1, 2, 3]
    assert calls == 3
    assert sleeps == [0.25, 0.5]


def test_retrying_records_retries_a_shorter_clean_replay_prefix(monkeypatch) -> None:
    calls = 0
    sleeps = []

    def source(epoch: int):
        nonlocal calls
        del epoch
        calls += 1
        limit = {1: 2, 2: 1}.get(calls, 4)
        for index in range(limit):
            yield {"id": index}
        if calls == 1:
            raise ConnectionError("initial disconnect")

    monkeypatch.setattr("causal_trainer.hf_retry.time.sleep", sleeps.append)

    records = list(
        iter_retrying_records(
            source,
            0,
            initial_delay=0.25,
            max_delay=1.0,
        )
    )

    assert [record["id"] for record in records] == [0, 1, 2, 3]
    assert calls == 3
    assert sleeps == [0.25, 0.5]


def test_retrying_records_accepts_natural_eof_without_a_cardinality_contract() -> None:
    def source(epoch: int):
        del epoch
        yield {"id": 0}
        yield {"id": 1}

    assert list(iter_retrying_records(source, 0)) == [{"id": 0}, {"id": 1}]


def test_retrying_records_fails_fast_for_non_network_errors(monkeypatch) -> None:
    calls = 0
    sleeps = []

    def source(epoch: int):
        nonlocal calls
        del epoch
        calls += 1
        yield {"id": 0}
        raise ValueError("invalid row")

    monkeypatch.setattr("causal_trainer.hf_retry.time.sleep", sleeps.append)

    with pytest.raises(ValueError, match="invalid row"):
        list(iter_retrying_records(source, 0))
    assert calls == 1
    assert sleeps == []


def test_retrying_records_rejects_changed_prefix_after_reconnect(monkeypatch) -> None:
    calls = 0
    sleeps = []

    def source(epoch: int):
        nonlocal calls
        del epoch
        calls += 1
        if calls == 1:
            yield {"id": 0}
            yield {"id": 1}
            raise ConnectionError("temporary disconnect")
        yield {"id": 9}
        yield {"id": 1}
        yield {"id": 2}

    monkeypatch.setattr("causal_trainer.hf_retry.time.sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="source prefix changed"):
        list(
            iter_retrying_records(
                source,
                0,
                initial_delay=0.25,
                max_delay=1.0,
            )
        )

    assert calls == 2
    assert sleeps == [0.25]


def test_retry_hf_call_uses_unbounded_exponential_backoff(monkeypatch) -> None:
    calls = 0
    sleeps = []

    def operation():
        nonlocal calls
        calls += 1
        if calls < 4:
            raise TimeoutError("offline")
        return "ready"

    monkeypatch.setattr("causal_trainer.hf_retry.time.sleep", sleeps.append)

    assert retry_hf_call(operation, 0.5, 1.0, "metadata") == "ready"
    assert calls == 4
    assert sleeps == [0.5, 1.0, 1.0]


@pytest.mark.parametrize(("initial", "maximum"), [(float("nan"), 1.0), (1.0, float("inf"))])
def test_retry_hf_call_rejects_non_finite_delays(initial: float, maximum: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        retry_hf_call(lambda: None, initial, maximum)


def test_retryable_http_statuses_do_not_include_auth_or_missing_data() -> None:
    class HttpError(Exception):
        def __init__(self, status: int):
            super().__init__(f"HTTP {status}")
            self.response = SimpleNamespace(status_code=status)

    assert is_retryable_hf_error(HttpError(429))
    assert is_retryable_hf_error(HttpError(503))
    assert not is_retryable_hf_error(HttpError(401))
    assert not is_retryable_hf_error(HttpError(403))
    assert not is_retryable_hf_error(HttpError(404))


def test_retryable_http_status_can_be_stored_on_the_exception() -> None:
    class ClientResponseError(Exception):
        status = 503

    assert is_retryable_hf_error(ClientResponseError())


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("aiohttp.client_exceptions", "ClientPayloadError"),
        ("http.client", "RemoteDisconnected"),
        ("httpx", "DecodingError"),
        ("requests.exceptions", "SSLError"),
        ("ssl", "SSLError"),
        ("urllib.error", "URLError"),
    ],
)
def test_common_stream_transport_errors_are_retryable(module: str, name: str) -> None:
    error_type = type(name, (Exception,), {"__module__": module})

    assert is_retryable_hf_error(error_type("transient stream failure"))


def test_network_exception_subclasses_are_classified_by_their_transport_base() -> None:
    transport_base = type(
        "ReadTimeout",
        (Exception,),
        {"__module__": "httpx"},
    )

    class WrappedReadTimeout(transport_base):
        pass

    assert is_retryable_hf_error(WrappedReadTimeout("transient stream failure"))


def test_permanent_aiohttp_configuration_errors_are_not_retryable() -> None:
    client_error = type(
        "ClientError",
        (Exception,),
        {"__module__": "aiohttp.client_exceptions"},
    )
    invalid_url = type(
        "InvalidURL",
        (client_error,),
        {"__module__": "aiohttp.client_exceptions"},
    )

    assert not is_retryable_hf_error(invalid_url("malformed URL"))
