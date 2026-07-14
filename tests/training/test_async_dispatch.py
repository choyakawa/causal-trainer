from collections import deque

from causal_trainer.training.runner import (
    _enforce_metric_window,
    _synchronize_metric_window,
)


class _Metric:
    def __init__(self) -> None:
        self.blocks = 0

    def block_until_ready(self):
        self.blocks += 1
        return self


class _Tree:
    @staticmethod
    def map(function, tree):
        return {key: function(value) for key, value in tree.items()}


class _FakeJax:
    tree = _Tree()


def test_metric_window_blocks_only_when_the_bound_is_exceeded() -> None:
    metrics = [_Metric() for _ in range(3)]
    pending = deque((step, {"loss": metric}) for step, metric in enumerate(metrics, start=1))

    _enforce_metric_window(_FakeJax, pending, 2)

    assert [step for step, _ in pending] == [2, 3]
    assert [metric.blocks for metric in metrics] == [1, 0, 0]


def test_metric_window_synchronizes_newest_step_and_releases_references() -> None:
    metrics = [_Metric() for _ in range(2)]
    pending = deque([(4, {"loss": metrics[0]}), (5, {"loss": metrics[1]})])

    synchronized = _synchronize_metric_window(_FakeJax, pending)

    assert synchronized is not None and synchronized[0] == 5
    assert metrics[1].blocks == 1
    assert not pending

