from __future__ import annotations

import logging
from typing import Any


def configure_logging(is_primary: bool) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO if is_primary else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("causal_trainer")


def memory_metrics(jax_module) -> dict[str, float]:
    values: dict[str, float] = {}
    for index, device in enumerate(jax_module.local_devices()):
        stats = device.memory_stats()
        if not stats:
            continue
        for source, target in (
            ("bytes_in_use", "bytes_in_use_gib"),
            ("peak_bytes_in_use", "peak_bytes_in_use_gib"),
        ):
            if source in stats:
                values[f"device_{index}/{target}"] = float(stats[source]) / 2**30
    return values


class ExperimentLogger:
    def __init__(self, enabled: bool, project: str, run_name: str | None, config: dict[str, Any]):
        self._run = None
        if enabled:
            try:
                import wandb
            except ImportError as error:
                raise RuntimeError("--use_wandb requires installing causal-trainer[wandb]") from error
            self._run = wandb.init(project=project, name=run_name, config=config)

    def log(self, values: dict[str, Any], step: int) -> None:
        if self._run is not None:
            self._run.log(values, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()


__all__ = ["ExperimentLogger", "configure_logging", "memory_metrics"]
