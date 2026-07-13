from __future__ import annotations

from pathlib import Path

from external_workloads.csv_provider import write_task_csv
from external_workloads.providers import TimelineTaskProvider
from external_workloads.synthetic_provider import generate_synthetic_task_provider


def generate_standard_task_dataset(
    config: dict, output_path: str | Path | None = None
) -> TimelineTaskProvider:
    provider = generate_synthetic_task_provider(config)
    if output_path is not None:
        write_task_csv(provider, output_path)
    return provider
