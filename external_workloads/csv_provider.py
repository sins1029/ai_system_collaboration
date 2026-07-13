from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from datacenter_env import TaskSpec

from external_workloads.models import TaskDatasetMetadata
from external_workloads.providers import TimelineTaskProvider


def write_task_csv(provider: TimelineTaskProvider, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "task_id": task.task_id,
                "arrival_time": task.arrival_time.isoformat(),
                "duration_steps": task.duration_steps,
                "cpu_cores": task.cpu_cores,
                "gpu_units": task.gpu_units,
                "memory_gb": task.memory_gb,
                "deadline_time": task.deadline_time.isoformat(),
                "priority": task.priority,
                "deferrable": task.deferrable,
            }
            for task in provider.tasks
        ]
    ).to_csv(output, index=False)
    return output


def load_task_csv(
    path: str | Path,
    *,
    dataset_name: str = "csv_tasks",
    seed: int = 0,
    step_minutes: int = 15,
    number_of_steps: int = 96,
) -> TimelineTaskProvider:
    frame = pd.read_csv(path)
    tasks = tuple(
        TaskSpec(
            task_id=str(row.task_id),
            arrival_time=datetime.fromisoformat(str(row.arrival_time)),
            duration_steps=int(row.duration_steps),
            cpu_cores=float(row.cpu_cores),
            gpu_units=float(row.gpu_units),
            memory_gb=float(row.memory_gb),
            deadline_time=datetime.fromisoformat(str(row.deadline_time)),
            priority=int(row.priority),
            deferrable=str(row.deferrable).lower() in {"true", "1"},
        )
        for row in frame.itertuples(index=False)
    )
    ordered = tuple(sorted(tasks, key=lambda task: (task.arrival_time, task.task_id)))
    start = min(task.arrival_time for task in ordered)
    end = max(task.arrival_time for task in ordered)
    return TimelineTaskProvider(
        ordered,
        TaskDatasetMetadata(
            dataset_name,
            seed,
            start,
            end,
            step_minutes,
            number_of_steps,
            len(ordered),
            {"source_path": str(path)},
        ),
    )
