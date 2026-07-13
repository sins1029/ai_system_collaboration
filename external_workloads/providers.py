from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from datacenter_env import TaskArrivalBatch, TaskSpec

from external_workloads.models import TaskDatasetMetadata
from external_workloads.validation import validate_task_dataset


@dataclass(frozen=True)
class TimelineTaskProvider:
    tasks: tuple[TaskSpec, ...]
    metadata: TaskDatasetMetadata

    def __post_init__(self) -> None:
        validate_task_dataset(self.tasks)

    def arrivals_at(self, timestamp: datetime) -> TaskArrivalBatch:
        return TaskArrivalBatch(
            timestamp,
            tuple(task for task in self.tasks if task.arrival_time == timestamp),
        )
