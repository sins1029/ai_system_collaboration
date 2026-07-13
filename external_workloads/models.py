from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TaskDatasetMetadata:
    name: str
    seed: int
    start_timestamp: datetime
    end_timestamp: datetime
    step_minutes: int
    number_of_steps: int
    task_count: int
    generation_parameters: Mapping[str, Any]
