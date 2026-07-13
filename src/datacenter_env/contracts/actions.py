from __future__ import annotations

from dataclasses import dataclass
import math

from datacenter_env.exceptions import InputValidationError
from datacenter_env.contracts.tasks import TaskSchedulingDecision


@dataclass(frozen=True, slots=True)
class DataCenterAction:
    cooling_target_kw: float
    task_decision: TaskSchedulingDecision | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.cooling_target_kw)):
            raise InputValidationError("cooling_target_kw must be finite")
