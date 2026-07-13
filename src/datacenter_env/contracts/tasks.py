from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping

from datacenter_env.exceptions import InputValidationError


class TaskStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    SLA_VIOLATED = "sla_violated"
    UNSCHEDULABLE = "unschedulable"


class TaskEventType(str, Enum):
    ARRIVED = "arrived"
    STARTED = "started"
    DEFERRED = "deferred"
    COMPLETED = "completed"
    SLA_VIOLATED = "sla_violated"
    UNSCHEDULABLE = "unschedulable"
    RESOURCE_BLOCKED = "resource_blocked"
    FORCED_RESOURCE_BLOCKED = "forced_resource_blocked"


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    arrival_time: datetime
    duration_steps: int
    cpu_cores: float
    gpu_units: float
    memory_gb: float
    deadline_time: datetime
    priority: int = 0
    deferrable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise InputValidationError("task_id must be nonempty")
        if not isinstance(self.arrival_time, datetime) or not isinstance(
            self.deadline_time, datetime
        ):
            raise InputValidationError("task timestamps must be datetime values")
        if int(self.duration_steps) != self.duration_steps or self.duration_steps < 1:
            raise InputValidationError("duration_steps must be an integer >= 1")
        resources = (self.cpu_cores, self.gpu_units, self.memory_gb)
        if not all(math.isfinite(float(value)) and float(value) >= 0 for value in resources):
            raise InputValidationError("task resources must be finite and nonnegative")
        if self.cpu_cores <= 0 and self.gpu_units <= 0:
            raise InputValidationError("task must request CPU or GPU compute resources")
        arrival_aware = self.arrival_time.utcoffset() is not None
        deadline_aware = self.deadline_time.utcoffset() is not None
        if arrival_aware != deadline_aware:
            raise InputValidationError("arrival and deadline timezone awareness must match")
        if self.deadline_time <= self.arrival_time:
            raise InputValidationError("deadline_time must be after arrival_time")


@dataclass(frozen=True, slots=True)
class TaskArrivalBatch:
    timestamp: datetime
    tasks: tuple[TaskSpec, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise InputValidationError("arrival batch timestamp must be a datetime")
        identifiers = tuple(task.task_id for task in self.tasks)
        if len(set(identifiers)) != len(identifiers):
            raise InputValidationError("task ids must be unique within an arrival batch")
        if any(task.arrival_time != self.timestamp for task in self.tasks):
            raise InputValidationError("every task arrival_time must equal the batch timestamp")


@dataclass(frozen=True, slots=True)
class ResourceCapacity:
    cpu_cores: float
    gpu_units: float
    memory_gb: float

    def __post_init__(self) -> None:
        values = (self.cpu_cores, self.gpu_units, self.memory_gb)
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in values):
            raise InputValidationError("resource capacity values must be finite and positive")


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    cpu_cores: float = 0.0
    gpu_units: float = 0.0
    memory_gb: float = 0.0


@dataclass(frozen=True, slots=True)
class ResourceAvailability:
    cpu_cores: float
    gpu_units: float
    memory_gb: float


@dataclass(frozen=True, slots=True)
class TaskView:
    spec: TaskSpec
    status: TaskStatus
    remaining_steps: int
    wait_steps: int
    deferral_count: int
    latest_start_time: datetime


@dataclass(frozen=True, slots=True)
class RunningTaskView(TaskView):
    first_start_time: datetime


@dataclass(frozen=True, slots=True)
class EnvironmentalView:
    electricity_price_per_kwh: float
    carbon_intensity_kg_per_kwh: float
    outdoor_temperature_c: float
    renewable_power_kw: float


@dataclass(frozen=True, slots=True)
class TaskSchedulingObservation:
    timestamp: datetime
    waiting_tasks: tuple[TaskView, ...]
    running_tasks: tuple[RunningTaskView, ...]
    total_resources: ResourceCapacity
    used_resources: ResourceUsage
    available_resources: ResourceAvailability
    environment: EnvironmentalView


@dataclass(frozen=True, slots=True)
class TaskSchedulingDecision:
    start_task_ids: tuple[str, ...] = ()
    deferred_task_ids: tuple[str, ...] = ()
    classification_complete: bool = False

    def __post_init__(self) -> None:
        if len(set(self.start_task_ids)) != len(self.start_task_ids):
            raise InputValidationError("scheduling decision contains duplicate task ids")
        if len(set(self.deferred_task_ids)) != len(self.deferred_task_ids):
            raise InputValidationError("scheduling decision contains duplicate deferred ids")
        if set(self.start_task_ids) & set(self.deferred_task_ids):
            raise InputValidationError("a task cannot be both started and deferred")


@dataclass(frozen=True, slots=True)
class TaskEvent:
    timestamp: datetime
    task_id: str
    event_type: TaskEventType
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.details is not None:
            object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    task_id: str
    final_status: TaskStatus
    first_start_time: datetime | None
    completion_time: datetime | None
    wait_steps: int
    deferral_count: int
    sla_violated: bool
    lateness_minutes: float | None
    resource_blocked_count: int = 0


@dataclass(frozen=True, slots=True)
class TaskStepResult:
    arrived_task_ids: tuple[str, ...] = ()
    started_task_ids: tuple[str, ...] = ()
    completed_task_ids: tuple[str, ...] = ()
    newly_sla_violated_task_ids: tuple[str, ...] = ()
    unschedulable_task_ids: tuple[str, ...] = ()
    resource_blocked_task_ids: tuple[str, ...] = ()
    forced_resource_blocked_task_ids: tuple[str, ...] = ()
    waiting_count: int = 0
    running_count: int = 0
    completed_count: int = 0
    at_risk_count: int = 0
    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    workload_fraction: float = 0.0
    events: tuple[TaskEvent, ...] = ()
    arrived_tasks: tuple[TaskSpec, ...] = ()
