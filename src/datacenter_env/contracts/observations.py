from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from datacenter_env.contracts.tasks import RunningTaskView, TaskView


@dataclass(frozen=True, slots=True)
class DataCenterObservation:
    timestamp: datetime
    measured_temperature_c: float
    previous_applied_cooling_kw: float
    workload_fraction: float
    electricity_price_per_kwh: float
    carbon_intensity_kg_per_kwh: float
    outdoor_temperature_c: float
    renewable_power_kw: float
    workload_mode: str = "legacy_aggregate"
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    available_cpu_cores: float | None = None
    available_gpu_units: float | None = None
    available_memory_gb: float | None = None
    waiting_task_count: int = 0
    running_task_count: int = 0
    completed_task_count: int = 0
    at_risk_task_count: int = 0
    waiting_tasks: tuple[TaskView, ...] = ()
    running_tasks: tuple[RunningTaskView, ...] = ()
