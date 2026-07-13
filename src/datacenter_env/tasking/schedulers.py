from __future__ import annotations

from dataclasses import dataclass

from datacenter_env.contracts import ForecastWindow
from datacenter_env.contracts.tasks import (
    ResourceUsage,
    TaskSchedulingDecision,
    TaskSchedulingObservation,
    TaskView,
)
from datacenter_env.exceptions import ConfigurationError
from datacenter_env.tasking.resources import add_usage


def _pack(observation: TaskSchedulingObservation, ordered: list[TaskView]) -> tuple[str, ...]:
    usage = ResourceUsage(
        observation.used_resources.cpu_cores,
        observation.used_resources.gpu_units,
        observation.used_resources.memory_gb,
    )
    selected: list[str] = []
    capacity = observation.total_resources
    for view in ordered:
        candidate = add_usage(usage, view.spec)
        if (
            candidate.cpu_cores <= capacity.cpu_cores + 1e-9
            and candidate.gpu_units <= capacity.gpu_units + 1e-9
            and candidate.memory_gb <= capacity.memory_gb + 1e-9
        ):
            selected.append(view.spec.task_id)
            usage = candidate
    return tuple(selected)


class FifoImmediateScheduler:
    name = "fifo_immediate"
    max_forecast_steps = 1

    def reset(self, seed: int | None = None) -> None:
        del seed

    def schedule(self, observation, forecast_window=None) -> TaskSchedulingDecision:
        del forecast_window
        ordered = sorted(
            observation.waiting_tasks,
            key=lambda view: (view.spec.arrival_time, view.spec.task_id),
        )
        return TaskSchedulingDecision(_pack(observation, ordered))


class EarliestDeadlineFirstScheduler:
    name = "earliest_deadline_first"
    max_forecast_steps = 1

    def reset(self, seed: int | None = None) -> None:
        del seed

    def schedule(self, observation, forecast_window=None) -> TaskSchedulingDecision:
        del forecast_window
        ordered = sorted(
            observation.waiting_tasks,
            key=lambda view: (
                view.spec.deadline_time,
                -view.spec.priority,
                view.spec.arrival_time,
                view.spec.task_id,
            ),
        )
        return TaskSchedulingDecision(_pack(observation, ordered))


@dataclass
class EnergyAwareDeferralScheduler:
    price_weight: float = 0.45
    carbon_weight: float = 0.35
    renewable_weight: float = 0.20
    forecast_steps: int = 8
    price_scale: float = 1.0
    carbon_scale: float = 1.0
    renewable_scale_kw: float = 1000.0

    name = "energy_aware_deferral"

    def __post_init__(self) -> None:
        values = (
            self.price_weight,
            self.carbon_weight,
            self.renewable_weight,
            self.price_scale,
            self.carbon_scale,
            self.renewable_scale_kw,
        )
        if any(value < 0 for value in values[:3]) or any(value <= 0 for value in values[3:]):
            raise ConfigurationError("energy-aware scheduler weights/scales are invalid")
        if self.forecast_steps < 1:
            raise ConfigurationError("scheduler forecast_steps must be positive")

    @property
    def max_forecast_steps(self) -> int:
        return self.forecast_steps

    def reset(self, seed: int | None = None) -> None:
        del seed

    def schedule(
        self,
        observation: TaskSchedulingObservation,
        forecast_window: ForecastWindow | None = None,
    ) -> TaskSchedulingDecision:
        steps = (forecast_window.steps if forecast_window is not None else ())[: self.forecast_steps]
        ordered = sorted(
            observation.waiting_tasks,
            key=lambda view: (
                not (not view.spec.deferrable),
                view.spec.deadline_time,
                -view.spec.priority,
                view.spec.arrival_time,
                view.spec.task_id,
            ),
        )
        eligible: list[TaskView] = []
        for view in ordered:
            if not view.spec.deferrable or observation.timestamp >= view.latest_start_time:
                eligible.append(view)
                continue
            feasible_future = tuple(
                step for step in steps[1:] if step.timestamp <= view.latest_start_time
            )
            if not feasible_future or not steps:
                eligible.append(view)
                continue
            current_score = self._score(steps[0])
            future_best = min(self._score(step) for step in feasible_future)
            if future_best >= current_score - 1e-12:
                eligible.append(view)
        return TaskSchedulingDecision(_pack(observation, eligible))

    def _score(self, step) -> float:
        return (
            self.price_weight * min(1.0, step.electricity_price_per_kwh / self.price_scale)
            + self.carbon_weight
            * min(1.0, step.carbon_intensity_kg_per_kwh / self.carbon_scale)
            - self.renewable_weight
            * min(1.0, step.renewable_power_kw / self.renewable_scale_kw)
        )


def build_task_scheduler(config):
    name = config.scheduler_name
    if name == "fifo_immediate":
        return FifoImmediateScheduler()
    if name == "earliest_deadline_first":
        return EarliestDeadlineFirstScheduler()
    if name == "energy_aware_deferral":
        values = config.scheduler
        return EnergyAwareDeferralScheduler(
            price_weight=float(values.get("price_weight", 0.45)),
            carbon_weight=float(values.get("carbon_weight", 0.35)),
            renewable_weight=float(values.get("renewable_weight", 0.20)),
            forecast_steps=int(values.get("forecast_steps", 8)),
            price_scale=float(values.get("price_scale", 1.0)),
            carbon_scale=float(values.get("carbon_scale", 1.0)),
            renewable_scale_kw=float(values.get("renewable_scale_kw", 1000.0)),
        )
    raise ConfigurationError(f"unknown task scheduler: {name}")
