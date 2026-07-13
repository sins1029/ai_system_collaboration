from __future__ import annotations

from datetime import datetime
import math

import gymnasium as gym
import numpy as np

from datacenter_env.config import DataCenterSystemConfig
from datacenter_env.contracts import (
    DataCenterObservation,
    ResourceAvailability,
    ResourceUsage,
    StepResult,
    TaskSchedulingObservation,
    TaskView,
)
from datacenter_env.gym.config import ObservationConfig


class ObservationEncoder:
    def __init__(
        self,
        config: ObservationConfig,
        system_config: DataCenterSystemConfig,
        max_simulation_steps: int,
    ):
        self.config = config
        self.system_config = system_config
        self.max_simulation_steps = max_simulation_steps
        self.clipping_count = 0
        self.space = gym.spaces.Dict(
            {
                "global": gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32),
                "candidate_task": gym.spaces.Box(-1.0, 1.0, (10,), dtype=np.float32),
                "resources": gym.spaces.Box(0.0, 1.0, (6,), dtype=np.float32),
                "queue_summary": gym.spaces.Box(-1.0, 1.0, (8,), dtype=np.float32),
                "environment": gym.spaces.Box(0.0, 1.0, (4,), dtype=np.float32),
                "thermal": gym.spaces.Box(0.0, 1.0, (4,), dtype=np.float32),
                "action_mask": gym.spaces.MultiBinary(2),
            }
        )

    def reset(self) -> None:
        self.clipping_count = 0

    def encode(
        self,
        *,
        domain_observation: DataCenterObservation,
        task_observation: TaskSchedulingObservation,
        candidate: TaskView | None,
        waiting_tasks: tuple[TaskView, ...],
        virtual_usage: ResourceUsage,
        virtual_availability: ResourceAvailability,
        simulation_step_index: int,
        action_mask: np.ndarray,
        last_result: StepResult | None,
    ) -> dict[str, np.ndarray]:
        timestamp = task_observation.timestamp
        fraction = simulation_step_index / max(1, self.max_simulation_steps)
        hour = timestamp.hour + timestamp.minute / 60.0
        angle = 2.0 * math.pi * hour / 24.0
        capacity = task_observation.total_resources
        cpu_fraction = self._unit(virtual_usage.cpu_cores / capacity.cpu_cores)
        gpu_fraction = self._unit(virtual_usage.gpu_units / capacity.gpu_units)
        memory_fraction = self._unit(virtual_usage.memory_gb / capacity.memory_gb)
        queue_slacks = [
            (view.latest_start_time - timestamp).total_seconds()
            / (60.0 * self.system_config.step_minutes)
            for view in waiting_tasks
        ]
        candidate_values = np.zeros(10, dtype=np.float32)
        if candidate is not None:
            spec = candidate.spec
            slack = (
                candidate.latest_start_time - timestamp
            ).total_seconds() / (60.0 * self.system_config.step_minutes)
            candidate_values = np.asarray(
                [
                    self._unit(spec.cpu_cores / capacity.cpu_cores),
                    self._unit(spec.gpu_units / capacity.gpu_units),
                    self._unit(spec.memory_gb / capacity.memory_gb),
                    self._unit(spec.duration_steps / self.config.max_duration_steps),
                    self._signed(slack / self.config.max_slack_steps),
                    self._unit(candidate.wait_steps / self.config.max_wait_steps),
                    self._unit(spec.priority / self.config.max_priority),
                    float(spec.deferrable),
                    float(timestamp >= candidate.latest_start_time),
                    float(bool(action_mask[1])),
                ],
                dtype=np.float32,
            )
        mean_wait = (
            sum(view.wait_steps for view in waiting_tasks) / len(waiting_tasks)
            if waiting_tasks
            else 0.0
        )
        mean_cpu = (
            sum(view.spec.cpu_cores / capacity.cpu_cores for view in waiting_tasks)
            / len(waiting_tasks)
            if waiting_tasks
            else 0.0
        )
        mean_gpu = (
            sum(view.spec.gpu_units / capacity.gpu_units for view in waiting_tasks)
            / len(waiting_tasks)
            if waiting_tasks
            else 0.0
        )
        mean_memory = (
            sum(view.spec.memory_gb / capacity.memory_gb for view in waiting_tasks)
            / len(waiting_tasks)
            if waiting_tasks
            else 0.0
        )
        at_risk = sum(timestamp > view.latest_start_time for view in waiting_tasks)
        environment = task_observation.environment
        grid_power = last_result.physical.grid_power_kw if last_result else 0.0
        aggregation = self.system_config.load_aggregation
        utilization_values = (cpu_fraction, gpu_fraction, memory_fraction)
        if str(aggregation.get("mode", "weighted_sum")) == "max_utilization":
            workload_fraction = max(utilization_values)
        else:
            workload_fraction = sum(
                weight * value
                for weight, value in zip(
                    (
                        float(aggregation.get("cpu_weight", 0.45)),
                        float(aggregation.get("gpu_weight", 0.45)),
                        float(aggregation.get("memory_weight", 0.10)),
                    ),
                    utilization_values,
                )
            )
        return {
            "global": np.asarray(
                [
                    self._unit(fraction),
                    math.sin(angle),
                    math.cos(angle),
                    self._unit(1.0 - fraction),
                ],
                dtype=np.float32,
            ),
            "candidate_task": candidate_values,
            "resources": np.asarray(
                [
                    cpu_fraction,
                    gpu_fraction,
                    memory_fraction,
                    self._unit(virtual_availability.cpu_cores / capacity.cpu_cores),
                    self._unit(virtual_availability.gpu_units / capacity.gpu_units),
                    self._unit(virtual_availability.memory_gb / capacity.memory_gb),
                ],
                dtype=np.float32,
            ),
            "queue_summary": np.asarray(
                [
                    self._unit(len(waiting_tasks) / self.config.max_queue_length),
                    self._unit(len(task_observation.running_tasks) / self.config.max_queue_length),
                    self._unit(at_risk / self.config.max_queue_length),
                    self._unit(mean_wait / self.config.max_wait_steps),
                    self._signed(min(queue_slacks, default=0.0) / self.config.max_slack_steps),
                    self._unit(mean_cpu),
                    self._unit(mean_gpu),
                    self._unit(mean_memory),
                ],
                dtype=np.float32,
            ),
            "environment": np.asarray(
                [
                    self._unit(environment.electricity_price_per_kwh / self.config.price_scale),
                    self._unit(environment.carbon_intensity_kg_per_kwh / self.config.carbon_scale),
                    self._range(
                        environment.outdoor_temperature_c,
                        self.config.outdoor_temperature_min_c,
                        self.config.outdoor_temperature_max_c,
                    ),
                    self._unit(environment.renewable_power_kw / self.config.renewable_power_scale_kw),
                ],
                dtype=np.float32,
            ),
            "thermal": np.asarray(
                [
                    self._range(
                        domain_observation.measured_temperature_c,
                        self.config.measured_temperature_min_c,
                        self.config.measured_temperature_max_c,
                    ),
                    self._unit(
                        domain_observation.previous_applied_cooling_kw
                        / float(self.system_config.datacenter["cooling"]["max_cooling_kw"])
                    ),
                    self._unit(workload_fraction),
                    self._unit(grid_power / self.config.grid_power_scale_kw),
                ],
                dtype=np.float32,
            ),
            "action_mask": np.asarray(action_mask, dtype=np.int8),
        }

    def _unit(self, value: float) -> float:
        return self._clip(float(value), 0.0, 1.0)

    def _signed(self, value: float) -> float:
        return self._clip(float(value), -1.0, 1.0)

    def _range(self, value: float, lower: float, upper: float) -> float:
        return self._unit((float(value) - lower) / (upper - lower))

    def _clip(self, value: float, lower: float, upper: float) -> float:
        clipped = min(upper, max(lower, value))
        if clipped != value:
            self.clipping_count += 1
        return clipped
