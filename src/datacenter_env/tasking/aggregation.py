from __future__ import annotations

from datacenter_env.contracts.tasks import ResourceCapacity, ResourceUsage
from datacenter_env.exceptions import ConfigurationError


class TaskLoadAggregator:
    def __init__(
        self,
        capacity: ResourceCapacity,
        mode: str = "weighted_sum",
        cpu_weight: float = 0.45,
        gpu_weight: float = 0.45,
        memory_weight: float = 0.10,
    ):
        self.capacity = capacity
        self.mode = str(mode)
        self.weights = (float(cpu_weight), float(gpu_weight), float(memory_weight))
        if self.mode not in {"weighted_sum", "max_utilization"}:
            raise ConfigurationError(f"unknown task load aggregation mode: {self.mode}")
        if any(weight < 0 for weight in self.weights):
            raise ConfigurationError("task load weights must be nonnegative")
        if self.mode == "weighted_sum" and abs(sum(self.weights) - 1.0) > 1e-9:
            raise ConfigurationError("task load weights must sum to 1")

    def utilizations(self, usage: ResourceUsage) -> tuple[float, float, float]:
        return (
            self._clip(usage.cpu_cores / self.capacity.cpu_cores),
            self._clip(usage.gpu_units / self.capacity.gpu_units),
            self._clip(usage.memory_gb / self.capacity.memory_gb),
        )

    def aggregate(self, usage: ResourceUsage) -> float:
        values = self.utilizations(usage)
        if self.mode == "max_utilization":
            return max(values)
        return self._clip(sum(weight * value for weight, value in zip(self.weights, values)))

    @staticmethod
    def _clip(value: float) -> float:
        return min(1.0, max(0.0, float(value)))
