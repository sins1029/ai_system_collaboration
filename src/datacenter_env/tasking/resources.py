from __future__ import annotations

from datacenter_env.contracts.tasks import (
    ResourceAvailability,
    ResourceCapacity,
    ResourceUsage,
    TaskSpec,
)
from datacenter_env.exceptions import SchedulingError


class ResourcePool:
    def __init__(self, capacity: ResourceCapacity):
        self.capacity = capacity
        self._usage = ResourceUsage()

    @property
    def usage(self) -> ResourceUsage:
        return self._usage

    @property
    def availability(self) -> ResourceAvailability:
        return ResourceAvailability(
            self.capacity.cpu_cores - self._usage.cpu_cores,
            self.capacity.gpu_units - self._usage.gpu_units,
            self.capacity.memory_gb - self._usage.memory_gb,
        )

    def reset(self) -> None:
        self._usage = ResourceUsage()

    def can_allocate(self, task: TaskSpec, usage: ResourceUsage | None = None) -> bool:
        active = usage or self._usage
        epsilon = 1e-9
        return (
            active.cpu_cores + task.cpu_cores <= self.capacity.cpu_cores + epsilon
            and active.gpu_units + task.gpu_units <= self.capacity.gpu_units + epsilon
            and active.memory_gb + task.memory_gb <= self.capacity.memory_gb + epsilon
        )

    def allocate(self, task: TaskSpec) -> None:
        if not self.can_allocate(task):
            raise SchedulingError(f"insufficient resources for task {task.task_id}")
        self._usage = ResourceUsage(
            self._usage.cpu_cores + task.cpu_cores,
            self._usage.gpu_units + task.gpu_units,
            self._usage.memory_gb + task.memory_gb,
        )

    def release(self, task: TaskSpec) -> None:
        self._usage = ResourceUsage(
            max(0.0, self._usage.cpu_cores - task.cpu_cores),
            max(0.0, self._usage.gpu_units - task.gpu_units),
            max(0.0, self._usage.memory_gb - task.memory_gb),
        )

    def task_fits_capacity(self, task: TaskSpec) -> bool:
        return self.can_allocate(task, ResourceUsage())


def add_usage(usage: ResourceUsage, task: TaskSpec) -> ResourceUsage:
    return ResourceUsage(
        usage.cpu_cores + task.cpu_cores,
        usage.gpu_units + task.gpu_units,
        usage.memory_gb + task.memory_gb,
    )
