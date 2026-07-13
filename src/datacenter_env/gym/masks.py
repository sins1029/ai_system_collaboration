from __future__ import annotations

from datetime import datetime

import numpy as np

from datacenter_env.contracts import ResourceAvailability, TaskView


def task_action_mask(
    candidate: TaskView | None,
    timestamp: datetime,
    available: ResourceAvailability,
) -> np.ndarray:
    if candidate is None:
        return np.asarray([0, 0], dtype=np.int8)
    task = candidate.spec
    defer_legal = task.deferrable and timestamp < candidate.latest_start_time
    start_legal = (
        task.cpu_cores <= available.cpu_cores + 1e-9
        and task.gpu_units <= available.gpu_units + 1e-9
        and task.memory_gb <= available.memory_gb + 1e-9
    )
    return np.asarray([int(defer_legal), int(start_legal)], dtype=np.int8)


def order_task_candidates(
    tasks: tuple[TaskView, ...], order: str
) -> tuple[TaskView, ...]:
    if order == "fifo":
        key = lambda view: (view.spec.arrival_time, view.spec.task_id)
    elif order == "priority":
        key = lambda view: (
            -view.spec.priority,
            view.spec.deadline_time,
            view.spec.arrival_time,
            view.spec.task_id,
        )
    elif order == "energy_aware":
        key = lambda view: (
            view.spec.deferrable,
            view.spec.deadline_time,
            -view.spec.priority,
            view.spec.arrival_time,
            view.spec.task_id,
        )
    else:
        key = lambda view: (
            view.spec.deadline_time,
            -view.spec.priority,
            view.spec.arrival_time,
            view.spec.task_id,
        )
    return tuple(sorted(tasks, key=key))
