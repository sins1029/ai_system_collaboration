from __future__ import annotations

from datacenter_env import TaskSpec


def validate_task_dataset(tasks: tuple[TaskSpec, ...]) -> None:
    identifiers = tuple(task.task_id for task in tasks)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("task dataset contains duplicate task ids")
    if tuple(sorted(tasks, key=lambda task: (task.arrival_time, task.task_id))) != tasks:
        raise ValueError("task dataset must be ordered by arrival_time and task_id")
