from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from datacenter_env import TaskSpec

from external_workloads.models import TaskDatasetMetadata
from external_workloads.providers import TimelineTaskProvider


def generate_synthetic_task_provider(config: dict) -> TimelineTaskProvider:
    seed = int(config.get("seed", 2026))
    steps = int(config.get("steps", 96))
    step_minutes = int(config.get("time_step_minutes", 15))
    start = datetime.fromisoformat(str(config.get("start_timestamp", "2026-01-01T00:00:00")))
    timezone = str(config.get("timezone", "Asia/Shanghai"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=ZoneInfo(timezone))
    rng = np.random.default_rng(seed)
    tasks: list[TaskSpec] = []
    task_number = 0
    for step in range(steps):
        hour = step * step_minutes / 60.0
        daytime = np.exp(-0.5 * ((hour - 13.0) / 5.0) ** 2)
        arrival_rate = float(config.get("base_arrival_rate", 1.4)) + float(
            config.get("peak_arrival_rate", 3.2)
        ) * daytime
        for _ in range(int(rng.poisson(arrival_rate))):
            task_number += 1
            arrival = start + timedelta(minutes=step * step_minutes)
            gpu_task = bool(rng.random() < float(config.get("gpu_task_fraction", 0.38)))
            duration = int(rng.choice([1, 1, 2, 2, 3, 4, 6]))
            cpu = float(rng.choice([20, 40, 60, 80, 120, 160]))
            gpu = float(rng.choice([4, 8, 12, 16])) if gpu_task else 0.0
            memory = float(rng.choice([32, 64, 96, 128, 192, 256]))
            deferrable = bool(rng.random() >= float(config.get("nondeferrable_fraction", 0.30)))
            if deferrable:
                slack = int(rng.choice([1, 2, 3, 4, 6, 8, 12]))
            else:
                slack = int(rng.choice([0, 0, 1, 2]))
            deadline = arrival + timedelta(minutes=(duration + slack) * step_minutes)
            tasks.append(
                TaskSpec(
                    task_id=f"task-{task_number:05d}",
                    arrival_time=arrival,
                    duration_steps=duration,
                    cpu_cores=cpu,
                    gpu_units=gpu,
                    memory_gb=memory,
                    deadline_time=deadline,
                    priority=int(rng.integers(0, 6)),
                    deferrable=deferrable,
                )
            )
    ordered = tuple(sorted(tasks, key=lambda task: (task.arrival_time, task.task_id)))
    metadata = TaskDatasetMetadata(
        name=str(config.get("dataset_name", "synthetic_single_center_tasks_24h_v1")),
        seed=seed,
        start_timestamp=start,
        end_timestamp=start + timedelta(minutes=(steps - 1) * step_minutes),
        step_minutes=step_minutes,
        number_of_steps=steps,
        task_count=len(ordered),
        generation_parameters=dict(config),
    )
    return TimelineTaskProvider(ordered, metadata)
