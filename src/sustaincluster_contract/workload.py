from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import numpy as np

from sustaincluster_contract.runtime import (
    RuntimeEstimateRequest,
    RuntimeInformationContract,
    attach_runtime_information,
)


WORKLOAD_TASK_FIELDS = (
    "job_name",
    "start_time",
    "end_time",
    "start_dt",
    "duration_min",
    "cpu_usage",
    "gpu_wrk_util",
    "avg_mem",
    "avg_gpu_wrk_mem",
    "bandwidth_gb",
    "weekday_name",
    "weekday_num",
)
WORKLOAD_FIELD_INDEX = {
    field: index for index, field in enumerate(WORKLOAD_TASK_FIELDS)
}


def field_value(task_data: Sequence[Any], field: str) -> Any:
    try:
        index = WORKLOAD_FIELD_INDEX[field]
    except KeyError as exc:
        raise KeyError(f"Unknown workload field {field!r}") from exc
    if len(task_data) != len(WORKLOAD_TASK_FIELDS):
        raise ValueError(
            "tasks_matrix row does not match the 12-field workload schema: "
            f"got {len(task_data)} values"
        )
    return task_data[index]


def extract_tasks_from_row(
    row: Any,
    scale: int = 1,
    datacenter_configs: list[dict[str, Any]] | None = None,
    current_time_utc: Any = None,
    logger: Any = None,
    task_scale: int = 5,
    group_size: int = 1,
    *,
    contract: RuntimeInformationContract | None = None,
) -> list[Any]:
    """Map one workload row to Tasks using the explicit field contract."""
    from rl_components.task import Task
    from utils.workload_utils import assign_task_origins

    if scale < 1:
        raise ValueError("scale must be at least 1")
    if task_scale <= 0:
        raise ValueError("task_scale must be positive")
    group_size = max(1, int(group_size))
    contract = contract or RuntimeInformationContract()
    individual_tasks: list[Any] = []

    for task_data in row["tasks_matrix"]:
        source_job_name = str(field_value(task_data, "job_name"))
        true_duration = float(field_value(task_data, "duration_min"))
        cores_req = (
            float(task_scale)
            * float(field_value(task_data, "cpu_usage"))
            / 100.0
        )
        gpu_req = (
            float(task_scale)
            * float(field_value(task_data, "gpu_wrk_util"))
            / 100.0
        )
        mem_req = float(task_scale) * float(field_value(task_data, "avg_mem"))
        bandwidth_gb = float(field_value(task_data, "bandwidth_gb"))
        estimate_request = RuntimeEstimateRequest(
            source_job_name=source_job_name,
            arrival_time=current_time_utc,
            cores_req=cores_req,
            gpu_req=gpu_req,
            mem_req=mem_req,
            bandwidth_gb=bandwidth_gb,
        )
        estimated_duration = contract.estimate_duration(
            estimate_request,
            true_duration,
        )

        task = Task(
            source_job_name,
            current_time_utc,
            true_duration,
            cores_req,
            gpu_req,
            mem_req,
            bandwidth_gb,
        )
        individual_tasks.append(
            attach_runtime_information(
                task,
                source_job_name=source_job_name,
                true_duration=true_duration,
                estimated_duration=estimated_duration,
                contract=contract,
            )
        )

        for index in range(scale - 1):
            varied_task = Task(
                f"{source_job_name}_scaled_{index}",
                current_time_utc,
                true_duration,
                max(0.5, cores_req * np.random.uniform(0.8, 1.2)),
                max(0.0, gpu_req * np.random.uniform(0.8, 1.2)),
                max(0.5, mem_req * np.random.uniform(0.8, 1.2)),
                max(0.1, bandwidth_gb * np.random.uniform(0.8, 1.2)),
            )
            individual_tasks.append(
                attach_runtime_information(
                    varied_task,
                    source_job_name=source_job_name,
                    true_duration=true_duration,
                    estimated_duration=estimated_duration,
                    contract=contract,
                )
            )

    if datacenter_configs and current_time_utc is not None and individual_tasks:
        assign_task_origins(
            individual_tasks,
            datacenter_configs,
            current_time_utc,
            logger=logger,
        )

    if group_size == 1:
        final_tasks = individual_tasks
    else:
        final_tasks = _group_tasks(
            individual_tasks,
            group_size,
            contract,
            Task,
        )

    if logger:
        level = logging.INFO if group_size == 1 else logging.DEBUG
        logger.log(
            level,
            "contract extractor: returning %d tasks/groups at %s",
            len(final_tasks),
            current_time_utc,
        )
    return final_tasks


def _group_tasks(
    tasks: list[Any],
    group_size: int,
    contract: RuntimeInformationContract,
    task_class: Any,
) -> list[Any]:
    grouped: list[Any] = []
    for group_index in range(math.ceil(len(tasks) / group_size)):
        current = tasks[
            group_index * group_size : (group_index + 1) * group_size
        ]
        if not current:
            continue
        true_duration = max(float(task.true_duration) for task in current)
        estimated_duration = max(
            float(task.estimated_duration) for task in current
        )
        source_job_name = f"Group_{group_index + 1}_({current[0].source_job_name})"
        task = task_class(
            source_job_name,
            current[0].arrival_time,
            true_duration,
            sum(item.cores_req for item in current),
            sum(item.gpu_req for item in current),
            sum(item.mem_req for item in current),
            sum(item.bandwidth_gb for item in current),
        )
        attach_runtime_information(
            task,
            source_job_name=source_job_name,
            true_duration=true_duration,
            estimated_duration=estimated_duration,
            contract=contract,
        )
        task.origin_dc_id = current[0].origin_dc_id
        grouped.append(task)
    return grouped
