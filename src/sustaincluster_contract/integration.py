from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np

from sustaincluster_contract.runtime import (
    RuntimeInformationContract,
    controller_duration_minutes,
)
from sustaincluster_contract.workload import extract_tasks_from_row


def bind_cluster_task_extractor(
    cluster: Any,
    contract: RuntimeInformationContract,
) -> Any:
    """Bind the corrected parser to one cluster instance without vendor edits."""
    cluster.information_contract = contract

    def get_tasks_for_timestep(self: Any, current_time: Any) -> list[Any]:
        if self.tasks is None:
            return []
        adjusted_time = current_time.replace(year=2020)
        matches = self.tasks[self.tasks["interval_15m"] == adjusted_time]
        if matches.empty:
            return []
        tasks = extract_tasks_from_row(
            matches.iloc[0],
            scale=1,
            datacenter_configs=self.get_config_list(),
            current_time_utc=current_time,
            logger=self.logger,
            task_scale=5,
            group_size=1,
            contract=self.information_contract,
        )
        if self.logger:
            self.logger.info(
                "contract get_tasks_for_timestep: found %d tasks at %s",
                len(tasks),
                current_time,
            )
        return tasks

    cluster.get_tasks_for_timestep = MethodType(
        get_tasks_for_timestep, cluster
    )
    return cluster


def bind_task_scheduling_env(
    env: Any,
    contract: RuntimeInformationContract,
) -> Any:
    """Make native RL observations use controller-visible runtime estimates."""
    env.information_contract = contract
    env.information_mode = contract.information_mode
    env._generate_per_task_obs_list = MethodType(
        _generate_per_task_obs_list, env
    )
    env._aggregate_task_observations = MethodType(
        _aggregate_task_observations, env
    )
    if not hasattr(env, "_step_without_scheduler_waiting_accounting"):
        env._step_without_scheduler_waiting_accounting = env.step
        env.step = MethodType(_step_with_scheduler_waiting_accounting, env)
    return env


def _step_with_scheduler_waiting_accounting(
    self: Any, actions: Any
) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    """Account once for work retained by the global scheduler's defer action."""
    current_tasks = tuple(self.current_tasks)
    deferred = _scheduler_deferred_tasks(self, actions, current_tasks)
    deferred_ids = {id(task) for task in deferred}
    original_values = {
        id(task): (
            int(getattr(task, "wait_intervals", 0)),
            int(getattr(task, "scheduler_wait_intervals", 0)),
            bool(getattr(task, "temporarily_deferred", False)),
        )
        for task in current_tasks
    }
    deadlines = {id(task): task.sla_deadline for task in current_tasks}

    for task in current_tasks:
        if id(task) in deferred_ids:
            task.increment_wait_intervals()
            task.scheduler_wait_intervals = (
                int(getattr(task, "scheduler_wait_intervals", 0)) + 1
            )
        else:
            task.temporarily_deferred = False

    try:
        observation, reward, terminated, truncated, info = (
            self._step_without_scheduler_waiting_accounting(actions)
        )
    except Exception:
        for task in current_tasks:
            wait, scheduler_wait, temporarily_deferred = original_values[id(task)]
            task.wait_intervals = wait
            task.scheduler_wait_intervals = scheduler_wait
            task.temporarily_deferred = temporarily_deferred
        raise

    for task in current_tasks:
        if task.sla_deadline != deadlines[id(task)]:
            raise RuntimeError("scheduler defer must not move task SLA deadlines")

    result_info = dict(info)
    result_info.update(
        {
            "scheduler_deferred_tasks": len(deferred),
            "scheduler_wait_intervals_added": len(deferred),
            "scheduler_deferred_task_ids": tuple(
                str(task.job_name) for task in deferred
            ),
        }
    )
    return observation, reward, terminated, truncated, result_info


def _scheduler_deferred_tasks(
    env: Any,
    actions: Any,
    current_tasks: tuple[Any, ...],
) -> tuple[Any, ...]:
    if env.disable_defer_action or not current_tasks:
        return ()

    if env.single_action_mode:
        action_values = (int(actions),) * len(current_tasks)
    else:
        action_values = tuple(int(action) for action in actions)
        if len(action_values) != len(current_tasks):
            return ()

    return tuple(
        task
        for task, action in zip(current_tasks, action_values)
        if action == 0 and env.current_time <= task.sla_deadline
    )


def _generate_per_task_obs_list(self: Any) -> list[np.ndarray]:
    day_of_year = self.current_time.dayofyear
    hour_of_day = self.current_time.hour + self.current_time.minute / 60.0
    time_features = [
        np.sin(2 * np.pi * day_of_year / 365.0),
        np.cos(2 * np.pi * day_of_year / 365.0),
        np.sin(2 * np.pi * hour_of_day / 24.0),
        np.cos(2 * np.pi * hour_of_day / 24.0),
    ]
    dc_state_features: list[float] = []
    for dc in self.cluster_manager.datacenters.values():
        dc_state_features.extend(
            [
                dc.available_cores / dc.total_cores if dc.total_cores > 0 else 0,
                dc.available_gpus / dc.total_gpus if dc.total_gpus > 0 else 0,
                dc.available_mem / dc.total_mem_GB if dc.total_mem_GB > 0 else 0,
                float(dc.ci_manager.get_current_ci(norm=False) / 1000.0),
                float(dc.price_manager.get_current_price()) / 100.0,
            ]
        )

    observations = []
    for task in self.current_tasks:
        time_to_deadline = max(
            0.0,
            (task.sla_deadline - self.current_time).total_seconds() / 60.0,
        )
        task_features = [
            float(task.origin_dc_id),
            task.cores_req,
            task.gpu_req,
            controller_duration_minutes(task, self.information_mode),
            time_to_deadline,
        ]
        observations.append(
            np.asarray(
                time_features + task_features + dc_state_features,
                dtype=np.float32,
            )
        )
    return observations


def _aggregate_task_observations(
    self: Any,
    list_of_per_task_obs: list[np.ndarray],
    current_tasks_list: list[Any],
) -> np.ndarray:
    if not list_of_per_task_obs:
        return np.zeros(self.obs_dim_aggregated, dtype=np.float32)
    time_features = np.asarray(list_of_per_task_obs[0][:4], dtype=np.float32)
    task_count = float(len(current_tasks_list))
    if task_count:
        task_features = np.asarray(
            [
                task_count,
                np.mean([task.cores_req for task in current_tasks_list]),
                np.mean([task.gpu_req for task in current_tasks_list]),
                np.mean(
                    [
                        controller_duration_minutes(
                            task, self.information_mode
                        )
                        for task in current_tasks_list
                    ]
                ),
                np.min(
                    [
                        max(
                            0.0,
                            (task.sla_deadline - self.current_time).total_seconds()
                            / 60.0,
                        )
                        for task in current_tasks_list
                    ]
                ),
            ],
            dtype=np.float32,
        )
    else:
        task_features = np.zeros(5, dtype=np.float32)
    dc_features = np.asarray(list_of_per_task_obs[0][9:], dtype=np.float32)
    result = np.concatenate([time_features, task_features, dc_features])
    if result.shape[0] < self.obs_dim_aggregated:
        result = np.pad(result, (0, self.obs_dim_aggregated - result.shape[0]))
    return result[: self.obs_dim_aggregated]
