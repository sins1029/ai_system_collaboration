from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np

from run_one_step_closed_loop import build_env, has_resource_overflow
from stress_scenarios import (
    ScenarioDataCenter,
    ScenarioTask,
    StressScenario,
    build_stress_scenarios,
)
from sustaincluster_mpc import (
    ActionMapping,
    DataCenterSnapshot,
    ExogenousSignalsSnapshot,
    ForecastNoiseConfig,
    FutureArrivalAggregate,
    HorizonDataCenterSnapshot,
    HorizonState,
    HorizonStateAdapter,
    NetworkLinkSnapshot,
    ObjectiveWeights,
    OneStepOptimizer,
    OptimizationConfig,
    RollingHorizonConfig,
    RollingHorizonOptimizer,
    RollingObjectiveWeights,
    SchedulerState,
    SustainClusterActionAdapter,
    SustainClusterStateAdapter,
    TaskDestinationSnapshot,
    TaskSnapshot,
)


COMPONENTS = (
    "electricity",
    "carbon",
    "transmission",
    "waiting_defer",
    "sla_risk",
    "terminal_backlog",
)
ONE_STEP_COMPONENT_MAP = {
    "defer": "waiting_defer",
}
REAL_ROLLING_WEIGHTS = RollingObjectiveWeights(
    electricity=1.0,
    carbon=1.0,
    transmission=1.0,
    waiting_defer=100.0,
    sla_risk=10.0,
    terminal_backlog=100.0,
)
STRESS_ROLLING_WEIGHTS = RollingObjectiveWeights(
    electricity=1000.0,
    carbon=0.1,
    transmission=1.0,
    waiting_defer=0.2,
    sla_risk=50.0,
    terminal_backlog=500.0,
)


@dataclass
class RuntimeTask:
    spec: ScenarioTask
    wait_steps: int = 0


@dataclass(frozen=True)
class ScheduledTask:
    spec: ScenarioTask
    dc_id: int
    dispatch_step: int
    execution_start_step: int
    completion_step: int
    wait_steps: int


def _zero_components() -> dict[str, float]:
    return {name: 0.0 for name in COMPONENTS}


def _accumulate(target: dict[str, float], source: object) -> None:
    values = asdict(source)
    for source_name, value in values.items():
        target_name = ONE_STEP_COMPONENT_MAP.get(source_name, source_name)
        if target_name in target:
            target[target_name] += float(value)


def _summary(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    return mean(values), min(values), max(values)


def _common_metrics(info: dict) -> Iterable[dict]:
    for dc_info in info["datacenter_infos"].values():
        yield dc_info["__common__"]


def _forecast_error(
    state: HorizonState, next_tasks: Iterable[object]
) -> tuple[float, float, float, float]:
    predicted = [item for item in state.future_arrivals if item.arrival_step == 1]
    predicted_values = (
        sum(item.task_count for item in predicted),
        sum(item.cpu_cores for item in predicted),
        sum(item.gpu_units for item in predicted),
        sum(item.memory_gb for item in predicted),
    )
    actual = [
        task
        for task in next_tasks
        if not bool(getattr(task, "temporarily_deferred", False))
    ]
    actual_values = (
        len(actual),
        sum(float(task.cores_req) for task in actual),
        sum(float(task.gpu_req) for task in actual),
        sum(float(task.mem_req) for task in actual),
    )
    return tuple(
        abs(float(predicted_value) - float(actual_value))
        for predicted_value, actual_value in zip(
            predicted_values, actual_values
        )
    )


def run_real_trace(
    repo: Path,
    *,
    algorithm: str,
    horizon: int,
    forecast_mode: str,
    steps: int,
    seed: int,
) -> dict:
    env = build_env(repo, allow_defer=True)
    env.reset(seed=seed)
    actions_adapter = SustainClusterActionAdapter.from_env(env)
    horizon_adapter = HorizonStateAdapter(
        ForecastNoiseConfig(
            demand_relative_std=0.20,
            duration_relative_std=0.20,
            arrival_step_std=0.75,
            seed=2026,
        )
    )
    one_step = OneStepOptimizer(
        OptimizationConfig(
            allow_defer=True,
            weights=ObjectiveWeights(1.0, 1.0, 1.0, 100.0, 10.0),
        )
    )
    rolling = RollingHorizonOptimizer()
    rolling_config = RollingHorizonConfig(
        allow_defer=True, weights=REAL_ROLLING_WEIGHTS
    )

    solve_times: list[float] = []
    variable_counts: list[int] = []
    wait_samples: list[float] = []
    cpu_util: list[float] = []
    gpu_util: list[float] = []
    forecast_errors: list[tuple[float, float, float, float]] = []
    components = _zero_components()
    full_plan_objective = 0.0
    task_decisions = 0
    defers = 0
    completed = 0
    sla_met = 0
    sla_violations = 0
    infeasible = 0
    overflow_steps = 0
    completed_steps = 0

    for _ in range(steps):
        actions_adapter.assert_matches_env(env)
        if algorithm == "one_step":
            current = SustainClusterStateAdapter(env).build_scheduler_state()
            result = one_step.solve(current, actions_adapter)
            current_tasks = current.tasks
            actions = result.environment_actions
            decisions = result.assignments
            variable_count = len(current.tasks) * (
                len(current.datacenters) + 1
            )
            selected_costs = result.costs
            plan_objective = result.objective_value or 0.0
            horizon_state = None
        else:
            horizon_state = horizon_adapter.build_horizon_state(
                env, horizon, forecast_mode
            )
            result = rolling.solve(
                horizon_state, rolling_config, actions_adapter
            )
            current_tasks = horizon_state.current.tasks
            actions = result.environment_actions
            decisions = result.first_step_decisions
            variable_count = result.integer_variable_count
            selected_costs = result.first_step_costs
            plan_objective = result.objective_value or 0.0

        if not result.feasible:
            infeasible += 1
            break
        actions_adapter.validate_actions(current_tasks, actions)
        solve_times.append(float(result.solve_seconds))
        variable_counts.append(variable_count)
        task_decisions += len(decisions)
        defers += sum(item.decision == "defer" for item in decisions)
        wait_samples.extend(float(task.wait_intervals) for task in current_tasks)
        _accumulate(components, selected_costs)
        full_plan_objective += plan_objective

        _, _, terminated, truncated, info = env.step(list(actions))
        completed_steps += 1
        if horizon_state is not None:
            forecast_errors.append(
                _forecast_error(horizon_state, env.current_tasks)
            )
        common = list(_common_metrics(info))
        completed += sum(int(item["finished_tasks_count"]) for item in common)
        sla_met += sum(int(item["__sla__"]["met"]) for item in common)
        sla_violations += sum(
            int(item["__sla__"]["violated"]) for item in common
        )
        cpu_util.extend(float(item["cpu_util_percent"]) / 100.0 for item in common)
        gpu_util.extend(float(item["gpu_util_percent"]) / 100.0 for item in common)
        overflow_steps += int(has_resource_overflow(env))
        if terminated or truncated:
            break

    average_solve, minimum_solve, maximum_solve = _summary(solve_times)
    forecast_mae = tuple(
        mean(values) if values else 0.0
        for values in zip(*forecast_errors)
    ) if forecast_errors else (0.0, 0.0, 0.0, 0.0)
    return {
        "track": "normal_trace",
        "algorithm": algorithm,
        "horizon": horizon,
        "forecast_mode": forecast_mode,
        "requested_steps": steps,
        "completed_steps": completed_steps,
        "task_decisions": task_decisions,
        "defer_ratio": defers / task_decisions if task_decisions else 0.0,
        "average_wait_steps_at_decision": mean(wait_samples) if wait_samples else 0.0,
        "completed_tasks": completed,
        "sla_met": sla_met,
        "sla_violations": sla_violations,
        "average_cpu_utilization": mean(cpu_util) if cpu_util else 0.0,
        "average_gpu_utilization": mean(gpu_util) if gpu_util else 0.0,
        "average_solve_seconds": average_solve,
        "minimum_solve_seconds": minimum_solve,
        "maximum_solve_seconds": maximum_solve,
        "average_integer_variables": mean(variable_counts) if variable_counts else 0.0,
        "maximum_integer_variables": max(variable_counts) if variable_counts else 0,
        "infeasible_count": infeasible,
        "resource_overflow_steps": overflow_steps,
        "executed_stage_cost": sum(components.values()),
        "full_plan_objective_sum": full_plan_objective,
        "cost_components": components,
        "forecast_mae": {
            "task_count": forecast_mae[0],
            "cpu_cores": forecast_mae[1],
            "gpu_units": forecast_mae[2],
            "memory_gb": forecast_mae[3],
        },
    }


def _at(values: tuple[float, ...], step: int) -> float:
    return float(values[min(max(0, step), len(values) - 1)])


def _resource_available(
    dc: ScenarioDataCenter,
    dc_id: int,
    step: int,
    scheduled: list[ScheduledTask],
) -> tuple[float, float, float]:
    base = (
        _at(dc.cpu_capacity, step),
        _at(dc.gpu_capacity, step),
        _at(dc.memory_capacity, step),
    )
    active = [
        task
        for task in scheduled
        if task.dc_id == dc_id
        and task.execution_start_step <= step < task.completion_step
    ]
    used = (
        sum(task.spec.cpu_cores for task in active),
        sum(task.spec.gpu_units for task in active),
        sum(task.spec.memory_gb for task in active),
    )
    return tuple(max(0.0, total - occupied) for total, occupied in zip(base, used))


def _forecast_specs(
    scenario: StressScenario,
    current_step: int,
    horizon: int,
    mode: str,
    noise_scale: float,
    noise_seed: int,
) -> list[ScenarioTask]:
    if mode == "no_future_arrivals":
        return []
    values = [
        task
        for task in scenario.tasks
        if current_step < task.arrival_step < current_step + horizon
    ]
    if mode == "oracle":
        return values
    rng = np.random.default_rng(
        noise_seed + current_step * 1009 + horizon * 9176
    )
    noisy: list[ScenarioTask] = []
    for task in values:
        shift = int(np.rint(rng.normal(0.0, 0.75 * noise_scale)))
        arrival = min(
            current_step + horizon - 1,
            max(current_step + 1, task.arrival_step + shift),
        )
        demand = max(
            0.0, float(rng.normal(1.0, 0.20 * noise_scale))
        )
        duration = max(
            1,
            int(
                math.ceil(
                    task.duration_steps
                    * max(0.1, rng.normal(1.0, 0.20 * noise_scale))
                )
            ),
        )
        noisy.append(
            ScenarioTask(
                task.task_id,
                arrival,
                arrival + max(0, task.deadline_step - task.arrival_step),
                duration,
                task.origin_dc_id,
                task.cpu_cores * demand,
                task.gpu_units * demand,
                task.memory_gb * demand,
                task.bandwidth_gb,
                task.priority,
            )
        )
    return noisy


def _current_dc_snapshot(
    dc: ScenarioDataCenter,
    available: tuple[float, float, float],
    step: int,
) -> DataCenterSnapshot:
    cpu_total = max(dc.cpu_capacity)
    gpu_total = max(dc.gpu_capacity)
    memory_total = max(dc.memory_capacity)
    return DataCenterSnapshot(
        dc_id=dc.dc_id,
        dc_name=f"DC{dc.dc_id}",
        location=f"scenario-{dc.dc_id}",
        cpu_total_cores=cpu_total,
        cpu_available_cores=available[0],
        cpu_reserved_cores=0.0,
        cpu_schedulable_cores=available[0],
        cpu_available_ratio=available[0] / max(cpu_total, 1.0),
        gpu_total_units=max(gpu_total, 1.0),
        gpu_available_units=available[1],
        gpu_reserved_units=0.0,
        gpu_schedulable_units=available[1],
        gpu_available_ratio=available[1] / max(gpu_total, 1.0),
        memory_total_gb=max(memory_total, 1.0),
        memory_available_gb=available[2],
        memory_reserved_gb=0.0,
        memory_schedulable_gb=available[2],
        memory_available_ratio=available[2] / max(memory_total, 1.0),
        running_task_count=0,
        queued_task_count=0,
        in_transit_task_count=0,
        resource_release_times_utc=(),
        electricity_price_usd_per_mwh=_at(dc.electricity_price, step),
        carbon_intensity_gco2_per_kwh=_at(dc.carbon_intensity, step),
        total_power_kw=None,
        it_power_kw=None,
        cooling_power_kw=None,
        internal_temperature_c=None,
        ambient_temperature_c=None,
        crac_setpoint_c=None,
    )


def _build_synthetic_state(
    scenario: StressScenario,
    pending: list[RuntimeTask],
    scheduled: list[ScheduledTask],
    current_step: int,
    horizon: int,
    forecast_mode: str,
    noise_scale: float,
    noise_seed: int,
) -> HorizonState:
    forecast = _forecast_specs(
        scenario,
        current_step,
        horizon,
        forecast_mode,
        noise_scale,
        noise_seed,
    )
    horizon_dcs: list[HorizonDataCenterSnapshot] = []
    current_dcs: list[DataCenterSnapshot] = []
    for dc in scenario.datacenters:
        available = [
            list(
                _resource_available(
                    dc, dc.dc_id, current_step + offset, scheduled
                )
            )
            for offset in range(horizon)
        ]
        forecast_reservations = [[0.0, 0.0, 0.0] for _ in range(horizon)]
        for task in forecast:
            if task.origin_dc_id != dc.dc_id:
                continue
            relative_arrival = task.arrival_step - current_step
            for offset in range(
                max(0, relative_arrival),
                min(horizon, relative_arrival + task.duration_steps),
            ):
                demand = (task.cpu_cores, task.gpu_units, task.memory_gb)
                for resource in range(3):
                    forecast_reservations[offset][resource] += demand[resource]
                    available[offset][resource] = max(
                        0.0, available[offset][resource] - demand[resource]
                    )
        zeros = (0.0,) * horizon
        horizon_dcs.append(
            HorizonDataCenterSnapshot(
                dc_id=dc.dc_id,
                dc_name=f"DC{dc.dc_id}",
                location=f"scenario-{dc.dc_id}",
                cpu_total_cores=max(dc.cpu_capacity),
                gpu_total_units=max(max(dc.gpu_capacity), 1.0),
                memory_total_gb=max(max(dc.memory_capacity), 1.0),
                cpu_available_cores=tuple(item[0] for item in available),
                gpu_available_units=tuple(item[1] for item in available),
                memory_available_gb=tuple(item[2] for item in available),
                known_cpu_reservations=zeros,
                known_gpu_reservations=zeros,
                known_memory_reservations=zeros,
                forecast_cpu_reservations=tuple(item[0] for item in forecast_reservations),
                forecast_gpu_reservations=tuple(item[1] for item in forecast_reservations),
                forecast_memory_reservations=tuple(item[2] for item in forecast_reservations),
                electricity_price_usd_per_mwh=tuple(
                    _at(dc.electricity_price, current_step + offset)
                    for offset in range(horizon)
                ),
                carbon_intensity_gco2_per_kwh=tuple(
                    _at(dc.carbon_intensity, current_step + offset)
                    for offset in range(horizon)
                ),
            )
        )
        current_dcs.append(
            _current_dc_snapshot(
                dc,
                _resource_available(dc, dc.dc_id, current_step, scheduled),
                current_step,
            )
        )

    tasks = tuple(
        TaskSnapshot(
            task_id=item.spec.task_id,
            original_index=index,
            origin_dc_id=item.spec.origin_dc_id,
            cpu_cores=item.spec.cpu_cores,
            gpu_units=item.spec.gpu_units,
            memory_gb=item.spec.memory_gb,
            duration_minutes=item.spec.duration_steps * 15.0,
            remaining_duration_minutes=item.spec.duration_steps * 15.0,
            arrival_time_utc=f"step:{item.spec.arrival_step}",
            sla_deadline_utc=f"step:{item.spec.deadline_step}",
            remaining_sla_minutes=(item.spec.deadline_step - current_step) * 15.0,
            bandwidth_gb=item.spec.bandwidth_gb,
            wait_intervals=item.wait_steps,
            was_deferred=item.wait_steps > 0,
        )
        for index, item in enumerate(pending)
    )
    links = tuple(
        NetworkLinkSnapshot(origin.dc_id, destination.dc_id, 0.0)
        for origin in scenario.datacenters
        for destination in scenario.datacenters
    )
    destinations = tuple(
        TaskDestinationSnapshot(
            task.task_id, task.original_index, dc.dc_id, 0.0, 0.0
        )
        for task in tasks
        for dc in scenario.datacenters
    )
    current = SchedulerState(
        tasks,
        tuple(current_dcs),
        links,
        destinations,
        ExogenousSignalsSnapshot(f"step:{current_step}", 15.0),
        True,
    )
    aggregates = tuple(
        FutureArrivalAggregate(
            arrival_step=arrival,
            origin_dc_id=origin,
            task_count=len(group),
            cpu_cores=sum(task.cpu_cores for task in group),
            gpu_units=sum(task.gpu_units for task in group),
            memory_gb=sum(task.memory_gb for task in group),
            maximum_duration_steps=max(task.duration_steps for task in group),
            minimum_deadline_step=min(
                task.deadline_step - current_step for task in group
            ),
        )
        for (arrival, origin), group in _group_forecast(
            forecast, current_step
        )
    )
    return HorizonState(
        current=current,
        horizon=horizon,
        forecast_mode=forecast_mode,
        timestep_minutes=15.0,
        datacenters=tuple(horizon_dcs),
        running_tasks=(),
        transit_tasks=(),
        future_arrivals=aggregates,
    )


def _group_forecast(
    forecast: list[ScenarioTask], current_step: int
) -> list[tuple[tuple[int, int], list[ScenarioTask]]]:
    groups: dict[tuple[int, int], list[ScenarioTask]] = {}
    for task in forecast:
        key = (task.arrival_step - current_step, task.origin_dc_id)
        groups.setdefault(key, []).append(task)
    return sorted(groups.items())


def _synthetic_action_adapter(
    scenario: StressScenario,
) -> SustainClusterActionAdapter:
    return SustainClusterActionAdapter(
        ActionMapping(
            tuple(
                (dc.dc_id, index + 1)
                for index, dc in enumerate(scenario.datacenters)
            ),
            0,
            len(scenario.datacenters) + 1,
        )
    )


def _find_execution_slot(
    task: ScenarioTask,
    dc: ScenarioDataCenter,
    scheduled: list[ScheduledTask],
    earliest: int,
    final_step: int,
) -> int:
    for start in range(earliest, final_step + 1):
        feasible = True
        for step in range(start, start + task.duration_steps):
            available = _resource_available(dc, dc.dc_id, step, scheduled)
            demand = (task.cpu_cores, task.gpu_units, task.memory_gb)
            if any(required > capacity + 1e-9 for required, capacity in zip(demand, available)):
                feasible = False
                break
        if feasible:
            return start
    return final_step + 1


def run_stress_scenario(
    scenario: StressScenario,
    *,
    algorithm: str,
    horizon: int,
    forecast_mode: str,
    noise_scale: float = 1.0,
    noise_seed: int = 2026,
) -> dict:
    pending: list[RuntimeTask] = []
    scheduled: list[ScheduledTask] = []
    actions_adapter = _synthetic_action_adapter(scenario)
    one_step = OneStepOptimizer(
        OptimizationConfig(
            allow_defer=True,
            weights=ObjectiveWeights(1000.0, 0.1, 1.0, 300.0, 50.0),
        )
    )
    rolling = RollingHorizonOptimizer()
    rolling_config = RollingHorizonConfig(
        allow_defer=True, weights=STRESS_ROLLING_WEIGHTS
    )
    solve_times: list[float] = []
    variable_counts: list[int] = []
    components = _zero_components()
    task_decisions = 0
    defers = 0
    infeasible = 0
    full_plan_objective = 0.0

    for step in range(scenario.steps):
        pending.extend(
            RuntimeTask(task)
            for task in scenario.tasks
            if task.arrival_step == step
        )
        state = _build_synthetic_state(
            scenario,
            pending,
            scheduled,
            step,
            horizon,
            forecast_mode,
            noise_scale,
            noise_seed,
        )
        if algorithm == "one_step":
            result = one_step.solve(state.current, actions_adapter)
            decisions = result.assignments
            selected_costs = result.costs
            variable_count = len(pending) * (len(scenario.datacenters) + 1)
        else:
            result = rolling.solve(state, rolling_config, actions_adapter)
            decisions = result.first_step_decisions
            selected_costs = result.first_step_costs
            variable_count = result.integer_variable_count
        if not result.feasible:
            infeasible += 1
            break

        solve_times.append(float(result.solve_seconds))
        variable_counts.append(variable_count)
        full_plan_objective += float(result.objective_value or 0.0)
        _accumulate(components, selected_costs)
        task_decisions += len(decisions)
        defers += sum(item.decision == "defer" for item in decisions)

        kept: list[RuntimeTask] = []
        dc_by_id = {dc.dc_id: dc for dc in scenario.datacenters}
        for runtime, decision in zip(pending, decisions):
            if decision.decision == "defer":
                runtime.wait_steps += 1
                kept.append(runtime)
                continue
            dc = dc_by_id[int(decision.dc_id)]
            start = _find_execution_slot(
                runtime.spec, dc, scheduled, step + 1, scenario.steps + 8
            )
            scheduled.append(
                ScheduledTask(
                    runtime.spec,
                    dc.dc_id,
                    step,
                    start,
                    start + runtime.spec.duration_steps,
                    runtime.wait_steps + max(0, start - (step + 1)),
                )
            )
        pending = kept

    completion_count = sum(
        task.completion_step <= scenario.steps for task in scheduled
    )
    sla_violations = sum(
        task.completion_step > task.spec.deadline_step for task in scheduled
    ) + len(pending)
    sla_met = len(scheduled) - sum(
        task.completion_step > task.spec.deadline_step for task in scheduled
    )
    wait_values = [task.wait_steps for task in scheduled] + [
        task.wait_steps for task in pending
    ]
    cpu_used = 0.0
    gpu_used = 0.0
    cpu_total = 0.0
    gpu_total = 0.0
    for step in range(scenario.steps):
        for dc in scenario.datacenters:
            active = [
                task
                for task in scheduled
                if task.dc_id == dc.dc_id
                and task.execution_start_step <= step < task.completion_step
            ]
            cpu_used += sum(task.spec.cpu_cores for task in active)
            gpu_used += sum(task.spec.gpu_units for task in active)
            cpu_total += _at(dc.cpu_capacity, step)
            gpu_total += _at(dc.gpu_capacity, step)
    average_solve, minimum_solve, maximum_solve = _summary(solve_times)
    return {
        "track": scenario.key,
        "scenario_name": scenario.name,
        "algorithm": algorithm,
        "horizon": horizon,
        "forecast_mode": forecast_mode,
        "noise_scale": noise_scale if forecast_mode == "noisy_oracle" else None,
        "noise_seed": noise_seed if forecast_mode == "noisy_oracle" else None,
        "task_count": len(scenario.tasks),
        "task_decisions": task_decisions,
        "dispatched_tasks": len(scheduled),
        "completed_tasks": completion_count,
        "remaining_backlog": len(pending),
        "defer_ratio": defers / task_decisions if task_decisions else 0.0,
        "average_wait_steps": mean(wait_values) if wait_values else 0.0,
        "sla_met": sla_met,
        "sla_violations": sla_violations,
        "average_cpu_utilization": cpu_used / cpu_total if cpu_total else 0.0,
        "average_gpu_utilization": gpu_used / gpu_total if gpu_total else 0.0,
        "average_solve_seconds": average_solve,
        "minimum_solve_seconds": minimum_solve,
        "maximum_solve_seconds": maximum_solve,
        "average_integer_variables": mean(variable_counts) if variable_counts else 0.0,
        "maximum_integer_variables": max(variable_counts) if variable_counts else 0,
        "infeasible_count": infeasible,
        "executed_stage_cost": sum(components.values()),
        "full_plan_objective_sum": full_plan_objective,
        "cost_components": components,
        "dispatches": [asdict(item) for item in scheduled],
        "backlog_task_ids": [item.spec.task_id for item in pending],
    }


def experiment_matrix(
    repo: Path, steps: int, seed: int, include_real: bool
) -> dict:
    records: list[dict] = []
    if include_real:
        records.append(
            run_real_trace(
                repo,
                algorithm="one_step",
                horizon=1,
                forecast_mode="not_applicable",
                steps=steps,
                seed=seed,
            )
        )
        for horizon in (1, 2, 4, 8):
            for mode in ("no_future_arrivals", "oracle", "noisy_oracle"):
                records.append(
                    run_real_trace(
                        repo,
                        algorithm="rolling_horizon",
                        horizon=horizon,
                        forecast_mode=mode,
                        steps=steps,
                        seed=seed,
                    )
                )
    for scenario in build_stress_scenarios():
        records.append(
            run_stress_scenario(
                scenario,
                algorithm="one_step",
                horizon=1,
                forecast_mode="not_applicable",
            )
        )
        for horizon in (1, 2, 4, 8):
            for mode in ("no_future_arrivals", "oracle", "noisy_oracle"):
                records.append(
                    run_stress_scenario(
                        scenario,
                        algorithm="rolling_horizon",
                        horizon=horizon,
                        forecast_mode=mode,
                    )
                )
    gpu_burst = build_stress_scenarios()[0]
    for noise_scale in (0.0, 0.5, 1.0, 2.0):
        for noise_seed in range(2026, 2031):
            record = run_stress_scenario(
                gpu_burst,
                algorithm="rolling_horizon",
                horizon=4,
                forecast_mode="noisy_oracle",
                noise_scale=noise_scale,
                noise_seed=noise_seed,
            )
            record["analysis"] = "noise_sensitivity"
            records.append(record)
    return {
        "metadata": {
            "seed": seed,
            "normal_trace_steps": steps,
            "real_trace_included": include_real,
            "solver": "scipy.optimize.milp / HiGHS",
            "real_rolling_weights": asdict(REAL_ROLLING_WEIGHTS),
            "stress_rolling_weights": asdict(STRESS_ROLLING_WEIGHTS),
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stress-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    output = args.output.resolve()
    result = experiment_matrix(
        args.repo.resolve(), args.steps, args.seed, not args.stress_only
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "records": len(result["records"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
