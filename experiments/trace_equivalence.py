from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from datacenter_env import DataCenterSystem, ForecastWindow, NullRunStore, RunMetadata
from datacenter_env.contracts import TaskStatus
from datacenter_env.gym.adapters import RuleBasedGymPolicy
from datacenter_env.gym.masks import order_task_candidates
from datacenter_env.tasking import build_task_scheduler
from experiments.gymnasium_env import (
    build_standard_gym_components,
    build_standard_gym_environment,
)


SCHEDULER_ALIASES = {
    "fifo": "fifo_immediate",
    "edf": "earliest_deadline_first",
    "energy_aware": "energy_aware_deferral",
}

FLOAT_FIELDS = {
    "workload_fraction",
    "energy_cost",
    "carbon_kg",
    "grid_power_kw",
    "temperature_c",
}

DOMAIN_FIELDS = (
    "simulation_step_index",
    "timestamp",
    "arrived_task_ids",
    "waiting_task_ids_before",
    "running_task_ids_before",
    "scheduler_planned_start_ids",
    "candidate_order",
    "actual_started_task_ids",
    "deferred_task_ids",
    "resource_blocked_task_ids",
    "forced_resource_blocked_task_ids",
    "available_resources",
    "workload_fraction",
    "completed_task_ids",
    "newly_violated_task_ids",
    "waiting_task_ids_after",
    "running_task_ids_after",
    "energy_cost",
    "carbon_kg",
    "grid_power_kw",
    "temperature_c",
)


def run_native_trace(root: Path, scheduler_name: str):
    config, signal_factory, task_factory = build_standard_gym_components(
        root, scheduler_name=scheduler_name
    )
    signals = tuple(signal_factory())
    task_provider = task_factory()
    system = DataCenterSystem.from_config(config.system_config, store=NullRunStore())
    system.start_run(
        RunMetadata(
            name="native_trace_equivalence",
            controller_name="baseline",
            scheduler_name=scheduler_name,
            dataset_name="trace-equivalence",
            seed=2026,
        ),
        seed=2026,
    )
    scheduler = build_task_scheduler(config.system_config)
    scheduler.reset(2026)
    traces: list[dict[str, Any]] = []
    for index, current in enumerate(signals[: config.max_simulation_steps]):
        task_observation = system.prepare_task_step(
            current, task_provider.arrivals_at(current.timestamp)
        )
        forecast = ForecastWindow(
            signals[index : index + scheduler.max_forecast_steps]
        )
        decision = scheduler.schedule(task_observation, forecast)
        result = system.step_with_task_decision(
            current, decision, ForecastWindow((current,))
        )
        outcomes = system.environment.task_outcomes()
        ordered = order_task_candidates(
            task_observation.waiting_tasks, config.candidate_order
        )
        deferred = tuple(
            event.task_id
            for event in result.tasking.events
            if event.event_type.value == "deferred"
        )
        traces.append(
            {
                "simulation_step_index": index,
                "timestamp": current.timestamp.isoformat(),
                "arrived_task_ids": result.tasking.arrived_task_ids,
                "waiting_task_ids_before": tuple(
                    view.spec.task_id for view in task_observation.waiting_tasks
                ),
                "running_task_ids_before": tuple(
                    view.spec.task_id for view in task_observation.running_tasks
                ),
                "scheduler_planned_start_ids": decision.start_task_ids,
                "candidate_order": tuple(view.spec.task_id for view in ordered),
                "gym_actions": (),
                "actual_started_task_ids": result.tasking.started_task_ids,
                "deferred_task_ids": deferred,
                "resource_blocked_task_ids": result.tasking.resource_blocked_task_ids,
                "forced_resource_blocked_task_ids": (
                    result.tasking.forced_resource_blocked_task_ids
                ),
                "auto_skipped_candidate_ids": (),
                "available_resources": (
                    task_observation.available_resources.cpu_cores,
                    task_observation.available_resources.gpu_units,
                    task_observation.available_resources.memory_gb,
                ),
                "workload_fraction": result.tasking.workload_fraction,
                "completed_task_ids": result.tasking.completed_task_ids,
                "newly_violated_task_ids": (
                    result.tasking.newly_sla_violated_task_ids
                ),
                "waiting_task_ids_after": tuple(
                    sorted(
                        item.task_id
                        for item in outcomes
                        if item.final_status is TaskStatus.WAITING
                    )
                ),
                "running_task_ids_after": tuple(
                    sorted(
                        item.task_id
                        for item in outcomes
                        if item.final_status is TaskStatus.RUNNING
                    )
                ),
                "energy_cost": result.accounting.energy_cost,
                "carbon_kg": result.accounting.carbon_kg,
                "grid_power_kw": result.physical.grid_power_kw,
                "temperature_c": result.physical.true_temperature_c,
            }
        )
    summary = system.finish_run()
    return tuple(traces), dict(summary.metrics)


def run_gym_trace(
    root: Path, scheduler_name: str, candidate_order: str | None = None
):
    env = build_standard_gym_environment(
        root, scheduler_name=scheduler_name, candidate_order=candidate_order
    )
    policy = RuleBasedGymPolicy(
        env, build_task_scheduler(env.config.system_config)
    )
    observation, info = env.reset(seed=2026)
    policy.reset(2026)
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.action(observation, info)
        observation, _, terminated, truncated, info = env.step(action)
    traces = tuple(dict(trace) for trace in env.simulation_traces)
    metrics = dict(info["final_metrics"])
    env.close()
    return traces, metrics


def compare_scheduler_traces(root: Path, scheduler: str) -> dict[str, Any]:
    scheduler_name = SCHEDULER_ALIASES.get(scheduler, scheduler)
    native, native_metrics = run_native_trace(root, scheduler_name)
    gym, gym_metrics = run_gym_trace(root, scheduler_name)
    normalized_native = tuple(_normalize_trace(trace) for trace in native)
    normalized_gym = tuple(_normalize_trace(trace) for trace in gym)
    mismatches = tuple(
        index
        for index, (left, right) in enumerate(
            zip(normalized_native, normalized_gym)
        )
        if left != right
    )
    if len(normalized_native) != len(normalized_gym):
        mismatches += tuple(
            range(min(len(native), len(gym)), max(len(native), len(gym)))
        )
    event_fields = (
        "arrived_task_ids",
        "actual_started_task_ids",
        "deferred_task_ids",
        "resource_blocked_task_ids",
        "forced_resource_blocked_task_ids",
        "completed_task_ids",
        "newly_violated_task_ids",
    )
    event_mismatches = sum(
        any(left[field] != right[field] for field in event_fields)
        for left, right in zip(normalized_native, normalized_gym)
    )
    plan_mismatches = sum(
        set(trace["scheduler_planned_start_ids"])
        != set(trace["actual_started_task_ids"])
        for trace in gym
    )
    first = mismatches[0] if mismatches else None
    pre_fix_risk = _find_pre_fix_trigger(root, scheduler_name)
    metric_names = (
        "tasks_completed",
        "tasks_unfinished",
        "sla_violation_count",
        "energy_cost",
        "carbon_kg",
        "total_grid_energy_kwh",
        "temperature_violation_count",
    )
    metric_comparison = {
        name: {
            "native": native_metrics[name],
            "gym": gym_metrics[name],
            "absolute_difference": abs(
                float(native_metrics[name]) - float(gym_metrics[name])
            ),
            "relative_difference": _relative_difference(
                float(native_metrics[name]), float(gym_metrics[name])
            ),
        }
        for name in metric_names
    }
    return {
        "scheduler": scheduler_name,
        "native_metrics": native_metrics,
        "gym_metrics": gym_metrics,
        "metric_comparison": metric_comparison,
        "native_hash": _trajectory_hash(normalized_native),
        "gym_hash": _trajectory_hash(normalized_gym),
        "mismatch_steps": mismatches,
        "mismatch_count": len(mismatches),
        "event_mismatch_count": event_mismatches,
        "planned_executed_mismatch_count": plan_mismatches,
        "first_divergence_step": first,
        "first_divergence_timestamp": (
            native[first]["timestamp"] if first is not None and first < len(native) else None
        ),
        "native_state": native[first] if first is not None and first < len(native) else None,
        "gym_state": gym[first] if first is not None and first < len(gym) else None,
        "pre_fix_risk": pre_fix_risk,
    }


def _normalize_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field in DOMAIN_FIELDS:
        value = trace[field]
        if field in FLOAT_FIELDS:
            normalized[field] = round(float(value), 9)
        elif field == "available_resources":
            normalized[field] = tuple(round(float(item), 9) for item in value)
        else:
            normalized[field] = value
    return normalized


def _trajectory_hash(traces: tuple[Mapping[str, Any], ...]) -> str:
    payload = json.dumps(traces, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _relative_difference(native: float, gym: float) -> float:
    difference = abs(native - gym)
    return difference / abs(native) if native else difference


def _find_pre_fix_trigger(root: Path, scheduler_name: str) -> dict[str, Any] | None:
    traces, _ = run_gym_trace(root, scheduler_name, candidate_order="edf")
    for trace in traces:
        order = trace["candidate_order"]
        planned = trace["scheduler_planned_start_ids"]
        for blocked_id in trace["auto_skipped_candidate_ids"]:
            blocked_position = order.index(blocked_id)
            missed = tuple(
                task_id
                for task_id in planned
                if order.index(task_id) > blocked_position
            )
            if missed:
                return {
                    "simulation_step_index": trace["simulation_step_index"],
                    "timestamp": trace["timestamp"],
                    "blocked_task_id": blocked_id,
                    "blocked_position": blocked_position,
                    "missed_planned_start_ids": missed,
                    "native_planned_start_ids": planned,
                }
    return None
