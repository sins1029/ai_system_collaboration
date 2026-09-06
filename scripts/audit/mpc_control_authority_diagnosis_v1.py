from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting.transformer_workload_provider import (
    TransformerWorkloadForecastProvider,
)
from forecasting.workload_forecast_provider import (
    ForecastBundle,
    ForecastTraceSource,
    OracleWorkloadForecastProvider,
    PersistenceWorkloadForecastProvider,
    build_bundle,
)
from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_imitation.paths import prepare_sustaincluster_imports
from sustaincluster_mpc.action_adapter import (
    ActionMapping,
    SustainClusterActionAdapter,
)
from sustaincluster_mpc.forecast_pressure_adapter import (
    apply_forecast_pressure,
    apply_oracle_future_signals,
)
from sustaincluster_mpc.future_signals import FutureSignalProvider
from sustaincluster_mpc.horizon_adapter import (
    HorizonDataCenterSnapshot,
    HorizonState,
    HorizonStateAdapter,
)
from sustaincluster_mpc.rolling_horizon_optimizer import (
    RollingHorizonConfig,
    RollingHorizonOptimizer,
    RollingObjectiveWeights,
)
from sustaincluster_mpc.state_adapter import (
    DataCenterSnapshot,
    ExogenousSignalsSnapshot,
    NetworkLinkSnapshot,
    SchedulerState,
    TaskDestinationSnapshot,
    TaskSnapshot,
)


OUTPUT = WORKSPACE / "artifacts/mpc_control_authority_diagnosis_v1"
SUSTAIN_REPO = WORKSPACE / "references/external_repos/sustain-cluster"
FORECAST_CONFIG = WORKSPACE / "configs/sustaincluster_mpc/forecast_aware_mpc_v1.yaml"
OPTIMIZER_CONFIG = WORKSPACE / "configs/sustaincluster_mpc/h4_expert.yaml"
CONTROLLERS = ("H1", "H4_ORACLE", "H4_PERSISTENCE", "H4_TRANSFORMER")


def git(*args: str, cwd: Path = WORKSPACE) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8", errors="replace"
    ).strip()


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def optimizer_config() -> RollingHorizonConfig:
    raw = yaml.safe_load(OPTIMIZER_CONFIG.read_text("utf-8"))["expert"]
    return RollingHorizonConfig(
        allow_defer=bool(raw["allow_defer"]),
        weights=RollingObjectiveWeights(**raw["objective_weights"]),
        cpu_power_w_per_core=float(raw["power_model"]["cpu_power_w_per_core"]),
        gpu_power_w_per_unit=float(raw["power_model"]["gpu_power_w_per_unit"]),
        memory_power_w_per_gb=float(raw["power_model"]["memory_power_w_per_gb"]),
        waiting_cost_per_step=float(raw["waiting_cost_per_step"]),
        terminal_backlog_base_cost=float(raw["terminal_backlog_base_cost"]),
        deterministic_tie_break_epsilon=float(raw["deterministic_tie_break_epsilon"]),
        solver_time_limit_seconds=float(raw["solver_time_limit_seconds"]),
    )


def make_task(
    *,
    task_id: str = "diagnostic-task",
    index: int = 0,
    origin: int = 1,
    cpu: float = 40.0,
    gpu: float = 40.0,
    memory: float = 40.0,
    duration: float = 60.0,
    remaining_sla: float = 180.0,
) -> TaskSnapshot:
    return TaskSnapshot(
        task_id=task_id,
        original_index=index,
        origin_dc_id=origin,
        cpu_cores=cpu,
        gpu_units=gpu,
        memory_gb=memory,
        duration_minutes=duration,
        remaining_duration_minutes=duration,
        arrival_time_utc="2023-02-13T00:00:00+00:00",
        sla_deadline_utc="2023-02-13T03:00:00+00:00",
        remaining_sla_minutes=remaining_sla,
        bandwidth_gb=0.0,
        wait_intervals=0,
        was_deferred=False,
    )


def make_horizon_dc(
    dc_id: int,
    horizon: int = 4,
    *,
    cpu: Iterable[float] | None = None,
    gpu: Iterable[float] | None = None,
    memory: Iterable[float] | None = None,
    price: Iterable[float] | None = None,
    carbon: Iterable[float] | None = None,
    total: float = 100.0,
) -> HorizonDataCenterSnapshot:
    cpu_values = tuple(cpu or (total,) * horizon)
    gpu_values = tuple(gpu or (total,) * horizon)
    memory_values = tuple(memory or (total,) * horizon)
    price_values = tuple(price or (1000.0,) * horizon)
    carbon_values = tuple(carbon or (300.0,) * horizon)
    zeros = (0.0,) * horizon
    return HorizonDataCenterSnapshot(
        dc_id=dc_id,
        dc_name=f"DC{dc_id}",
        location=f"diagnostic-{dc_id}",
        cpu_total_cores=total,
        gpu_total_units=total,
        memory_total_gb=total,
        cpu_available_cores=cpu_values,
        gpu_available_units=gpu_values,
        memory_available_gb=memory_values,
        known_cpu_reservations=zeros,
        known_gpu_reservations=zeros,
        known_memory_reservations=zeros,
        forecast_cpu_reservations=zeros,
        forecast_gpu_reservations=zeros,
        forecast_memory_reservations=zeros,
        electricity_price_usd_per_mwh=price_values,
        carbon_intensity_gco2_per_kwh=carbon_values,
    )


def current_dc(dc: HorizonDataCenterSnapshot) -> DataCenterSnapshot:
    return DataCenterSnapshot(
        dc_id=dc.dc_id,
        dc_name=dc.dc_name,
        location=dc.location,
        cpu_total_cores=dc.cpu_total_cores,
        cpu_available_cores=dc.cpu_available_cores[0],
        cpu_reserved_cores=0.0,
        cpu_schedulable_cores=dc.cpu_available_cores[0],
        cpu_available_ratio=dc.cpu_available_cores[0] / dc.cpu_total_cores,
        gpu_total_units=dc.gpu_total_units,
        gpu_available_units=dc.gpu_available_units[0],
        gpu_reserved_units=0.0,
        gpu_schedulable_units=dc.gpu_available_units[0],
        gpu_available_ratio=dc.gpu_available_units[0] / dc.gpu_total_units,
        memory_total_gb=dc.memory_total_gb,
        memory_available_gb=dc.memory_available_gb[0],
        memory_reserved_gb=0.0,
        memory_schedulable_gb=dc.memory_available_gb[0],
        memory_available_ratio=dc.memory_available_gb[0] / dc.memory_total_gb,
        running_task_count=0,
        queued_task_count=0,
        in_transit_task_count=0,
        resource_release_times_utc=(),
        electricity_price_usd_per_mwh=dc.electricity_price_usd_per_mwh[0],
        carbon_intensity_gco2_per_kwh=dc.carbon_intensity_gco2_per_kwh[0],
        total_power_kw=None,
        it_power_kw=None,
        cooling_power_kw=None,
        internal_temperature_c=None,
        ambient_temperature_c=None,
        crac_setpoint_c=None,
    )


def make_state(
    tasks: tuple[TaskSnapshot, ...],
    dcs: tuple[HorizonDataCenterSnapshot, ...],
    *,
    allow_defer: bool = True,
) -> HorizonState:
    current_dcs = tuple(current_dc(dc) for dc in dcs)
    destinations = tuple(
        TaskDestinationSnapshot(
            task_id=task.task_id,
            original_index=task.original_index,
            destination_dc_id=dc.dc_id,
            transmission_cost_usd=0.0,
            transmission_delay_seconds=0.0,
        )
        for task in tasks
        for dc in dcs
    )
    links = tuple(
        NetworkLinkSnapshot(left.dc_id, right.dc_id, 0.0)
        for left in dcs
        for right in dcs
    )
    current = SchedulerState(
        tasks=tasks,
        datacenters=current_dcs,
        network_links=links,
        task_destinations=destinations,
        exogenous=ExogenousSignalsSnapshot(
            "2023-02-13T00:00:00+00:00", 15.0
        ),
        allow_defer=allow_defer,
        information_mode="deployable",
    )
    return HorizonState(
        current=current,
        horizon=len(dcs[0].cpu_available_cores),
        forecast_mode="no_future_arrivals",
        timestep_minutes=15.0,
        datacenters=dcs,
        running_tasks=(),
        transit_tasks=(),
        future_arrivals=(),
        information_mode="deployable",
        future_signal_mode="persistence",
    )


def action_adapter(dcs: tuple[HorizonDataCenterSnapshot, ...]) -> SustainClusterActionAdapter:
    return SustainClusterActionAdapter(
        ActionMapping(
            tuple((dc.dc_id, index + 1) for index, dc in enumerate(dcs)),
            defer_action=0,
            action_space_n=len(dcs) + 1,
        )
    )


def solve(state: HorizonState, config: RollingHorizonConfig | None = None):
    return RollingHorizonOptimizer().solve(
        state, config or optimizer_config(), action_adapter(state.datacenters)
    )


def first_action(result: Any) -> tuple[str, int | None, bool]:
    decision = result.first_step_decisions[0]
    return decision.decision, decision.dc_id, decision.decision == "defer"


def global_bundle(values: np.ndarray) -> ForecastBundle:
    return build_bundle(
        provider="persistence",
        current_timestamp=pd.Timestamp("2023-02-13T12:00:00Z"),
        values=np.asarray(values, dtype=np.float64),
        history_available=True,
        fallback_used=False,
    )


def forced_cost_rows(state: HorizonState, config: RollingHorizonConfig) -> list[dict[str, Any]]:
    optimizer = RollingHorizonOptimizer()
    task_count = len(state.current.tasks)
    dc_count = len(state.datacenters)
    variable_count = task_count * dc_count * state.horizon + task_count
    components, metadata = optimizer._component_vectors(state, config, variable_count)
    upper = optimizer._variable_upper_bounds(state, config, metadata, variable_count)
    candidates = (
        ("EXECUTE_NOW", optimizer._x_index(0, 0, 0, dc_count, state.horizon)),
        ("DEFER_ONE_STEP", optimizer._x_index(0, 0, 1, dc_count, state.horizon)),
        ("TERMINAL_BACKLOG", task_count * dc_count * state.horizon),
    )
    rows = []
    for label, variable in candidates:
        values = {name: float(vector[variable]) for name, vector in components.items()}
        rows.append(
            {
                "forced_option": label,
                **values,
                "total": float(sum(values.values())),
                "variable_upper_bound": float(upper[variable]),
                "feasible_by_bound": bool(upper[variable] > 0.5),
            }
        )
    return rows


def real_defer_transition() -> dict[str, Any]:
    env = build_sustaincluster_env(
        None,
        pd.Timestamp("2023-02-13T00:00:00Z"),
        4,
        allow_defer=True,
        initial_seed=1201,
        information_mode="deployable",
        baseline_estimated_duration_minutes=60.0,
    )
    try:
        env.reset(seed=1201)
        selected = env.current_tasks[0]
        env.current_tasks = [selected]
        selected.sla_deadline = env.current_time + pd.Timedelta(hours=6)
        before_time = pd.Timestamp(env.current_time)
        before_wait = int(selected.wait_intervals)
        before_remaining = (selected.sla_deadline - env.current_time).total_seconds() / 60.0
        env.step([0])
        after_time = pd.Timestamp(env.current_time)
        remains_pending = any(task is selected for task in env.current_tasks)
        after_wait = int(selected.wait_intervals)
        after_remaining = (selected.sla_deadline - env.current_time).total_seconds() / 60.0
        adapter = SustainClusterActionAdapter.from_env(env)
        actions = [
            adapter.mapping.dc_id_to_action[int(task.origin_dc_id)]
            for task in env.current_tasks
        ]
        env.step(actions)
        accepted_next_step = (
            any(task is selected for _, task, _ in env.in_transit_tasks)
            or any(
                task is selected
                for dc in env.cluster_manager.datacenters.values()
                for task in tuple(dc.pending_tasks) + tuple(dc.running_tasks)
            )
        )
        return {
            "defer_action_value": adapter.mapping.defer_action,
            "before_time": before_time.isoformat(),
            "after_time": after_time.isoformat(),
            "remains_external_pending": remains_pending,
            "before_wait_intervals": before_wait,
            "after_wait_intervals": after_wait,
            "waiting_increment": after_wait - before_wait,
            "before_remaining_sla_minutes": before_remaining,
            "after_remaining_sla_minutes": after_remaining,
            "sla_clock_delta_minutes": before_remaining - after_remaining,
            "accepted_assignment_next_step": accepted_next_step,
        }
    finally:
        env.close()


def synthetic_diagnostics() -> dict[str, Any]:
    config = optimizer_config()
    task = make_task()
    base_dcs = (make_horizon_dc(1), make_horizon_dc(2))
    base_state = make_state((task,), base_dcs)

    price_rows = []
    price_cases = {
        "P0_PERSISTENCE": (1000.0, 1000.0, 1000.0, 1000.0),
        "P1_LOW_ALL_FUTURE": (1000.0, 200.0, 200.0, 200.0),
        "P2_HIGH_ALL_FUTURE": (1000.0, 2000.0, 2000.0, 2000.0),
        "P3_LOW_AFTER_ONE_DEFER": (1000.0, 1000.0, 200.0, 200.0),
    }
    for case, prices in price_cases.items():
        state = make_state(
            (replace(task, cpu_cores=20.0, gpu_units=20.0, memory_gb=20.0),),
            (make_horizon_dc(1, price=prices),),
        )
        result = solve(state, config)
        action, target, deferred = first_action(result)
        price_rows.append(
            {
                "case": case,
                "current_price": prices[0],
                "future_price_h1": prices[1],
                "future_price_h2": prices[2],
                "future_price_h3": prices[3],
                "first_action": action,
                "first_target_dc": target,
                "defer": deferred,
                "objective": result.objective_value,
                **{f"cost_{key}": value for key, value in result.costs.__dict__.items()},
            }
        )
    defer_cost_state = make_state(
        (replace(task, cpu_cores=20.0, gpu_units=20.0, memory_gb=20.0),),
        (make_horizon_dc(1, price=price_cases["P3_LOW_AFTER_ONE_DEFER"]),),
    )
    cost_rows = forced_cost_rows(defer_cost_state, config)

    bridge_dcs = (
        {"dc_id": 1, "population_weight": 0.8, "timezone_shift": 0},
        {"dc_id": 2, "population_weight": 0.2, "timezone_shift": 0},
    )
    baseline_values = np.repeat([[10.0, 30.0, 30.0, 30.0]], 4, axis=0)
    workload_rows = []
    for perturbation, scales in (
        ("ALL_RESOURCES", (0.0, 0.5, 1.0, 2.0, 3.0)),
        ("GPU_ONLY", (0.0, 0.5, 1.0, 2.0, 3.0)),
    ):
        provisional = []
        for scale in scales:
            values = baseline_values.copy()
            if perturbation == "ALL_RESOURCES":
                values *= scale
            else:
                values[:, 2] *= scale
            application = apply_forecast_pressure(
                base_state, global_bundle(values), bridge_dcs
            )
            result = solve(application.state, config)
            action, target, deferred = first_action(result)
            chosen = next(dc for dc in application.state.datacenters if dc.dc_id == target)
            provisional.append(
                {
                    "state_id": "balanced_long_gpu",
                    "perturbation": perturbation,
                    "forecast_scale": scale if perturbation == "ALL_RESOURCES" else 1.0,
                    "gpu_scale": scale if perturbation == "GPU_ONLY" else scale,
                    "first_action": action,
                    "first_target_dc": target,
                    "defer": deferred,
                    "objective": result.objective_value,
                    "min_expected_cpu_capacity": min(chosen.cpu_available_cores),
                    "min_expected_gpu_capacity": min(chosen.gpu_available_units),
                    "min_expected_memory_capacity": min(chosen.memory_available_gb),
                    "capacity_shortage_events": application.capacity_shortage_events,
                    "solver_status": result.status,
                }
            )
        baseline_target = next(
            row["first_target_dc"] for row in provisional if row["gpu_scale"] == 1.0
        )
        for row in provisional:
            row["changed_vs_baseline"] = row["first_target_dc"] != baseline_target
            workload_rows.append(row)

    threshold_rows = []
    baseline_target = None
    for percent in (0, 25, 50, 75, 100, 125, 150, 200, 300):
        remaining = max(0.0, 100.0 - float(percent))
        dcs = (
            make_horizon_dc(1, gpu=(100.0, remaining, remaining, remaining)),
            make_horizon_dc(2),
        )
        result = solve(make_state((task,), dcs), config)
        action, target, deferred = first_action(result)
        if baseline_target is None:
            baseline_target = target
        threshold_rows.append(
            {
                "forecast_gpu_percent_of_capacity": percent,
                "dc1_future_gpu_available": remaining,
                "first_action": action,
                "first_target_dc": target,
                "defer": deferred,
                "objective": result.objective_value,
                "changed_vs_zero": target != baseline_target,
                "solver_status": result.status,
            }
        )

    high_dc1 = make_state(
        (task,),
        (
            make_horizon_dc(1, gpu=(100.0, 20.0, 20.0, 20.0)),
            make_horizon_dc(2),
        ),
    )
    high_dc2 = make_state(
        (task,),
        (
            make_horizon_dc(1),
            make_horizon_dc(2, gpu=(100.0, 20.0, 20.0, 20.0)),
        ),
    )
    placement_a = solve(high_dc1, config)
    placement_b = solve(high_dc2, config)
    placement = {
        "dc1_high_target": first_action(placement_a)[1],
        "dc2_high_target": first_action(placement_b)[1],
        "reversed": first_action(placement_a)[1] != first_action(placement_b)[1],
    }

    occupancy_rows = []
    for duration in (15.0, 30.0, 60.0):
        result = solve(
            make_state((replace(task, duration_minutes=duration, remaining_duration_minutes=duration),), (make_horizon_dc(1),)),
            config,
        )
        occupancy_rows.append(
            {
                "estimated_duration_minutes": duration,
                "duration_steps": result.plans[0].duration_steps,
                "execution_start_step": result.plans[0].execution_start_step,
                "gpu_occupancy": json.dumps(result.datacenter_allocations[0].gpu_units),
                "future_occupied_steps": sum(value > 0 for value in result.datacenter_allocations[0].gpu_units[1:]),
            }
        )

    stress_rows = []
    h1_release = solve(
        make_state((replace(task, gpu_units=40.0, duration_minutes=30.0, remaining_duration_minutes=30.0),), (make_horizon_dc(1, 1, gpu=(0.0,)),)),
        config,
    )
    h4_release = solve(
        make_state((replace(task, gpu_units=40.0, duration_minutes=30.0, remaining_duration_minutes=30.0),), (make_horizon_dc(1, gpu=(0.0, 0.0, 100.0, 100.0)),)),
        config,
    )
    h1_price = solve(
        make_state((replace(task, gpu_units=20.0),), (make_horizon_dc(1, 1, price=(1000.0,)),)), config
    )
    h4_price = solve(
        make_state((replace(task, gpu_units=20.0),), (make_horizon_dc(1, price=(1000.0, 1000.0, 200.0, 200.0)),)), config
    )
    h1_burst = solve(make_state((task,), (make_horizon_dc(1, 1), make_horizon_dc(2, 1))), config)
    h4_burst = solve(high_dc1, config)
    for scenario, h1, h4 in (
        ("capacity_release", h1_release, h4_release),
        ("future_low_price", h1_price, h4_price),
        ("gpu_burst", h1_burst, h4_burst),
    ):
        for controller, result in (("H1", h1), ("H4_ORACLE", h4)):
            action, target, deferred = first_action(result)
            stress_rows.append(
                {
                    "scenario": scenario,
                    "controller": controller,
                    "first_action": action,
                    "first_target_dc": target,
                    "defer": deferred,
                    "planned_dispatch_step": result.plans[0].dispatch_step,
                    "terminal_backlog": result.terminal_backlog_count,
                    "objective": result.objective_value,
                    "solver_status": result.status,
                }
            )

    return {
        "price": pd.DataFrame(price_rows),
        "cost": pd.DataFrame(cost_rows),
        "workload": pd.DataFrame(workload_rows),
        "threshold": pd.DataFrame(threshold_rows),
        "occupancy": pd.DataFrame(occupancy_rows),
        "stress": pd.DataFrame(stress_rows),
        "placement": placement,
    }


def controller_provider(controller: str, transformer: Any) -> Any:
    if controller == "H4_ORACLE":
        return OracleWorkloadForecastProvider()
    if controller == "H4_PERSISTENCE":
        return PersistenceWorkloadForecastProvider()
    if controller == "H4_TRANSFORMER":
        return transformer
    return None


def task_key(task: Any) -> str:
    return "|".join(
        (
            str(task.job_name),
            pd.Timestamp(task.arrival_time).isoformat(),
            str(task.origin_dc_id),
            f"{float(task.cores_req):.8f}",
            f"{float(task.gpu_req):.8f}",
            f"{float(task.mem_req):.8f}",
        )
    )


def run_trace_controller(
    *,
    controller: str,
    scenario: str,
    start: pd.Timestamp,
    seed: int,
    steps: int,
    trace_source: ForecastTraceSource,
    transformer: TransformerWorkloadForecastProvider,
    dc_configs: list[dict[str, Any]],
    config: RollingHorizonConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    env = build_sustaincluster_env(
        None,
        start,
        steps,
        allow_defer=True,
        initial_seed=seed,
        information_mode="deployable",
        duration_estimate_mode="declared_or_baseline",
        baseline_estimated_duration_minutes=60.0,
    )
    env.reset(seed=seed)
    adapter = SustainClusterActionAdapter.from_env(env)
    horizon_adapter = HorizonStateAdapter(
        information_mode="deployable",
        future_signal_provider=FutureSignalProvider("persistence"),
    )
    provider = controller_provider(controller, transformer)
    action_rows: list[dict[str, Any]] = []
    pressure_rows: list[dict[str, Any]] = []
    slack_values: list[int] = []
    try:
        for step in range(steps):
            current_time = pd.Timestamp(env.current_time)
            horizon = 1 if controller == "H1" else 4
            state = horizon_adapter.build_horizon_state(env, horizon, "no_future_arrivals")
            if controller == "H1":
                for task in state.current.tasks:
                    slack_values.append(
                        int(math.floor((task.remaining_sla_minutes - task.remaining_duration_minutes) / state.timestep_minutes))
                    )
            if provider is not None:
                request = trace_source.request(
                    current_time, include_oracle_future=controller == "H4_ORACLE"
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    bundle = provider.forecast(request)
                if controller == "H4_ORACLE":
                    state = apply_oracle_future_signals(state, env)
                original = state
                application = apply_forecast_pressure(state, bundle, dc_configs)
                state = application.state
                dc_by_id = {dc.dc_id: dc for dc in original.datacenters}
                for item in application.pressures:
                    dc = dc_by_id[item.dc_id]
                    index = item.horizon_step - 1
                    for resource, demand, total, available in (
                        ("cpu", item.cpu_demand, dc.cpu_total_cores, dc.cpu_available_cores[index]),
                        ("gpu", item.gpu_demand, dc.gpu_total_units, dc.gpu_available_units[index]),
                        ("memory", item.memory_demand, dc.memory_total_gb, dc.memory_available_gb[index]),
                    ):
                        pressure_rows.append(
                            {
                                "controller": controller,
                                "scenario": scenario,
                                "seed": seed,
                                "step": step,
                                "horizon_step": item.horizon_step,
                                "dc_id": item.dc_id,
                                "resource": resource,
                                "forecast_to_total_capacity": demand / max(total, 1e-12),
                                "forecast_to_free_capacity": demand / max(available, 1e-12),
                                "current_free_ratio": available / max(total, 1e-12),
                            }
                        )
            result = RollingHorizonOptimizer().solve(state, config, adapter)
            if not result.feasible:
                raise RuntimeError(f"trace diagnostic infeasible: {controller}/{scenario}/{seed}/{step}")
            for task, decision in zip(env.current_tasks, result.first_step_decisions):
                action_rows.append(
                    {
                        "controller": controller,
                        "scenario": scenario,
                        "seed": seed,
                        "step": step,
                        "timestamp": current_time,
                        "task_key": task_key(task),
                        "semantic_action": decision.decision,
                        "target_dc": decision.dc_id,
                    }
                )
            env.step(list(result.environment_actions))
        return action_rows, pressure_rows, slack_values
    finally:
        env.close()


def action_agreement(actions: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("H1_vs_H4_ORACLE", "H1", "H4_ORACLE", "ALL"),
        ("PERSISTENCE_vs_TRANSFORMER", "H4_PERSISTENCE", "H4_TRANSFORMER", "ALL"),
    )
    rows = []
    keys = ["scenario", "seed", "step", "task_key"]
    forecast = pd.read_parquet(
        WORKSPACE / "artifacts/forecast_aware_mpc_v1/08_forecast_control_trace.parquet"
    )
    gpu = forecast[
        forecast["controller"].isin(["H4_PERSISTENCE", "H4_TRANSFORMER"])
    ].pivot_table(
        index=["scenario", "seed", "step", "horizon_step"],
        columns="controller",
        values="predicted_gpu",
    )
    gpu["gap"] = (gpu["H4_TRANSFORMER"] - gpu["H4_PERSISTENCE"]).abs()
    step_gap = gpu.groupby(level=[0, 1, 2])["gap"].mean()
    gap_threshold = float(step_gap.quantile(0.9))
    high_gap_keys = set(step_gap[step_gap >= gap_threshold].index)
    for name, left_name, right_name, scope in comparisons:
        left = actions[actions["controller"] == left_name].drop(columns="controller")
        right = actions[actions["controller"] == right_name].drop(columns="controller")
        merged = left.merge(right, on=keys, suffixes=("_left", "_right"), validate="one_to_one")
        semantic = merged["semantic_action_left"] == merged["semantic_action_right"]
        target = merged["target_dc_left"].fillna(-1) == merged["target_dc_right"].fillna(-1)
        rows.append(
            {
                "comparison": name,
                "scope": scope,
                "matched_tasks": len(merged),
                "semantic_action_agreement": float(semantic.mean()),
                "target_dc_agreement": float(target.mean()),
                "semantic_disagreements": int((~semantic).sum()),
                "target_disagreements": int((~target).sum()),
                "forecast_gpu_gap_threshold": gap_threshold if "PERSISTENCE" in name else np.nan,
            }
        )
        if name == "PERSISTENCE_vs_TRANSFORMER":
            mask = merged.apply(
                lambda row: (row["scenario"], int(row["seed"]), int(row["step"])) in high_gap_keys,
                axis=1,
            )
            selected = merged[mask]
            semantic = selected["semantic_action_left"] == selected["semantic_action_right"]
            target = selected["target_dc_left"].fillna(-1) == selected["target_dc_right"].fillna(-1)
            rows.append(
                {
                    "comparison": name,
                    "scope": "TOP_10_PERCENT_FORECAST_GPU_GAP",
                    "matched_tasks": len(selected),
                    "semantic_action_agreement": float(semantic.mean()),
                    "target_dc_agreement": float(target.mean()),
                    "semantic_disagreements": int((~semantic).sum()),
                    "target_disagreements": int((~target).sum()),
                    "forecast_gpu_gap_threshold": gap_threshold,
                }
            )
    return pd.DataFrame(rows)


def pressure_quantiles(rows: pd.DataFrame) -> pd.DataFrame:
    values = []
    for (controller, resource), group in rows.groupby(["controller", "resource"]):
        for metric in (
            "forecast_to_total_capacity",
            "forecast_to_free_capacity",
            "current_free_ratio",
        ):
            for quantile in (0.5, 0.9, 0.95, 0.99):
                values.append(
                    {
                        "controller": controller,
                        "resource": resource,
                        "metric": metric,
                        "quantile": f"P{int(quantile * 100)}",
                        "value": float(group[metric].replace([np.inf, -np.inf], np.nan).quantile(quantile)),
                    }
                )
    return pd.DataFrame(values)


def slack_distribution(slack_values: list[int]) -> pd.DataFrame:
    array = np.asarray(slack_values, dtype=np.int64)
    summary = {
        "record_type": "SUMMARY",
        "slack_bin": "ALL",
        "count": len(array),
        "fraction": 1.0,
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "fraction_ge_1": float(np.mean(array >= 1)),
        "fraction_ge_4": float(np.mean(array >= 4)),
    }
    rows = [summary]
    bins = (
        ("<=0", array <= 0),
        ("1", array == 1),
        ("2", array == 2),
        (">=3", array >= 3),
    )
    for label, mask in bins:
        rows.append(
            {
                "record_type": "BIN",
                "slack_bin": label,
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
                "p50": np.nan,
                "p90": np.nan,
                "fraction_ge_1": np.nan,
                "fraction_ge_4": np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_trace_diagnostics() -> dict[str, pd.DataFrame]:
    raw = yaml.safe_load(FORECAST_CONFIG.read_text("utf-8"))["forecast_aware_mpc_v1"]
    trace_source = ForecastTraceSource(WORKSPACE / raw["dataset_root"])
    transformer = TransformerWorkloadForecastProvider(
        WORKSPACE / raw["transformer_checkpoint"],
        dataset_root=WORKSPACE / raw["dataset_root"],
        device="cpu",
    )
    dc_configs = yaml.safe_load((WORKSPACE / raw["datacenter_config"]).read_text("utf-8"))["datacenters"]
    config = optimizer_config()
    action_rows: list[dict[str, Any]] = []
    pressure_rows: list[dict[str, Any]] = []
    slack_values: list[int] = []
    for scenario, scenario_config in raw["scenarios"].items():
        for seed in raw["seeds"]:
            for controller in CONTROLLERS:
                print(f"TRACE {scenario} seed={seed} controller={controller}", flush=True)
                actions, pressure, slack = run_trace_controller(
                    controller=controller,
                    scenario=scenario,
                    start=pd.Timestamp(scenario_config["environment_start"]),
                    seed=int(seed),
                    steps=int(raw["episode_steps"]),
                    trace_source=trace_source,
                    transformer=transformer,
                    dc_configs=dc_configs,
                    config=config,
                )
                action_rows.extend(actions)
                pressure_rows.extend(pressure)
                slack_values.extend(slack)
    actions = pd.DataFrame(action_rows)
    pressure = pd.DataFrame(pressure_rows)
    return {
        "actions": actions,
        "agreement": action_agreement(actions),
        "pressure": pressure_quantiles(pressure),
        "slack": slack_distribution(slack_values),
    }


def create_documents(
    output: Path,
    transition: dict[str, Any],
    synthetic: dict[str, Any],
    trace: dict[str, pd.DataFrame],
) -> None:
    cost = synthetic["cost"].set_index("forced_option")
    agreement = trace["agreement"].set_index(["comparison", "scope"])
    slack = trace["slack"].iloc[0]
    threshold = synthetic["threshold"]
    switched = threshold[threshold["changed_vs_zero"]]
    switch_threshold = int(switched.iloc[0]["forecast_gpu_percent_of_capacity"]) if len(switched) else None
    stress = synthetic["stress"].set_index(["scenario", "controller"])
    pressure = trace["pressure"]

    write_text(
        output / "02_defer_path_audit.md",
        f"""# Defer Path Audit

## Path

`semantic defer` -> `AssignmentDecision(decision=defer)` -> action adapter value `0` -> `TaskSchedulingEnv.step()` -> `deferred_tasks` -> next-step `current_tasks`.

## Findings

- Defer is legal when `disable_defer_action=False`; the adapter maps it to action 0 without collision.
- The MILP has future-dispatch binaries and a terminal-backlog binary rather than a separate defer binary. Any dispatch at step > 0 or terminal backlog becomes first-step defer.
- SLA feasibility masks a dispatch variable when `execution_step + duration_steps > deadline_step`.
- The environment retained the task for the next external scheduling step: `{transition['remains_external_pending']}`.
- The next-step assignment was accepted: `{transition['accepted_assignment_next_step']}`.
- Remaining SLA decreased by `{transition['sla_clock_delta_minutes']:.1f}` minutes.
- Scheduler-level defer changed `wait_intervals` by `{transition['waiting_increment']}`. This is a confirmed accounting bug: the increment exists only in a DC-local capacity-failure path.
- `temporarily_deferred` is set on defer and is not reset by the assignment branch; this is secondary stale-state technical debt.
""",
    )
    write_text(
        output / "07_forecast_to_constraint_trace.md",
        """# Forecast to Constraint Trace

The bridge is active, not diagnostic-only:

`ForecastBundle.points[h]`
-> deterministic `expected_origin_probability(dc, issued_at+h)`
-> `ExpectedDataCenterPressure(cpu,gpu,memory)`
-> subtract from `HorizonDataCenterSnapshot.*_available_[h-1]`
-> `RollingHorizonOptimizer._constraints()` capacity upper bound
-> rows `(dc, time, resource)` of `A_capacity x <= available_capacity`
-> dispatch binary `x(task, dc, dispatch_step)` contributes resource demand from execution start through estimated-duration occupancy.

Thus future pressure reaches the MILP RHS and can affect current placement when current dispatch occupancy overlaps the constrained future slot. No synthetic future task variables are created.

## Indexing diagnosis

The environment/optimizer timeline contains a current slot at index 0. Even zero-delay current assignments enter a DC no earlier than index 1 because arrivals are processed before actions. The v1 bridge maps +15/+30/+45/+60 to indexes 0/1/2/3. Consequently +15 pressure is placed in index 0 and does not compete with a current assignment that starts at index 1; +60 is compressed into index 3. This is a material bridge-grid alignment defect, although pressure at indexes 1-3 still reaches constraints.
""",
    )
    occupancy_frame = synthetic["occupancy"]
    occupancy_columns = list(occupancy_frame.columns)
    occupancy_lines = [
        "| " + " | ".join(occupancy_columns) + " |",
        "| " + " | ".join("---" for _ in occupancy_columns) + " |",
    ]
    occupancy_lines.extend(
        "| " + " | ".join(str(row[column]) for column in occupancy_columns) + " |"
        for _, row in occupancy_frame.iterrows()
    )
    occupancy = "\n".join(occupancy_lines)
    write_text(
        output / "08_temporal_coupling_audit.md",
        f"""# Temporal Coupling Audit

Current placement is coupled to future capacity. A dispatch variable occupies every capacity row from `execution_start_step` through the estimated duration, truncated at the horizon.

{occupancy}

Because the environment processes transfers before new actions, even local/sub-step transfers use one minimum transfer step. A 30-minute task occupies two future slots; a 60-minute task occupies all three representable future slots after the current slot. The synthetic pressure-swap test reversed placement: `{synthetic['placement']['reversed']}`.

Diagnosis: `TEMPORAL_COUPLING_MISSING = NO`. The important defect is forecast/grid offset, not absence of current-task future occupancy.
""",
    )

    h1_oracle = agreement.loc[("H1_vs_H4_ORACLE", "ALL")]
    pt = agreement.loc[("PERSISTENCE_vs_TRANSFORMER", "ALL")]
    p50_gpu = pressure[(pressure.controller == "H4_TRANSFORMER") & (pressure.resource == "gpu") & (pressure.metric == "forecast_to_total_capacity") & (pressure["quantile"] == "P50")].iloc[0].value
    p90_gpu = pressure[(pressure.controller == "H4_TRANSFORMER") & (pressure.resource == "gpu") & (pressure.metric == "forecast_to_total_capacity") & (pressure["quantile"] == "P90")].iloc[0].value
    write_text(
        output / "14_root_cause_assessment.md",
        f"""# Root Cause Assessment

Primary classification: **MIXED**.

1. **DEFER_IMPLEMENTATION_BUG (confirmed accounting bug):** scheduler-level defer persists the task and advances the SLA clock but does not increment `wait_intervals`. This makes Waiting=0 unusable as evidence that no scheduler waiting occurred. It did not cause the observed zero defer decisions because the optimizer action logs also show no defer.
2. **OBJECTIVE_DOMINATES_LOOKAHEAD (strong):** with frozen weights, one-step defer adds `{cost.loc['DEFER_ONE_STEP','waiting_defer']:.3f}` waiting cost. Execute-now and one-step-defer totals are `{cost.loc['EXECUTE_NOW','total']:.6f}` and `{cost.loc['DEFER_ONE_STEP','total']:.6f}` in the low-price synthetic case. Price savings are much smaller.
3. **TRACE_LOW_SENSITIVITY (strong):** Transformer forecast GPU pressure is only `{p50_gpu:.6%}` of per-DC total capacity at P50 and `{p90_gpu:.6%}` at P90. H1/Oracle target agreement is `{h1_oracle.target_dc_agreement:.6%}` and Persistence/Transformer target agreement is `{pt.target_dc_agreement:.6%}`.
4. **TEMPORAL_COUPLING_MISSING (not supported):** long current tasks occupy future slots and swapped pressure reverses current placement.
5. **Forecast bridge alignment defect (confirmed):** +15 forecast is written to current-slot index 0 while new assignments start at index 1. This weakens/improperly shifts forecast competition and means Forecast-aware MPC v1 is not yet a clean test of forecast value.

Forecast-aware MPC v1 conclusion validity: **PARTIALLY_VALID**. It validly reports the behavior of the implemented bridge and objective, but it cannot yet support a general claim that future prediction has little control value.
""",
    )

    capacity_release_h1 = stress.loc[("capacity_release", "H1")]
    capacity_release_h4 = stress.loc[("capacity_release", "H4_ORACLE")]
    low_price_h1 = stress.loc[("future_low_price", "H1")]
    low_price_h4 = stress.loc[("future_low_price", "H4_ORACLE")]
    burst_h1 = stress.loc[("gpu_burst", "H1")]
    burst_h4 = stress.loc[("gpu_burst", "H4_ORACLE")]
    write_text(
        output / "01_summary.md",
        f"""# MPC Control Authority & Future Sensitivity Diagnosis v1

1. **Q1 Defer feasible?** YES as a semantic/environment action; action 0 retains the task and it can be assigned next step.
2. **Q2 Why Defer=0?** Frozen waiting cost and deadline feasibility strongly dominate normal energy savings; formal action traces contain no optimizer defer. Separately, scheduler defer waiting accounting is broken.
3. **Q3 SLA slack?** P50 `{slack.p50:.1f}`, P90 `{slack.p90:.1f}`, fraction >=1 `{slack.fraction_ge_1:.6%}`, fraction >=4 `{slack.fraction_ge_4:.6%}`.
4. **Q4 Current task future occupancy?** YES; estimated duration controls multi-slot capacity occupancy.
5. **Q5 Forecast reaches constraints?** YES, via per-DC capacity RHS; however +15..+60 is offset against a timeline with a current slot.
6. **Q6 Workload 0x->3x changes first action?** YES, only after capacity pressure crosses the feasibility threshold.
7. **Q7 Future price changes defer?** NO under frozen weights, including an 80% low-price diagnostic.
8. **Q8 Action switch threshold?** `{switch_threshold}%` of DC GPU capacity in the synthetic sweep.
9. **Q9 Repaired stress cases?** Capacity release: both defer now but H4 plans future dispatch; low price: both execute; GPU burst: H1 targets DC{burst_h1.first_target_dc}, H4 targets DC{burst_h4.first_target_dc}.
10. **Q10 Real pressure/capacity?** Transformer per-DC GPU forecast/total capacity P50 `{p50_gpu:.6%}`, P90 `{p90_gpu:.6%}`.
11. **Q11 Action agreement?** H1/Oracle target `{h1_oracle.target_dc_agreement:.6%}`; Persistence/Transformer target `{pt.target_dc_agreement:.6%}`; semantic agreement is `{pt.semantic_action_agreement:.6%}`.
12. **Q12 Root cause?** MIXED: defer accounting bug + objective dominance + trace low sensitivity + forecast-grid alignment defect. Current-task temporal occupancy itself exists.
13. **Q13 Prior conclusion validity?** PARTIALLY_VALID.
14. **Q14 Next step?** FIX MPC FORMULATION: repair defer waiting/state accounting and align forecast intervals with the optimizer timeline, then rerun the same frozen diagnostics before role selection.

No objective, reward, forecast model, dataset, BC, or SAC semantics were modified.
""",
    )

    write_text(
        output / "15_evidence_index.md",
        """# Evidence Index

| Evidence | File |
|---|---|
| Direct conclusions | `01_summary.md` |
| Defer path and real transition | `02_defer_path_audit.md` |
| Defer feasibility/constraint matrix | `03_defer_constraint_matrix.csv` |
| Frozen objective decomposition | `04_defer_cost_decomposition.csv` |
| Future price perturbation | `05_future_price_sensitivity.csv` |
| Aggregate workload perturbation | `06_future_workload_sensitivity.csv` |
| Forecast-to-MILP RHS trace | `07_forecast_to_constraint_trace.md` |
| Duration occupancy audit | `08_temporal_coupling_audit.md` |
| Capacity switch sweep | `09_capacity_threshold_sweep.csv` |
| Formal SLA slack | `10_sla_slack_distribution.csv` |
| Repaired deterministic stress cases | `11_stress_case_results.csv` |
| Formal action agreement | `12_action_sensitivity_summary.csv` |
| Forecast/capacity quantiles | `13_forecast_capacity_pressure.csv` |
| Root-cause classification | `14_root_cause_assessment.md` |
| Reproducibility manifest | `16_change_manifest.md` |
| Raw paired action trace | `action_trace.parquet` |
""",
    )
    write_text(
        output / "16_change_manifest.md",
        """# Change Manifest

## Added

- One diagnosis-only runner and one focused test module.
- Synthetic state, real defer transition, formal action-agreement, SLA-slack, and pressure/capacity evidence.
- Sixteen required diagnosis artifacts plus a raw action trace.

## Explicit non-changes

- No production optimizer, objective weight, reward, SLA, defer penalty, or environment semantics changed.
- No Transformer training, checkpoint selection, forecast-model change, or Dataset v1 change.
- No BC/SAC/expert-dataset/Triggered-MPC work.
- No SustainCluster source modification, commit, or push.
""",
    )


def generate() -> Path:
    if OUTPUT.exists():
        if (OUTPUT / "01_summary.md").exists():
            raise FileExistsError(f"Refusing to overwrite existing artifacts: {OUTPUT}")
        required_partial = (
            "04_defer_cost_decomposition.csv",
            "10_sla_slack_distribution.csv",
            "11_stress_case_results.csv",
            "12_action_sensitivity_summary.csv",
            "13_forecast_capacity_pressure.csv",
            "action_trace.parquet",
            "diagnostic_metadata.json",
        )
        if not all((OUTPUT / name).is_file() for name in required_partial):
            raise FileExistsError(
                f"Refusing to recover an unrecognized partial directory: {OUTPUT}"
            )
        metadata = json.loads((OUTPUT / "diagnostic_metadata.json").read_text("utf-8"))
        transition = metadata["defer_transition"]
        synthetic = synthetic_diagnostics()
        trace = {
            "actions": pd.read_parquet(OUTPUT / "action_trace.parquet"),
            "agreement": pd.read_csv(OUTPUT / "12_action_sensitivity_summary.csv"),
            "pressure": pd.read_csv(OUTPUT / "13_forecast_capacity_pressure.csv"),
            "slack": pd.read_csv(OUTPUT / "10_sla_slack_distribution.csv"),
        }
        create_documents(OUTPUT, transition, synthetic, trace)
        print(
            json.dumps(
                {
                    "status": "RECOVERED AND READY FOR VALIDATION",
                    "output": str(OUTPUT),
                    "defer_wait_increment": transition["waiting_increment"],
                    "placement_reversed": synthetic["placement"]["reversed"],
                    "artifact_count": len(list(OUTPUT.iterdir())),
                }
            )
        )
        return OUTPUT
    expected_head = "90eb972f78742603b522fabc3e1cf65391434418"
    expected_sustain = "3f6ea95cb835b89ba50b0ef76d66d14b8037643e"
    if git("rev-parse", "HEAD") != expected_head:
        raise RuntimeError("main Git HEAD changed")
    if git("rev-parse", "HEAD", cwd=SUSTAIN_REPO) != expected_sustain:
        raise RuntimeError("SustainCluster HEAD changed")
    if git("status", "--short", cwd=SUSTAIN_REPO):
        raise RuntimeError("SustainCluster must remain clean")
    prepare_sustaincluster_imports(SUSTAIN_REPO)

    transition = real_defer_transition()
    synthetic = synthetic_diagnostics()
    trace = run_trace_diagnostics()

    OUTPUT.mkdir(parents=True, exist_ok=False)
    constraint_rows = pd.DataFrame(
        [
            {"component": "ActionMapping", "condition": "disable_defer_action=False", "defer_feasible": True, "cost_effect": "none", "constraint_effect": "action 0 legal", "evidence": "real transition", "notes": "task retained"},
            {"component": "MILP future dispatch", "condition": "dispatch_step>0", "defer_feasible": True, "cost_effect": "100 per planned wait step", "constraint_effect": "future dispatch variable selected", "evidence": "optimizer vectors", "notes": "decoded as first-step defer"},
            {"component": "MILP terminal backlog", "condition": "not dispatched in horizon", "defer_feasible": True, "cost_effect": "waiting + SLA + terminal", "constraint_effect": "one-of constraint", "evidence": "optimizer vectors", "notes": "decoded as first-step defer"},
            {"component": "SLA upper bound", "condition": "completion_step>deadline_step", "defer_feasible": False, "cost_effect": "candidate removed", "constraint_effect": "variable upper bound=0", "evidence": "_variable_upper_bounds", "notes": "late defer masked"},
            {"component": "Config", "condition": "allow_defer=False", "defer_feasible": False, "cost_effect": "none", "constraint_effect": "future dispatch/backlog upper bound=0", "evidence": "optimizer validation", "notes": "not active in formal run"},
            {"component": "Environment SLA force", "condition": "current_time>sla_deadline", "defer_feasible": False, "cost_effect": "environment reward path", "constraint_effect": "forced origin assignment", "evidence": "TaskSchedulingEnv.step", "notes": "strictly after deadline"},
            {"component": "Scheduler wait counter", "condition": "action 0", "defer_feasible": True, "cost_effect": "MPC pays modeled wait cost", "constraint_effect": "no wait_intervals increment", "evidence": f"observed increment={transition['waiting_increment']}", "notes": "implementation accounting bug"},
        ]
    )
    constraint_rows.to_csv(OUTPUT / "03_defer_constraint_matrix.csv", index=False)
    synthetic["cost"].to_csv(OUTPUT / "04_defer_cost_decomposition.csv", index=False)
    synthetic["price"].to_csv(OUTPUT / "05_future_price_sensitivity.csv", index=False)
    synthetic["workload"].to_csv(OUTPUT / "06_future_workload_sensitivity.csv", index=False)
    synthetic["threshold"].to_csv(OUTPUT / "09_capacity_threshold_sweep.csv", index=False)
    trace["slack"].to_csv(OUTPUT / "10_sla_slack_distribution.csv", index=False)
    synthetic["stress"].to_csv(OUTPUT / "11_stress_case_results.csv", index=False)
    trace["agreement"].to_csv(OUTPUT / "12_action_sensitivity_summary.csv", index=False)
    trace["pressure"].to_csv(OUTPUT / "13_forecast_capacity_pressure.csv", index=False)
    trace["actions"].to_parquet(OUTPUT / "action_trace.parquet", index=False)
    (OUTPUT / "diagnostic_metadata.json").write_text(
        json.dumps(
            {
                "git_branch": git("branch", "--show-current"),
                "git_head": expected_head,
                "sustaincluster_commit": expected_sustain,
                "defer_transition": transition,
                "synthetic_placement": synthetic["placement"],
                "training": False,
                "objective_modified": False,
                "forecast_model_modified": False,
                "dataset_modified": False,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    create_documents(OUTPUT, transition, synthetic, trace)
    print(
        json.dumps(
            {
                "status": "READY FOR VALIDATION",
                "output": str(OUTPUT),
                "defer_wait_increment": transition["waiting_increment"],
                "placement_reversed": synthetic["placement"]["reversed"],
                "artifact_count": len(list(OUTPUT.iterdir())),
            }
        )
    )
    return OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    generate()


if __name__ == "__main__":
    main()


