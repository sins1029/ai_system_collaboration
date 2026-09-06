from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np

from forecasting.workload_forecast_provider import ForecastBundle
from sustaincluster_mpc.forecast_pressure_adapter import distribute_global_forecast
from sustaincluster_mpc.horizon_adapter import HorizonState
from sustaincluster_mpc.timeline_contract import (
    FORECAST_FUTURE_INTERVALS,
    REPAIRED_H4_CAPACITY_NODES,
    forecast_step_to_capacity_index,
)


CALIBRATION_SEEDS = (1101, 1102, 1103, 1104, 1105)
EVALUATION_SEEDS = (1201, 1202, 1203, 1204, 1205)
PRIMARY_TRIGGER_QUANTILE = 0.95
RESOURCE_ORDER = ("cpu", "gpu", "memory")
TRIGGER_TRACE_COLUMNS = (
    "seed",
    "scenario",
    "step",
    "timestamp",
    "risk_score",
    "threshold",
    "triggered",
    "risk_dc",
    "risk_horizon",
    "risk_resource",
    "gpu_pressure",
    "cpu_pressure",
    "mem_pressure",
    "selected_controller",
    "selected_semantic_action",
    "selected_target_dc",
    "solver_ms",
    "history_fallback",
)


@dataclass(frozen=True)
class RiskComponent:
    dc_id: int
    horizon_step: int
    horizon_minutes: int
    resource: Literal["cpu", "gpu", "memory"]
    predicted_existing_occupancy: float
    predicted_future_arrival_demand: float
    capacity: float
    pressure: float


@dataclass(frozen=True)
class TriggerRiskAssessment:
    risk_score: float
    risk_dc: int
    risk_horizon: int
    risk_resource: Literal["cpu", "gpu", "memory"]
    cpu_pressure: float
    gpu_pressure: float
    mem_pressure: float
    components: tuple[RiskComponent, ...]


@dataclass(frozen=True)
class TriggerThresholds:
    p90: float
    p95_primary: float
    p99: float
    calibration_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    selection_rule: str = "risk quantile only; no reward tuning"


@dataclass(frozen=True)
class TriggerSelection:
    triggered: bool
    selected_controller: Literal["H1", "H4_TRANSFORMER"]
    planning_state: HorizonState


@dataclass(frozen=True)
class TriggeredTransformerMPC:
    """Thin wrapper that selects an unmodified H1 or Transformer-H4 path."""

    threshold: float
    datacenter_configs: Sequence[Mapping[str, object]]

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")

    def evaluate(
        self,
        h1_state: HorizonState,
        h4_transformer_state: HorizonState,
        risk_state: HorizonState,
        bundle: ForecastBundle,
    ) -> tuple[TriggerRiskAssessment, TriggerSelection]:
        risk = compute_deployable_risk(
            risk_state, bundle, self.datacenter_configs
        )
        selection = select_planning_state(
            h1_state,
            h4_transformer_state,
            risk.risk_score,
            self.threshold,
        )
        return risk, selection


def compute_deployable_risk(
    state: HorizonState,
    bundle: ForecastBundle,
    datacenter_configs: Sequence[Mapping[str, object]],
) -> TriggerRiskAssessment:
    """Compute max deployable pressure without changing the planning state."""
    if state.horizon != REPAIRED_H4_CAPACITY_NODES:
        raise ValueError("trigger risk requires current + four future nodes")
    if state.information_mode != "deployable":
        raise ValueError("trigger risk requires deployable state information")
    if state.future_signal_mode == "oracle" or bundle.provider == "oracle":
        raise ValueError("trigger risk cannot consume oracle future information")
    if len(bundle.points) != FORECAST_FUTURE_INTERVALS:
        raise ValueError("trigger risk requires four future forecast points")

    pressures = distribute_global_forecast(bundle, datacenter_configs)
    pressure_by_key = {
        (item.dc_id, item.horizon_step): item for item in pressures
    }
    datacenters = {dc.dc_id: dc for dc in state.datacenters}
    if set(datacenters) != {
        int(config["dc_id"]) for config in datacenter_configs
    }:
        raise ValueError("datacenter config and HorizonState ids do not match")

    existing = {
        (dc_id, horizon_step, resource): 0.0
        for dc_id in datacenters
        for horizon_step in range(1, FORECAST_FUTURE_INTERVALS + 1)
        for resource in RESOURCE_ORDER
    }
    for task in state.running_tasks:
        if task.dc_id not in datacenters:
            raise ValueError(f"running task references unknown dc_id={task.dc_id}")
        for horizon_step in range(1, FORECAST_FUTURE_INTERVALS + 1):
            if task.release_step <= horizon_step:
                continue
            existing[(task.dc_id, horizon_step, "cpu")] += task.cpu_cores
            existing[(task.dc_id, horizon_step, "gpu")] += task.gpu_units
            existing[(task.dc_id, horizon_step, "memory")] += task.memory_gb

    components: list[RiskComponent] = []
    resource_max = {resource: 0.0 for resource in RESOURCE_ORDER}
    winner: RiskComponent | None = None
    for dc_id in sorted(datacenters):
        dc = datacenters[dc_id]
        capacities = {
            "cpu": dc.cpu_total_cores,
            "gpu": dc.gpu_total_units,
            "memory": dc.memory_total_gb,
        }
        for point in bundle.points:
            horizon_step = forecast_step_to_capacity_index(
                point.horizon_step,
                point.horizon_minutes,
                state.timestep_minutes,
            )
            expected = pressure_by_key[(dc_id, horizon_step)]
            arrivals = {
                "cpu": expected.cpu_demand,
                "gpu": expected.gpu_demand,
                "memory": expected.memory_demand,
            }
            for resource in RESOURCE_ORDER:
                capacity = float(capacities[resource])
                if not math.isfinite(capacity) or capacity <= 0:
                    raise ValueError("trigger capacities must be finite and positive")
                existing_value = existing[(dc_id, horizon_step, resource)]
                arrival_value = float(arrivals[resource])
                pressure = (existing_value + arrival_value) / capacity
                component = RiskComponent(
                    dc_id=dc_id,
                    horizon_step=horizon_step,
                    horizon_minutes=int(point.horizon_minutes),
                    resource=resource,  # type: ignore[arg-type]
                    predicted_existing_occupancy=existing_value,
                    predicted_future_arrival_demand=arrival_value,
                    capacity=capacity,
                    pressure=pressure,
                )
                components.append(component)
                resource_max[resource] = max(resource_max[resource], pressure)
                if winner is None or pressure > winner.pressure:
                    winner = component

    assert winner is not None
    return TriggerRiskAssessment(
        risk_score=winner.pressure,
        risk_dc=winner.dc_id,
        risk_horizon=winner.horizon_minutes,
        risk_resource=winner.resource,
        cpu_pressure=resource_max["cpu"],
        gpu_pressure=resource_max["gpu"],
        mem_pressure=resource_max["memory"],
        components=tuple(components),
    )


def calibrate_trigger_thresholds(
    risk_scores: Sequence[float],
    *,
    calibration_seeds: Sequence[int] = CALIBRATION_SEEDS,
    evaluation_seeds: Sequence[int] = EVALUATION_SEEDS,
) -> TriggerThresholds:
    calibration = tuple(int(seed) for seed in calibration_seeds)
    evaluation = tuple(int(seed) for seed in evaluation_seeds)
    if set(calibration) & set(evaluation):
        raise ValueError("calibration and evaluation seeds must be disjoint")
    if calibration != CALIBRATION_SEEDS:
        raise ValueError("Triggered MPC v1 calibration seeds are frozen")
    if evaluation != EVALUATION_SEEDS:
        raise ValueError("Triggered MPC v1 evaluation seeds are frozen")
    values = np.asarray(tuple(float(value) for value in risk_scores), dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("risk scores must be finite, nonnegative and non-empty")
    return TriggerThresholds(
        p90=float(np.percentile(values, 90)),
        p95_primary=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
        calibration_seeds=calibration,
        evaluation_seeds=evaluation,
    )


def select_planning_state(
    h1_state: HorizonState,
    h4_transformer_state: HorizonState,
    risk_score: float,
    threshold: float,
) -> TriggerSelection:
    if not math.isfinite(risk_score) or risk_score < 0:
        raise ValueError("risk_score must be finite and nonnegative")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    triggered = risk_score >= threshold
    return TriggerSelection(
        triggered=triggered,
        selected_controller="H4_TRANSFORMER" if triggered else "H1",
        planning_state=h4_transformer_state if triggered else h1_state,
    )


def safe_benefit_retention(
    h1_value: float,
    always_h4_value: float,
    triggered_value: float,
    *,
    direction: Literal["reward", "cost"],
    tolerance: float = 1e-9,
) -> float | None:
    values = (float(h1_value), float(always_h4_value), float(triggered_value))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("benefit retention values must be finite")
    if direction == "reward":
        denominator = always_h4_value - h1_value
        numerator = triggered_value - h1_value
    elif direction == "cost":
        denominator = h1_value - always_h4_value
        numerator = h1_value - triggered_value
    else:
        raise ValueError(f"unsupported benefit direction={direction!r}")
    if denominator <= tolerance:
        return None
    return numerator / denominator


def validate_trigger_trace_schema(columns: Sequence[str]) -> None:
    missing = [name for name in TRIGGER_TRACE_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"trigger trace is missing required columns: {missing}")
