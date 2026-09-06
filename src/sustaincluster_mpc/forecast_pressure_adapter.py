from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from forecasting.workload_forecast_provider import ForecastBundle
from sustaincluster_mpc.future_signals import FutureSignalProvider
from sustaincluster_mpc.horizon_adapter import HorizonState
from sustaincluster_mpc.timeline_contract import (
    FORECAST_FUTURE_INTERVALS,
    REPAIRED_H4_CAPACITY_NODES,
    forecast_step_to_capacity_index,
)


@dataclass(frozen=True)
class ExpectedDataCenterPressure:
    horizon_step: int
    horizon_minutes: int
    dc_id: int
    origin_probability: float
    expected_task_count: float
    cpu_demand: float
    gpu_demand: float
    memory_demand: float


@dataclass(frozen=True)
class ForecastPressureApplication:
    state: HorizonState
    pressures: tuple[ExpectedDataCenterPressure, ...]
    capacity_shortage_events: int
    cpu_shortage: float
    gpu_shortage: float
    memory_shortage: float


def expected_origin_probabilities(
    datacenter_configs: Sequence[Mapping[str, Any]],
    current_time_utc: str | pd.Timestamp,
) -> dict[int, float]:
    timestamp = pd.Timestamp(current_time_utc)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    scores: dict[int, float] = {}
    for config in datacenter_configs:
        dc_id = int(config["dc_id"])
        population = float(config.get("population_weight", 0.1))
        timezone_shift = float(config.get("timezone_shift", 0.0))
        local_hour = (timestamp + pd.Timedelta(hours=timezone_shift)).hour
        activity = 1.0 if 8 <= local_hour < 20 else 0.3
        score = population * activity
        if not math.isfinite(score) or score < 0:
            raise ValueError("origin score must be finite and nonnegative")
        scores[dc_id] = score
    total = sum(scores.values())
    if total <= 0:
        raise ValueError("origin probability score sum must be positive")
    return {dc_id: score / total for dc_id, score in scores.items()}


def distribute_global_forecast(
    bundle: ForecastBundle,
    datacenter_configs: Sequence[Mapping[str, Any]],
) -> tuple[ExpectedDataCenterPressure, ...]:
    rows: list[ExpectedDataCenterPressure] = []
    issued = pd.Timestamp(bundle.issued_at_timestamp)
    for point in bundle.points:
        probabilities = expected_origin_probabilities(
            datacenter_configs,
            issued + pd.Timedelta(minutes=point.horizon_minutes),
        )
        for dc_id, probability in probabilities.items():
            rows.append(
                ExpectedDataCenterPressure(
                    point.horizon_step,
                    point.horizon_minutes,
                    dc_id,
                    probability,
                    point.global_task_count * probability,
                    point.global_cpu_demand * probability,
                    point.global_gpu_demand * probability,
                    point.global_memory_demand * probability,
                )
            )
    return tuple(rows)


def apply_forecast_pressure(
    state: HorizonState,
    bundle: ForecastBundle,
    datacenter_configs: Sequence[Mapping[str, Any]],
) -> ForecastPressureApplication:
    if state.horizon != REPAIRED_H4_CAPACITY_NODES:
        raise ValueError(
            "Repaired H4 requires one current capacity node plus four future nodes"
        )
    if len(bundle.points) != FORECAST_FUTURE_INTERVALS:
        raise ValueError("Repaired H4 requires four future forecast intervals")
    pressures = distribute_global_forecast(bundle, datacenter_configs)
    by_dc = {dc.dc_id: dc for dc in state.datacenters}
    if set(by_dc) != {int(config["dc_id"]) for config in datacenter_configs}:
        raise ValueError("datacenter config and HorizonState ids do not match")
    grouped: dict[int, list[ExpectedDataCenterPressure]] = {}
    for item in pressures:
        grouped.setdefault(item.dc_id, []).append(item)

    shortage_events = 0
    shortage = np.zeros(3, dtype=np.float64)
    updated = []
    for dc_id, dc in by_dc.items():
        cpu = np.asarray(dc.cpu_available_cores, dtype=np.float64).copy()
        gpu = np.asarray(dc.gpu_available_units, dtype=np.float64).copy()
        memory = np.asarray(dc.memory_available_gb, dtype=np.float64).copy()
        reserve_cpu = np.asarray(dc.forecast_cpu_reservations, dtype=np.float64).copy()
        reserve_gpu = np.asarray(dc.forecast_gpu_reservations, dtype=np.float64).copy()
        reserve_memory = np.asarray(dc.forecast_memory_reservations, dtype=np.float64).copy()
        for item in grouped.get(dc_id, ()):
            index = forecast_step_to_capacity_index(
                item.horizon_step,
                item.horizon_minutes,
                state.timestep_minutes,
            )
            demand = np.asarray(
                [item.cpu_demand, item.gpu_demand, item.memory_demand],
                dtype=np.float64,
            )
            available = np.asarray([cpu[index], gpu[index], memory[index]])
            deficit = np.maximum(demand - available, 0.0)
            if np.any(deficit > 1e-12):
                shortage_events += 1
                shortage += deficit
            reserve_cpu[index] += demand[0]
            reserve_gpu[index] += demand[1]
            reserve_memory[index] += demand[2]
            cpu[index] = max(0.0, cpu[index] - demand[0])
            gpu[index] = max(0.0, gpu[index] - demand[1])
            memory[index] = max(0.0, memory[index] - demand[2])
        updated.append(
            replace(
                dc,
                cpu_available_cores=tuple(float(value) for value in cpu),
                gpu_available_units=tuple(float(value) for value in gpu),
                memory_available_gb=tuple(float(value) for value in memory),
                forecast_cpu_reservations=tuple(float(value) for value in reserve_cpu),
                forecast_gpu_reservations=tuple(float(value) for value in reserve_gpu),
                forecast_memory_reservations=tuple(float(value) for value in reserve_memory),
            )
        )
    applied = replace(
        state,
        forecast_mode=bundle.provider,  # type: ignore[arg-type]
        datacenters=tuple(updated),
        future_arrivals=(),
    )
    return ForecastPressureApplication(
        applied,
        pressures,
        shortage_events,
        float(shortage[0]),
        float(shortage[1]),
        float(shortage[2]),
    )


def apply_oracle_future_signals(state: HorizonState, env: Any) -> HorizonState:
    if state.horizon != REPAIRED_H4_CAPACITY_NODES:
        raise ValueError(
            "Repaired H4 oracle signals require current + four future nodes"
        )
    provider = FutureSignalProvider("oracle")
    raw_by_id = {
        int(dc.dc_id): dc for dc in env.cluster_manager.datacenters.values()
    }
    datacenters = tuple(
        replace(
            dc,
            electricity_price_usd_per_mwh=provider.electricity_price(
                raw_by_id[dc.dc_id], env.current_time, state.horizon
            ),
            carbon_intensity_gco2_per_kwh=provider.carbon_intensity(
                raw_by_id[dc.dc_id], env.current_time, state.horizon
            ),
        )
        for dc in state.datacenters
    )
    return replace(
        state,
        datacenters=datacenters,
        future_signal_mode="oracle",
    )
