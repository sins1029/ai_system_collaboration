from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datacenter_env import ExogenousInput, ForecastWindow
from models.exogenous_signals import ExogenousSignals
from optimization.baseline import immediate_dispatch
from optimization.temporal_shift import price_aware_dispatch


@dataclass(frozen=True)
class ScenarioSignalProvider:
    """Package-external owner of complete input timelines and forecast slicing."""

    steps: tuple[ExogenousInput, ...]

    def __iter__(self):
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def window(self, start_index: int, horizon_steps: int) -> ForecastWindow:
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be at least one")
        if start_index < 0 or start_index >= len(self.steps):
            raise IndexError("forecast start is outside the signal provider")
        return ForecastWindow(self.steps[start_index : start_index + horizon_steps])


def provider_from_legacy_signals(
    signals: ExogenousSignals,
    batch_service: pd.Series,
) -> ScenarioSignalProvider:
    if len(signals) != len(batch_service):
        raise ValueError("batch service length must match exogenous signal length")
    backlog = 0.0
    steps: list[ExogenousInput] = []
    for index in range(len(signals)):
        row = signals.current(index)
        backlog += float(row["batch_workload"])
        service = min(backlog, max(0.0, float(batch_service.iloc[index])))
        backlog -= service
        steps.append(
            ExogenousInput(
                timestamp=row["timestamp"].to_pydatetime(),
                workload_fraction=max(0.0, float(row["online_workload"]) + service),
                electricity_price_per_kwh=float(row["electricity_price"]),
                carbon_intensity_kg_per_kwh=float(
                    row["carbon_intensity_kg_per_kwh"]
                ),
                outdoor_temperature_c=float(row["outdoor_temperature_c"]),
                renewable_power_kw=float(row["renewable_power_kw"]),
            )
        )
    return ScenarioSignalProvider(tuple(steps))


def build_scenario_provider(
    signals: ExogenousSignals,
    dispatch_strategy: str,
    optimization_config: dict,
) -> ScenarioSignalProvider:
    if dispatch_strategy == "immediate":
        service = immediate_dispatch(signals)
    elif dispatch_strategy == "price_aware_temporal_shift":
        service = price_aware_dispatch(signals, optimization_config)
    else:
        raise ValueError(f"unknown dispatch strategy: {dispatch_strategy}")
    return provider_from_legacy_signals(signals, service)
