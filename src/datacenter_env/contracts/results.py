from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping

import pandas as pd

from datacenter_env.contracts.actions import DataCenterAction
from datacenter_env.contracts.observations import DataCenterObservation


@dataclass(frozen=True, slots=True)
class PhysicalResult:
    previous_true_temperature_c: float
    previous_proposed_cooling_kw: float
    previous_applied_cooling_kw: float
    true_temperature_c: float
    measured_temperature_c: float
    controller_measured_temperature_c: float
    proposed_cooling_kw: float
    constrained_cooling_kw: float
    applied_cooling_kw: float
    it_power_kw: float
    heat_kw: float
    cooling_power_kw: float
    dc_power_kw: float
    grid_power_kw: float
    renewable_available_kw: float
    renewable_used_kw: float
    renewable_curtailed_kw: float
    total_load: float
    actuator_alpha: float


@dataclass(frozen=True, slots=True)
class AccountingResult:
    dc_energy_kwh: float
    grid_energy_kwh: float
    cooling_energy_kwh: float
    renewable_available_kwh: float
    renewable_used_kwh: float
    renewable_curtailed_kwh: float
    energy_cost: float
    carbon_kg: float


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    power_balance_error: float
    renewable_balance_error: float
    thermal_balance_error: float
    temperature_violation: bool
    below_min_temperature: bool
    above_max_temperature: bool
    invalid_value_count: int
    negative_power_count: int
    actuator_tracking_error_kw: float
    ramp_limited: bool
    ramp_up_limited: bool
    ramp_down_limited: bool
    thermal_infeasible: bool
    actuator_induced_thermal_infeasible: bool
    predicted_next_temperature_c: float
    prediction_error_c: float
    cooling_min_feasible_kw: float
    cooling_max_feasible_kw: float
    cooling_constraint_intervention_kw: float


@dataclass(frozen=True, slots=True)
class ControllerResult:
    controller_name: str
    optimizer_attempted: bool = False
    optimizer_success: bool = False
    optimizer_failure: bool = False
    fallback_used: bool = False
    prediction_infeasibility: bool = False
    actuator_infeasibility: bool = False
    fallback_reason: str | None = None
    candidate_evaluations: int | None = None
    objective_total: float | None = None
    objective_energy_cost: float | None = None
    objective_carbon: float | None = None
    objective_temperature: float | None = None
    objective_control_movement: float | None = None
    target_temperature_c: float | None = None
    control_floor_temperature_c: float | None = None
    precooling_active: bool = False


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    action: DataCenterAction
    result: ControllerResult
    predicted_next_temperature_c: float | None = None
    target_temperature_c: float | None = None
    control_floor_temperature_c: float | None = None
    precooling_active: bool = False


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: DataCenterObservation
    physical: PhysicalResult
    accounting: AccountingResult
    diagnostics: DiagnosticResult
    controller: ControllerResult
    terminated: bool = False
    truncated: bool = False

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "timestamp": self.observation.timestamp,
            "workload_fraction": self.observation.workload_fraction,
        }
        record.update(asdict(self.physical))
        record.update(asdict(self.accounting))
        record.update(asdict(self.diagnostics))
        record.update(asdict(self.controller))
        return record


@dataclass(frozen=True, slots=True)
class RunMetadata:
    name: str = "single_center"
    controller_name: str = "baseline"
    condition_name: str = "ideal"
    dataset_name: str = "external_inputs"
    seed: int = 0
    source_path: str = "external-provider"
    config_snapshot: Mapping[str, Any] | None = None
    package_version: str | None = None


@dataclass(frozen=True, slots=True)
class RunHandle:
    run_id: int | None


@dataclass(frozen=True)
class RunSummary:
    status: str
    run_id: int | None
    controller_name: str
    condition_name: str
    metrics: Mapping[str, float]
    steps: tuple[StepResult, ...]
    error_message: str | None = None
    records: tuple[Mapping[str, Any], ...] = ()

    def to_frame(self) -> pd.DataFrame:
        if self.records:
            return pd.DataFrame(self.records)
        return pd.DataFrame(step.to_record() for step in self.steps)
