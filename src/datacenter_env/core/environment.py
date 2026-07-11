from __future__ import annotations

from datetime import datetime, timedelta
import math
from typing import Any, Mapping

from datacenter_env.config import DataCenterSystemConfig
from datacenter_env.contracts import (
    AccountingResult,
    ControllerDecision,
    ControllerResult,
    DataCenterAction,
    DataCenterObservation,
    DiagnosticResult,
    ExogenousInput,
    PhysicalResult,
    StepResult,
)
from datacenter_env.core.actuator import CoolingActuator
from datacenter_env.core.measurement import TemperatureMeasurement
from datacenter_env.core.plant import DataCenterPlant
from datacenter_env.core.prediction import ThermalPredictionModel
from datacenter_env.core.primitives import it_power_kw, next_temperature_c
from datacenter_env.core.safety import apply_thermal_constraints
from datacenter_env.core.state import DataCenterSnapshot
from datacenter_env.exceptions import InputValidationError, TimeAlignmentError


class DataCenterEnvironment:
    """Controller-independent single-step thermo-electric environment."""

    def __init__(self, config: DataCenterSystemConfig):
        self.config = DataCenterSystemConfig.coerce(config)
        self._seed: int | None = None
        self._build_components(seed=None)
        self._reset_state()

    @classmethod
    def from_config(
        cls, config: DataCenterSystemConfig | Mapping[str, Any]
    ) -> "DataCenterEnvironment":
        return cls(DataCenterSystemConfig.coerce(config))

    def _build_components(self, seed: int | None) -> None:
        dc = self.config.datacenter
        self._plant = DataCenterPlant(
            dc, self.config.step_hours, self.config.plant_parameters
        )
        self._predictor = ThermalPredictionModel(
            dc, self.config.prediction_parameters, self.config.step_hours
        )
        self._actuator = CoolingActuator(
            self.config.actuator,
            self.config.step_minutes,
            float(dc["cooling"]["max_cooling_kw"]),
        )
        configured_seed = self.config.measurement.get("seed")
        measurement_seed = int(configured_seed) if configured_seed is not None else seed
        self._measurement = TemperatureMeasurement(
            self.config.measurement, seed=measurement_seed
        )

    def _reset_state(self) -> None:
        initial = float(self.config.datacenter["thermal"]["initial_temp_c"])
        self._true_temperature_c = initial
        self._measured_temperature_c = self._measurement.measure_temperature_c(initial)
        self._actuator_state = self._actuator.initial_state()
        self._previous_proposed_cooling_kw = 0.0
        self._step_index = 0
        self._last_timestamp: datetime | None = None

    def reset(
        self,
        seed: int | None = None,
        initial_input: ExogenousInput | None = None,
    ) -> DataCenterObservation:
        self._seed = seed
        self._build_components(seed)
        self._reset_state()
        if initial_input is not None:
            return self.get_observation(initial_input)
        return DataCenterObservation(
            timestamp=datetime.min,
            measured_temperature_c=self._measured_temperature_c,
            previous_applied_cooling_kw=self._actuator_state.applied_cooling_kw,
            workload_fraction=0.0,
            electricity_price_per_kwh=0.0,
            carbon_intensity_kg_per_kwh=0.0,
            outdoor_temperature_c=0.0,
            renewable_power_kw=0.0,
        )

    def get_observation(self, current_input: ExogenousInput) -> DataCenterObservation:
        return DataCenterObservation(
            timestamp=current_input.timestamp,
            measured_temperature_c=self._measured_temperature_c,
            previous_applied_cooling_kw=self._actuator_state.applied_cooling_kw,
            workload_fraction=current_input.workload_fraction,
            electricity_price_per_kwh=current_input.electricity_price_per_kwh,
            carbon_intensity_kg_per_kwh=current_input.carbon_intensity_kg_per_kwh,
            outdoor_temperature_c=current_input.outdoor_temperature_c,
            renewable_power_kw=current_input.renewable_power_kw,
        )

    def step(
        self,
        action: DataCenterAction,
        current_input: ExogenousInput,
        predicted_next_temperature_c: float | None = None,
        *,
        controller_decision: ControllerDecision | None = None,
    ) -> StepResult:
        self._validate_time(current_input)
        dc = self.config.datacenter
        thermal = dc["thermal"]
        power = dc["power"]
        controller_measured = self._measured_temperature_c
        previous_true = self._true_temperature_c
        previous_applied = self._actuator_state.applied_cooling_kw
        previous_proposed = self._previous_proposed_cooling_kw
        proposed = float(action.cooling_target_kw)
        heat = it_power_kw(
            current_input.workload_fraction,
            float(dc["capacity"]["max_load"]),
            float(power["p_idle_kw"]),
            float(power["p_peak_kw"]),
        )
        floor = (
            controller_decision.control_floor_temperature_c
            if controller_decision is not None
            else None
        )
        constrained = apply_thermal_constraints(
            proposed,
            controller_measured,
            heat,
            current_input.outdoor_temperature_c,
            dc,
            self.config.step_hours,
            floor,
        )
        constrained_target, actuator_infeasible = self._actuator.actuator_aware_target(
            constrained.cooling_kw,
            self._actuator_state,
            constrained.minimum_feasible_kw,
            constrained.maximum_feasible_kw,
        )
        actuator_result = self._actuator.step(constrained_target, self._actuator_state)
        applied = actuator_result.applied_cooling_kw
        plant = self._plant.step(
            previous_true,
            current_input.workload_fraction,
            applied,
            current_input.outdoor_temperature_c,
            current_input.renewable_power_kw,
        )
        predicted = (
            float(predicted_next_temperature_c)
            if predicted_next_temperature_c is not None
            else self._predictor.predict_step(
                controller_measured,
                heat,
                applied,
                current_input.outdoor_temperature_c,
            )
        )
        measured = self._measurement.measure_temperature_c(plant.temperature_c)
        accounting = AccountingResult(
            dc_energy_kwh=plant.total_power_kw * self.config.step_hours,
            grid_energy_kwh=plant.grid_power_kw * self.config.step_hours,
            cooling_energy_kwh=plant.cooling_power_kw * self.config.step_hours,
            renewable_available_kwh=plant.renewable_available_kw * self.config.step_hours,
            renewable_used_kwh=plant.renewable_used_kw * self.config.step_hours,
            renewable_curtailed_kwh=plant.renewable_curtailed_kw * self.config.step_hours,
            energy_cost=(
                plant.grid_power_kw
                * self.config.step_hours
                * current_input.electricity_price_per_kwh
            ),
            carbon_kg=(
                plant.grid_power_kw
                * self.config.step_hours
                * current_input.carbon_intensity_kg_per_kwh
            ),
        )
        physical = PhysicalResult(
            previous_true_temperature_c=previous_true,
            previous_proposed_cooling_kw=previous_proposed,
            previous_applied_cooling_kw=previous_applied,
            true_temperature_c=plant.temperature_c,
            measured_temperature_c=measured,
            controller_measured_temperature_c=controller_measured,
            proposed_cooling_kw=proposed,
            constrained_cooling_kw=constrained_target,
            applied_cooling_kw=applied,
            it_power_kw=plant.it_power_kw,
            heat_kw=plant.heat_kw,
            cooling_power_kw=plant.cooling_power_kw,
            dc_power_kw=plant.total_power_kw,
            grid_power_kw=plant.grid_power_kw,
            renewable_available_kw=plant.renewable_available_kw,
            renewable_used_kw=plant.renewable_used_kw,
            renewable_curtailed_kw=plant.renewable_curtailed_kw,
            total_load=plant.total_load,
            actuator_alpha=actuator_result.alpha,
        )
        diagnostics = self._diagnose(
            physical,
            predicted,
            constrained,
            actuator_result.tracking_error_kw,
            actuator_result.ramp_limited,
            actuator_result.ramp_up_limited,
            actuator_result.ramp_down_limited,
            actuator_infeasible,
            current_input,
        )
        controller = self._controller_result(controller_decision)

        self._true_temperature_c = plant.temperature_c
        self._measured_temperature_c = measured
        self._actuator_state = actuator_result.state
        self._previous_proposed_cooling_kw = proposed
        self._last_timestamp = current_input.timestamp
        self._step_index += 1
        observation = self.get_observation(current_input)
        return StepResult(
            observation=observation,
            physical=physical,
            accounting=accounting,
            diagnostics=diagnostics,
            controller=controller,
        )

    def snapshot(self) -> DataCenterSnapshot:
        return DataCenterSnapshot(
            true_temperature_c=self._true_temperature_c,
            measured_temperature_c=self._measured_temperature_c,
            previous_proposed_cooling_kw=self._previous_proposed_cooling_kw,
            actuator_state=self._actuator_state,
            step_index=self._step_index,
            last_timestamp=self._last_timestamp,
        )

    def _validate_time(self, current_input: ExogenousInput) -> None:
        if self._last_timestamp is None:
            return
        expected = self._last_timestamp + timedelta(minutes=self.config.step_minutes)
        if current_input.timestamp != expected:
            raise TimeAlignmentError(
                f"expected timestamp {expected.isoformat()}, got {current_input.timestamp.isoformat()}"
            )

    def _controller_result(
        self, decision: ControllerDecision | None
    ) -> ControllerResult:
        if decision is None:
            return ControllerResult(controller_name="external")
        result = decision.result
        return ControllerResult(
            controller_name=result.controller_name,
            optimizer_attempted=result.optimizer_attempted,
            optimizer_success=result.optimizer_success,
            optimizer_failure=result.optimizer_failure,
            fallback_used=result.fallback_used,
            prediction_infeasibility=result.prediction_infeasibility,
            actuator_infeasibility=result.actuator_infeasibility,
            fallback_reason=result.fallback_reason,
            candidate_evaluations=result.candidate_evaluations,
            objective_total=result.objective_total,
            objective_energy_cost=result.objective_energy_cost,
            objective_carbon=result.objective_carbon,
            objective_temperature=result.objective_temperature,
            objective_control_movement=result.objective_control_movement,
            target_temperature_c=decision.target_temperature_c,
            control_floor_temperature_c=decision.control_floor_temperature_c,
            precooling_active=decision.precooling_active,
        )

    def _diagnose(
        self,
        physical: PhysicalResult,
        predicted: float,
        constrained: Any,
        tracking_error: float,
        ramp_limited: bool,
        ramp_up_limited: bool,
        ramp_down_limited: bool,
        actuator_infeasible: bool,
        current_input: ExogenousInput,
    ) -> DiagnosticResult:
        dc = self.config.datacenter
        thermal = dc["thermal"]
        expected_grid = (
            physical.it_power_kw
            + physical.cooling_power_kw
            + float(dc["power"]["p_aux_kw"])
            - physical.renewable_used_kw
        )
        expected_temp = next_temperature_c(
            physical.previous_true_temperature_c,
            physical.heat_kw * float(self.config.plant_parameters["heat_transfer_scale"]),
            physical.applied_cooling_kw
            * float(self.config.plant_parameters["cooling_effectiveness_scale"]),
            current_input.outdoor_temperature_c,
            self.config.step_hours,
            float(thermal["thermal_resistance_c_per_kw"]),
            float(thermal["thermal_capacitance_kwh_per_c"])
            * float(self.config.plant_parameters["thermal_capacity_scale"]),
        )
        below = physical.true_temperature_c < float(thermal["min_temp_c"])
        above = physical.true_temperature_c > float(thermal["max_temp_c"])
        checked = (
            physical.total_load,
            physical.it_power_kw,
            physical.heat_kw,
            physical.true_temperature_c,
            physical.applied_cooling_kw,
            physical.cooling_power_kw,
            physical.grid_power_kw,
        )
        negative = sum(value < 0 for value in checked if math.isfinite(value))
        return DiagnosticResult(
            power_balance_error=physical.grid_power_kw - expected_grid,
            renewable_balance_error=(
                physical.renewable_available_kw
                - physical.renewable_used_kw
                - physical.renewable_curtailed_kw
            ),
            thermal_balance_error=physical.true_temperature_c - expected_temp,
            temperature_violation=below or above,
            below_min_temperature=below,
            above_max_temperature=above,
            invalid_value_count=sum(not math.isfinite(value) for value in checked),
            negative_power_count=negative,
            actuator_tracking_error_kw=tracking_error,
            ramp_limited=ramp_limited,
            ramp_up_limited=ramp_up_limited,
            ramp_down_limited=ramp_down_limited,
            thermal_infeasible=bool(constrained.thermal_infeasible),
            actuator_induced_thermal_infeasible=actuator_infeasible,
            predicted_next_temperature_c=predicted,
            prediction_error_c=predicted - physical.true_temperature_c,
            cooling_min_feasible_kw=float(constrained.minimum_feasible_kw),
            cooling_max_feasible_kw=float(constrained.maximum_feasible_kw),
            cooling_constraint_intervention_kw=float(constrained.intervention_kw),
        )
