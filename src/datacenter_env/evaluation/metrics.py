from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from datacenter_env.config import DataCenterSystemConfig
from datacenter_env.contracts import StepResult
from datacenter_env.evaluation.records import step_to_record


@dataclass
class MetricAggregator:
    config: DataCenterSystemConfig
    _steps: list[StepResult] = field(default_factory=list)

    def reset(self) -> None:
        self._steps.clear()

    def update(self, result: StepResult) -> None:
        self._steps.append(result)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(step_to_record(step, self.config) for step in self._steps)

    def summarize(self) -> dict[str, float]:
        frame = self.frame()
        if frame.empty:
            return {}
        grid_energy = float(frame["grid_energy_kwh"].sum())
        dc_energy = float(frame["dc_energy_kwh"].sum())
        cooling_energy = float(frame["cooling_energy_kwh"].sum())
        renewable_available = float(
            (frame["renewable_available_kw"] * self.config.step_hours).sum()
        )
        renewable_used = float(
            (frame["renewable_used_kw"] * self.config.step_hours).sum()
        )
        renewable_curtailed = float(
            (frame["renewable_curtailed_kw"] * self.config.step_hours).sum()
        )
        attempted = frame["optimizer_attempted"] > 0
        successful = frame["optimizer_success"] > 0
        prediction_error = frame["one_step_temperature_prediction_error_c"]
        below = frame["below_min_temperature"] > 0
        above = frame["above_max_temperature"] > 0
        metrics = {
            "energy_kwh": grid_energy,
            "total_grid_energy_kwh": grid_energy,
            "total_energy_kwh": dc_energy,
            "total_dc_energy_kwh": dc_energy,
            "cooling_energy_kwh": cooling_energy,
            "renewable_energy_available_kwh": renewable_available,
            "renewable_energy_used_kwh": renewable_used,
            "renewable_energy_curtailed_kwh": renewable_curtailed,
            "renewable_utilization_rate": (
                renewable_used / renewable_available if renewable_available > 0 else 0.0
            ),
            "cost": float(frame["energy_cost_step"].sum()),
            "energy_cost": float(frame["energy_cost_step"].sum()),
            "carbon_kg": float(frame["carbon_kg_step"].sum()),
            "average_temperature_c": float(frame["temp_c"].mean()),
            "max_temp_c": float(frame["temp_c"].max()),
            "maximum_temperature_c": float(frame["temp_c"].max()),
            "thermal_violations": float(frame["temperature_violation"].sum()),
            "temperature_violation_count": float(frame["temperature_violation"].sum()),
            "peak_grid_kw": float(frame["grid_power_kw"].max()),
            "peak_grid_power_kw": float(frame["grid_power_kw"].max()),
            "final_backlog": 0.0,
            "mean_cooling_command": float(frame["cooling_kw"].mean()),
            "max_cooling_command": float(frame["cooling_kw"].max()),
            "control_movement_total": float(frame["control_movement_kw"].sum()),
            "proposed_control_movement_total": float(
                frame["proposed_control_movement_kw"].sum()
            ),
            "applied_control_movement_total": float(
                frame["applied_control_movement_kw"].sum()
            ),
            "mean_actuator_tracking_error_kw": float(
                frame["actuator_tracking_error_kw"].mean()
            ),
            "max_actuator_tracking_error_kw": float(
                frame["actuator_tracking_error_kw"].max()
            ),
            "total_actuator_tracking_error_kw": float(
                frame["actuator_tracking_error_kw"].sum()
            ),
            "mean_actuator_lag_kw": float(frame["actuator_tracking_error_kw"].mean()),
            "ramp_limited_count": float(frame["actuator_ramp_limited"].sum()),
            "ramp_up_limited_count": float(frame["actuator_ramp_up_limited"].sum()),
            "ramp_down_limited_count": float(frame["actuator_ramp_down_limited"].sum()),
            "actuator_induced_thermal_infeasibility_count": float(
                frame["actuator_induced_thermal_infeasibility"].sum()
            ),
            "optimizer_success_count": float(frame["optimizer_success"].sum()),
            "optimizer_failure_count": float(frame["optimizer_failure"].sum()),
            "optimizer_fallback_count": float(frame["optimizer_fallback"].sum()),
            "prediction_infeasibility_count": float(frame["prediction_infeasibility"].sum()),
            "actuator_infeasibility_count": float(frame["actuator_infeasibility"].sum()),
            "mean_optimizer_candidate_evaluations": (
                float(frame.loc[attempted, "optimizer_candidate_evaluations"].mean())
                if attempted.any()
                else 0.0
            ),
            "mean_objective_value": self._successful_mean(frame, successful, "objective_total"),
            "mean_objective_energy_cost": self._successful_mean(
                frame, successful, "objective_energy_cost"
            ),
            "mean_objective_carbon": self._successful_mean(
                frame, successful, "objective_carbon"
            ),
            "mean_objective_temperature": self._successful_mean(
                frame, successful, "objective_temperature"
            ),
            "mean_objective_control_movement": self._successful_mean(
                frame, successful, "objective_control_movement"
            ),
            "mean_abs_temperature_prediction_error_c": float(prediction_error.abs().mean()),
            "max_abs_temperature_prediction_error_c": float(prediction_error.abs().max()),
            "rmse_temperature_prediction_error_c": float(
                (prediction_error.pow(2).mean()) ** 0.5
            ),
            "temperature_prediction_bias_c": float(prediction_error.mean()),
            "max_abs_power_balance_error": float(frame["power_balance_error"].abs().max()),
            "mean_abs_power_balance_error": float(frame["power_balance_error"].abs().mean()),
            "max_abs_renewable_balance_error": float(
                frame["renewable_balance_error"].abs().max()
            ),
            "mean_abs_renewable_balance_error": float(
                frame["renewable_balance_error"].abs().mean()
            ),
            "max_abs_thermal_balance_error": float(frame["thermal_balance_error"].abs().max()),
            "mean_abs_thermal_balance_error": float(
                frame["thermal_balance_error"].abs().mean()
            ),
            "below_min_temperature_count": float(below.sum()),
            "above_max_temperature_count": float(above.sum()),
            "minimum_temperature_c": float(frame["temp_c"].min()),
            "mean_temperature_deviation_from_setpoint": float(
                frame["temperature_deviation_from_setpoint_c"].mean()
            ),
            "max_temperature_deviation_from_setpoint": float(
                frame["temperature_deviation_from_setpoint_c"].max()
            ),
            "longest_below_min_streak": float(self._longest_streak(below)),
            "longest_above_max_streak": float(self._longest_streak(above)),
            "thermal_infeasibility_count": float(frame["thermal_infeasible"].sum()),
            "cooling_constraint_intervention_count": float(
                frame["cooling_constraint_intervention"].sum()
            ),
            "cooling_constraint_intervention_energy_kwh": float(
                frame["cooling_constraint_intervention_energy_kwh"].sum()
            ),
            "invalid_value_count": float(frame["invalid_value_count"].sum()),
            "negative_power_count": float(frame["negative_power_count"].sum()),
        }
        return metrics

    @staticmethod
    def _successful_mean(frame: pd.DataFrame, successful: pd.Series, column: str) -> float:
        if not successful.any():
            return 0.0
        values = frame.loc[successful, column].dropna()
        return float(values.mean()) if not values.empty else 0.0

    @staticmethod
    def _longest_streak(values: pd.Series) -> int:
        longest = current = 0
        for value in values:
            current = current + 1 if bool(value) else 0
            longest = max(longest, current)
        return longest
