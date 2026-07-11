from __future__ import annotations

import pandas as pd

from evaluation.accounting import carbon_emissions_kg, energy_cost
from models.diagnostics import summarize_diagnostics


def summarize(result: pd.DataFrame, step_hours: float) -> dict[str, float]:
    grid_energy = float((result["grid_power_kw"] * step_hours).sum())
    dc_energy = float((result["total_power_kw"] * step_hours).sum())
    cooling_energy = float((result["cooling_power_kw"] * step_hours).sum())
    renewable_available = float((result["renewable_available_kw"] * step_hours).sum())
    renewable_used = float((result["renewable_used_kw"] * step_hours).sum())
    renewable_curtailed = float((result["renewable_curtailed_kw"] * step_hours).sum())
    renewable_rate = renewable_used / renewable_available if renewable_available > 0 else 0.0
    cost = float(sum(
        energy_cost(power, price, step_hours)
        for power, price in zip(result["grid_power_kw"], result["electricity_price"])
    ))
    carbon_kg = float(sum(
        carbon_emissions_kg(power, intensity, step_hours)
        for power, intensity in zip(
            result["grid_power_kw"], result["carbon_intensity_kg_per_kwh"]
        )
    ))
    max_temp = float(result["temp_c"].max())
    avg_temp = float(result["temp_c"].mean())
    violations = int(result["temperature_violation"].sum())
    attempted = result["optimizer_attempted"] > 0
    successful = result["optimizer_success"] > 0
    metrics = {
        "energy_kwh": grid_energy,
        "total_grid_energy_kwh": grid_energy,
        "total_energy_kwh": dc_energy,
        "total_dc_energy_kwh": dc_energy,
        "cooling_energy_kwh": cooling_energy,
        "renewable_energy_available_kwh": renewable_available,
        "renewable_energy_used_kwh": renewable_used,
        "renewable_energy_curtailed_kwh": renewable_curtailed,
        "renewable_utilization_rate": renewable_rate,
        "cost": cost,
        "energy_cost": cost,
        "carbon_kg": carbon_kg,
        "average_temperature_c": avg_temp,
        "max_temp_c": max_temp,
        "maximum_temperature_c": max_temp,
        "thermal_violations": float(violations),
        "temperature_violation_count": float(violations),
        "peak_grid_kw": float(result["grid_power_kw"].max()),
        "peak_grid_power_kw": float(result["grid_power_kw"].max()),
        "final_backlog": float(result["backlog"].iloc[-1]),
        "mean_cooling_command": float(result["cooling_kw"].mean()),
        "max_cooling_command": float(result["cooling_kw"].max()),
        "control_movement_total": float(result["control_movement_kw"].sum()),
        "proposed_control_movement_total": float(
            result["proposed_control_movement_kw"].sum()
        ),
        "applied_control_movement_total": float(
            result["applied_control_movement_kw"].sum()
        ),
        "mean_actuator_tracking_error_kw": float(
            result["actuator_tracking_error_kw"].mean()
        ),
        "max_actuator_tracking_error_kw": float(
            result["actuator_tracking_error_kw"].max()
        ),
        "total_actuator_tracking_error_kw": float(
            result["actuator_tracking_error_kw"].sum()
        ),
        "mean_actuator_lag_kw": float(result["actuator_tracking_error_kw"].mean()),
        "ramp_limited_count": float(result["actuator_ramp_limited"].sum()),
        "ramp_up_limited_count": float(result["actuator_ramp_up_limited"].sum()),
        "ramp_down_limited_count": float(result["actuator_ramp_down_limited"].sum()),
        "actuator_induced_thermal_infeasibility_count": float(
            result["actuator_induced_thermal_infeasibility"].sum()
        ),
        "optimizer_success_count": float(result["optimizer_success"].sum()),
        "optimizer_failure_count": float(result["optimizer_failure"].sum()),
        "optimizer_fallback_count": float(result["optimizer_fallback"].sum()),
        "prediction_infeasibility_count": float(result["prediction_infeasibility"].sum()),
        "actuator_infeasibility_count": float(result["actuator_infeasibility"].sum()),
        "mean_optimizer_candidate_evaluations": (
            float(result.loc[attempted, "optimizer_candidate_evaluations"].mean())
            if attempted.any()
            else 0.0
        ),
        "mean_objective_value": (
            float(result.loc[successful, "objective_total"].mean()) if successful.any() else 0.0
        ),
        "mean_objective_energy_cost": (
            float(result.loc[successful, "objective_energy_cost"].mean())
            if successful.any()
            else 0.0
        ),
        "mean_objective_carbon": (
            float(result.loc[successful, "objective_carbon"].mean())
            if successful.any()
            else 0.0
        ),
        "mean_objective_temperature": (
            float(result.loc[successful, "objective_temperature"].mean())
            if successful.any()
            else 0.0
        ),
        "mean_objective_control_movement": (
            float(result.loc[successful, "objective_control_movement"].mean())
            if successful.any()
            else 0.0
        ),
    }
    prediction_error = result["one_step_temperature_prediction_error_c"]
    metrics.update(
        {
            "mean_abs_temperature_prediction_error_c": float(prediction_error.abs().mean()),
            "max_abs_temperature_prediction_error_c": float(prediction_error.abs().max()),
            "rmse_temperature_prediction_error_c": float(
                (prediction_error.pow(2).mean()) ** 0.5
            ),
            "temperature_prediction_bias_c": float(prediction_error.mean()),
        }
    )
    metrics.update(summarize_diagnostics(result))
    return metrics
