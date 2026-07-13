from __future__ import annotations

from typing import Any

from datacenter_env.config import DataCenterSystemConfig
from datacenter_env.contracts import StepResult


def step_to_record(step: StepResult, config: DataCenterSystemConfig) -> dict[str, Any]:
    obs = step.observation
    physical = step.physical
    accounting = step.accounting
    diagnostics = step.diagnostics
    controller = step.controller
    thermal = config.datacenter["thermal"]
    proposed_movement = abs(
        physical.proposed_cooling_kw - physical.previous_proposed_cooling_kw
    )
    applied_movement = abs(
        physical.applied_cooling_kw - physical.previous_applied_cooling_kw
    )
    tasking = step.tasking
    return {
        "timestamp": obs.timestamp,
        "online_load": obs.workload_fraction,
        "batch_load": 0.0,
        "online_workload": obs.workload_fraction,
        "batch_workload": 0.0,
        "batch_service": 0.0,
        "backlog": 0.0,
        "total_load": physical.total_load,
        "prev_temp_c": physical.previous_true_temperature_c,
        "price": obs.electricity_price_per_kwh,
        "electricity_price": obs.electricity_price_per_kwh,
        "carbon": obs.carbon_intensity_kg_per_kwh * 1000.0,
        "carbon_intensity_kg_per_kwh": obs.carbon_intensity_kg_per_kwh,
        "outdoor_temp": obs.outdoor_temperature_c,
        "outdoor_temperature_c": obs.outdoor_temperature_c,
        "renewable": obs.renewable_power_kw / 100.0,
        "renewable_kw": obs.renewable_power_kw,
        "renewable_available_kw": physical.renewable_available_kw,
        "renewable_used_kw": physical.renewable_used_kw,
        "renewable_curtailed_kw": physical.renewable_curtailed_kw,
        "it_power_kw": physical.it_power_kw,
        "heat_kw": physical.heat_kw,
        "proposed_cooling_kw": physical.proposed_cooling_kw,
        "constrained_cooling_kw": physical.constrained_cooling_kw,
        "applied_cooling_kw": physical.applied_cooling_kw,
        "cooling_command_kw": physical.proposed_cooling_kw,
        "cooling_kw": physical.applied_cooling_kw,
        "proposed_control_movement_kw": proposed_movement,
        "applied_control_movement_kw": applied_movement,
        "control_movement_kw": applied_movement,
        "actuator_tracking_error_kw": diagnostics.actuator_tracking_error_kw,
        "actuator_ramp_limited": int(diagnostics.ramp_limited),
        "actuator_ramp_up_limited": int(diagnostics.ramp_up_limited),
        "actuator_ramp_down_limited": int(diagnostics.ramp_down_limited),
        "actuator_alpha": physical.actuator_alpha,
        "actuator_induced_thermal_infeasibility": int(
            diagnostics.actuator_induced_thermal_infeasible
        ),
        "cooling_min_feasible_kw": diagnostics.cooling_min_feasible_kw,
        "cooling_max_feasible_kw": diagnostics.cooling_max_feasible_kw,
        "cooling_constraint_intervention": int(
            diagnostics.cooling_constraint_intervention_kw > 1e-9
        ),
        "cooling_constraint_intervention_kw": (
            diagnostics.cooling_constraint_intervention_kw
        ),
        "cooling_constraint_intervention_energy_kwh": (
            diagnostics.cooling_constraint_intervention_kw * config.step_hours
        ),
        "cooling_power_kw": physical.cooling_power_kw,
        "p_aux_kw": float(config.datacenter["power"]["p_aux_kw"]),
        "total_power_kw": physical.dc_power_kw,
        "grid_power_kw": physical.grid_power_kw,
        "temp_c": physical.true_temperature_c,
        "true_temperature_c": physical.true_temperature_c,
        "measured_temperature_c": physical.measured_temperature_c,
        "controller_measured_temperature_c": physical.controller_measured_temperature_c,
        "predicted_next_temperature_c": diagnostics.predicted_next_temperature_c,
        "actual_next_temperature_c": physical.true_temperature_c,
        "one_step_temperature_prediction_error_c": diagnostics.prediction_error_c,
        "temp_min_c": float(thermal["min_temp_c"]),
        "temp_setpoint_c": float(thermal["setpoint_temp_c"]),
        "temp_max_c": float(thermal["max_temp_c"]),
        "temp_deadband_c": float(thermal["deadband_c"]),
        "temp_control_target_c": controller.target_temperature_c,
        "precool_floor_c": controller.control_floor_temperature_c,
        "precooling_active": int(controller.precooling_active),
        "power_balance_error": diagnostics.power_balance_error,
        "renewable_balance_error": diagnostics.renewable_balance_error,
        "thermal_balance_error": diagnostics.thermal_balance_error,
        "temperature_is_finite": int(diagnostics.invalid_value_count == 0),
        "temperature_within_physical_range": int(
            -50.0 <= physical.true_temperature_c <= 100.0
        ),
        "temperature_step_change": (
            physical.true_temperature_c - physical.previous_true_temperature_c
        ),
        "below_min_temperature": int(diagnostics.below_min_temperature),
        "above_max_temperature": int(diagnostics.above_max_temperature),
        "temperature_violation": int(diagnostics.temperature_violation),
        "temperature_deviation_from_setpoint_c": abs(
            physical.true_temperature_c - float(thermal["setpoint_temp_c"])
        ),
        "thermal_infeasible": int(diagnostics.thermal_infeasible),
        "optimizer_attempted": int(controller.optimizer_attempted),
        "optimizer_success": int(controller.optimizer_success),
        "optimizer_failure": int(controller.optimizer_failure),
        "optimizer_fallback": int(controller.fallback_used),
        "prediction_infeasibility": int(controller.prediction_infeasibility),
        "actuator_infeasibility": int(
            controller.actuator_infeasibility
            or diagnostics.actuator_induced_thermal_infeasible
        ),
        "fallback_reason": controller.fallback_reason or "",
        "optimizer_candidate_evaluations": controller.candidate_evaluations or 0,
        "objective_total": controller.objective_total,
        "objective_energy_cost": controller.objective_energy_cost,
        "objective_carbon": controller.objective_carbon,
        "objective_temperature": controller.objective_temperature,
        "objective_control_movement": controller.objective_control_movement,
        "invalid_value_count": diagnostics.invalid_value_count,
        "negative_power_count": diagnostics.negative_power_count,
        "workload_mode": obs.workload_mode,
        "cpu_utilization": tasking.cpu_utilization if tasking else 0.0,
        "gpu_utilization": tasking.gpu_utilization if tasking else 0.0,
        "memory_utilization": tasking.memory_utilization if tasking else 0.0,
        "waiting_task_count": tasking.waiting_count if tasking else 0,
        "running_task_count": tasking.running_count if tasking else 0,
        "completed_task_count": tasking.completed_count if tasking else 0,
        "at_risk_task_count": tasking.at_risk_count if tasking else 0,
        "dc_energy_kwh": accounting.dc_energy_kwh,
        "grid_energy_kwh": accounting.grid_energy_kwh,
        "cooling_energy_kwh": accounting.cooling_energy_kwh,
        "energy_cost_step": accounting.energy_cost,
        "carbon_kg_step": accounting.carbon_kg,
    }
