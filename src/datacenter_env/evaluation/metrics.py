from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from datacenter_env.config import DataCenterSystemConfig
from datacenter_env.contracts import StepResult
from datacenter_env.contracts.tasks import TaskOutcome, TaskStatus
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

    def summarize(
        self, task_outcomes: tuple[TaskOutcome, ...] = ()
    ) -> dict[str, float | None]:
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
        if self.config.workload_mode == "task_queue":
            metrics.update(self._task_metrics(frame, task_outcomes))
        return metrics

    def _task_metrics(
        self, frame: pd.DataFrame, outcomes: tuple[TaskOutcome, ...]
    ) -> dict[str, float | None]:
        completed_statuses = {TaskStatus.COMPLETED, TaskStatus.SLA_VIOLATED}
        completed = tuple(item for item in outcomes if item.final_status in completed_statuses)
        unfinished = tuple(
            item for item in outcomes if item.final_status in {TaskStatus.WAITING, TaskStatus.RUNNING}
        )
        violated = tuple(item for item in completed if item.sla_violated)
        lateness = [float(item.lateness_minutes or 0.0) for item in completed]
        waits = [float(item.wait_steps) for item in completed]
        deferrals = [float(item.deferral_count) for item in outcomes]
        resource_blocks = [float(item.resource_blocked_count) for item in outcomes]
        completed_ids = {item.task_id for item in completed}
        duration_by_id = {
            task.task_id: task.duration_steps
            for step in self._steps
            if step.tasking is not None
            for task in step.tasking.arrived_tasks
        }
        completed_count = len(completed)
        hours = len(frame) * self.config.step_hours
        cost = float(frame["energy_cost_step"].sum())
        carbon = float(frame["carbon_kg_step"].sum())
        grid = float(frame["grid_energy_kwh"].sum())
        return {
            "tasks_arrived": float(len(outcomes)),
            "tasks_started": float(sum(item.first_start_time is not None for item in outcomes)),
            "tasks_completed": float(completed_count),
            "tasks_unschedulable": float(
                sum(item.final_status is TaskStatus.UNSCHEDULABLE for item in outcomes)
            ),
            "tasks_unfinished": float(len(unfinished)),
            "sla_violation_count": float(len(violated)),
            "sla_violation_rate": (
                float(len(violated) / completed_count) if completed_count else 0.0
            ),
            "peak_at_risk_task_count": float(frame["at_risk_task_count"].max()),
            "mean_lateness_minutes": float(pd.Series(lateness).mean()) if lateness else 0.0,
            "max_lateness_minutes": max(lateness, default=0.0),
            "mean_wait_steps": float(pd.Series(waits).mean()) if waits else 0.0,
            "max_wait_steps": max(waits, default=0.0),
            "p95_wait_steps": float(pd.Series(waits).quantile(0.95)) if waits else 0.0,
            "total_deferrals": float(sum(deferrals)),
            "mean_deferrals": float(pd.Series(deferrals).mean()) if deferrals else 0.0,
            "total_resource_blocked": float(sum(resource_blocks)),
            "mean_resource_blocked": (
                float(pd.Series(resource_blocks).mean()) if resource_blocks else 0.0
            ),
            "completed_per_hour": float(completed_count / hours) if hours else 0.0,
            "completed_task_steps": float(
                sum(duration_by_id.get(task_id, 0) for task_id in completed_ids)
            ),
            "mean_cpu_utilization": float(frame["cpu_utilization"].mean()),
            "peak_cpu_utilization": float(frame["cpu_utilization"].max()),
            "mean_gpu_utilization": float(frame["gpu_utilization"].mean()),
            "peak_gpu_utilization": float(frame["gpu_utilization"].max()),
            "mean_memory_utilization": float(frame["memory_utilization"].mean()),
            "peak_memory_utilization": float(frame["memory_utilization"].max()),
            "mean_queue_length": float(frame["waiting_task_count"].mean()),
            "max_queue_length": float(frame["waiting_task_count"].max()),
            "mean_running_tasks": float(frame["running_task_count"].mean()),
            "cost_per_completed_task": cost / completed_count if completed_count else None,
            "carbon_per_completed_task_kg": carbon / completed_count if completed_count else None,
            "grid_energy_per_completed_task_kwh": (
                grid / completed_count if completed_count else None
            ),
        }

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
