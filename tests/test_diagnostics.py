from __future__ import annotations

import unittest

import pandas as pd

from evaluation.metrics import summarize
from models.diagnostics import add_step_diagnostics, summarize_diagnostics
from models.thermal import next_temperature_c


DC_CONFIG = {
    "power": {"p_aux_kw": 12.0},
    "thermal": {
        "min_temp_c": 18.0,
        "setpoint_temp_c": 24.0,
        "max_temp_c": 32.0,
        "deadband_c": 0.5,
        "thermal_resistance_c_per_kw": 0.18,
        "thermal_capacitance_kwh_per_c": 55.0,
    },
}


def diagnostic_row(temp_c: float, cooling_power_kw: float = 45.0) -> dict:
    return {
        "timestamp": pd.Timestamp("2026-01-01", tz="Asia/Shanghai"),
        "online_workload": 0.3,
        "batch_workload": 0.2,
        "electricity_price": 1.0,
        "carbon_intensity_kg_per_kwh": 0.5,
        "outdoor_temperature_c": 10.0,
        "prev_temp_c": 24.0,
        "total_load": 0.5,
        "it_power_kw": 170.0,
        "heat_kw": 170.0,
        "cooling_kw": 180.0,
        "cooling_power_kw": cooling_power_kw,
        "cooling_command_kw": 180.0,
        "proposed_cooling_kw": 180.0,
        "constrained_cooling_kw": 180.0,
        "applied_cooling_kw": 180.0,
        "control_movement_kw": 180.0,
        "proposed_control_movement_kw": 180.0,
        "applied_control_movement_kw": 180.0,
        "actuator_tracking_error_kw": 0.0,
        "actuator_ramp_limited": 0,
        "actuator_ramp_up_limited": 0,
        "actuator_ramp_down_limited": 0,
        "actuator_induced_thermal_infeasibility": 0,
        "p_aux_kw": 12.0,
        "renewable_available_kw": 20.0,
        "renewable_used_kw": 20.0,
        "renewable_curtailed_kw": 0.0,
        "total_power_kw": 227.0,
        "grid_power_kw": 207.0,
        "temp_c": temp_c,
        "temp_min_c": 18.0,
        "temp_setpoint_c": 24.0,
        "temp_max_c": 32.0,
        "cooling_constraint_intervention": 0,
        "cooling_constraint_intervention_kw": 0.0,
        "thermal_infeasible": 0,
        "backlog": 0.0,
        "optimizer_attempted": 0,
        "optimizer_success": 0,
        "optimizer_failure": 0,
        "optimizer_fallback": 0,
        "prediction_infeasibility": 0,
        "actuator_infeasibility": 0,
        "optimizer_candidate_evaluations": 0,
        "objective_total": 0.0,
        "objective_energy_cost": 0.0,
        "objective_carbon": 0.0,
        "objective_temperature": 0.0,
        "objective_control_movement": 0.0,
        "one_step_temperature_prediction_error_c": 0.0,
    }


class DiagnosticsTest(unittest.TestCase):
    def test_normal_input_has_small_balance_errors(self) -> None:
        expected_temp = next_temperature_c(
            24.0, 170.0, 180.0, 10.0, 0.25, 0.18, 55.0
        )
        result = pd.DataFrame(add_step_diagnostics([diagnostic_row(expected_temp)], DC_CONFIG, 0.25))
        metrics = summarize(result, 0.25)
        self.assertLess(metrics["max_abs_power_balance_error"], 1e-9)
        self.assertLess(metrics["max_abs_renewable_balance_error"], 1e-9)
        self.assertLess(metrics["max_abs_thermal_balance_error"], 1e-9)
        self.assertEqual(metrics["invalid_value_count"], 0.0)

    def test_nan_and_negative_power_are_detected(self) -> None:
        row = diagnostic_row(120.0, float("nan"))
        row["it_power_kw"] = -1.0
        row["heat_kw"] = -1.0
        row["thermal_infeasible"] = 1
        result = pd.DataFrame(add_step_diagnostics([row], DC_CONFIG, 0.25))
        self.assertGreater(int(result["invalid_value_count"].iloc[0]), 0)
        self.assertGreater(int(result["negative_power_count"].iloc[0]), 0)
        self.assertEqual(int(result["temperature_violation"].iloc[0]), 1)
        self.assertEqual(int(summarize_diagnostics(result)["thermal_infeasibility_count"]), 1)


if __name__ == "__main__":
    unittest.main()
