from __future__ import annotations

from copy import deepcopy
import unittest

import pandas as pd

from models.exogenous_signals import ExogenousSignals, SignalMetadata
from models.thermal_constraints import apply_thermal_constraints, validate_thermal_control_config
from optimization.thermal_control import TemperatureFeedbackCoolingController


DC_CONFIG = {
    "capacity": {"max_load": 1.0},
    "power": {"p_idle_kw": 80.0, "p_peak_kw": 260.0, "p_aux_kw": 12.0},
    "thermal": {
        "initial_temp_c": 24.0, "min_temp_c": 18.0, "setpoint_temp_c": 24.0,
        "max_temp_c": 32.0, "deadband_c": 0.5,
        "thermal_resistance_c_per_kw": 0.18, "thermal_capacitance_kwh_per_c": 55.0,
    },
    "cooling": {
        "max_cooling_kw": 280.0, "temperature_feedback_gain_kw_per_c": 30.0,
        "cop_base": 4.0, "cop_temp_slope": 0.035, "cop_min": 2.2,
    },
}
OPT_CONFIG = {
    "cooling": {
        "low_price_threshold": 0.4, "high_price_threshold": 0.7,
        "precooling": {
            "enabled": True, "lookahead_steps": 2,
            "max_precool_delta_c": 3.0, "precool_floor_c": 20.5,
        },
    }
}


def signals() -> ExogenousSignals:
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=2, freq="15min"),
        "online_workload": [0.4, 0.4], "batch_workload": [0.1, 0.1],
        "electricity_price": [0.2, 1.0],
        "carbon_intensity_kg_per_kwh": [0.5, 0.6],
        "outdoor_temperature_c": [20.0, 20.0], "renewable_power_kw": [0.0, 0.0],
    })
    return ExogenousSignals(frame, SignalMetadata("test", "Asia/Shanghai", 15, 1, {}))


class ThermalControlTest(unittest.TestCase):
    def command(self, controller, temperature: float):
        return controller.command(
            current_temp_c=temperature, heat_kw=180.0, outdoor_temp_c=20.0,
            electricity_price=0.2, forecast=signals().window(0, controller.lookahead_steps),
            previous_cooling_kw=100.0,
        )

    def test_baseline_temperature_feedback_changes_command(self) -> None:
        controller = TemperatureFeedbackCoolingController("load_following", DC_CONFIG, OPT_CONFIG)
        self.assertLess(
            self.command(controller, 22.0).requested_cooling_kw,
            self.command(controller, 24.0).requested_cooling_kw,
        )
        self.assertGreater(
            self.command(controller, 26.0).requested_cooling_kw,
            self.command(controller, 24.0).requested_cooling_kw,
        )
        self.assertEqual(controller.lookahead_steps, 1)

    def test_controller_decision_depends_on_measured_not_hidden_true_state(self) -> None:
        controller = TemperatureFeedbackCoolingController("load_following", DC_CONFIG, OPT_CONFIG)
        first = self.command(controller, 24.0)
        hidden_true_temperature = 30.0
        del hidden_true_temperature
        second = self.command(controller, 24.0)
        self.assertEqual(first.requested_cooling_kw, second.requested_cooling_kw)

    def test_precooling_respects_floor_in_next_step_prediction(self) -> None:
        controller = TemperatureFeedbackCoolingController(
            "price_aware_precooling", DC_CONFIG, OPT_CONFIG
        )
        command = self.command(controller, 20.55)
        action = apply_thermal_constraints(
            command.requested_cooling_kw, 20.55, 180.0, 20.0, DC_CONFIG, 0.25,
            control_floor_temp_c=command.control_floor_temp_c,
        )
        self.assertTrue(command.precooling_active)
        self.assertGreaterEqual(action.predicted_temp_c, 20.5 - 1e-9)

    def test_invalid_temperature_limits_fail(self) -> None:
        config = deepcopy(DC_CONFIG)
        config["thermal"]["min_temp_c"] = 32.0
        with self.assertRaisesRegex(ValueError, "min_temp_c"):
            validate_thermal_control_config(config, OPT_CONFIG)

    def test_invalid_setpoint_fails(self) -> None:
        config = deepcopy(DC_CONFIG)
        config["thermal"]["setpoint_temp_c"] = 40.0
        with self.assertRaisesRegex(ValueError, "setpoint_temp_c"):
            validate_thermal_control_config(config, OPT_CONFIG)

    def test_invalid_precool_floor_fails(self) -> None:
        config = deepcopy(OPT_CONFIG)
        config["cooling"]["precooling"]["precool_floor_c"] = 17.0
        with self.assertRaisesRegex(ValueError, "precool_floor_c"):
            validate_thermal_control_config(DC_CONFIG, config)

    def test_excessive_cooling_is_constrained(self) -> None:
        action = apply_thermal_constraints(10_000.0, 18.1, 100.0, 10.0, DC_CONFIG, 0.25)
        self.assertTrue(action.intervention)
        self.assertGreaterEqual(action.predicted_temp_c, 18.0 - 1e-9)

    def test_physical_infeasibility_is_reported(self) -> None:
        action = apply_thermal_constraints(0.0, 0.0, 0.0, 0.0, DC_CONFIG, 0.25)
        self.assertTrue(action.thermal_infeasible)
        self.assertLess(action.predicted_temp_c, 18.0)


if __name__ == "__main__":
    unittest.main()
