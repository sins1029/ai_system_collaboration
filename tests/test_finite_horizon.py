from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.mvp import load_yaml, simulate
from models.conditions import resolve_condition
from models.exogenous_signals import ExogenousSignals
from models.it_power import it_power_kw
from models.synthetic_signals import generate_standard_signals
from models.thermal_constraints import validate_thermal_control_config
from optimization.baseline import immediate_dispatch
from optimization.finite_horizon import FiniteHorizonCoolingController


ROOT = Path(__file__).resolve().parents[1]


class FiniteHorizonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dc_config = load_yaml(ROOT / "configs" / "datacenter.yaml")
        self.opt_config = load_yaml(ROOT / "configs" / "optimization.yaml")
        signal_config = load_yaml(ROOT / "configs" / "signals.yaml")
        self.signals = generate_standard_signals(signal_config)

    def command(self, controller, signals: ExogenousSignals, temperature: float = 24.0):
        window = signals.window(0, controller.lookahead_steps)
        frame = window.to_frame()
        services = frame["batch_workload"].astype(float).tolist()
        first = frame.iloc[0]
        total_load = float(first["online_workload"] + first["batch_workload"])
        heat_kw = it_power_kw(
            total_load,
            float(self.dc_config["capacity"]["max_load"]),
            float(self.dc_config["power"]["p_idle_kw"]),
            float(self.dc_config["power"]["p_peak_kw"]),
        )
        return controller.command(
            current_temp_c=temperature,
            heat_kw=heat_kw,
            outdoor_temp_c=float(first["outdoor_temperature_c"]),
            electricity_price=float(first["electricity_price"]),
            forecast=window,
            previous_cooling_kw=0.0,
            batch_service_forecast=services,
        )

    def test_receding_horizon_executes_first_action_and_resolves(self) -> None:
        controller = FiniteHorizonCoolingController(self.dc_config, self.opt_config, 0.25)
        first = self.command(controller, self.signals)
        second = self.command(controller, self.signals, temperature=24.2)
        self.assertTrue(first.optimizer_success)
        self.assertEqual(first.requested_cooling_kw, first.planned_cooling_kw[0])
        self.assertEqual(len(first.planned_cooling_kw), controller.lookahead_steps)
        self.assertEqual(controller.solve_count, 2)
        self.assertEqual(
            first.objective_total,
            first.objective_energy_cost
            + first.objective_carbon
            + first.objective_temperature
            + first.objective_control_movement,
        )
        self.assertTrue(second.optimizer_success)

    def test_data_after_horizon_cannot_change_current_action(self) -> None:
        changed_frame = self.signals.to_frame()
        changed_frame.loc[8:, "electricity_price"] = 100.0
        changed_frame.loc[8:, "carbon_intensity_kg_per_kwh"] = 100.0
        changed = ExogenousSignals(changed_frame, self.signals.metadata)
        first_controller = FiniteHorizonCoolingController(self.dc_config, self.opt_config, 0.25)
        second_controller = FiniteHorizonCoolingController(self.dc_config, self.opt_config, 0.25)
        self.assertAlmostEqual(
            self.command(first_controller, self.signals).requested_cooling_kw,
            self.command(second_controller, changed).requested_cooling_kw,
        )

    def test_solver_failure_uses_explicit_fallback(self) -> None:
        controller = FiniteHorizonCoolingController(self.dc_config, self.opt_config, 0.25)
        with patch.object(controller, "_solve", side_effect=RuntimeError("forced")):
            command = self.command(controller, self.signals)
        self.assertTrue(command.optimizer_failure)
        self.assertTrue(command.optimizer_fallback)
        self.assertFalse(command.optimizer_success)

    def test_invalid_mpc_floor_fails_validation(self) -> None:
        config = deepcopy(self.opt_config)
        config["finite_horizon"]["mpc_temperature_floor_c"] = 17.0
        with self.assertRaisesRegex(ValueError, "mpc_temperature_floor_c"):
            validate_thermal_control_config(self.dc_config, config)

    def test_optimizer_prediction_uses_rate_limited_applied_actions(self) -> None:
        actuator_config = {
            "mode": "rate_limited", "initial_applied_cooling_kw": 0.0,
            "max_ramp_up_kw_per_step": 30.0, "max_ramp_down_kw_per_step": 50.0,
        }
        controller = FiniteHorizonCoolingController(
            self.dc_config, self.opt_config, 0.25, actuator_config=actuator_config
        )
        command = self.command(controller, self.signals)
        previous = 0.0
        for applied in command.planned_applied_cooling_kw:
            self.assertLessEqual(applied - previous, 30.0 + 1e-9)
            self.assertLessEqual(previous - applied, 50.0 + 1e-9)
            previous = applied

    def test_fallback_still_passes_through_actuator_and_plant(self) -> None:
        robust = load_yaml(ROOT / "configs" / "robustness.yaml")
        condition = resolve_condition(robust, "rate_limited")
        limited_signals = ExogenousSignals(self.signals.to_frame().iloc[:2], self.signals.metadata)
        controller = FiniteHorizonCoolingController(
            self.dc_config, self.opt_config, 0.25, actuator_config=condition["actuator"]
        )
        service = immediate_dispatch(limited_signals)
        with patch.object(controller, "_solve", side_effect=RuntimeError("forced")):
            result = simulate(
                limited_signals,
                service,
                controller,
                self.dc_config,
                0.25,
                condition=condition,
            )
        self.assertEqual(int(result["optimizer_fallback"].sum()), 2)
        self.assertTrue((result["applied_cooling_kw"] >= 0.0).all())
        self.assertTrue(result["temp_c"].notna().all())


if __name__ == "__main__":
    unittest.main()
