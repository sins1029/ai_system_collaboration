from __future__ import annotations

from pathlib import Path
import unittest

from evaluation.metrics import summarize
from experiments.mvp import load_yaml, simulate
from experiments.mvp import prepare_signals
from optimization.scenarios import build_controls, load_scenarios


ROOT = Path(__file__).resolve().parents[1]


class SingleCenterClosedLoopTest(unittest.TestCase):
    def test_all_scenarios_are_thermally_and_numerically_valid(self) -> None:
        dc_config = load_yaml(ROOT / "configs" / "datacenter.yaml")
        exp_config = load_yaml(ROOT / "configs" / "experiment.yaml")
        opt_config = load_yaml(ROOT / "configs" / "optimization.yaml")
        signal_config = load_yaml(ROOT / "configs" / "signals.yaml")
        signals = prepare_signals(ROOT, exp_config, signal_config)
        step_hours = float(exp_config["time_step_minutes"]) / 60.0

        for scenario in load_scenarios(exp_config).values():
            service, controller = build_controls(
                scenario, signals, dc_config, opt_config, step_hours
            )
            result = simulate(signals, service, controller, dc_config, step_hours)
            metrics = summarize(result, step_hours)
            self.assertEqual(metrics["temperature_violation_count"], 0.0)
            self.assertEqual(metrics["thermal_infeasibility_count"], 0.0)
            self.assertEqual(metrics["invalid_value_count"], 0.0)
            self.assertLessEqual(metrics["max_abs_power_balance_error"], 1e-9)
            self.assertLessEqual(metrics["max_abs_renewable_balance_error"], 1e-9)


if __name__ == "__main__":
    unittest.main()
