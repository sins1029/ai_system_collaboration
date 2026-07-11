from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from datacenter_env import (
    DataCenterSystem,
    DataCenterSystemConfig,
    NullRunStore,
    RunMetadata,
)
from experiments.mvp import load_yaml, prepare_signals
from external_signals import build_scenario_provider
from models.conditions import resolve_condition
from optimization.scenarios import load_scenarios


ROOT = Path(__file__).resolve().parents[1]


class PackageRegressionTest(unittest.TestCase):
    def test_ideal_results_match_pre_package_baselines(self) -> None:
        dc = load_yaml(ROOT / "configs" / "datacenter.yaml")
        experiment = load_yaml(ROOT / "configs" / "experiment.yaml")
        optimization = load_yaml(ROOT / "configs" / "optimization.yaml")
        signals = prepare_signals(
            ROOT,
            experiment,
            load_yaml(ROOT / "configs" / "signals.yaml"),
        )
        condition = resolve_condition(
            load_yaml(ROOT / "configs" / "robustness.yaml"), "ideal"
        )
        expected = {
            "baseline": (2421.4561, 2189.4853),
            "heuristic": (2364.9363, 2168.3593),
            "finite_horizon": (2332.7363, 2125.1807),
        }
        for name, scenario in load_scenarios(experiment).items():
            opt = deepcopy(optimization)
            provider = build_scenario_provider(signals, scenario.dispatch_strategy, opt)
            config = DataCenterSystemConfig.from_dict(
                dc,
                opt,
                controller_name=name,
                step_minutes=int(experiment["time_step_minutes"]),
                actuator=condition["actuator"],
                plant_parameters=condition["plant"],
                prediction_parameters=condition["prediction"],
                measurement=condition["measurement"],
            )
            system = DataCenterSystem.from_config(config, store=NullRunStore())
            system.start_run(RunMetadata(controller_name=name))
            for index, current in enumerate(provider):
                system.step(
                    current,
                    provider.window(index, system.controller.max_forecast_steps),
                )
            metrics = system.finish_run().metrics
            self.assertAlmostEqual(metrics["energy_cost"], expected[name][0], places=4)
            self.assertAlmostEqual(metrics["carbon_kg"], expected[name][1], places=4)
            self.assertEqual(metrics["temperature_violation_count"], 0.0)
