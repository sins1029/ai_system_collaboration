from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import unittest

import datacenter_env
from datacenter_env import (
    DataCenterAction,
    DataCenterEnvironment,
    DataCenterSystemConfig,
    ExogenousInput,
    ForecastWindow,
)
from datacenter_env.control import build_controller
from datacenter_env.storage import schema_text
from models.config_loader import load_simple_yaml
from models.datacenter import DatacenterModel


ROOT = Path(__file__).resolve().parents[1]


def package_config(controller: str = "baseline", **kwargs) -> DataCenterSystemConfig:
    return DataCenterSystemConfig.from_dict(
        load_simple_yaml(ROOT / "configs" / "datacenter.yaml"),
        load_simple_yaml(ROOT / "configs" / "optimization.yaml"),
        controller_name=controller,
        **kwargs,
    )


def external_input(minute: int = 0, workload: float = 0.5) -> ExogenousInput:
    return ExogenousInput(
        timestamp=datetime(2026, 1, 1) + timedelta(minutes=minute),
        workload_fraction=workload,
        electricity_price_per_kwh=0.5,
        carbon_intensity_kg_per_kwh=0.4,
        outdoor_temperature_c=25.0,
        renewable_power_kw=100.0,
    )


class PackageApiTest(unittest.TestCase):
    def test_top_level_public_api_is_small_and_complete(self) -> None:
        expected = {
            "DataCenterSystem",
            "DataCenterEnvironment",
            "DataCenterSystemConfig",
            "ExogenousInput",
            "ForecastWindow",
            "DataCenterAction",
            "DataCenterObservation",
            "StepResult",
            "RunSummary",
            "SQLiteRunStore",
            "NullRunStore",
            "run_single_center",
            "__version__",
        }
        self.assertTrue(expected.issubset(set(datacenter_env.__all__)))
        self.assertFalse(hasattr(datacenter_env, "CoolingActuator"))

    def test_package_schema_resource_is_installed(self) -> None:
        schema = schema_text()
        self.assertIn("CREATE TABLE IF NOT EXISTS experiment_runs", schema)
        self.assertIn("schema_versions", schema)

    def test_environment_step_matches_legacy_plant_for_feasible_action(self) -> None:
        config = package_config()
        current = external_input()
        env = DataCenterEnvironment.from_config(config)
        env.reset(seed=1)
        result = env.step(DataCenterAction(150.0), current)
        legacy = DatacenterModel(
            dict(config.datacenter), config.step_hours
        ).plant_step(
            current_temp_c=24.0,
            online_load=0.5,
            batch_service=0.0,
            cooling_kw=150.0,
            outdoor_temp_c=25.0,
            renewable_available_kw=100.0,
        )
        self.assertAlmostEqual(result.physical.true_temperature_c, legacy.temp_c, places=12)
        self.assertAlmostEqual(result.physical.grid_power_kw, legacy.grid_power_kw, places=12)

    def test_all_controllers_share_act_interface(self) -> None:
        current = external_input()
        future = ForecastWindow(tuple(external_input(15 * i) for i in range(8)))
        for name in ("baseline", "heuristic", "finite_horizon"):
            controller = build_controller(package_config(name))
            controller.reset(seed=4)
            window = ForecastWindow((current,)) if name == "baseline" else future
            decision = controller.act(
                DataCenterEnvironment.from_config(package_config(name)).get_observation(current),
                window,
            )
            self.assertIsInstance(decision.action, DataCenterAction)
            self.assertEqual(decision.result.controller_name, name)

    def test_datacenter_package_does_not_import_external_signal_owners(self) -> None:
        package_root = ROOT / "src" / "datacenter_env"
        source = "\n".join(path.read_text(encoding="utf-8") for path in package_root.rglob("*.py"))
        self.assertNotIn("models.synthetic_signals", source)
        self.assertNotIn("models.exogenous_signals", source)
        self.assertNotIn("read_csv(", source)

    def test_config_is_not_mutated_by_instances(self) -> None:
        raw = load_simple_yaml(ROOT / "configs" / "datacenter.yaml")
        config = DataCenterSystemConfig.from_dict(
            raw, load_simple_yaml(ROOT / "configs" / "optimization.yaml")
        )
        env = DataCenterEnvironment.from_config(config)
        env.reset(seed=2)
        env.step(DataCenterAction(120.0), external_input())
        self.assertEqual(raw["thermal"]["initial_temp_c"], 24.0)
        with self.assertRaises(TypeError):
            config.datacenter["thermal"]["initial_temp_c"] = 30.0
