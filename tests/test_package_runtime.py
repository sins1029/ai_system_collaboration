from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import unittest

from datacenter_env import (
    DataCenterAction,
    DataCenterEnvironment,
    DataCenterSystem,
    DataCenterSystemConfig,
    ExogenousInput,
    ForecastWindow,
    NullRunStore,
    RunMetadata,
)
from datacenter_env.exceptions import RunStateError, TimeAlignmentError
from models.config_loader import load_simple_yaml


ROOT = Path(__file__).resolve().parents[1]


def config(controller: str = "baseline", noisy: bool = False) -> DataCenterSystemConfig:
    return DataCenterSystemConfig.from_dict(
        load_simple_yaml(ROOT / "configs" / "datacenter.yaml"),
        load_simple_yaml(ROOT / "configs" / "optimization.yaml"),
        controller_name=controller,
        measurement=(
            {"mode": "gaussian", "temperature_noise_std_c": 0.15, "seed": 8}
            if noisy
            else None
        ),
    )


def signal(index: int, workload: float = 0.5) -> ExogenousInput:
    return ExogenousInput(
        datetime(2026, 1, 1) + timedelta(minutes=15 * index),
        workload,
        0.5,
        0.4,
        25.0,
        100.0,
    )


class PackageRuntimeTest(unittest.TestCase):
    def test_baseline_rejects_future_information(self) -> None:
        system = DataCenterSystem.from_config(config(), store=NullRunStore())
        with self.assertRaises(TimeAlignmentError):
            system.step(signal(0), ForecastWindow((signal(0), signal(1))))

    def test_environment_can_be_driven_without_controller(self) -> None:
        env = DataCenterEnvironment.from_config(config())
        env.reset(seed=3)
        result = env.step(DataCenterAction(140.0), signal(0))
        self.assertEqual(result.controller.controller_name, "external")
        self.assertAlmostEqual(result.physical.proposed_cooling_kw, 140.0)

    def test_reset_with_same_seed_is_reproducible(self) -> None:
        env = DataCenterEnvironment.from_config(config(noisy=True))
        first_run = []
        for _ in range(2):
            env.reset(seed=17)
            first_run.append(
                tuple(
                    env.step(DataCenterAction(140.0), signal(i)).physical.measured_temperature_c
                    for i in range(2)
                )
            )
        self.assertEqual(first_run[0], first_run[1])

    def test_multiple_instances_have_isolated_state(self) -> None:
        a = DataCenterSystem.from_config(config(noisy=True), store=NullRunStore())
        b = DataCenterSystem.from_config(config(noisy=True), store=NullRunStore())
        a.reset(seed=1)
        b.reset(seed=2)
        b_before = b.environment.snapshot()
        a.step(signal(0, 0.8))
        self.assertEqual(b.environment.snapshot(), b_before)
        a.reset(seed=9)
        self.assertEqual(b.environment.snapshot(), b_before)

    def test_null_store_runs_end_to_end(self) -> None:
        system = DataCenterSystem.from_config(config(), store=NullRunStore())
        system.start_run(RunMetadata(controller_name="baseline", seed=4))
        for index in range(3):
            system.step(signal(index))
        summary = system.finish_run()
        self.assertEqual(summary.status, "completed")
        self.assertEqual(len(summary.steps), 3)
        self.assertIn("energy_cost", summary.metrics)

    def test_run_state_errors_are_explicit(self) -> None:
        system = DataCenterSystem.from_config(config(), store=NullRunStore())
        with self.assertRaises(RunStateError):
            system.finish_run()
        system.start_run(RunMetadata(controller_name="baseline"))
        with self.assertRaises(RunStateError):
            system.start_run(RunMetadata(controller_name="baseline"))
