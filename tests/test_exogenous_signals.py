from __future__ import annotations

from copy import deepcopy
import unittest

import pandas as pd

from models.exogenous_signals import ExogenousSignals, SignalMetadata
from models.synthetic_signals import generate_standard_signals


CONFIG = {
    "seed": 42, "start_timestamp": "2026-01-01T00:00:00",
    "timezone": "Asia/Shanghai", "steps": 96, "time_step_minutes": 15,
    "renewable_enabled": True, "renewable_peak_kw": 320.0,
    "workload_noise_std": 0.01, "batch_noise_std": 0.01,
    "price_noise_std": 0.01, "carbon_noise_std": 0.01,
    "temperature_noise_std_c": 0.1,
}


class ExogenousSignalsTest(unittest.TestCase):
    def test_seed_is_reproducible_and_units_are_canonical(self) -> None:
        first = generate_standard_signals(CONFIG).to_frame()
        second = generate_standard_signals(CONFIG).to_frame()
        pd.testing.assert_frame_equal(first, second)
        self.assertLess(first["carbon_intensity_kg_per_kwh"].max(), 1.0)
        self.assertGreater(first["renewable_power_kw"].max(), 100.0)

    def test_renewable_can_be_disabled(self) -> None:
        config = deepcopy(CONFIG)
        config["renewable_enabled"] = False
        self.assertEqual(generate_standard_signals(config).to_frame()["renewable_power_kw"].sum(), 0.0)

    def test_duplicate_and_missing_timestamps_fail(self) -> None:
        frame = generate_standard_signals(CONFIG).to_frame()
        metadata = SignalMetadata("test", "Asia/Shanghai", 15, 1, {})
        duplicate = frame.iloc[:3].copy()
        duplicate.loc[2, "timestamp"] = duplicate.loc[1, "timestamp"]
        with self.assertRaisesRegex(ValueError, "strictly increasing|duplicates"):
            ExogenousSignals(duplicate, metadata)
        missing = frame.iloc[[0, 2]].copy()
        with self.assertRaisesRegex(ValueError, "missing or misaligned"):
            ExogenousSignals(missing, metadata)

    def test_signal_window_blocks_out_of_horizon_access(self) -> None:
        window = generate_standard_signals(CONFIG).window(0, 8)
        self.assertEqual(len(window), 8)
        with self.assertRaises(IndexError):
            window.row(8)

    def test_nonfinite_signal_fails(self) -> None:
        frame = generate_standard_signals(CONFIG).to_frame().iloc[:2].copy()
        frame.loc[1, "electricity_price"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            ExogenousSignals(frame, SignalMetadata("test", "Asia/Shanghai", 15, 1, {}))


if __name__ == "__main__":
    unittest.main()
