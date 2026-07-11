from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.compare_single_center_runs import latest_pair, latest_scenario_runs, load_metrics, pct_change
from storage.database import connect, create_run, initialize_database, insert_metrics, upsert_dataset

import pandas as pd


class CompareTest(unittest.TestCase):
    def test_percentage_change(self) -> None:
        self.assertAlmostEqual(pct_change(100.0, 80.0), -20.0)
        self.assertAlmostEqual(pct_change(100.0, 120.0), 20.0)

    def test_zero_baseline_percentage_is_none(self) -> None:
        self.assertIsNone(pct_change(0.0, 10.0))

    def test_latest_pair_and_missing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = connect(Path(tmpdir) / "compare.sqlite")
            initialize_database(conn)
            data = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-01-01", periods=1, freq="15min"),
                    "online_load": [0.1],
                    "batch_load": [0.1],
                    "price": [0.3],
                    "carbon": [500.0],
                    "outdoor_temp": [10.0],
                    "renewable": [0.0],
                }
            )
            dataset_id = upsert_dataset(conn, "sample", "sample.csv", 15, data)
            baseline_id = create_run(conn, "single", "baseline", dataset_id, 42, {}, "abc")
            optimized_id = create_run(conn, "single", "optimized", dataset_id, 42, {}, "abc")
            insert_metrics(conn, baseline_id, {"energy_cost": 100.0, "only_baseline": 1.0})
            insert_metrics(conn, optimized_id, {"energy_cost": 80.0})

            self.assertEqual(latest_pair(conn), (baseline_id, optimized_id))
            self.assertEqual(load_metrics(conn, baseline_id)["energy_cost"], 100.0)
            self.assertEqual(load_metrics(conn, optimized_id)["energy_cost"], 80.0)
            self.assertNotIn("only_baseline", load_metrics(conn, optimized_id))
            conn.close()

    def test_latest_three_scenario_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = connect(Path(tmpdir) / "three.sqlite")
            initialize_database(conn)
            data = pd.DataFrame({
                "timestamp": pd.date_range("2026-01-01", periods=1, freq="15min"),
                "online_load": [0.1], "batch_load": [0.1], "price": [0.3],
                "carbon": [500.0], "outdoor_temp": [10.0], "renewable": [0.0],
            })
            dataset_id = upsert_dataset(conn, "sample", "sample.csv", 15, data)
            expected = {}
            for scenario in ["baseline", "heuristic", "finite_horizon"]:
                expected[scenario] = create_run(
                    conn, "coordination", scenario, dataset_id, 42, {}, "abc"
                )
            self.assertEqual(latest_scenario_runs(conn), expected)
            conn.close()


if __name__ == "__main__":
    unittest.main()
