from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from storage.database import (
    connect,
    current_git_commit,
    create_run,
    initialize_database,
    insert_metrics,
    insert_results,
    upsert_dataset,
)


class DatabaseTest(unittest.TestCase):
    def test_current_git_commit_handles_missing_git(self) -> None:
        with patch("storage.database.shutil.which", return_value=None):
            self.assertIsNone(current_git_commit(Path.cwd()))

    def test_database_round_trip_for_two_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = connect(Path(tmpdir) / "test.sqlite")
            initialize_database(conn)
            data = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-01-01", periods=2, freq="15min"),
                    "online_load": [0.1, 0.2],
                    "batch_load": [0.2, 0.1],
                    "price": [0.3, 0.4],
                    "carbon": [500.0, 510.0],
                    "outdoor_temp": [10.0, 11.0],
                    "renewable": [0.0, 0.0],
                }
            )
            metadata = {
                "signal_source": "test-synthetic", "random_seed": 42,
                "timezone": "Asia/Shanghai", "start_timestamp": "2026-01-01T00:00:00+08:00",
                "end_timestamp": "2026-01-01T00:15:00+08:00", "number_of_steps": 2,
            }
            dataset_id = upsert_dataset(conn, "sample", "sample.csv", 15, data, metadata)
            self.assertEqual(count(conn, "input_timeseries"), 2)
            stored = conn.execute(
                "SELECT signal_source, random_seed, number_of_steps FROM datasets WHERE id = ?",
                (dataset_id,),
            ).fetchone()
            self.assertEqual(stored, ("test-synthetic", 42, 2))

            result = fake_result(data)
            for scenario in ["baseline", "optimized"]:
                run_id = create_run(conn, "single", scenario, dataset_id, 42, {"a": 1}, "abc123", {"name": scenario})
                insert_results(conn, run_id, result)
                insert_metrics(conn, run_id, {"energy_cost": 1.0, "total_grid_energy_kwh": 2.0})

            self.assertEqual(count(conn, "experiment_runs"), 2)
            self.assertEqual(count(conn, "simulation_results"), 4)
            self.assertEqual(count(conn, "run_metrics"), 4)
            scenarios = [row[0] for row in conn.execute("SELECT scenario FROM experiment_runs ORDER BY id")]
            self.assertEqual(scenarios, ["baseline", "optimized"])
            conn.close()


def count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def fake_result(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["batch_service"] = result["batch_load"]
    result["backlog"] = 0.0
    result["total_load"] = result["online_load"] + result["batch_service"]
    result["prev_temp_c"] = 24.0
    result["renewable_kw"] = 0.0
    result["it_power_kw"] = 100.0
    result["heat_kw"] = 100.0
    result["cooling_command_kw"] = 80.0
    result["cooling_kw"] = 80.0
    result["cooling_min_feasible_kw"] = 0.0
    result["cooling_max_feasible_kw"] = 280.0
    result["cooling_constraint_intervention"] = 0
    result["cooling_constraint_intervention_kw"] = 0.0
    result["cooling_constraint_intervention_energy_kwh"] = 0.0
    result["cooling_power_kw"] = 20.0
    result["p_aux_kw"] = 12.0
    result["total_power_kw"] = 132.0
    result["grid_power_kw"] = 132.0
    result["temp_c"] = 24.1
    result["temp_min_c"] = 18.0
    result["temp_setpoint_c"] = 24.0
    result["temp_max_c"] = 32.0
    result["temp_deadband_c"] = 0.5
    result["temp_control_target_c"] = 24.0
    result["precool_floor_c"] = 18.0
    result["precooling_active"] = 0
    result["power_balance_error"] = 0.0
    result["thermal_balance_error"] = 0.0
    result["temperature_is_finite"] = 1
    result["temperature_within_physical_range"] = 1
    result["temperature_step_change"] = 0.1
    result["below_min_temperature"] = 0
    result["above_max_temperature"] = 0
    result["temperature_violation"] = 0
    result["temperature_deviation_from_setpoint_c"] = 0.1
    result["thermal_infeasible"] = 0
    result["invalid_value_count"] = 0
    result["negative_power_count"] = 0
    return result


if __name__ == "__main__":
    unittest.main()
