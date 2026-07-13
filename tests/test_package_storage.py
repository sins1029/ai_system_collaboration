from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import tempfile
import unittest

from datacenter_env import (
    DataCenterSystem,
    DataCenterSystemConfig,
    ExogenousInput,
    RunMetadata,
    SQLiteRunStore,
)
from datacenter_env.exceptions import TimeAlignmentError
from datacenter_env.storage.database import initialize_database
from models.config_loader import load_simple_yaml


ROOT = Path(__file__).resolve().parents[1]


def config() -> DataCenterSystemConfig:
    return DataCenterSystemConfig.from_dict(
        load_simple_yaml(ROOT / "configs" / "datacenter.yaml"),
        load_simple_yaml(ROOT / "configs" / "optimization.yaml"),
    )


def signal(index: int) -> ExogenousInput:
    return ExogenousInput(
        datetime(2026, 1, 1) + timedelta(minutes=15 * index),
        0.5,
        0.5,
        0.4,
        25.0,
        100.0,
    )


class PackageStorageTest(unittest.TestCase):
    def test_sqlite_round_trip_preserves_action_chain_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.sqlite"
            store = SQLiteRunStore(path)
            system = DataCenterSystem.from_config(config(), store=store)
            handle = system.start_run(
                RunMetadata(controller_name="baseline", dataset_name="roundtrip")
            )
            system.step(signal(0))
            summary = system.finish_run()
            store.close()
            connection = sqlite3.connect(path)
            run = connection.execute(
                "SELECT status, package_version FROM experiment_runs WHERE id = ?",
                (handle.run_id,),
            ).fetchone()
            action = connection.execute(
                """
                SELECT proposed_cooling_kw, constrained_cooling_kw, applied_cooling_kw
                FROM simulation_results WHERE run_id = ?
                """,
                (handle.run_id,),
            ).fetchone()
            inputs = connection.execute(
                "SELECT COUNT(*) FROM run_input_timeseries WHERE run_id = ?",
                (handle.run_id,),
            ).fetchone()[0]
            metrics = connection.execute(
                "SELECT COUNT(*) FROM run_metrics WHERE run_id = ?", (handle.run_id,)
            ).fetchone()[0]
            connection.close()
            self.assertEqual(run[0], "completed")
            self.assertTrue(run[1])
            self.assertEqual(inputs, 1)
            self.assertGreater(metrics, 20)
            self.assertEqual(action[0], action[1])
            self.assertEqual(action[1], action[2])
            self.assertAlmostEqual(summary.metrics["temperature_violation_count"], 0.0)

    def test_failed_run_is_not_marked_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.sqlite"
            store = SQLiteRunStore(path)
            system = DataCenterSystem.from_config(config(), store=store)
            handle = system.start_run(RunMetadata(controller_name="baseline"))
            system.step(signal(0))
            with self.assertRaises(TimeAlignmentError):
                system.step(signal(2))
            store.close()
            connection = sqlite3.connect(path)
            status, message = connection.execute(
                "SELECT status, error_message FROM experiment_runs WHERE id = ?",
                (handle.run_id,),
            ).fetchone()
            connection.close()
            self.assertEqual(status, "failed")
            self.assertIn("expected timestamp", message)

    def test_legacy_database_receives_schema_version_migration(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE experiment_runs(id INTEGER PRIMARY KEY, name TEXT, scenario TEXT, "
            "dataset_id INTEGER, seed INTEGER, config_json TEXT, started_at TEXT)"
        )
        initialize_database(connection)
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(experiment_runs)")
        }
        version = connection.execute("SELECT MAX(version) FROM schema_versions").fetchone()[0]
        connection.close()
        self.assertIn("status", columns)
        self.assertIn("package_version", columns)
        self.assertEqual(version, 4)
