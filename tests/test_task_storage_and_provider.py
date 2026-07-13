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
    TaskArrivalBatch,
    TaskSpec,
)
from external_workloads import generate_standard_task_dataset, load_task_csv
from models.config_loader import load_simple_yaml


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 1, 1)


def config(path: Path) -> DataCenterSystemConfig:
    return DataCenterSystemConfig.from_dict(
        load_simple_yaml(ROOT / "configs" / "datacenter.yaml"),
        load_simple_yaml(ROOT / "configs" / "optimization.yaml"),
        database_path=str(path),
        workload={"mode": "task_queue"},
        capacity={
            "total_cpu_cores": 100,
            "total_gpu_units": 10,
            "total_memory_gb": 200,
        },
        scheduler={"name": "fifo_immediate"},
    )


class TaskStorageAndProviderTest(unittest.TestCase):
    def test_standard_provider_is_reproducible_and_hides_future_arrivals(self) -> None:
        raw = load_simple_yaml(ROOT / "configs" / "tasks.yaml")
        first = generate_standard_task_dataset(raw)
        second = generate_standard_task_dataset(raw)
        self.assertEqual(first.tasks, second.tasks)
        self.assertEqual(first.metadata.number_of_steps, 96)
        now = first.metadata.start_timestamp
        batch = first.arrivals_at(now)
        self.assertTrue(all(item.arrival_time == now for item in batch.tasks))
        self.assertLess(len(batch.tasks), len(first.tasks))

    def test_csv_provider_round_trip(self) -> None:
        raw = load_simple_yaml(ROOT / "configs" / "tasks.yaml")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.csv"
            original = generate_standard_task_dataset(raw, path)
            loaded = load_task_csv(
                path,
                dataset_name=original.metadata.name,
                seed=original.metadata.seed,
                step_minutes=original.metadata.step_minutes,
                number_of_steps=original.metadata.number_of_steps,
            )
            self.assertEqual(loaded.tasks, original.tasks)

    def test_sqlite_round_trip_persists_task_spec_events_outcome_and_step_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite"
            store = SQLiteRunStore(path, commit_each_step=True)
            system = DataCenterSystem.from_config(config(path), store=store)
            handle = system.start_run(
                RunMetadata(dataset_name="task-roundtrip", task_dataset_id="task-roundtrip")
            )
            task = TaskSpec(
                "db-task", NOW, 1, 50, 0, 20, NOW + timedelta(minutes=15)
            )
            current = ExogenousInput(NOW, 0, 0.5, 0.4, 25, 10)
            system.step(current, task_arrivals=TaskArrivalBatch(NOW, (task,)))
            system.finish_run()
            store.close()
            connection = sqlite3.connect(path)
            row = connection.execute(
                """
                SELECT r.scheduler_name, r.cooling_controller_name,
                       s.workload_mode, s.cpu_utilization
                FROM experiment_runs r JOIN simulation_results s ON s.run_id = r.id
                WHERE r.id = ?
                """,
                (handle.run_id,),
            ).fetchone()
            task_count = connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            events = connection.execute(
                "SELECT event_type FROM run_task_events ORDER BY rowid"
            ).fetchall()
            outcome = connection.execute(
                "SELECT final_status, completion_time FROM run_task_outcomes"
            ).fetchone()
            connection.close()
            self.assertEqual(row[:3], ("fifo_immediate", "baseline", "task_queue"))
            self.assertAlmostEqual(row[3], 0.5)
            self.assertEqual(task_count, 1)
            self.assertEqual([item[0] for item in events], ["arrived", "started", "completed"])
            self.assertEqual(outcome[0], "completed")
            self.assertIsNotNone(outcome[1])
