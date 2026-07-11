from __future__ import annotations

from importlib.resources import files
import sqlite3
from pathlib import Path


SCHEMA_VERSION = 2


def schema_text() -> str:
    return files("datacenter_env.storage").joinpath("schema.sql").read_text(encoding="utf-8")


def connect(path: str | Path) -> sqlite3.Connection:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(schema_text())
    _migrate_legacy_database(connection)
    connection.execute(
        "INSERT OR IGNORE INTO schema_versions(version) VALUES (?)", (SCHEMA_VERSION,)
    )
    connection.commit()


def _migrate_legacy_database(connection: sqlite3.Connection) -> None:
    additions = {
        "experiment_runs": {
            "controller_name": "TEXT",
            "condition_name": "TEXT",
            "actuator_mode": "TEXT",
            "mismatch_scenario": "TEXT",
            "measurement_mode": "TEXT",
            "scenario_config_json": "TEXT",
            "plant_config_json": "TEXT",
            "prediction_config_json": "TEXT",
            "package_version": "TEXT",
            "status": "TEXT NOT NULL DEFAULT 'completed'",
            "error_message": "TEXT",
            "completed_at": "TEXT",
        },
        "simulation_results": {
            "proposed_cooling_kw": "REAL",
            "constrained_cooling_kw": "REAL",
            "applied_cooling_kw": "REAL",
            "controller_measured_temperature_c": "REAL",
            "measured_temperature_c": "REAL",
            "predicted_next_temperature_c": "REAL",
            "actual_next_temperature_c": "REAL",
            "one_step_temperature_prediction_error_c": "REAL",
            "fallback_reason": "TEXT",
        },
    }
    for table, columns in additions.items():
        existing = {
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
        }
        for name, definition in columns.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
