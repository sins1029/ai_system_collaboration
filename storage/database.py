from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    _ensure_columns(
        conn,
        "datasets",
        {
            "signal_source": "TEXT",
            "metadata_json": "TEXT",
            "random_seed": "INTEGER",
            "timezone": "TEXT",
            "start_timestamp": "TEXT",
            "end_timestamp": "TEXT",
            "number_of_steps": "INTEGER",
        },
    )
    _ensure_columns(
        conn,
        "input_timeseries",
        {
            "online_workload": "REAL",
            "batch_workload": "REAL",
            "electricity_price": "REAL",
            "carbon_intensity_kg_per_kwh": "REAL",
            "outdoor_temperature_c": "REAL",
            "renewable_power_kw": "REAL",
        },
    )
    _ensure_columns(
        conn,
        "experiment_runs",
        {
            "scenario_config_json": "TEXT",
            "controller_name": "TEXT",
            "condition_name": "TEXT",
            "actuator_mode": "TEXT",
            "mismatch_scenario": "TEXT",
            "measurement_mode": "TEXT",
            "plant_config_json": "TEXT",
            "prediction_config_json": "TEXT",
        },
    )
    _ensure_columns(
        conn,
        "simulation_results",
        {
            "prev_temp_c": "REAL",
            "renewable_kw": "REAL",
            "online_workload": "REAL",
            "batch_workload": "REAL",
            "electricity_price": "REAL",
            "carbon_intensity_kg_per_kwh": "REAL",
            "outdoor_temperature_c": "REAL",
            "renewable_available_kw": "REAL",
            "renewable_used_kw": "REAL",
            "renewable_curtailed_kw": "REAL",
            "heat_kw": "REAL",
            "p_aux_kw": "REAL",
            "cooling_command_kw": "REAL",
            "proposed_cooling_kw": "REAL",
            "constrained_cooling_kw": "REAL",
            "applied_cooling_kw": "REAL",
            "control_movement_kw": "REAL",
            "proposed_control_movement_kw": "REAL",
            "applied_control_movement_kw": "REAL",
            "actuator_tracking_error_kw": "REAL",
            "actuator_ramp_limited": "INTEGER",
            "actuator_ramp_up_limited": "INTEGER",
            "actuator_ramp_down_limited": "INTEGER",
            "actuator_alpha": "REAL",
            "actuator_induced_thermal_infeasibility": "INTEGER",
            "cooling_min_feasible_kw": "REAL",
            "cooling_max_feasible_kw": "REAL",
            "cooling_constraint_intervention": "INTEGER",
            "cooling_constraint_intervention_kw": "REAL",
            "cooling_constraint_intervention_energy_kwh": "REAL",
            "power_balance_error": "REAL",
            "renewable_balance_error": "REAL",
            "thermal_balance_error": "REAL",
            "temp_min_c": "REAL",
            "true_temperature_c": "REAL",
            "measured_temperature_c": "REAL",
            "controller_measured_temperature_c": "REAL",
            "predicted_next_temperature_c": "REAL",
            "actual_next_temperature_c": "REAL",
            "one_step_temperature_prediction_error_c": "REAL",
            "temp_setpoint_c": "REAL",
            "temp_deadband_c": "REAL",
            "temp_control_target_c": "REAL",
            "precool_floor_c": "REAL",
            "precooling_active": "INTEGER",
            "temperature_is_finite": "INTEGER",
            "temperature_within_physical_range": "INTEGER",
            "temperature_step_change": "REAL",
            "below_min_temperature": "INTEGER",
            "above_max_temperature": "INTEGER",
            "temperature_violation": "INTEGER",
            "temperature_deviation_from_setpoint_c": "REAL",
            "thermal_infeasible": "INTEGER",
            "optimizer_attempted": "INTEGER",
            "optimizer_success": "INTEGER",
            "optimizer_failure": "INTEGER",
            "optimizer_fallback": "INTEGER",
            "prediction_infeasibility": "INTEGER",
            "actuator_infeasibility": "INTEGER",
            "fallback_reason": "TEXT",
            "optimizer_candidate_evaluations": "INTEGER",
            "objective_total": "REAL",
            "objective_energy_cost": "REAL",
            "objective_carbon": "REAL",
            "objective_temperature": "REAL",
            "objective_control_movement": "REAL",
            "invalid_value_count": "INTEGER",
            "negative_power_count": "INTEGER",
        },
    )
    conn.commit()


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for column, column_type in columns.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def _value(row: pd.Series, canonical: str, legacy: str | None = None, default: float = 0.0) -> float:
    if canonical in row.index:
        return float(row[canonical])
    if legacy is not None and legacy in row.index:
        return float(row[legacy])
    return float(default)


def _canonical_carbon(row: pd.Series) -> float:
    if "carbon_intensity_kg_per_kwh" in row.index:
        return float(row["carbon_intensity_kg_per_kwh"])
    return float(row["carbon"]) / 1000.0


def _canonical_renewable(row: pd.Series) -> float:
    if "renewable_power_kw" in row.index:
        return float(row["renewable_power_kw"])
    return float(row["renewable"]) * 100.0


def upsert_dataset(
    conn: sqlite3.Connection,
    name: str,
    source_path: str | Path,
    time_step_minutes: int,
    data: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
) -> int:
    metadata = metadata or {}
    conn.execute(
        """
        INSERT INTO datasets(
            name, source_path, time_step_minutes, signal_source, metadata_json,
            random_seed, timezone, start_timestamp, end_timestamp, number_of_steps
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            source_path = excluded.source_path,
            time_step_minutes = excluded.time_step_minutes,
            signal_source = excluded.signal_source,
            metadata_json = excluded.metadata_json,
            random_seed = excluded.random_seed,
            timezone = excluded.timezone,
            start_timestamp = excluded.start_timestamp,
            end_timestamp = excluded.end_timestamp,
            number_of_steps = excluded.number_of_steps
        """,
        (
            name,
            str(source_path),
            time_step_minutes,
            metadata.get("signal_source"),
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            metadata.get("random_seed"),
            metadata.get("timezone"),
            metadata.get("start_timestamp"),
            metadata.get("end_timestamp"),
            metadata.get("number_of_steps", len(data)),
        ),
    )
    dataset_id = int(conn.execute("SELECT id FROM datasets WHERE name = ?", (name,)).fetchone()[0])
    conn.execute("DELETE FROM input_timeseries WHERE dataset_id = ?", (dataset_id,))
    rows = [
        (
            dataset_id,
            row["timestamp"].isoformat(sep=" ") if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
            _value(row, "online_workload", "online_load"),
            _value(row, "batch_workload", "batch_load"),
            _value(row, "electricity_price", "price"),
            _value(row, "carbon_intensity_kg_per_kwh", "carbon") * (
                1000.0 if "carbon_intensity_kg_per_kwh" in row.index else 1.0
            ),
            _value(row, "outdoor_temperature_c", "outdoor_temp"),
            _value(row, "renewable_power_kw", "renewable") / (
                100.0 if "renewable_power_kw" in row.index else 1.0
            ),
            _value(row, "online_workload", "online_load"),
            _value(row, "batch_workload", "batch_load"),
            _value(row, "electricity_price", "price"),
            _canonical_carbon(row),
            _value(row, "outdoor_temperature_c", "outdoor_temp"),
            _canonical_renewable(row),
        )
        for _, row in data.iterrows()
    ]
    conn.executemany(
        """
        INSERT INTO input_timeseries(
            dataset_id, timestamp, online_load, batch_load, price, carbon, outdoor_temp, renewable,
            online_workload, batch_workload, electricity_price, carbon_intensity_kg_per_kwh,
            outdoor_temperature_c, renewable_power_kw
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return dataset_id


def create_run(
    conn: sqlite3.Connection,
    name: str,
    scenario: str,
    dataset_id: int,
    seed: int,
    config: dict[str, Any],
    git_commit: str | None,
    scenario_config: dict[str, Any] | None = None,
    controller_name: str | None = None,
    condition_name: str | None = None,
    actuator_mode: str | None = None,
    mismatch_scenario: str | None = None,
    measurement_mode: str | None = None,
    plant_config: dict[str, Any] | None = None,
    prediction_config: dict[str, Any] | None = None,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO experiment_runs(
            name, scenario, controller_name, condition_name, actuator_mode,
            mismatch_scenario, measurement_mode, dataset_id, seed, git_commit,
            config_json, scenario_config_json, plant_config_json, prediction_config_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            scenario,
            controller_name or scenario,
            condition_name,
            actuator_mode,
            mismatch_scenario,
            measurement_mode,
            dataset_id,
            seed,
            git_commit,
            json.dumps(config, ensure_ascii=False, sort_keys=True),
            json.dumps(scenario_config or {}, ensure_ascii=False, sort_keys=True),
            json.dumps(plant_config or {}, ensure_ascii=False, sort_keys=True),
            json.dumps(prediction_config or {}, ensure_ascii=False, sort_keys=True),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def insert_results(conn: sqlite3.Connection, run_id: int, result: pd.DataFrame) -> None:
    columns = [
        "run_id", "step_index", "timestamp", "online_load", "batch_load", "batch_service",
        "backlog", "total_load", "prev_temp_c", "price", "carbon", "outdoor_temp",
        "renewable", "renewable_kw", "online_workload", "batch_workload",
        "electricity_price", "carbon_intensity_kg_per_kwh", "outdoor_temperature_c",
        "renewable_available_kw", "renewable_used_kw", "renewable_curtailed_kw",
        "it_power_kw", "heat_kw", "proposed_cooling_kw", "constrained_cooling_kw",
        "applied_cooling_kw", "cooling_command_kw", "cooling_kw", "control_movement_kw",
        "proposed_control_movement_kw", "applied_control_movement_kw",
        "actuator_tracking_error_kw", "actuator_ramp_limited", "actuator_ramp_up_limited",
        "actuator_ramp_down_limited", "actuator_alpha",
        "actuator_induced_thermal_infeasibility",
        "cooling_min_feasible_kw", "cooling_max_feasible_kw", "cooling_constraint_intervention",
        "cooling_constraint_intervention_kw", "cooling_constraint_intervention_energy_kwh",
        "cooling_power_kw", "p_aux_kw", "total_power_kw", "grid_power_kw", "temp_c",
        "true_temperature_c", "measured_temperature_c", "controller_measured_temperature_c",
        "predicted_next_temperature_c", "actual_next_temperature_c",
        "one_step_temperature_prediction_error_c",
        "temp_min_c", "temp_setpoint_c", "temp_max_c", "temp_deadband_c",
        "temp_control_target_c", "precool_floor_c", "precooling_active", "power_balance_error",
        "renewable_balance_error", "thermal_balance_error", "temperature_is_finite",
        "temperature_within_physical_range", "temperature_step_change", "below_min_temperature",
        "above_max_temperature", "temperature_violation", "temperature_deviation_from_setpoint_c",
        "thermal_infeasible", "optimizer_attempted", "optimizer_success", "optimizer_failure",
        "optimizer_fallback", "prediction_infeasibility", "actuator_infeasibility",
        "fallback_reason", "optimizer_candidate_evaluations", "objective_total",
        "objective_energy_cost", "objective_carbon", "objective_temperature",
        "objective_control_movement", "invalid_value_count", "negative_power_count",
    ]
    rows = []
    for step_index, row in result.reset_index(drop=True).iterrows():
        online = _value(row, "online_workload", "online_load")
        batch = _value(row, "batch_workload", "batch_load")
        price = _value(row, "electricity_price", "price")
        carbon = _canonical_carbon(row)
        outdoor = _value(row, "outdoor_temperature_c", "outdoor_temp")
        renewable_available = _value(row, "renewable_available_kw", "renewable_kw")
        renewable_used = _value(row, "renewable_used_kw", default=0.0)
        renewable_curtailed = _value(
            row, "renewable_curtailed_kw", default=max(0.0, renewable_available - renewable_used)
        )
        timestamp = row["timestamp"].isoformat(sep=" ") if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"])
        record = {
            "run_id": run_id, "step_index": int(step_index), "timestamp": timestamp,
            "online_load": online, "batch_load": batch,
            "batch_service": _value(row, "batch_service"), "backlog": _value(row, "backlog"),
            "total_load": _value(row, "total_load"), "prev_temp_c": _value(row, "prev_temp_c"),
            "price": price, "carbon": carbon * 1000.0, "outdoor_temp": outdoor,
            "renewable": renewable_available / 100.0, "renewable_kw": renewable_available,
            "online_workload": online, "batch_workload": batch, "electricity_price": price,
            "carbon_intensity_kg_per_kwh": carbon, "outdoor_temperature_c": outdoor,
            "renewable_available_kw": renewable_available, "renewable_used_kw": renewable_used,
            "renewable_curtailed_kw": renewable_curtailed,
        }
        for column in columns[22:]:
            if column == "fallback_reason":
                record[column] = str(row.get(column, ""))
            else:
                record[column] = _value(row, column)
        rows.append(tuple(record[column] for column in columns))
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT INTO simulation_results({', '.join(columns)}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()


def insert_metrics(conn: sqlite3.Connection, run_id: int, metrics: dict[str, float]) -> None:
    conn.executemany(
        """
        INSERT INTO run_metrics(run_id, metric, value)
        VALUES (?, ?, ?)
        ON CONFLICT(run_id, metric) DO UPDATE SET value = excluded.value
        """,
        [(run_id, key, float(value)) for key, value in metrics.items()],
    )
    conn.commit()


def current_git_commit(root: Path) -> str | None:
    git = r"C:\Users\26550\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe"
    try:
        result = subprocess.run(
            [git, "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()
