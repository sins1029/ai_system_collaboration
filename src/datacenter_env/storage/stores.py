from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from datacenter_env.config import DataCenterSystemConfig
from datacenter_env.contracts import (
    ExogenousInput,
    RunHandle,
    RunMetadata,
    RunSummary,
    StepResult,
)
from datacenter_env.evaluation.records import step_to_record
from datacenter_env.exceptions import RunStateError, StorageError
from datacenter_env.storage.database import connect, initialize_database
from datacenter_env.version import __version__


class NullRunStore:
    def __init__(self) -> None:
        self._next_id = 1
        self._states: dict[int, str] = {}

    def bind_config(self, config: DataCenterSystemConfig) -> None:
        del config

    def initialize(self) -> None:
        return None

    def create_run(self, metadata: RunMetadata) -> RunHandle:
        del metadata
        run_id = self._next_id
        self._next_id += 1
        self._states[run_id] = "running"
        return RunHandle(run_id)

    def append_input(self, run_id: int | None, external_input: ExogenousInput) -> None:
        del external_input
        self._require_running(run_id)

    def append_step(self, run_id: int | None, step_result: StepResult) -> None:
        del step_result
        self._require_running(run_id)

    def finish_run(self, run_id: int | None, summary: RunSummary) -> None:
        del summary
        self._require_running(run_id)
        self._states[int(run_id)] = "completed"

    def fail_run(self, run_id: int | None, error: Exception) -> None:
        del error
        self._require_running(run_id)
        self._states[int(run_id)] = "failed"

    def _require_running(self, run_id: int | None) -> None:
        if run_id is None or self._states.get(run_id) != "running":
            raise RunStateError("run is not active")


class SQLiteRunStore:
    def __init__(self, path: str | Path, commit_each_step: bool = False):
        self.path = Path(path)
        self.commit_each_step = bool(commit_each_step)
        self._connection: sqlite3.Connection | None = None
        self._config: DataCenterSystemConfig | None = None
        self._step_indices: dict[int, int] = {}

    def bind_config(self, config: DataCenterSystemConfig) -> None:
        self._config = config

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise StorageError("store has not been initialized")
        return self._connection

    def initialize(self) -> None:
        try:
            if self._connection is None:
                self._connection = connect(self.path)
            initialize_database(self._connection)
        except Exception as error:
            raise StorageError(f"failed to initialize SQLite store: {error}") from error

    def create_run(self, metadata: RunMetadata) -> RunHandle:
        connection = self.connection
        try:
            dataset_id = self._ensure_dataset(metadata)
            config_snapshot = dict(metadata.config_snapshot or {})
            config = self._config
            cursor = connection.execute(
                """
                INSERT INTO experiment_runs(
                    name, scenario, controller_name, condition_name, actuator_mode,
                    mismatch_scenario, measurement_mode, dataset_id, seed, config_json,
                    scenario_config_json, plant_config_json, prediction_config_json,
                    package_version, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')
                """,
                (
                    metadata.name,
                    metadata.controller_name,
                    metadata.controller_name,
                    metadata.condition_name,
                    str(config.actuator["mode"]) if config else None,
                    config.mismatch_scenario if config else None,
                    str(config.measurement["mode"]) if config else None,
                    dataset_id,
                    metadata.seed,
                    json.dumps(config_snapshot, ensure_ascii=False, sort_keys=True),
                    json.dumps({"controller_name": metadata.controller_name}),
                    json.dumps(config.to_dict()["plant_parameters"] if config else {}),
                    json.dumps(config.to_dict()["prediction_parameters"] if config else {}),
                    metadata.package_version or __version__,
                ),
            )
            run_id = int(cursor.lastrowid)
            connection.commit()
            self._step_indices[run_id] = 0
            return RunHandle(run_id)
        except Exception as error:
            connection.rollback()
            raise StorageError(f"failed to create run: {error}") from error

    def append_input(self, run_id: int | None, external_input: ExogenousInput) -> None:
        run = self._require_running(run_id)
        step_index = self._step_indices[run]
        try:
            self.connection.execute(
                """
                INSERT INTO run_input_timeseries(
                    run_id, step_index, timestamp, workload_fraction,
                    electricity_price_per_kwh, carbon_intensity_kg_per_kwh,
                    outdoor_temperature_c, renewable_power_kw
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run,
                    step_index,
                    external_input.timestamp.isoformat(sep=" "),
                    external_input.workload_fraction,
                    external_input.electricity_price_per_kwh,
                    external_input.carbon_intensity_kg_per_kwh,
                    external_input.outdoor_temperature_c,
                    external_input.renewable_power_kw,
                ),
            )
        except Exception as error:
            raise StorageError(f"failed to append run input: {error}") from error

    def append_step(self, run_id: int | None, step_result: StepResult) -> None:
        run = self._require_running(run_id)
        if self._config is None:
            raise StorageError("SQLiteRunStore is not bound to a system config")
        step_index = self._step_indices[run]
        record = step_to_record(step_result, self._config)
        columns = self._simulation_columns()
        values = [self._sqlite_value(record.get(column)) for column in columns]
        try:
            self.connection.execute(
                f"INSERT INTO simulation_results(run_id, step_index, {', '.join(columns)}) "
                f"VALUES (?, ?, {', '.join('?' for _ in columns)})",
                (run, step_index, *values),
            )
            self._step_indices[run] += 1
            if self.commit_each_step:
                self.connection.commit()
        except Exception as error:
            raise StorageError(f"failed to append simulation step: {error}") from error

    def finish_run(self, run_id: int | None, summary: RunSummary) -> None:
        run = self._require_running(run_id)
        try:
            self.connection.executemany(
                """
                INSERT INTO run_metrics(run_id, metric, value) VALUES (?, ?, ?)
                ON CONFLICT(run_id, metric) DO UPDATE SET value = excluded.value
                """,
                [(run, key, float(value)) for key, value in summary.metrics.items()],
            )
            self.connection.execute(
                """
                UPDATE experiment_runs
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP, error_message = NULL
                WHERE id = ? AND status = 'running'
                """,
                (run,),
            )
            self.connection.commit()
        except Exception as error:
            self.connection.rollback()
            raise StorageError(f"failed to finish run: {error}") from error
        finally:
            self._step_indices.pop(run, None)

    def fail_run(self, run_id: int | None, error: Exception) -> None:
        run = self._require_running(run_id)
        try:
            self.connection.rollback()
            self.connection.execute(
                """
                UPDATE experiment_runs
                SET status = 'failed', completed_at = CURRENT_TIMESTAMP, error_message = ?
                WHERE id = ? AND status = 'running'
                """,
                (str(error), run),
            )
            self.connection.commit()
        except Exception as storage_error:
            raise StorageError(f"failed to mark run as failed: {storage_error}") from storage_error
        finally:
            self._step_indices.pop(run, None)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _ensure_dataset(self, metadata: RunMetadata) -> int:
        connection = self.connection
        connection.execute(
            """
            INSERT INTO datasets(name, source_path, time_step_minutes, signal_source, metadata_json)
            VALUES (?, ?, ?, 'external-provider', '{}')
            ON CONFLICT(name) DO UPDATE SET source_path = excluded.source_path
            """,
            (
                metadata.dataset_name,
                metadata.source_path,
                self._config.step_minutes if self._config else 0,
            ),
        )
        row = connection.execute(
            "SELECT id FROM datasets WHERE name = ?", (metadata.dataset_name,)
        ).fetchone()
        return int(row[0])

    def _require_running(self, run_id: int | None) -> int:
        if run_id is None or run_id not in self._step_indices:
            raise RunStateError("run is not active")
        row = self.connection.execute(
            "SELECT status FROM experiment_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None or row[0] != "running":
            raise RunStateError("run is not in running state")
        return int(run_id)

    @staticmethod
    def _sqlite_value(value: Any) -> Any:
        if isinstance(value, bool):
            return int(value)
        return value

    @staticmethod
    def _simulation_columns() -> list[str]:
        return [
            "timestamp", "online_load", "batch_load", "batch_service", "backlog",
            "total_load", "prev_temp_c", "price", "carbon", "outdoor_temp", "renewable",
            "renewable_kw", "online_workload", "batch_workload", "electricity_price",
            "carbon_intensity_kg_per_kwh", "outdoor_temperature_c", "renewable_available_kw",
            "renewable_used_kw", "renewable_curtailed_kw", "it_power_kw", "heat_kw",
            "proposed_cooling_kw", "constrained_cooling_kw", "applied_cooling_kw",
            "cooling_command_kw", "cooling_kw", "control_movement_kw",
            "proposed_control_movement_kw", "applied_control_movement_kw",
            "actuator_tracking_error_kw", "actuator_ramp_limited", "actuator_ramp_up_limited",
            "actuator_ramp_down_limited", "actuator_alpha",
            "actuator_induced_thermal_infeasibility", "cooling_min_feasible_kw",
            "cooling_max_feasible_kw", "cooling_constraint_intervention",
            "cooling_constraint_intervention_kw", "cooling_constraint_intervention_energy_kwh",
            "cooling_power_kw", "p_aux_kw", "total_power_kw", "grid_power_kw", "temp_c",
            "true_temperature_c", "measured_temperature_c",
            "controller_measured_temperature_c", "predicted_next_temperature_c",
            "actual_next_temperature_c", "one_step_temperature_prediction_error_c",
            "temp_min_c", "temp_setpoint_c", "temp_max_c", "temp_deadband_c",
            "temp_control_target_c", "precool_floor_c", "precooling_active",
            "power_balance_error", "renewable_balance_error", "thermal_balance_error",
            "temperature_is_finite", "temperature_within_physical_range",
            "temperature_step_change", "below_min_temperature", "above_max_temperature",
            "temperature_violation", "temperature_deviation_from_setpoint_c",
            "thermal_infeasible", "optimizer_attempted", "optimizer_success",
            "optimizer_failure", "optimizer_fallback", "prediction_infeasibility",
            "actuator_infeasibility", "fallback_reason", "optimizer_candidate_evaluations",
            "objective_total", "objective_energy_cost", "objective_carbon",
            "objective_temperature", "objective_control_movement", "invalid_value_count",
            "negative_power_count",
        ]
