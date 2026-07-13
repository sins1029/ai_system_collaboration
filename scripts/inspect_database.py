from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.mvp import load_yaml  # noqa: E402


def scalar(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0])


def run_health(conn: sqlite3.Connection, run_id: int, dataset_id: int) -> dict[str, int]:
    expected_steps = scalar(conn, f"SELECT COUNT(*) FROM input_timeseries WHERE dataset_id = {dataset_id}")
    actual_steps = scalar(conn, f"SELECT COUNT(*) FROM simulation_results WHERE run_id = {run_id}")
    distinct_timestamps = scalar(
        conn,
        f"SELECT COUNT(DISTINCT timestamp) FROM simulation_results WHERE run_id = {run_id}",
    )
    invalid_values = scalar(
        conn,
        f"""
        SELECT COUNT(*)
        FROM simulation_results
        WHERE run_id = {run_id}
          AND (
            total_load != total_load OR it_power_kw != it_power_kw OR heat_kw != heat_kw
            OR cooling_power_kw != cooling_power_kw OR grid_power_kw != grid_power_kw
            OR temp_c != temp_c OR ABS(total_load) > 1e308 OR ABS(it_power_kw) > 1e308
            OR ABS(cooling_power_kw) > 1e308 OR ABS(grid_power_kw) > 1e308 OR ABS(temp_c) > 1e308
          )
        """,
    )
    negative_power = scalar(
        conn,
        f"""
        SELECT COUNT(*)
        FROM simulation_results
        WHERE run_id = {run_id}
          AND (it_power_kw < -1e-9 OR heat_kw < -1e-9 OR cooling_power_kw < -1e-9 OR grid_power_kw < -1e-9)
        """,
    )
    missing_steps = max(0, expected_steps - actual_steps)
    duplicate_steps = max(0, actual_steps - distinct_timestamps)
    return {
        "steps": actual_steps,
        "metrics": scalar(conn, f"SELECT COUNT(*) FROM run_metrics WHERE run_id = {run_id}"),
        "invalid_values": invalid_values,
        "negative_power": negative_power,
        "missing_steps": missing_steps,
        "duplicate_steps": duplicate_steps,
        "temperature_violations": scalar(
            conn,
            f"SELECT COALESCE(SUM(temperature_violation), 0) FROM simulation_results WHERE run_id = {run_id}",
        ),
        "thermal_infeasible": scalar(
            conn,
            f"SELECT COALESCE(SUM(thermal_infeasible), 0) FROM simulation_results WHERE run_id = {run_id}",
        ),
        "cooling_interventions": scalar(
            conn,
            f"SELECT COALESCE(SUM(cooling_constraint_intervention), 0) FROM simulation_results WHERE run_id = {run_id}",
        ),
    }


def main() -> None:
    exp_config = load_yaml(ROOT / "configs" / "experiment.yaml")
    db_path = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else ROOT / exp_config["database_path"]
    )
    if not db_path.exists():
        raise SystemExit(f"database does not exist: {db_path}")

    conn = sqlite3.connect(db_path)
    print(f"database: {db_path}")
    print(f"datasets: {scalar(conn, 'SELECT COUNT(*) FROM datasets')}")
    print(f"input_timeseries: {scalar(conn, 'SELECT COUNT(*) FROM input_timeseries')}")
    print(f"run_input_timeseries: {scalar(conn, 'SELECT COUNT(*) FROM run_input_timeseries')}")
    print(f"experiment_runs: {scalar(conn, 'SELECT COUNT(*) FROM experiment_runs')}")
    print(f"simulation_results: {scalar(conn, 'SELECT COUNT(*) FROM simulation_results')}")
    print(f"run_metrics: {scalar(conn, 'SELECT COUNT(*) FROM run_metrics')}")
    print(f"tasks: {scalar(conn, 'SELECT COUNT(*) FROM tasks')}")
    print(f"run_task_events: {scalar(conn, 'SELECT COUNT(*) FROM run_task_events')}")
    print(f"run_task_outcomes: {scalar(conn, 'SELECT COUNT(*) FROM run_task_outcomes')}")
    print(f"run_agent_decisions: {scalar(conn, 'SELECT COUNT(*) FROM run_agent_decisions')}")
    print(
        "run_gym_episode_summaries: "
        f"{scalar(conn, 'SELECT COUNT(*) FROM run_gym_episode_summaries')}"
    )
    print(f"schema_version: {scalar(conn, 'SELECT MAX(version) FROM schema_versions')}")
    print("")
    print("datasets_metadata:")
    for row in conn.execute(
        """
        SELECT id, name, signal_source, timezone, random_seed, start_timestamp,
               end_timestamp, number_of_steps
        FROM datasets ORDER BY id DESC
        """
    ):
        print(
            f"dataset_id={row[0]} name={row[1]} source={row[2]} timezone={row[3]} "
            f"seed={row[4]} range={row[5]}..{row[6]} steps={row[7]}"
        )
    print("")
    print("latest_runs:")
    for row in conn.execute(
        """
        SELECT r.id, r.name, r.scenario, d.id, d.name, r.git_commit, r.started_at,
               r.controller_name, r.condition_name, r.actuator_mode,
               r.mismatch_scenario, r.measurement_mode, r.status,
               r.package_version, r.error_message
        FROM experiment_runs r
        JOIN datasets d ON d.id = r.dataset_id
        ORDER BY r.id DESC
        LIMIT 10
        """
    ):
        (
            run_id, name, scenario, dataset_id, dataset_name, commit, started_at,
            controller_name, condition_name, actuator_mode, mismatch_scenario,
            measurement_mode, status, package_version, error_message,
        ) = row
        health = run_health(conn, int(run_id), int(dataset_id))
        print(
            f"run_id={run_id} name={name} scenario={scenario} dataset={dataset_name} "
            f"controller={controller_name} condition={condition_name} actuator={actuator_mode} "
            f"mismatch={mismatch_scenario} measurement={measurement_mode} "
            f"status={status} package={package_version} error={error_message} "
            f"commit={commit} started_at={started_at} steps={health['steps']} "
            f"metrics={health['metrics']} invalid={health['invalid_values']} "
            f"negative_power={health['negative_power']} missing_steps={health['missing_steps']} "
            f"duplicate_steps={health['duplicate_steps']} temp_violations={health['temperature_violations']} "
            f"thermal_infeasible={health['thermal_infeasible']} cooling_interventions={health['cooling_interventions']}"
        )
        optimizer = conn.execute(
            """
            SELECT COALESCE(SUM(optimizer_success), 0), COALESCE(SUM(optimizer_failure), 0),
                   COALESCE(SUM(optimizer_fallback), 0)
            FROM simulation_results WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if any(int(value) for value in optimizer):
            print(
                f"  optimizer_success={optimizer[0]} optimizer_failure={optimizer[1]} "
                f"optimizer_fallback={optimizer[2]}"
            )
    conn.close()


if __name__ == "__main__":
    main()
