from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.mvp import load_yaml  # noqa: E402


def main() -> None:
    config = load_yaml(ROOT / "configs" / "tasks.yaml")
    connection = sqlite3.connect(ROOT / str(config["database_path"]))
    runs = connection.execute(
        """
        SELECT scheduler_name, MAX(id)
        FROM experiment_runs
        WHERE name = 'single_center_task_scheduling_v0.2' AND status = 'completed'
        GROUP BY scheduler_name ORDER BY scheduler_name
        """
    ).fetchall()
    for scheduler, run_id in runs:
        print(f"[{scheduler}] run_id={run_id}")
        event_counts = connection.execute(
            """
            SELECT event_type, COUNT(*) FROM run_task_events
            WHERE run_id = ? GROUP BY event_type ORDER BY event_type
            """,
            (run_id,),
        ).fetchall()
        print("  events: " + ", ".join(f"{name}={count}" for name, count in event_counts))
        critical = connection.execute(
            """
            SELECT task_id, final_status, wait_steps, deferral_count, lateness_minutes
            FROM run_task_outcomes WHERE run_id = ?
            ORDER BY sla_violated DESC, COALESCE(lateness_minutes, 0) DESC,
                     wait_steps DESC, task_id LIMIT 8
            """,
            (run_id,),
        ).fetchall()
        for task_id, status, wait, deferrals, lateness in critical:
            print(
                f"  task={task_id} status={status} wait={wait} "
                f"deferrals={deferrals} lateness_min={lateness}"
            )
        windows = connection.execute(
            """
            SELECT timestamp, waiting_task_count, running_task_count,
                   at_risk_task_count, cpu_utilization, gpu_utilization,
                   electricity_price, renewable_available_kw
            FROM simulation_results WHERE run_id = ?
            ORDER BY waiting_task_count DESC, timestamp LIMIT 5
            """,
            (run_id,),
        ).fetchall()
        for row in windows:
            print(
                "  busy_window=" + " | ".join(str(value) for value in row)
            )
    connection.close()


if __name__ == "__main__":
    main()
