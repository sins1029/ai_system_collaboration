from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.mvp import load_yaml  # noqa: E402


METRICS = (
    "tasks_completed",
    "tasks_unfinished",
    "sla_violation_rate",
    "mean_wait_steps",
    "p95_wait_steps",
    "total_deferrals",
    "energy_cost",
    "carbon_kg",
    "mean_cpu_utilization",
    "mean_gpu_utilization",
    "mean_memory_utilization",
    "mean_queue_length",
    "total_grid_energy_kwh",
    "peak_grid_power_kw",
    "temperature_violation_count",
)


def main() -> None:
    config = load_yaml(ROOT / "configs" / "tasks.yaml")
    path = ROOT / str(config["database_path"])
    connection = sqlite3.connect(path)
    rows = connection.execute(
        """
        WITH latest AS (
            SELECT scheduler_name, MAX(id) AS run_id
            FROM experiment_runs
            WHERE name = 'single_center_task_scheduling_v0.2' AND status = 'completed'
            GROUP BY scheduler_name
        )
        SELECT latest.scheduler_name, metrics.metric, metrics.value
        FROM latest JOIN run_metrics metrics ON metrics.run_id = latest.run_id
        """
    ).fetchall()
    connection.close()
    values = {(str(scheduler), str(metric)): float(value) for scheduler, metric, value in rows}
    schedulers = (
        "fifo_immediate",
        "earliest_deadline_first",
        "energy_aware_deferral",
    )
    print("metric".ljust(36) + "".join(name[:24].rjust(26) for name in schedulers))
    for metric in METRICS:
        cells = []
        for scheduler in schedulers:
            value = values.get((scheduler, metric))
            cells.append(("n/a" if value is None else f"{value:.4f}").rjust(26))
        print(metric.ljust(36) + "".join(cells))
    if values:
        best_sla = min(
            schedulers,
            key=lambda name: values.get((name, "sla_violation_rate"), float("inf")),
        )
        best_cost = min(
            schedulers,
            key=lambda name: values.get((name, "energy_cost"), float("inf")),
        )
        print(f"\nlowest_sla_violation: {best_sla}")
        print(f"lowest_energy_cost: {best_cost}")


if __name__ == "__main__":
    main()
