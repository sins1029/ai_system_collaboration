from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.task_scheduling import run_task_scheduling  # noqa: E402


def main() -> None:
    metrics = run_task_scheduling(ROOT)
    for scheduler, values in metrics.items():
        print(f"[{scheduler}]")
        for key in (
            "tasks_completed",
            "tasks_unfinished",
            "sla_violation_rate",
            "mean_wait_steps",
            "energy_cost",
            "carbon_kg",
            "mean_cpu_utilization",
            "mean_gpu_utilization",
            "temperature_violation_count",
        ):
            value = values.get(key)
            print(f"  {key}: {'None' if value is None else f'{value:.4f}'}")


if __name__ == "__main__":
    main()
