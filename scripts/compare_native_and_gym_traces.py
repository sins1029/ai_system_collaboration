from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.trace_equivalence import compare_scheduler_traces  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scheduler",
        required=True,
        choices=("fifo", "edf", "energy_aware"),
    )
    args = parser.parse_args()
    comparison = compare_scheduler_traces(ROOT, args.scheduler)
    print(f"scheduler: {comparison['scheduler']}")
    print("final_metrics:")
    for name, values in comparison["metric_comparison"].items():
        print(
            f"  {name}: native={values['native']:.9f} gym={values['gym']:.9f} "
            f"abs={values['absolute_difference']:.12f} "
            f"rel={values['relative_difference']:.12f}"
        )
    print(f"native_trajectory_hash: {comparison['native_hash']}")
    print(f"gym_trajectory_hash: {comparison['gym_hash']}")
    print(f"per_step_mismatch_count: {comparison['mismatch_count']}")
    print(
        "planned_executed_mismatch_count: "
        f"{comparison['planned_executed_mismatch_count']}"
    )
    print(f"event_mismatch_count: {comparison['event_mismatch_count']}")
    print(
        "first_divergence: "
        + (
            "none"
            if comparison["first_divergence_step"] is None
            else f"step={comparison['first_divergence_step']} "
            f"timestamp={comparison['first_divergence_timestamp']}"
        )
    )
    risk = comparison["pre_fix_risk"]
    if risk is not None:
        print(
            "pre_fix_divergence_trigger: "
            f"step={risk['simulation_step_index']} timestamp={risk['timestamp']} "
            f"blocked_task={risk['blocked_task_id']} "
            f"missed_planned_starts={risk['missed_planned_start_ids']}"
        )


if __name__ == "__main__":
    main()
