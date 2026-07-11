from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.mvp import load_yaml  # noqa: E402
from scripts.compare_single_center_runs import load_metrics  # noqa: E402


METRICS = [
    ("energy_cost", "Energy cost"),
    ("carbon_kg", "Carbon"),
    ("temperature_violation_count", "Temperature violations"),
    ("mean_temperature_deviation_from_setpoint", "Temperature deviation"),
    ("rmse_temperature_prediction_error_c", "Prediction RMSE"),
    ("mean_actuator_tracking_error_kw", "Actuator tracking error"),
    ("applied_control_movement_total", "Applied control movement"),
    ("ramp_limited_count", "Ramp limited count"),
    ("optimizer_fallback_count", "Optimizer fallback"),
]


def latest_run(
    conn: sqlite3.Connection, controller: str, condition: str
) -> int | None:
    row = conn.execute(
        """
        SELECT id FROM experiment_runs
        WHERE name = 'single_center_robustness'
          AND controller_name = ? AND condition_name = ?
        ORDER BY id DESC LIMIT 1
        """,
        (controller, condition),
    ).fetchone()
    return None if row is None else int(row[0])


def degradation(ideal: float, candidate: float) -> str:
    if ideal == 0:
        return f"absolute change {candidate - ideal:+.4f}"
    return f"{(candidate - ideal) / abs(ideal) * 100.0:+.2f}%"


def main() -> None:
    exp = load_yaml(ROOT / "configs" / "experiment.yaml")
    robust = load_yaml(ROOT / "configs" / "robustness.yaml")
    controllers = list(exp["scenarios"])
    conditions = list(robust["conditions"])
    conn = sqlite3.connect(ROOT / exp["database_path"])

    for controller in controllers:
        run_ids = {condition: latest_run(conn, controller, condition) for condition in conditions}
        if any(run_id is None for run_id in run_ids.values()):
            missing = [name for name, run_id in run_ids.items() if run_id is None]
            raise SystemExit(
                "missing robustness runs for " + controller + ": " + ", ".join(missing)
            )
        values = {
            condition: load_metrics(conn, int(run_id))
            for condition, run_id in run_ids.items()
        }
        print(f"Controller: {controller}")
        width = 30
        print(f"{'Metric':<30}" + "".join(f"{name:>{width}}" for name in conditions))
        print("-" * (30 + width * len(conditions)))
        for metric, label in METRICS:
            print(
                f"{label:<30}"
                + "".join(
                    f"{values[name].get(metric, 0.0):>{width}.4f}"
                    for name in conditions
                )
            )
        print("Degradation from ideal:")
        for condition in conditions[1:]:
            print(
                f"  {condition}: cost={degradation(values['ideal']['energy_cost'], values[condition]['energy_cost'])}, "
                f"carbon={degradation(values['ideal']['carbon_kg'], values[condition]['carbon_kg'])}, "
                f"temperature_deviation={degradation(values['ideal']['mean_temperature_deviation_from_setpoint'], values[condition]['mean_temperature_deviation_from_setpoint'])}, "
                f"violations absolute change={values[condition]['temperature_violation_count'] - values['ideal']['temperature_violation_count']:+.0f}"
            )
        print("")
    conn.close()


if __name__ == "__main__":
    main()
