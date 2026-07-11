from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.mvp import load_yaml  # noqa: E402


EVENT_COLUMNS = [
    "timestamp",
    "true_temperature_c",
    "measured_temperature_c",
    "predicted_next_temperature_c",
    "actual_next_temperature_c",
    "proposed_cooling_kw",
    "constrained_cooling_kw",
    "applied_cooling_kw",
    "electricity_price",
    "carbon_intensity_kg_per_kwh",
    "outdoor_temperature_c",
    "renewable_available_kw",
]


def print_event(label: str, row: pd.Series, cause: str) -> None:
    print(f"{label}: {row['timestamp']}")
    for column in EVENT_COLUMNS[1:]:
        print(f"  {column}={float(row[column]):.4f}")
    print(f"  cause={cause}")


def main() -> None:
    exp = load_yaml(ROOT / "configs" / "experiment.yaml")
    conn = sqlite3.connect(ROOT / exp["database_path"])
    run = conn.execute(
        """
        SELECT id FROM experiment_runs
        WHERE name = 'single_center_robustness'
          AND controller_name = 'finite_horizon'
          AND condition_name = 'combined_nonideal'
        ORDER BY id DESC LIMIT 1
        """
    ).fetchone()
    if run is None:
        raise SystemExit("combined_nonideal finite_horizon run not found")
    columns = ", ".join(
        EVENT_COLUMNS
        + [
            "one_step_temperature_prediction_error_c",
            "actuator_tracking_error_kw",
            "temperature_deviation_from_setpoint_c",
            "applied_control_movement_kw",
            "temp_min_c",
            "temp_max_c",
        ]
    )
    frame = pd.read_sql_query(
        f"SELECT {columns} FROM simulation_results WHERE run_id = ? ORDER BY step_index",
        conn,
        params=(int(run[0]),),
    )
    prediction_row = frame.loc[frame["one_step_temperature_prediction_error_c"].abs().idxmax()]
    tracking_row = frame.loc[frame["actuator_tracking_error_kw"].idxmax()]
    movement_row = frame.loc[frame["applied_control_movement_kw"].idxmax()]
    safety_margin = pd.concat(
        [
            frame["true_temperature_c"] - frame["temp_min_c"],
            frame["temp_max_c"] - frame["true_temperature_c"],
        ],
        axis=1,
    ).min(axis=1)
    danger_row = frame.loc[safety_margin.idxmin()]

    print(f"run_id={run[0]} controller=finite_horizon condition=combined_nonideal")
    print_event(
        "maximum prediction error",
        prediction_row,
        "prediction-plant parameter mismatch plus temperature measurement noise",
    )
    print_event(
        "maximum actuator tracking error",
        tracking_row,
        "first-order actuator cannot instantly follow the constrained target",
    )
    print_event(
        "smallest thermal safety margin",
        danger_row,
        "true temperature is closest to a configured physical boundary",
    )
    print_event(
        "maximum applied action change",
        movement_row,
        "largest realized actuator movement between adjacent steps",
    )
    conn.close()


if __name__ == "__main__":
    main()
