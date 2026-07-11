from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.mvp import load_yaml  # noqa: E402
from scripts.compare_single_center_runs import latest_scenario_runs  # noqa: E402


SIGNALS = {
    "highest price": "electricity_price",
    "highest carbon intensity": "carbon_intensity_kg_per_kwh",
    "highest outdoor temperature": "outdoor_temperature_c",
    "highest renewable power": "renewable_available_kw",
}


def highest_window(frame: pd.DataFrame, column: str, steps: int = 4) -> tuple[int, int]:
    rolling = frame[column].rolling(steps).mean()
    end = int(rolling.idxmax())
    return end - steps + 1, end


def main() -> None:
    exp_config = load_yaml(ROOT / "configs" / "experiment.yaml")
    db_path = ROOT / exp_config["database_path"]
    conn = sqlite3.connect(db_path)
    run_ids = latest_scenario_runs(conn)
    frames: dict[str, pd.DataFrame] = {}
    for scenario, run_id in run_ids.items():
        frames[scenario] = pd.read_sql_query(
            """
            SELECT step_index, timestamp, online_workload, batch_workload, batch_service,
                   electricity_price, carbon_intensity_kg_per_kwh, outdoor_temperature_c,
                   renewable_available_kw, renewable_used_kw, it_power_kw, cooling_power_kw,
                   grid_power_kw, temp_c, cooling_kw
            FROM simulation_results WHERE run_id = ? ORDER BY step_index
            """,
            conn,
            params=(run_id,),
        )

    print("Key windows (four 15-minute steps):")
    baseline = frames["baseline"]
    for label, column in SIGNALS.items():
        start, end = highest_window(baseline, column)
        print(
            f"- {label}: {baseline.loc[start, 'timestamp']} -> {baseline.loc[end, 'timestamp']} "
            f"mean_{column}={baseline.loc[start:end, column].mean():.4f}"
        )
        for scenario, frame in frames.items():
            window = frame.loc[start:end]
            print(
                f"  {scenario}: batch_service={window['batch_service'].sum():.4f}, "
                f"grid_energy_kwh={(window['grid_power_kw'] * 0.25).sum():.4f}, "
                f"cooling_energy_kwh={(window['cooling_power_kw'] * 0.25).sum():.4f}, "
                f"renewable_used_kwh={(window['renewable_used_kw'] * 0.25).sum():.4f}, "
                f"mean_temp_c={window['temp_c'].mean():.4f}, "
                f"mean_control_kw={window['cooling_kw'].mean():.4f}"
            )

    print("")
    print("96-step coordination timeseries:")
    print(
        "scenario | timestamp | workload | electricity_price | carbon_intensity | "
        "outdoor_temperature | renewable_available | renewable_used | P_IT | P_cool | "
        "P_grid | temperature | control_action"
    )
    for scenario, frame in frames.items():
        for _, row in frame.iterrows():
            workload = float(row["online_workload"]) + float(row["batch_service"])
            print(
                f"{scenario} | {row['timestamp']} | {workload:.4f} | "
                f"{row['electricity_price']:.4f} | {row['carbon_intensity_kg_per_kwh']:.4f} | "
                f"{row['outdoor_temperature_c']:.4f} | {row['renewable_available_kw']:.4f} | "
                f"{row['renewable_used_kw']:.4f} | {row['it_power_kw']:.4f} | "
                f"{row['cooling_power_kw']:.4f} | {row['grid_power_kw']:.4f} | "
                f"{row['temp_c']:.4f} | {row['cooling_kw']:.4f}"
            )
    conn.close()


if __name__ == "__main__":
    main()
