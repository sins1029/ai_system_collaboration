from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.mvp import load_yaml  # noqa: E402


DISPLAY_COLUMNS = [
    "timestamp",
    "total_load",
    "prev_temp_c",
    "heat_kw",
    "cooling_command_kw",
    "cooling_kw",
    "cooling_power_kw",
    "temp_c",
    "temp_min_c",
    "temp_max_c",
    "temperature_violation",
]


def longest_streak(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def latest_run_ids(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    return [
        (str(scenario), int(run_id))
        for scenario, run_id in conn.execute(
            "SELECT scenario, MAX(id) FROM experiment_runs GROUP BY scenario ORDER BY scenario"
        )
    ]


def analyze_run(conn: sqlite3.Connection, scenario: str, run_id: int) -> None:
    columns = ", ".join(DISPLAY_COLUMNS)
    rows = conn.execute(
        f"SELECT {columns} FROM simulation_results WHERE run_id = ? ORDER BY step_index",
        (run_id,),
    ).fetchall()
    if not rows:
        print(f"{scenario}: no simulation results")
        return

    print(f"scenario={scenario} run_id={run_id}")
    print(" | ".join(DISPLAY_COLUMNS))
    for row in rows[:12]:
        print(" | ".join(_format(value) for value in row))

    temp_index = DISPLAY_COLUMNS.index("temp_c")
    min_index = DISPLAY_COLUMNS.index("temp_min_c")
    max_index = DISPLAY_COLUMNS.index("temp_max_c")
    temperatures = [float(row[temp_index]) for row in rows]
    below = [float(row[temp_index]) < float(row[min_index]) for row in rows]
    above = [float(row[temp_index]) > float(row[max_index]) for row in rows]
    print(
        "summary: "
        f"below_min_count={sum(below)} above_max_count={sum(above)} "
        f"longest_below_min_streak={longest_streak(below)} "
        f"longest_above_max_streak={longest_streak(above)} "
        f"minimum_temperature={min(temperatures):.4f} "
        f"maximum_temperature={max(temperatures):.4f}"
    )
    print("")


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    exp_config = load_yaml(ROOT / "configs" / "experiment.yaml")
    db_path = ROOT / exp_config["database_path"]
    if not db_path.exists():
        raise SystemExit(f"database does not exist: {db_path}")
    conn = sqlite3.connect(db_path)
    for scenario, run_id in latest_run_ids(conn):
        analyze_run(conn, scenario, run_id)
    conn.close()


if __name__ == "__main__":
    main()
