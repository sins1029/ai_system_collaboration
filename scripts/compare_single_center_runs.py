from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.mvp import load_yaml  # noqa: E402


DISPLAY_SCENARIOS = ["baseline", "heuristic", "finite_horizon"]
SCENARIO_LABELS = {
    "baseline": "Baseline",
    "heuristic": "Heuristic",
    "finite_horizon": "Finite Horizon",
}
PREFERRED_METRICS = [
    ("energy_cost", "Energy cost"),
    ("total_grid_energy_kwh", "Grid energy"),
    ("cooling_energy_kwh", "Cooling energy"),
    ("carbon_kg", "Carbon"),
    ("renewable_utilization_rate", "Renewable utilization"),
    ("peak_grid_power_kw", "Peak grid power"),
    ("temperature_violation_count", "Temperature violations"),
    ("mean_temperature_deviation_from_setpoint", "Mean temperature deviation"),
    ("control_movement_total", "Control movement"),
]


def load_metrics(conn: sqlite3.Connection, run_id: int) -> dict[str, float]:
    return {
        metric: float(value)
        for metric, value in conn.execute(
            "SELECT metric, value FROM run_metrics WHERE run_id = ?", (run_id,)
        )
    }


def latest_scenario_runs(conn: sqlite3.Connection) -> dict[str, int]:
    anchor = conn.execute(
        """
        SELECT id, name, dataset_id
        FROM experiment_runs
        WHERE scenario = 'finite_horizon'
          AND (condition_name = 'ideal' OR condition_name IS NULL)
        ORDER BY id DESC LIMIT 1
        """
    ).fetchone()
    if anchor is None:
        raise SystemExit("No finite_horizon run found.")
    anchor_id, run_name, dataset_id = anchor
    run_ids = {"finite_horizon": int(anchor_id)}
    for scenario in ["baseline", "heuristic"]:
        aliases = [scenario] if scenario != "heuristic" else ["heuristic", "optimized"]
        placeholders = ", ".join("?" for _ in aliases)
        row = conn.execute(
            f"""
            SELECT id FROM experiment_runs
            WHERE scenario IN ({placeholders}) AND name = ? AND dataset_id = ? AND id <= ?
              AND (condition_name = 'ideal' OR condition_name IS NULL)
            ORDER BY id DESC LIMIT 1
            """,
            (*aliases, run_name, dataset_id, anchor_id),
        ).fetchone()
        if row is None:
            raise SystemExit(f"No comparable {scenario} run found.")
        run_ids[scenario] = int(row[0])
    return run_ids


def latest_pair(conn: sqlite3.Connection) -> tuple[int, int]:
    """Compatibility helper for older two-scenario tests and callers."""
    optimized = conn.execute(
        "SELECT id, name, dataset_id FROM experiment_runs WHERE scenario = 'optimized' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if optimized is None:
        raise SystemExit("No optimized run found.")
    optimized_id, run_name, dataset_id = optimized
    baseline = conn.execute(
        """
        SELECT id FROM experiment_runs
        WHERE scenario = 'baseline' AND name = ? AND dataset_id = ? AND id <= ?
        ORDER BY id DESC LIMIT 1
        """,
        (run_name, dataset_id, optimized_id),
    ).fetchone()
    if baseline is None:
        raise SystemExit("No comparable baseline run found.")
    return int(baseline[0]), int(optimized_id)


def pct_change(baseline: float, candidate: float) -> float | None:
    if baseline == 0:
        return None
    return (candidate - baseline) / abs(baseline) * 100.0


def _pass(value: bool) -> str:
    return "PASS" if value else "FAIL"


def main() -> None:
    exp_config = load_yaml(ROOT / "configs" / "experiment.yaml")
    db_path = ROOT / exp_config["database_path"]
    if not db_path.exists():
        raise SystemExit(f"database does not exist: {db_path}")
    conn = sqlite3.connect(db_path)
    run_ids = latest_scenario_runs(conn)
    metrics = {name: load_metrics(conn, run_id) for name, run_id in run_ids.items()}

    print("Run IDs: " + ", ".join(f"{name}={run_ids[name]}" for name in DISPLAY_SCENARIOS))
    print("")
    print(f"{'Metric':<30}{'Baseline':>16}{'Heuristic':>16}{'Finite Horizon':>18}")
    print("-" * 80)
    for metric, label in PREFERRED_METRICS:
        values = [metrics[name].get(metric) for name in DISPLAY_SCENARIOS]
        if any(value is None for value in values):
            print(f"{label:<30}{'missing':>16}{'missing':>16}{'missing':>18}")
            continue
        print(f"{label:<30}{values[0]:>16.4f}{values[1]:>16.4f}{values[2]:>18.4f}")

    baseline = metrics["baseline"]
    heuristic = metrics["heuristic"]
    finite = metrics["finite_horizon"]
    thermal_ok = all(
        item.get("temperature_violation_count", float("inf")) == 0
        and item.get("thermal_infeasibility_count", float("inf")) == 0
        for item in metrics.values()
    )
    numerical_ok = all(
        item.get("invalid_value_count", float("inf")) == 0
        and item.get("max_abs_power_balance_error", float("inf")) <= 1e-9
        and item.get("max_abs_renewable_balance_error", float("inf")) <= 1e-9
        for item in metrics.values()
    )
    economic_ok = finite.get("energy_cost", float("inf")) < baseline.get("energy_cost", float("-inf"))
    carbon_ok = finite.get("carbon_kg", float("inf")) <= baseline.get("carbon_kg", float("-inf"))

    print("")
    print("Conclusions:")
    print(f"Economic improvement: {_pass(economic_ok)}")
    print(f"Carbon improvement: {_pass(carbon_ok)}")
    print(f"Thermal feasibility: {_pass(thermal_ok)}")
    print(
        "Renewable utilization: "
        f"baseline={baseline.get('renewable_utilization_rate', 0.0):.2%}, "
        f"heuristic={heuristic.get('renewable_utilization_rate', 0.0):.2%}, "
        f"finite_horizon={finite.get('renewable_utilization_rate', 0.0):.2%}"
    )
    print(f"Numerical validity: {_pass(numerical_ok)}")
    print(
        "Finite horizon vs baseline cost: "
        f"{pct_change(baseline['energy_cost'], finite['energy_cost']):.2f}%"
    )
    conn.close()


if __name__ == "__main__":
    main()
