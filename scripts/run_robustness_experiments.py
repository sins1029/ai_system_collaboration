from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.robustness import run_robustness_experiments  # noqa: E402


def main() -> None:
    results = run_robustness_experiments(ROOT)
    for condition, controllers in results.items():
        print(f"[{condition}]")
        for controller, metrics in controllers.items():
            print(
                f"  {controller}: cost={metrics['energy_cost']:.4f} "
                f"violations={metrics['temperature_violation_count']:.0f} "
                f"prediction_rmse={metrics['rmse_temperature_prediction_error_c']:.4f} "
                f"tracking_error={metrics['mean_actuator_tracking_error_kw']:.4f} "
                f"fallback={metrics['optimizer_fallback_count']:.0f}"
            )


if __name__ == "__main__":
    main()
