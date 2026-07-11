from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.mvp import run_mvp  # noqa: E402


def main() -> None:
    _, metrics = run_mvp(ROOT)
    for scenario, values in metrics.items():
        print(f"[{scenario}]")
        for key, value in values.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
