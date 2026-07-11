from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = ROOT / ".cache" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

PACKAGES = [
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("matplotlib", "Matplotlib"),
    ("cvxpy", "CVXPY"),
    ("pyomo", "Pyomo"),
    ("highspy", "HiGHS Python bindings"),
    ("gymnasium", "Gymnasium"),
    ("stable_baselines3", "Stable-Baselines3"),
    ("tensorboard", "TensorBoard"),
]


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    failures: list[str] = []
    for module_name, label in PACKAGES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"OK  {label}: {version}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label}: {exc}")
            print(f"ERR {label}: {exc}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
