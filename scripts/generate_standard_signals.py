from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.mvp import load_yaml  # noqa: E402
from models.synthetic_signals import generate_to_path  # noqa: E402


def main() -> None:
    exp_config = load_yaml(ROOT / "configs" / "experiment.yaml")
    signal_config = load_yaml(ROOT / "configs" / "signals.yaml")
    output = ROOT / exp_config["input_csv"]
    signals = generate_to_path(signal_config, output)
    metadata = signals.dataset_metadata()
    print(f"generated: {output}")
    print(f"steps: {metadata['number_of_steps']}")
    print(f"range: {metadata['start_timestamp']} -> {metadata['end_timestamp']}")
    print(f"timezone: {metadata['timezone']}")
    print(f"seed: {metadata['random_seed']}")


if __name__ == "__main__":
    main()
