from __future__ import annotations

from pathlib import Path

from models.exogenous_signals import ExogenousSignals
from models.synthetic_signals import generate_to_path


def generate_standard_signal_dataset(
    config: dict,
    output_path: str | Path,
) -> ExogenousSignals:
    if str(config["source"]) != "synthetic-standard-v1":
        raise ValueError(f"unsupported configured signal source: {config['source']}")
    return generate_to_path(config, output_path)
