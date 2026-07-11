from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class TemperatureMeasurement:
    def __init__(self, config: Mapping[str, Any], seed: int | None = None):
        self.config = dict(config)
        self.mode = str(config["mode"])
        self.std_c = float(config.get("temperature_noise_std_c", 0.0))
        self.seed = int(config.get("seed", 0) if seed is None else seed)
        if self.mode not in {"none", "gaussian"}:
            raise ValueError(f"unknown measurement mode: {self.mode}")
        if self.std_c < 0:
            raise ValueError("temperature noise standard deviation must be nonnegative")
        self._rng = np.random.default_rng(self.seed)

    def measure_temperature_c(self, true_temperature_c: float) -> float:
        if self.mode == "none":
            return float(true_temperature_c)
        return float(true_temperature_c) + float(self._rng.normal(0.0, self.std_c))
