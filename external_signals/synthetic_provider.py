from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.exogenous_signals import ExogenousSignals, SignalMetadata


def generate_standard_signals(config: dict) -> ExogenousSignals:
    seed = int(config["seed"])
    steps = int(config["steps"])
    step_minutes = int(config["time_step_minutes"])
    timezone = str(config["timezone"])
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        start=str(config["start_timestamp"]),
        periods=steps,
        freq=f"{step_minutes}min",
        tz=timezone,
    )
    hour = np.arange(steps, dtype=float) * step_minutes / 60.0

    morning_activity = np.exp(-0.5 * ((hour - 8.0) / 2.0) ** 2)
    evening_activity = np.exp(-0.5 * ((hour - 17.0) / 2.3) ** 2)
    solar_shape = np.maximum(np.sin(np.pi * (hour - 6.0) / 12.0), 0.0) ** 1.6

    online = 0.32 + 0.28 * morning_activity + 0.34 * evening_activity
    online += rng.normal(0.0, float(config["workload_noise_std"]), steps)
    online = np.clip(online, 0.22, 0.82)

    batch = 0.12 + 0.13 * np.exp(-0.5 * ((hour - 5.5) / 2.4) ** 2)
    batch += 0.10 * np.exp(-0.5 * ((hour - 13.0) / 3.0) ** 2)
    batch += rng.normal(0.0, float(config["batch_noise_std"]), steps)
    batch = np.clip(batch, 0.05, 0.34)

    price = 0.43 + 0.32 * morning_activity + 0.50 * evening_activity
    price -= 0.14 * np.exp(-0.5 * ((hour - 2.0) / 2.7) ** 2)
    price += rng.normal(0.0, float(config["price_noise_std"]), steps)
    price = np.clip(price, 0.20, 1.10)

    carbon = 0.49 + 0.05 * morning_activity + 0.16 * evening_activity - 0.11 * solar_shape
    carbon += 0.025 * np.sin(2.0 * np.pi * (hour + 1.5) / 24.0)
    carbon += rng.normal(0.0, float(config["carbon_noise_std"]), steps)
    carbon = np.clip(carbon, 0.25, 0.80)

    outdoor = 15.0 + 10.5 * np.sin(2.0 * np.pi * (hour - 7.0) / 24.0)
    outdoor += rng.normal(0.0, float(config["temperature_noise_std_c"]), steps)

    renewable_peak = float(config["renewable_peak_kw"]) if bool(config["renewable_enabled"]) else 0.0
    renewable = renewable_peak * solar_shape

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "online_workload": online,
            "batch_workload": batch,
            "electricity_price": price,
            "carbon_intensity_kg_per_kwh": carbon,
            "outdoor_temperature_c": outdoor,
            "renewable_power_kw": renewable,
        }
    )
    metadata = SignalMetadata(
        source="synthetic-standard-v1",
        timezone=timezone,
        step_minutes=step_minutes,
        seed=seed,
        generation_parameters={key: value for key, value in config.items()},
    )
    return ExogenousSignals(frame, metadata)


def generate_to_path(config: dict, path: str | Path) -> ExogenousSignals:
    signals = generate_standard_signals(config)
    signals.to_csv(path)
    return signals
