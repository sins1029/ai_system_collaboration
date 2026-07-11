from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.exogenous_signals import ExogenousSignals, load_exogenous_signals


REQUIRED_COLUMNS = {
    "timestamp",
    "online_load",
    "batch_load",
    "price",
    "carbon",
    "outdoor_temp",
    "renewable",
}


def load_timeseries(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["timestamp"])
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"missing input columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("input timeseries is empty")
    return data.sort_values("timestamp").reset_index(drop=True)


def load_standard_timeseries(
    path: str | Path,
    step_minutes: int,
    timezone: str,
    source: str = "file",
    seed: int | None = None,
    generation_parameters: dict | None = None,
) -> ExogenousSignals:
    return load_exogenous_signals(
        path=path,
        step_minutes=step_minutes,
        timezone=timezone,
        source=source,
        seed=seed,
        generation_parameters=generation_parameters,
    )
