from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SIGNAL_COLUMNS = (
    "timestamp",
    "online_workload",
    "batch_workload",
    "electricity_price",
    "carbon_intensity_kg_per_kwh",
    "outdoor_temperature_c",
    "renewable_power_kw",
)

SIGNAL_UNITS = {
    "online_workload": "dimensionless capacity fraction",
    "batch_workload": "dimensionless capacity fraction",
    "electricity_price": "currency/kWh",
    "carbon_intensity_kg_per_kwh": "kgCO2/kWh",
    "outdoor_temperature_c": "degC",
    "renewable_power_kw": "kW",
}


@dataclass(frozen=True)
class SignalMetadata:
    source: str
    timezone: str
    step_minutes: int
    seed: int | None
    generation_parameters: dict[str, Any]


class SignalWindow:
    """A bounded forecast view that cannot expose samples beyond its horizon."""

    def __init__(self, frame: pd.DataFrame, start_index: int, requested_steps: int):
        self._frame = frame.reset_index(drop=True).copy()
        self.start_index = int(start_index)
        self.requested_steps = int(requested_steps)

    def __len__(self) -> int:
        return len(self._frame)

    def row(self, offset: int) -> pd.Series:
        if offset < 0 or offset >= len(self._frame):
            raise IndexError("signal offset is outside the permitted forecast window")
        return self._frame.iloc[offset].copy()

    def values(self, column: str) -> pd.Series:
        if column not in SIGNAL_COLUMNS:
            raise KeyError(f"unknown exogenous signal: {column}")
        return self._frame[column].copy()

    def to_frame(self) -> pd.DataFrame:
        return self._frame.copy()


class ExogenousSignals:
    def __init__(self, frame: pd.DataFrame, metadata: SignalMetadata):
        missing = set(SIGNAL_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"missing exogenous signal columns: {sorted(missing)}")
        self.metadata = metadata
        self._frame = frame.loc[:, SIGNAL_COLUMNS].copy()
        self._frame["timestamp"] = pd.to_datetime(self._frame["timestamp"])
        timestamps = self._frame["timestamp"]
        if timestamps.dt.tz is None:
            self._frame["timestamp"] = timestamps.dt.tz_localize(metadata.timezone)
        else:
            self._frame["timestamp"] = timestamps.dt.tz_convert(metadata.timezone)
        self._frame = self._frame.reset_index(drop=True)
        self.validate()

    def __len__(self) -> int:
        return len(self._frame)

    def current(self, step_index: int) -> pd.Series:
        if step_index < 0 or step_index >= len(self._frame):
            raise IndexError("signal step is outside the dataset")
        return self._frame.iloc[step_index].copy()

    def window(self, start_index: int, horizon_steps: int) -> SignalWindow:
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be at least 1")
        if start_index < 0 or start_index >= len(self._frame):
            raise IndexError("forecast start is outside the dataset")
        end_index = min(len(self._frame), start_index + horizon_steps)
        return SignalWindow(self._frame.iloc[start_index:end_index], start_index, horizon_steps)

    def to_frame(self) -> pd.DataFrame:
        return self._frame.copy()

    def to_csv(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self._frame.to_csv(output, index=False)

    def validate(self) -> None:
        if self._frame.empty:
            raise ValueError("exogenous signal dataset is empty")
        timestamps = self._frame["timestamp"]
        if not timestamps.is_monotonic_increasing:
            raise ValueError("timestamps must be strictly increasing")
        if timestamps.duplicated().any():
            raise ValueError("timestamps must not contain duplicates")
        expected_delta = pd.Timedelta(minutes=self.metadata.step_minutes)
        if len(timestamps) > 1 and not timestamps.diff().iloc[1:].eq(expected_delta).all():
            raise ValueError("timestamps contain missing or misaligned required steps")
        numeric = self._frame.drop(columns="timestamp").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("exogenous signals must contain only finite values")
        if (self._frame["renewable_power_kw"] < 0).any():
            raise ValueError("renewable power must be nonnegative")

    def dataset_metadata(self) -> dict[str, Any]:
        timestamps = self._frame["timestamp"]
        return {
            "signal_source": self.metadata.source,
            "timezone": self.metadata.timezone,
            "time_step_minutes": self.metadata.step_minutes,
            "random_seed": self.metadata.seed,
            "start_timestamp": timestamps.iloc[0].isoformat(),
            "end_timestamp": timestamps.iloc[-1].isoformat(),
            "number_of_steps": len(self._frame),
            "units": SIGNAL_UNITS,
            "generation_parameters": self.metadata.generation_parameters,
        }


def load_exogenous_signals(
    path: str | Path,
    step_minutes: int,
    timezone: str,
    source: str = "file",
    seed: int | None = None,
    generation_parameters: dict[str, Any] | None = None,
) -> ExogenousSignals:
    frame = pd.read_csv(path)
    if set(SIGNAL_COLUMNS).issubset(frame.columns):
        canonical = frame
    else:
        canonical = _normalize_legacy_frame(frame)
        source = f"{source}:legacy-normalized"
    metadata = SignalMetadata(
        source=source,
        timezone=timezone,
        step_minutes=int(step_minutes),
        seed=seed,
        generation_parameters=generation_parameters or {},
    )
    return ExogenousSignals(canonical, metadata)


def _normalize_legacy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    legacy = {"timestamp", "online_load", "batch_load", "price", "carbon", "outdoor_temp", "renewable"}
    missing = legacy.difference(frame.columns)
    if missing:
        raise ValueError(f"missing legacy input columns: {sorted(missing)}")
    return pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "online_workload": frame["online_load"],
            "batch_workload": frame["batch_load"],
            "electricity_price": frame["price"],
            "carbon_intensity_kg_per_kwh": frame["carbon"] / 1000.0,
            "outdoor_temperature_c": frame["outdoor_temp"],
            "renewable_power_kw": frame["renewable"] * 100.0,
        }
    )
