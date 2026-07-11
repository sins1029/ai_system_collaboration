from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

from datacenter_env.exceptions import InputValidationError


@dataclass(frozen=True, slots=True)
class ExogenousInput:
    timestamp: datetime
    workload_fraction: float
    electricity_price_per_kwh: float
    carbon_intensity_kg_per_kwh: float
    outdoor_temperature_c: float
    renewable_power_kw: float

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise InputValidationError("timestamp must be a datetime")
        numeric = (
            self.workload_fraction,
            self.electricity_price_per_kwh,
            self.carbon_intensity_kg_per_kwh,
            self.outdoor_temperature_c,
            self.renewable_power_kw,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise InputValidationError("exogenous input values must be finite")
        if self.workload_fraction < 0:
            raise InputValidationError("workload_fraction must be nonnegative")
        if self.electricity_price_per_kwh < 0:
            raise InputValidationError("electricity price must be nonnegative")
        if self.carbon_intensity_kg_per_kwh < 0:
            raise InputValidationError("carbon intensity must be nonnegative")
        if self.renewable_power_kw < 0:
            raise InputValidationError("renewable power must be nonnegative")


@dataclass(frozen=True, slots=True)
class ForecastWindow:
    steps: tuple[ExogenousInput, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise InputValidationError("forecast window must contain at least one step")
        timestamps = tuple(step.timestamp for step in self.steps)
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise InputValidationError("forecast timestamps must be strictly increasing")

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int) -> ExogenousInput:
        return self.steps[index]
