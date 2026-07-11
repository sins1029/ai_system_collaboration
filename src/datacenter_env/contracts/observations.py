from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class DataCenterObservation:
    timestamp: datetime
    measured_temperature_c: float
    previous_applied_cooling_kw: float
    workload_fraction: float
    electricity_price_per_kwh: float
    carbon_intensity_kg_per_kwh: float
    outdoor_temperature_c: float
    renewable_power_kw: float
