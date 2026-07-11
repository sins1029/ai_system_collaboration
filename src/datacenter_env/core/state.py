from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from datacenter_env.core.actuator import ActuatorState


@dataclass(frozen=True, slots=True)
class DataCenterSnapshot:
    true_temperature_c: float
    measured_temperature_c: float
    previous_proposed_cooling_kw: float
    actuator_state: ActuatorState
    step_index: int
    last_timestamp: datetime | None
