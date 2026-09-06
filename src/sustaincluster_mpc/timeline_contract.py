from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


FORECAST_FUTURE_INTERVALS = 4
REPAIRED_H4_CAPACITY_NODES = FORECAST_FUTURE_INTERVALS + 1
CURRENT_CAPACITY_NODE_INDEX = 0


@dataclass(frozen=True)
class CapacityTimelineNode:
    array_index: int
    semantic_stage: str
    absolute_offset_minutes: float
    timestamp: pd.Timestamp


def repaired_h4_capacity_timeline(
    current_time: Any,
    timestep_minutes: float = 15.0,
) -> tuple[CapacityTimelineNode, ...]:
    """Return the current node followed by four future capacity nodes."""
    if timestep_minutes <= 0:
        raise ValueError("timestep_minutes must be positive")
    current = pd.Timestamp(current_time)
    return tuple(
        CapacityTimelineNode(
            array_index=index,
            semantic_stage="current_state" if index == 0 else f"future_{index}",
            absolute_offset_minutes=float(index * timestep_minutes),
            timestamp=current + pd.Timedelta(minutes=index * timestep_minutes),
        )
        for index in range(REPAIRED_H4_CAPACITY_NODES)
    )


def forecast_step_to_capacity_index(
    horizon_step: int,
    horizon_minutes: int | float,
    timestep_minutes: float = 15.0,
) -> int:
    """Map +15/+30/+45/+60 forecasts to capacity nodes 1/2/3/4."""
    step = int(horizon_step)
    if step < 1 or step > FORECAST_FUTURE_INTERVALS:
        raise ValueError(
            f"forecast horizon_step must be 1..{FORECAST_FUTURE_INTERVALS}"
        )
    expected_minutes = step * float(timestep_minutes)
    if abs(float(horizon_minutes) - expected_minutes) > 1e-9:
        raise ValueError(
            "forecast timestamp does not match its horizon step: "
            f"step={step}, minutes={horizon_minutes}, expected={expected_minutes}"
        )
    return step
