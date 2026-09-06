from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class WaitingStatistics:
    average: float
    p50: float
    p95: float
    maximum: float


def waiting_statistics(values: Iterable[int | float]) -> WaitingStatistics:
    samples = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if samples.size == 0:
        return WaitingStatistics(0.0, 0.0, 0.0, 0.0)
    if not np.all(np.isfinite(samples)) or np.any(samples < 0):
        raise ValueError("waiting intervals must be finite and nonnegative")
    return WaitingStatistics(
        average=float(np.mean(samples)),
        p50=float(np.percentile(samples, 50)),
        p95=float(np.percentile(samples, 95)),
        maximum=float(np.max(samples)),
    )
