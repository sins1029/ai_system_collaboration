from __future__ import annotations


def utilization(total_load: float, max_load: float) -> float:
    if max_load <= 0:
        raise ValueError("max_load must be positive")
    return max(0.0, min(total_load / max_load, 1.0))


def it_power_kw(total_load: float, max_load: float, p_idle_kw: float, p_peak_kw: float) -> float:
    util = utilization(total_load, max_load)
    return p_idle_kw + (p_peak_kw - p_idle_kw) * util

