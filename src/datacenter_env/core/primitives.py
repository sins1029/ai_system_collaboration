from __future__ import annotations


def it_power_kw(total_load: float, max_load: float, p_idle_kw: float, p_peak_kw: float) -> float:
    if max_load <= 0:
        raise ValueError("max_load must be positive")
    utilization = min(max(float(total_load) / float(max_load), 0.0), 1.0)
    return float(p_idle_kw) + utilization * (float(p_peak_kw) - float(p_idle_kw))


def next_temperature_c(
    current_temp_c: float,
    heat_kw: float,
    cooling_kw: float,
    outdoor_temp_c: float,
    step_hours: float,
    thermal_resistance_c_per_kw: float,
    thermal_capacitance_kwh_per_c: float,
) -> float:
    passive_loss_kw = (float(current_temp_c) - float(outdoor_temp_c)) / float(
        thermal_resistance_c_per_kw
    )
    net_heat_kw = float(heat_kw) - float(cooling_kw) - passive_loss_kw
    return float(current_temp_c) + float(step_hours) * net_heat_kw / float(
        thermal_capacitance_kwh_per_c
    )


def cooling_cop(outdoor_temp_c: float, config: dict) -> float:
    raw = float(config["cop_base"]) - float(config["cop_temp_slope"]) * float(
        outdoor_temp_c
    )
    return max(float(config["cop_min"]), raw)


def cooling_power_kw(cooling_kw: float, outdoor_temp_c: float, config: dict) -> float:
    return max(0.0, float(cooling_kw)) / cooling_cop(outdoor_temp_c, config)


def renewable_allocation(available_kw: float, dc_power_kw: float) -> tuple[float, float]:
    available = max(0.0, float(available_kw))
    used = min(available, max(0.0, float(dc_power_kw)))
    return used, available - used
