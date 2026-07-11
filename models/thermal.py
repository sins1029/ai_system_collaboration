from __future__ import annotations


def next_temperature_c(
    current_temp_c: float,
    heat_kw: float,
    cooling_kw: float,
    outdoor_temp_c: float,
    step_hours: float,
    thermal_resistance_c_per_kw: float,
    thermal_capacitance_kwh_per_c: float,
) -> float:
    if thermal_capacitance_kwh_per_c <= 0:
        raise ValueError("thermal capacitance must be positive")
    passive_loss_kw = (current_temp_c - outdoor_temp_c) / thermal_resistance_c_per_kw
    delta_c = step_hours / thermal_capacitance_kwh_per_c
    return current_temp_c + delta_c * (heat_kw - cooling_kw - passive_loss_kw)

