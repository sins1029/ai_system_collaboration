from __future__ import annotations


def grid_energy_kwh(grid_power_kw: float, step_hours: float) -> float:
    return float(grid_power_kw) * float(step_hours)


def energy_cost(
    grid_power_kw: float,
    electricity_price_per_kwh: float,
    step_hours: float,
) -> float:
    return grid_energy_kwh(grid_power_kw, step_hours) * float(electricity_price_per_kwh)


def carbon_emissions_kg(
    grid_power_kw: float,
    carbon_intensity_kg_per_kwh: float,
    step_hours: float,
) -> float:
    return grid_energy_kwh(grid_power_kw, step_hours) * float(carbon_intensity_kg_per_kwh)


def renewable_allocation(available_kw: float, dc_power_kw: float) -> tuple[float, float]:
    available = max(0.0, float(available_kw))
    used = min(available, max(0.0, float(dc_power_kw)))
    return used, available - used
