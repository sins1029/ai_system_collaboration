from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from datacenter_env.core.primitives import next_temperature_c


@dataclass(frozen=True, slots=True)
class ConstrainedCoolingAction:
    cooling_kw: float
    minimum_feasible_kw: float
    maximum_feasible_kw: float
    predicted_temp_c: float
    intervention_kw: float
    thermal_infeasible: bool


def apply_thermal_constraints(
    requested_cooling_kw: float,
    current_temp_c: float,
    heat_kw: float,
    outdoor_temp_c: float,
    datacenter_config: Mapping[str, Any],
    step_hours: float,
    control_floor_temp_c: float | None = None,
) -> ConstrainedCoolingAction:
    thermal = datacenter_config["thermal"]
    cooling = datacenter_config["cooling"]
    min_temp = float(thermal["min_temp_c"])
    if control_floor_temp_c is not None:
        min_temp = max(min_temp, float(control_floor_temp_c))
    max_temp = float(thermal["max_temp_c"])
    resistance = float(thermal["thermal_resistance_c_per_kw"])
    capacitance = float(thermal["thermal_capacitance_kwh_per_c"])
    max_cooling = float(cooling["max_cooling_kw"])
    passive_loss_kw = (current_temp_c - outdoor_temp_c) / resistance
    minimum_kw = max(
        0.0,
        heat_kw - passive_loss_kw - (max_temp - current_temp_c) * capacitance / step_hours,
    )
    maximum_kw = min(
        max_cooling,
        heat_kw - passive_loss_kw - (min_temp - current_temp_c) * capacitance / step_hours,
    )
    feasible = minimum_kw <= maximum_kw + 1e-9
    requested = min(max(float(requested_cooling_kw), 0.0), max_cooling)
    if feasible:
        constrained = min(max(requested, minimum_kw), maximum_kw)
    elif minimum_kw > max_cooling:
        constrained = max_cooling
    else:
        constrained = 0.0
    predicted = next_temperature_c(
        current_temp_c=current_temp_c,
        heat_kw=heat_kw,
        cooling_kw=constrained,
        outdoor_temp_c=outdoor_temp_c,
        step_hours=step_hours,
        thermal_resistance_c_per_kw=resistance,
        thermal_capacitance_kwh_per_c=capacitance,
    )
    return ConstrainedCoolingAction(
        cooling_kw=constrained,
        minimum_feasible_kw=minimum_kw,
        maximum_feasible_kw=maximum_kw,
        predicted_temp_c=predicted,
        intervention_kw=abs(constrained - float(requested_cooling_kw)),
        thermal_infeasible=not feasible,
    )
