from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from datacenter_env.core.primitives import (
    cooling_power_kw,
    it_power_kw,
    next_temperature_c,
    renewable_allocation,
)


@dataclass(frozen=True, slots=True)
class PlantStep:
    total_load: float
    it_power_kw: float
    heat_kw: float
    cooling_power_kw: float
    total_power_kw: float
    grid_power_kw: float
    renewable_available_kw: float
    renewable_used_kw: float
    renewable_curtailed_kw: float
    temperature_c: float


class DataCenterPlant:
    def __init__(
        self,
        config: Mapping[str, Any],
        step_hours: float,
        plant_parameters: Mapping[str, Any],
    ):
        self.config = config
        self.step_hours = float(step_hours)
        self.parameters = plant_parameters

    def step(
        self,
        current_temperature_c: float,
        workload_fraction: float,
        cooling_kw: float,
        outdoor_temperature_c: float,
        renewable_power_kw: float,
    ) -> PlantStep:
        capacity = self.config["capacity"]
        power = self.config["power"]
        thermal = self.config["thermal"]
        cooling = self.config["cooling"]
        total_load = max(0.0, float(workload_fraction))
        p_it = it_power_kw(
            total_load,
            float(capacity["max_load"]),
            float(power["p_idle_kw"]),
            float(power["p_peak_kw"]),
        )
        q_cool = min(max(float(cooling_kw), 0.0), float(cooling["max_cooling_kw"]))
        p_cool = cooling_power_kw(q_cool, outdoor_temperature_c, dict(cooling))
        next_temp = next_temperature_c(
            current_temp_c=current_temperature_c,
            heat_kw=p_it * float(self.parameters["heat_transfer_scale"]),
            cooling_kw=q_cool * float(self.parameters["cooling_effectiveness_scale"]),
            outdoor_temp_c=outdoor_temperature_c,
            step_hours=self.step_hours,
            thermal_resistance_c_per_kw=float(thermal["thermal_resistance_c_per_kw"]),
            thermal_capacitance_kwh_per_c=(
                float(thermal["thermal_capacitance_kwh_per_c"])
                * float(self.parameters["thermal_capacity_scale"])
            ),
        )
        total_power = p_it + p_cool + float(power["p_aux_kw"])
        renewable_available = max(0.0, float(renewable_power_kw))
        renewable_used, renewable_curtailed = renewable_allocation(
            renewable_available, total_power
        )
        return PlantStep(
            total_load=total_load,
            it_power_kw=p_it,
            heat_kw=p_it,
            cooling_power_kw=p_cool,
            total_power_kw=total_power,
            grid_power_kw=total_power - renewable_used,
            renewable_available_kw=renewable_available,
            renewable_used_kw=renewable_used,
            renewable_curtailed_kw=renewable_curtailed,
            temperature_c=next_temp,
        )
