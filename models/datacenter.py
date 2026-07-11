from __future__ import annotations

from dataclasses import dataclass

from models.cooling import cooling_power_kw
from evaluation.accounting import renewable_allocation
from models.it_power import it_power_kw
from models.thermal import next_temperature_c


@dataclass(frozen=True)
class StepResult:
    total_load: float
    it_power_kw: float
    heat_kw: float
    cooling_kw: float
    cooling_power_kw: float
    total_power_kw: float
    grid_power_kw: float
    renewable_available_kw: float
    renewable_used_kw: float
    renewable_curtailed_kw: float
    temp_c: float


class DatacenterModel:
    def __init__(self, config: dict, step_hours: float, plant_parameters: dict | None = None):
        self.config = config
        self.step_hours = step_hours
        self.plant_parameters = dict(
            plant_parameters
            or {
                "thermal_capacity_scale": 1.0,
                "heat_transfer_scale": 1.0,
                "cooling_effectiveness_scale": 1.0,
            }
        )

    def plant_step(
        self,
        current_temp_c: float,
        online_load: float,
        batch_service: float,
        cooling_kw: float,
        outdoor_temp_c: float,
        renewable_available_kw: float,
    ) -> StepResult:
        capacity = self.config["capacity"]
        power = self.config["power"]
        thermal = self.config["thermal"]
        cooling = self.config["cooling"]

        total_load = max(0.0, online_load + batch_service)
        p_it = it_power_kw(
            total_load=total_load,
            max_load=float(capacity["max_load"]),
            p_idle_kw=float(power["p_idle_kw"]),
            p_peak_kw=float(power["p_peak_kw"]),
        )
        heat_kw = p_it
        q_cool = min(max(cooling_kw, 0.0), float(cooling["max_cooling_kw"]))
        p_cool = cooling_power_kw(q_cool, outdoor_temp_c, cooling)
        next_temp = next_temperature_c(
            current_temp_c=current_temp_c,
            heat_kw=heat_kw * float(self.plant_parameters["heat_transfer_scale"]),
            cooling_kw=q_cool * float(self.plant_parameters["cooling_effectiveness_scale"]),
            outdoor_temp_c=outdoor_temp_c,
            step_hours=self.step_hours,
            thermal_resistance_c_per_kw=float(thermal["thermal_resistance_c_per_kw"]),
            thermal_capacitance_kwh_per_c=(
                float(thermal["thermal_capacitance_kwh_per_c"])
                * float(self.plant_parameters["thermal_capacity_scale"])
            ),
        )
        total_power = p_it + p_cool + float(power["p_aux_kw"])
        renewable_available = max(0.0, renewable_available_kw)
        renewable_used, renewable_curtailed = renewable_allocation(renewable_available, total_power)
        grid_power = total_power - renewable_used
        return StepResult(
            total_load=total_load,
            it_power_kw=p_it,
            heat_kw=heat_kw,
            cooling_kw=q_cool,
            cooling_power_kw=p_cool,
            total_power_kw=total_power,
            grid_power_kw=grid_power,
            renewable_available_kw=renewable_available,
            renewable_used_kw=renewable_used,
            renewable_curtailed_kw=renewable_curtailed,
            temp_c=next_temp,
        )

    def step(self, **kwargs) -> StepResult:
        """Backward-compatible alias for the explicit plant interface."""
        if "renewable_kw" in kwargs:
            kwargs["renewable_available_kw"] = kwargs.pop("renewable_kw")
        return self.plant_step(**kwargs)
