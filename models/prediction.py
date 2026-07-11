from __future__ import annotations

from dataclasses import dataclass

from models.thermal import next_temperature_c


@dataclass(frozen=True)
class PredictionCoolingBounds:
    minimum_kw: float
    maximum_kw: float
    is_feasible: bool


class ThermalPredictionModel:
    """Controller-side model, kept separate from the simulated plant interface."""

    def __init__(self, dc_config: dict, prediction_config: dict, step_hours: float):
        self.dc_config = dc_config
        self.step_hours = float(step_hours)
        self.capacity_scale = float(prediction_config["thermal_capacity_scale"])
        self.heat_scale = float(prediction_config["heat_transfer_scale"])
        self.cooling_scale = float(prediction_config["cooling_effectiveness_scale"])
        if min(self.capacity_scale, self.heat_scale, self.cooling_scale) <= 0:
            raise ValueError("prediction model scales must be positive")

    def predict_step(
        self,
        current_temp_c: float,
        heat_kw: float,
        cooling_kw: float,
        outdoor_temp_c: float,
    ) -> float:
        thermal = self.dc_config["thermal"]
        return next_temperature_c(
            current_temp_c=current_temp_c,
            heat_kw=heat_kw * self.heat_scale,
            cooling_kw=cooling_kw * self.cooling_scale,
            outdoor_temp_c=outdoor_temp_c,
            step_hours=self.step_hours,
            thermal_resistance_c_per_kw=float(thermal["thermal_resistance_c_per_kw"]),
            thermal_capacitance_kwh_per_c=(
                float(thermal["thermal_capacitance_kwh_per_c"]) * self.capacity_scale
            ),
        )

    def feasible_cooling_bounds(
        self,
        current_temp_c: float,
        heat_kw: float,
        outdoor_temp_c: float,
        minimum_temp_c: float,
        maximum_temp_c: float,
    ) -> PredictionCoolingBounds:
        thermal = self.dc_config["thermal"]
        resistance = float(thermal["thermal_resistance_c_per_kw"])
        capacitance = float(thermal["thermal_capacitance_kwh_per_c"]) * self.capacity_scale
        passive_loss_kw = (current_temp_c - outdoor_temp_c) / resistance
        effective_heat_kw = heat_kw * self.heat_scale
        lower = (
            effective_heat_kw
            - passive_loss_kw
            - (maximum_temp_c - current_temp_c) * capacitance / self.step_hours
        ) / self.cooling_scale
        upper = (
            effective_heat_kw
            - passive_loss_kw
            - (minimum_temp_c - current_temp_c) * capacitance / self.step_hours
        ) / self.cooling_scale
        max_cooling = float(self.dc_config["cooling"]["max_cooling_kw"])
        minimum_kw = max(0.0, lower)
        maximum_kw = min(max_cooling, upper)
        return PredictionCoolingBounds(
            minimum_kw=minimum_kw,
            maximum_kw=maximum_kw,
            is_feasible=minimum_kw <= maximum_kw + 1e-9,
        )
