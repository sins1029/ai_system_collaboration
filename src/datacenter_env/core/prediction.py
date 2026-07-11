from __future__ import annotations

from typing import Any, Mapping

from datacenter_env.core.primitives import next_temperature_c


class ThermalPredictionModel:
    def __init__(
        self,
        datacenter_config: Mapping[str, Any],
        prediction_config: Mapping[str, Any],
        step_hours: float,
    ):
        self.datacenter_config = datacenter_config
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
        thermal = self.datacenter_config["thermal"]
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
