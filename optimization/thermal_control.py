from __future__ import annotations

from dataclasses import dataclass

from models.actuator import ActuatorState
from models.exogenous_signals import SignalWindow


@dataclass(frozen=True)
class CoolingCommand:
    requested_cooling_kw: float
    target_temp_c: float
    control_floor_temp_c: float
    precooling_active: bool
    optimizer_attempted: bool = False
    optimizer_success: bool = False
    optimizer_failure: bool = False
    optimizer_fallback: bool = False
    optimizer_candidate_evaluations: int = 0
    objective_total: float = 0.0
    objective_energy_cost: float = 0.0
    objective_carbon: float = 0.0
    objective_temperature: float = 0.0
    objective_control_movement: float = 0.0
    planned_cooling_kw: tuple[float, ...] = ()
    planned_applied_cooling_kw: tuple[float, ...] = ()
    prediction_infeasibility: bool = False
    actuator_infeasibility: bool = False
    fallback_reason: str = ""


class TemperatureFeedbackCoolingController:
    def __init__(self, strategy: str, dc_config: dict, opt_config: dict) -> None:
        self.strategy = strategy
        self.dc_config = dc_config
        self.opt_config = opt_config
        cooling_opt = opt_config["cooling"]
        self.low_price = float(cooling_opt["low_price_threshold"])
        self.high_price = float(cooling_opt["high_price_threshold"])
        self.lookahead_steps = (
            int(cooling_opt["precooling"]["lookahead_steps"])
            if strategy == "price_aware_precooling"
            else 1
        )

    def command(
        self,
        current_temp_c: float,
        heat_kw: float,
        outdoor_temp_c: float,
        electricity_price: float,
        forecast: SignalWindow,
        previous_cooling_kw: float,
        batch_service_forecast: list[float] | None = None,
        actuator_state: ActuatorState | None = None,
    ) -> CoolingCommand:
        del previous_cooling_kw, batch_service_forecast, actuator_state
        thermal = self.dc_config["thermal"]
        cooling = self.dc_config["cooling"]
        setpoint = float(thermal["setpoint_temp_c"])
        target = setpoint
        floor = float(thermal["min_temp_c"])
        precooling_active = False

        if self.strategy == "price_aware_precooling":
            precooling = self.opt_config["cooling"]["precooling"]
            floor = float(precooling["precool_floor_c"])
            precooling_active = bool(precooling["enabled"]) and self._should_precool(
                forecast, electricity_price
            )
            if precooling_active:
                target = max(floor, setpoint - float(precooling["max_precool_delta_c"]))
        elif self.strategy != "load_following":
            raise ValueError(f"unknown cooling strategy: {self.strategy}")

        resistance = float(thermal["thermal_resistance_c_per_kw"])
        feedforward_kw = heat_kw - (target - outdoor_temp_c) / resistance
        error_c = current_temp_c - target
        deadband_c = float(thermal["deadband_c"])
        gain = float(cooling["temperature_feedback_gain_kw_per_c"])
        if error_c > deadband_c:
            feedback_kw = gain * (error_c - deadband_c)
        elif error_c < -deadband_c:
            feedback_kw = gain * (error_c + deadband_c)
        else:
            feedback_kw = 0.0

        max_cooling = float(cooling["max_cooling_kw"])
        requested = min(max(feedforward_kw + feedback_kw, 0.0), max_cooling)
        return CoolingCommand(
            requested_cooling_kw=requested,
            target_temp_c=target,
            control_floor_temp_c=floor if precooling_active else float(thermal["min_temp_c"]),
            precooling_active=precooling_active,
        )

    def _should_precool(self, forecast: SignalWindow, current_price: float) -> bool:
        if current_price > self.low_price or len(forecast) <= 1:
            return False
        future_prices = forecast.values("electricity_price").iloc[1:]
        return bool((future_prices >= self.high_price).any())

    def metadata(self) -> dict:
        return {
            "lookahead_steps": self.lookahead_steps,
            "low_price_threshold": self.low_price,
            "high_price_threshold": self.high_price,
            "precooling_rule": "low-price now and high-price inside bounded lookahead",
        }
