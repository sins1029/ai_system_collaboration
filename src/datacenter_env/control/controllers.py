from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from datacenter_env.config import DataCenterSystemConfig
from datacenter_env.contracts import (
    ControllerDecision,
    ControllerResult,
    DataCenterAction,
    DataCenterObservation,
    ForecastWindow,
)
from datacenter_env.core.actuator import ActuatorState, CoolingActuator
from datacenter_env.core.prediction import ThermalPredictionModel
from datacenter_env.core.primitives import cooling_power_kw, it_power_kw


class PredictionInfeasibilityError(RuntimeError):
    pass


class FeedbackController:
    def __init__(self, config: DataCenterSystemConfig, strategy: str):
        self.config = config
        self.strategy = strategy
        cooling_opt = config.optimization["cooling"]
        self.low_price = float(cooling_opt["low_price_threshold"])
        self.high_price = float(cooling_opt["high_price_threshold"])

    @property
    def name(self) -> str:
        return "heuristic" if self.strategy == "price_aware_precooling" else "baseline"

    @property
    def max_forecast_steps(self) -> int:
        if self.strategy == "price_aware_precooling":
            return int(self.config.optimization["cooling"]["precooling"]["lookahead_steps"])
        return 1

    def reset(self, seed: int | None = None) -> None:
        del seed

    def act(
        self,
        observation: DataCenterObservation,
        forecast_window: ForecastWindow | None = None,
    ) -> ControllerDecision:
        dc = self.config.datacenter
        thermal = dc["thermal"]
        cooling = dc["cooling"]
        setpoint = float(thermal["setpoint_temp_c"])
        target = setpoint
        floor = float(thermal["min_temp_c"])
        precooling = False
        if self.strategy == "price_aware_precooling":
            precooling_cfg = self.config.optimization["cooling"]["precooling"]
            floor = float(precooling_cfg["precool_floor_c"])
            precooling = bool(precooling_cfg["enabled"]) and self._should_precool(
                forecast_window, observation.electricity_price_per_kwh
            )
            if precooling:
                target = max(
                    floor,
                    setpoint - float(precooling_cfg["max_precool_delta_c"]),
                )
        heat = it_power_kw(
            observation.workload_fraction,
            float(dc["capacity"]["max_load"]),
            float(dc["power"]["p_idle_kw"]),
            float(dc["power"]["p_peak_kw"]),
        )
        resistance = float(thermal["thermal_resistance_c_per_kw"])
        feedforward = heat - (target - observation.outdoor_temperature_c) / resistance
        error = observation.measured_temperature_c - target
        deadband = float(thermal["deadband_c"])
        gain = float(cooling["temperature_feedback_gain_kw_per_c"])
        if error > deadband:
            feedback = gain * (error - deadband)
        elif error < -deadband:
            feedback = gain * (error + deadband)
        else:
            feedback = 0.0
        requested = min(
            max(feedforward + feedback, 0.0),
            float(cooling["max_cooling_kw"]),
        )
        return ControllerDecision(
            action=DataCenterAction(requested),
            result=ControllerResult(controller_name=self.name),
            target_temperature_c=target,
            control_floor_temperature_c=(
                floor if precooling else float(thermal["min_temp_c"])
            ),
            precooling_active=precooling,
        )

    def _should_precool(
        self, forecast: ForecastWindow | None, current_price: float
    ) -> bool:
        if forecast is None or current_price > self.low_price or len(forecast) <= 1:
            return False
        return any(
            step.electricity_price_per_kwh >= self.high_price
            for step in forecast.steps[1:]
        )


@dataclass(frozen=True, slots=True)
class _Plan:
    temperature_c: float
    actuator_state: ActuatorState
    actions: tuple[float, ...]
    applied_actions: tuple[float, ...]
    objective_energy_cost: float
    objective_carbon: float
    objective_temperature: float
    objective_control_movement: float

    @property
    def objective_total(self) -> float:
        return (
            self.objective_energy_cost
            + self.objective_carbon
            + self.objective_temperature
            + self.objective_control_movement
        )


class FiniteHorizonController:
    def __init__(self, config: DataCenterSystemConfig):
        self.system_config = config
        self.dc = config.datacenter
        self.optimization = config.optimization
        self.config = config.optimization["finite_horizon"]
        self.predictor = ThermalPredictionModel(
            self.dc, config.prediction_parameters, config.step_hours
        )
        self.actuator = CoolingActuator(
            config.actuator,
            config.step_minutes,
            float(self.dc["cooling"]["max_cooling_kw"]),
        )
        self.fallback = FeedbackController(config, "load_following")
        self._actuator_state = self.actuator.initial_state()
        self.solve_count = 0

    @property
    def name(self) -> str:
        return "finite_horizon"

    @property
    def max_forecast_steps(self) -> int:
        return int(self.config["horizon_steps"])

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._actuator_state = self.actuator.initial_state()
        self.solve_count = 0
        self.fallback.reset()

    def act(
        self,
        observation: DataCenterObservation,
        forecast_window: ForecastWindow | None = None,
    ) -> ControllerDecision:
        self.solve_count += 1
        if forecast_window is None:
            return self._fallback(observation, None, RuntimeError("missing forecast window"))
        state = ActuatorState(
            observation.previous_applied_cooling_kw,
            self._actuator_state.delay_queue_kw,
        )
        try:
            plan, evaluations = self._solve(observation, forecast_window, state)
            if not plan.actions:
                raise RuntimeError("optimizer returned an empty action sequence")
            self._actuator_state = self.actuator.step(plan.actions[0], state).state
            return ControllerDecision(
                action=DataCenterAction(plan.actions[0]),
                result=ControllerResult(
                    controller_name=self.name,
                    optimizer_attempted=True,
                    optimizer_success=True,
                    candidate_evaluations=evaluations,
                    objective_total=plan.objective_total,
                    objective_energy_cost=plan.objective_energy_cost,
                    objective_carbon=plan.objective_carbon,
                    objective_temperature=plan.objective_temperature,
                    objective_control_movement=plan.objective_control_movement,
                ),
                target_temperature_c=float(self.dc["thermal"]["setpoint_temp_c"]),
                control_floor_temperature_c=float(self.config["mpc_temperature_floor_c"]),
            )
        except Exception as error:
            return self._fallback(observation, forecast_window, error)

    def _fallback(
        self,
        observation: DataCenterObservation,
        forecast: ForecastWindow | None,
        error: Exception,
    ) -> ControllerDecision:
        fallback = self.fallback.act(observation, forecast)
        prediction_infeasible = isinstance(error, PredictionInfeasibilityError)
        return ControllerDecision(
            action=fallback.action,
            result=ControllerResult(
                controller_name=self.name,
                optimizer_attempted=True,
                optimizer_failure=not prediction_infeasible,
                fallback_used=True,
                prediction_infeasibility=prediction_infeasible,
                fallback_reason=(
                    "prediction_infeasibility"
                    if prediction_infeasible
                    else "optimizer_failure"
                ),
            ),
            target_temperature_c=fallback.target_temperature_c,
            control_floor_temperature_c=fallback.control_floor_temperature_c,
        )

    def _solve(
        self,
        observation: DataCenterObservation,
        forecast: ForecastWindow,
        actuator_state: ActuatorState,
    ) -> tuple[_Plan, int]:
        objective = self.config["objective"]
        thermal = self.dc["thermal"]
        power = self.dc["power"]
        capacity = self.dc["capacity"]
        cooling_cfg = self.dc["cooling"]
        floor = float(self.config["mpc_temperature_floor_c"])
        max_temp = float(thermal["max_temp_c"])
        setpoint = float(thermal["setpoint_temp_c"])
        max_cooling = float(cooling_cfg["max_cooling_kw"])
        beam = [
            _Plan(
                observation.measured_temperature_c,
                actuator_state,
                (),
                (),
                0.0,
                0.0,
                0.0,
                0.0,
            )
        ]
        evaluations = 0
        for external in forecast.steps:
            heat = it_power_kw(
                external.workload_fraction,
                float(capacity["max_load"]),
                float(power["p_idle_kw"]),
                float(power["p_peak_kw"]),
            )
            expanded: list[_Plan] = []
            for plan in beam:
                for action in self._candidate_actions(
                    0.0, max_cooling, plan.actuator_state.applied_cooling_kw
                ):
                    evaluations += 1
                    actuator_result = self.actuator.step(action, plan.actuator_state)
                    applied = actuator_result.applied_cooling_kw
                    next_temp = self.predictor.predict_step(
                        plan.temperature_c,
                        heat,
                        applied,
                        external.outdoor_temperature_c,
                    )
                    if next_temp < floor - 1e-9 or next_temp > max_temp + 1e-9:
                        continue
                    p_cool = cooling_power_kw(
                        applied, external.outdoor_temperature_c, dict(cooling_cfg)
                    )
                    total_power = heat + p_cool + float(power["p_aux_kw"])
                    renewable_used = min(external.renewable_power_kw, total_power)
                    grid_energy = (total_power - renewable_used) * self.system_config.step_hours
                    energy_cost = (
                        float(objective["cost_weight"])
                        * external.electricity_price_per_kwh
                        * grid_energy
                    )
                    carbon = (
                        float(objective["carbon_weight"])
                        * external.carbon_intensity_kg_per_kwh
                        * grid_energy
                    )
                    temperature = float(objective["temperature_weight"]) * (
                        next_temp - setpoint
                    ) ** 2
                    movement = float(objective["control_movement_weight"]) * (
                        (applied - plan.actuator_state.applied_cooling_kw) / max_cooling
                    ) ** 2
                    expanded.append(
                        _Plan(
                            next_temp,
                            actuator_result.state,
                            plan.actions + (action,),
                            plan.applied_actions + (applied,),
                            plan.objective_energy_cost + energy_cost,
                            plan.objective_carbon + carbon,
                            plan.objective_temperature + temperature,
                            plan.objective_control_movement + movement,
                        )
                    )
            if not expanded:
                raise PredictionInfeasibilityError(
                    "no actuator-feasible action sequence in prediction horizon"
                )
            expanded.sort(key=lambda item: item.objective_total)
            beam = expanded[: int(self.config["beam_width"])]
        return min(beam, key=lambda item: item.objective_total), evaluations

    def _candidate_actions(self, lower: float, upper: float, previous: float) -> list[float]:
        candidates = list(np.linspace(lower, upper, int(self.config["candidate_count"])))
        candidates.append(min(max(previous, lower), upper))
        return sorted({round(float(value), 9) for value in candidates})
