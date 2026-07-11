from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.cooling import cooling_power_kw
from models.actuator import ActuatorState, CoolingActuator
from models.exogenous_signals import SignalWindow
from models.it_power import it_power_kw
from models.prediction import ThermalPredictionModel
from optimization.thermal_control import CoolingCommand, TemperatureFeedbackCoolingController


@dataclass(frozen=True)
class _Plan:
    temperature_c: float
    actuator_state: ActuatorState
    actions: tuple[float, ...]
    applied_actions: tuple[float, ...]
    predicted_temperatures: tuple[float, ...]
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


class PredictionInfeasibilityError(RuntimeError):
    pass


class FiniteHorizonCoolingController:
    def __init__(
        self,
        dc_config: dict,
        opt_config: dict,
        step_hours: float,
        actuator_config: dict | None = None,
    ):
        self.dc_config = dc_config
        self.opt_config = opt_config
        self.config = opt_config["finite_horizon"]
        self.lookahead_steps = int(self.config["horizon_steps"])
        self.step_hours = float(step_hours)
        self.prediction_model = ThermalPredictionModel(
            dc_config,
            self.config["prediction"],
            step_hours,
        )
        self.actuator = CoolingActuator(
            actuator_config or {"mode": "ideal", "initial_applied_cooling_kw": 0.0},
            step_minutes=step_hours * 60.0,
            max_cooling_kw=float(dc_config["cooling"]["max_cooling_kw"]),
        )
        self.fallback = TemperatureFeedbackCoolingController("load_following", dc_config, opt_config)
        self.solve_count = 0

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
        self.solve_count += 1
        try:
            plan, evaluations = self._solve(
                current_temp_c=current_temp_c,
                previous_cooling_kw=previous_cooling_kw,
                forecast=forecast,
                batch_service_forecast=batch_service_forecast or [],
                actuator_state=actuator_state or ActuatorState(previous_cooling_kw),
            )
            if not plan.actions:
                raise RuntimeError("optimizer returned an empty action sequence")
            return CoolingCommand(
                requested_cooling_kw=plan.actions[0],
                target_temp_c=float(self.dc_config["thermal"]["setpoint_temp_c"]),
                control_floor_temp_c=float(self.config["mpc_temperature_floor_c"]),
                precooling_active=False,
                optimizer_attempted=True,
                optimizer_success=True,
                optimizer_candidate_evaluations=evaluations,
                objective_total=plan.objective_total,
                objective_energy_cost=plan.objective_energy_cost,
                objective_carbon=plan.objective_carbon,
                objective_temperature=plan.objective_temperature,
                objective_control_movement=plan.objective_control_movement,
                planned_cooling_kw=plan.actions,
                planned_applied_cooling_kw=plan.applied_actions,
            )
        except Exception as error:
            fallback = self.fallback.command(
                current_temp_c=current_temp_c,
                heat_kw=heat_kw,
                outdoor_temp_c=outdoor_temp_c,
                electricity_price=electricity_price,
                forecast=forecast,
                previous_cooling_kw=previous_cooling_kw,
                actuator_state=actuator_state,
            )
            prediction_infeasible = isinstance(error, PredictionInfeasibilityError)
            return CoolingCommand(
                requested_cooling_kw=fallback.requested_cooling_kw,
                target_temp_c=fallback.target_temp_c,
                control_floor_temp_c=fallback.control_floor_temp_c,
                precooling_active=False,
                optimizer_attempted=True,
                optimizer_failure=not prediction_infeasible,
                optimizer_fallback=True,
                prediction_infeasibility=prediction_infeasible,
                fallback_reason=(
                    "prediction_infeasibility" if prediction_infeasible else "optimizer_failure"
                ),
            )

    def _solve(
        self,
        current_temp_c: float,
        previous_cooling_kw: float,
        forecast: SignalWindow,
        batch_service_forecast: list[float],
        actuator_state: ActuatorState,
    ) -> tuple[_Plan, int]:
        if len(forecast) == 0:
            raise RuntimeError("empty forecast window")
        frame = forecast.to_frame()
        if len(batch_service_forecast) < len(frame):
            raise RuntimeError("batch service forecast does not cover the optimization window")

        objective = self.config["objective"]
        thermal = self.dc_config["thermal"]
        power = self.dc_config["power"]
        capacity = self.dc_config["capacity"]
        cooling_cfg = self.dc_config["cooling"]
        floor = float(self.config["mpc_temperature_floor_c"])
        max_temp = float(thermal["max_temp_c"])
        setpoint = float(thermal["setpoint_temp_c"])
        max_cooling = float(cooling_cfg["max_cooling_kw"])

        beam = [
            _Plan(
                current_temp_c,
                actuator_state,
                (),
                (),
                (),
                0.0,
                0.0,
                0.0,
                0.0,
            )
        ]
        evaluations = 0
        for offset, row in frame.iterrows():
            total_load = max(0.0, float(row["online_workload"]) + float(batch_service_forecast[offset]))
            heat_kw = it_power_kw(
                total_load=total_load,
                max_load=float(capacity["max_load"]),
                p_idle_kw=float(power["p_idle_kw"]),
                p_peak_kw=float(power["p_peak_kw"]),
            )
            expanded: list[_Plan] = []
            for plan in beam:
                for action in self._candidate_actions(
                    0.0,
                    max_cooling,
                    plan.actuator_state.applied_cooling_kw,
                ):
                    evaluations += 1
                    actuator_result = self.actuator.predict_actuator_step(
                        action, plan.actuator_state
                    )
                    applied_action = actuator_result.applied_cooling_kw
                    next_temp = self.prediction_model.predict_step(
                        current_temp_c=plan.temperature_c,
                        heat_kw=heat_kw,
                        cooling_kw=applied_action,
                        outdoor_temp_c=float(row["outdoor_temperature_c"]),
                    )
                    if next_temp < floor - 1e-9 or next_temp > max_temp + 1e-9:
                        continue
                    p_cool = cooling_power_kw(
                        applied_action,
                        float(row["outdoor_temperature_c"]),
                        cooling_cfg,
                    )
                    total_power = heat_kw + p_cool + float(power["p_aux_kw"])
                    renewable_used = min(float(row["renewable_power_kw"]), total_power)
                    grid_energy = (total_power - renewable_used) * self.step_hours
                    energy_cost = (
                        float(objective["cost_weight"])
                        * float(row["electricity_price"])
                        * grid_energy
                    )
                    carbon = (
                        float(objective["carbon_weight"])
                        * float(row["carbon_intensity_kg_per_kwh"])
                        * grid_energy
                    )
                    temp_penalty = float(objective["temperature_weight"]) * (next_temp - setpoint) ** 2
                    movement = (
                        float(objective["control_movement_weight"])
                        * (
                            (
                                applied_action
                                - plan.actuator_state.applied_cooling_kw
                            )
                            / max_cooling
                        )
                        ** 2
                    )
                    expanded.append(
                        _Plan(
                            temperature_c=next_temp,
                            actuator_state=actuator_result.state,
                            actions=plan.actions + (action,),
                            applied_actions=plan.applied_actions + (applied_action,),
                            predicted_temperatures=plan.predicted_temperatures + (next_temp,),
                            objective_energy_cost=plan.objective_energy_cost + energy_cost,
                            objective_carbon=plan.objective_carbon + carbon,
                            objective_temperature=plan.objective_temperature + temp_penalty,
                            objective_control_movement=plan.objective_control_movement + movement,
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
        count = int(self.config["candidate_count"])
        candidates = list(np.linspace(lower, upper, count))
        candidates.append(min(max(previous, lower), upper))
        return sorted({round(float(value), 9) for value in candidates})

    def metadata(self) -> dict:
        return {
            "horizon_steps": self.lookahead_steps,
            "beam_width": int(self.config["beam_width"]),
            "candidate_count": int(self.config["candidate_count"]),
            "mpc_temperature_floor_c": float(self.config["mpc_temperature_floor_c"]),
            "objective": self.config["objective"],
            "prediction": self.config["prediction"],
            "actuator_mode": self.actuator.mode,
            "movement_penalty_definition": "sum squared applied-action changes normalized by cooling capacity",
            "execution_rule": "optimize horizon, execute first action, re-solve next step",
        }
