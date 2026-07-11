from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Any


@dataclass(frozen=True, slots=True)
class ActuatorState:
    applied_cooling_kw: float
    delay_queue_kw: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class ActuatorResult:
    applied_cooling_kw: float
    state: ActuatorState
    tracking_error_kw: float
    ramp_limited: bool
    ramp_up_limited: bool
    ramp_down_limited: bool
    alpha: float


class CoolingActuator:
    def __init__(self, config: Mapping[str, Any], step_minutes: float, max_cooling_kw: float):
        self.config = dict(config)
        self.mode = str(config["mode"])
        self.step_minutes = float(step_minutes)
        self.max_cooling_kw = float(max_cooling_kw)
        if self.mode not in {"ideal", "rate_limited", "first_order", "delayed"}:
            raise ValueError(f"unknown actuator mode: {self.mode}")
        if self.step_minutes <= 0 or self.max_cooling_kw <= 0:
            raise ValueError("actuator step and capacity must be positive")
        if self.mode == "first_order":
            self._alpha()
        if self.mode == "delayed" and int(config["delay_steps"]) < 1:
            raise ValueError("delayed actuator requires delay_steps >= 1")

    def initial_state(self) -> ActuatorState:
        initial = min(
            max(float(self.config.get("initial_applied_cooling_kw", 0.0)), 0.0),
            self.max_cooling_kw,
        )
        delay_steps = int(self.config.get("delay_steps", 0)) if self.mode == "delayed" else 0
        return ActuatorState(initial, tuple(initial for _ in range(delay_steps)))

    def step(self, target_cooling_kw: float, state: ActuatorState) -> ActuatorResult:
        target = min(max(float(target_cooling_kw), 0.0), self.max_cooling_kw)
        previous = float(state.applied_cooling_kw)
        ramp_up_limited = False
        ramp_down_limited = False
        alpha = 1.0
        queue = state.delay_queue_kw

        if self.mode == "ideal":
            applied = target
        elif self.mode == "rate_limited":
            upper = min(
                self.max_cooling_kw,
                previous + float(self.config["max_ramp_up_kw_per_step"]),
            )
            lower = max(0.0, previous - float(self.config["max_ramp_down_kw_per_step"]))
            applied = min(max(target, lower), upper)
            ramp_up_limited = target > upper + 1e-9
            ramp_down_limited = target < lower - 1e-9
        elif self.mode == "first_order":
            alpha = self._alpha()
            applied = previous + alpha * (target - previous)
        else:
            queue_values = list(state.delay_queue_kw)
            if not queue_values:
                raise ValueError("delayed actuator state has no delay queue")
            applied = queue_values.pop(0)
            queue_values.append(target)
            queue = tuple(queue_values)

        applied = min(max(applied, 0.0), self.max_cooling_kw)
        return ActuatorResult(
            applied_cooling_kw=applied,
            state=ActuatorState(applied, queue),
            tracking_error_kw=abs(target - applied),
            ramp_limited=ramp_up_limited or ramp_down_limited,
            ramp_up_limited=ramp_up_limited,
            ramp_down_limited=ramp_down_limited,
            alpha=alpha,
        )

    def reachable_applied_bounds(self, state: ActuatorState) -> tuple[float, float]:
        previous = float(state.applied_cooling_kw)
        if self.mode == "ideal":
            return 0.0, self.max_cooling_kw
        if self.mode == "rate_limited":
            return (
                max(0.0, previous - float(self.config["max_ramp_down_kw_per_step"])),
                min(
                    self.max_cooling_kw,
                    previous + float(self.config["max_ramp_up_kw_per_step"]),
                ),
            )
        if self.mode == "first_order":
            alpha = self._alpha()
            return previous * (1.0 - alpha), previous + alpha * (self.max_cooling_kw - previous)
        fixed = float(state.delay_queue_kw[0])
        return fixed, fixed

    def predict_actuator_step(
        self, target_cooling_kw: float, state: ActuatorState
    ) -> ActuatorResult:
        """Compatibility alias for side-effect-free actuator prediction."""
        return self.step(target_cooling_kw, state)

    def actuator_aware_target(
        self,
        requested_target_kw: float,
        state: ActuatorState,
        feasible_applied_min_kw: float,
        feasible_applied_max_kw: float,
    ) -> tuple[float, bool]:
        reachable_min, reachable_max = self.reachable_applied_bounds(state)
        intersection_min = max(reachable_min, feasible_applied_min_kw)
        intersection_max = min(reachable_max, feasible_applied_max_kw)
        if intersection_min > intersection_max + 1e-9:
            return min(max(requested_target_kw, 0.0), self.max_cooling_kw), True

        predicted = self.step(requested_target_kw, state).applied_cooling_kw
        desired_applied = min(max(predicted, intersection_min), intersection_max)
        if abs(desired_applied - predicted) <= 1e-9 or self.mode == "delayed":
            return min(max(requested_target_kw, 0.0), self.max_cooling_kw), False
        if self.mode in {"ideal", "rate_limited"}:
            target = desired_applied
        else:
            alpha = self._alpha()
            previous = float(state.applied_cooling_kw)
            target = previous + (desired_applied - previous) / alpha
        return min(max(target, 0.0), self.max_cooling_kw), False

    def _alpha(self) -> float:
        tau = float(self.config["time_constant_minutes"])
        if tau <= 0:
            raise ValueError("actuator time constant must be positive")
        alpha = 1.0 - math.exp(-self.step_minutes / tau)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("first-order actuator alpha must be within (0, 1]")
        return alpha
