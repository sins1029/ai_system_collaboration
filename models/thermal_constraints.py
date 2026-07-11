from __future__ import annotations

from dataclasses import dataclass

from models.thermal import next_temperature_c


@dataclass(frozen=True)
class CoolingBounds:
    minimum_kw: float
    maximum_kw: float
    is_feasible: bool


@dataclass(frozen=True)
class ConstrainedCoolingAction:
    cooling_kw: float
    minimum_feasible_kw: float
    maximum_feasible_kw: float
    predicted_temp_c: float
    intervention: bool
    intervention_kw: float
    thermal_infeasible: bool


def validate_thermal_control_config(dc_config: dict, opt_config: dict | None = None) -> None:
    thermal = dc_config["thermal"]
    cooling = dc_config["cooling"]
    min_temp = float(thermal["min_temp_c"])
    max_temp = float(thermal["max_temp_c"])
    setpoint = float(thermal["setpoint_temp_c"])
    deadband = float(thermal["deadband_c"])

    if min_temp >= max_temp:
        raise ValueError("thermal min_temp_c must be lower than max_temp_c")
    if not min_temp <= setpoint <= max_temp:
        raise ValueError("thermal setpoint_temp_c must be within [min_temp_c, max_temp_c]")
    if deadband < 0:
        raise ValueError("thermal deadband_c must be nonnegative")
    if setpoint - deadband < min_temp or setpoint + deadband > max_temp:
        raise ValueError("thermal setpoint deadband must remain within the safety limits")
    if float(thermal["thermal_resistance_c_per_kw"]) <= 0:
        raise ValueError("thermal resistance must be positive")
    if float(thermal["thermal_capacitance_kwh_per_c"]) <= 0:
        raise ValueError("thermal capacitance must be positive")
    if float(cooling["max_cooling_kw"]) < 0:
        raise ValueError("maximum cooling must be nonnegative")
    if float(cooling["temperature_feedback_gain_kw_per_c"]) < 0:
        raise ValueError("temperature feedback gain must be nonnegative")

    if opt_config is None:
        return

    cooling_opt = opt_config.get("cooling", {})
    precooling = cooling_opt.get("precooling", {})
    if not precooling:
        return
    floor = float(precooling["precool_floor_c"])
    max_delta = float(precooling["max_precool_delta_c"])
    lookahead = int(precooling["lookahead_steps"])
    if floor < min_temp:
        raise ValueError("precool_floor_c must be greater than or equal to min_temp_c")
    if floor > setpoint:
        raise ValueError("precool_floor_c must not exceed setpoint_temp_c")
    if max_delta < 0:
        raise ValueError("max_precool_delta_c must be nonnegative")
    if lookahead < 1:
        raise ValueError("precooling lookahead_steps must be at least 1")
    if float(cooling_opt["low_price_threshold"]) >= float(cooling_opt["high_price_threshold"]):
        raise ValueError("low price threshold must be below high price threshold")

    finite = opt_config.get("finite_horizon", {})
    if finite:
        mpc_floor = float(finite["mpc_temperature_floor_c"])
        if not min_temp <= mpc_floor <= setpoint:
            raise ValueError("mpc_temperature_floor_c must be within [min_temp_c, setpoint_temp_c]")
        if int(finite["horizon_steps"]) < 1:
            raise ValueError("finite horizon_steps must be at least 1")
        if int(finite["beam_width"]) < 1 or int(finite["candidate_count"]) < 2:
            raise ValueError("finite horizon search dimensions are invalid")
        if any(float(value) < 0 for value in finite["objective"].values()):
            raise ValueError("finite horizon objective weights must be nonnegative")


def evaluate_next_temperature(
    current_temp_c: float,
    heat_kw: float,
    cooling_kw: float,
    outdoor_temp_c: float,
    dc_config: dict,
    step_hours: float,
) -> float:
    thermal = dc_config["thermal"]
    return next_temperature_c(
        current_temp_c=current_temp_c,
        heat_kw=heat_kw,
        cooling_kw=cooling_kw,
        outdoor_temp_c=outdoor_temp_c,
        step_hours=step_hours,
        thermal_resistance_c_per_kw=float(thermal["thermal_resistance_c_per_kw"]),
        thermal_capacitance_kwh_per_c=float(thermal["thermal_capacitance_kwh_per_c"]),
    )


def compute_feasible_cooling_bounds(
    current_temp_c: float,
    heat_kw: float,
    outdoor_temp_c: float,
    dc_config: dict,
    step_hours: float,
    minimum_next_temp_c: float | None = None,
) -> CoolingBounds:
    if step_hours <= 0:
        raise ValueError("step_hours must be positive")

    thermal = dc_config["thermal"]
    cooling = dc_config["cooling"]
    min_temp = float(thermal["min_temp_c"])
    if minimum_next_temp_c is not None:
        min_temp = max(min_temp, float(minimum_next_temp_c))
    max_temp = float(thermal["max_temp_c"])
    resistance = float(thermal["thermal_resistance_c_per_kw"])
    capacitance = float(thermal["thermal_capacitance_kwh_per_c"])
    max_cooling = float(cooling["max_cooling_kw"])
    passive_loss_kw = (current_temp_c - outdoor_temp_c) / resistance

    cooling_for_max_temp = heat_kw - passive_loss_kw - (max_temp - current_temp_c) * capacitance / step_hours
    cooling_for_min_temp = heat_kw - passive_loss_kw - (min_temp - current_temp_c) * capacitance / step_hours
    minimum_kw = max(0.0, cooling_for_max_temp)
    maximum_kw = min(max_cooling, cooling_for_min_temp)
    return CoolingBounds(
        minimum_kw=minimum_kw,
        maximum_kw=maximum_kw,
        is_feasible=minimum_kw <= maximum_kw + 1e-9,
    )


def apply_thermal_constraints(
    requested_cooling_kw: float,
    current_temp_c: float,
    heat_kw: float,
    outdoor_temp_c: float,
    dc_config: dict,
    step_hours: float,
    control_floor_temp_c: float | None = None,
) -> ConstrainedCoolingAction:
    physical_bounds = compute_feasible_cooling_bounds(
        current_temp_c=current_temp_c,
        heat_kw=heat_kw,
        outdoor_temp_c=outdoor_temp_c,
        dc_config=dc_config,
        step_hours=step_hours,
    )
    max_cooling = float(dc_config["cooling"]["max_cooling_kw"])
    requested = min(max(float(requested_cooling_kw), 0.0), max_cooling)

    if physical_bounds.is_feasible:
        minimum_kw = physical_bounds.minimum_kw
        maximum_kw = physical_bounds.maximum_kw
        if control_floor_temp_c is not None:
            floor_bounds = compute_feasible_cooling_bounds(
                current_temp_c=current_temp_c,
                heat_kw=heat_kw,
                outdoor_temp_c=outdoor_temp_c,
                dc_config=dc_config,
                step_hours=step_hours,
                minimum_next_temp_c=control_floor_temp_c,
            )
            if floor_bounds.maximum_kw >= minimum_kw:
                maximum_kw = min(maximum_kw, floor_bounds.maximum_kw)
        applied = min(max(requested, minimum_kw), maximum_kw)
    elif physical_bounds.minimum_kw > max_cooling:
        minimum_kw = physical_bounds.minimum_kw
        maximum_kw = physical_bounds.maximum_kw
        applied = max_cooling
    else:
        minimum_kw = physical_bounds.minimum_kw
        maximum_kw = physical_bounds.maximum_kw
        applied = 0.0

    predicted_temp = evaluate_next_temperature(
        current_temp_c=current_temp_c,
        heat_kw=heat_kw,
        cooling_kw=applied,
        outdoor_temp_c=outdoor_temp_c,
        dc_config=dc_config,
        step_hours=step_hours,
    )
    intervention_kw = abs(applied - float(requested_cooling_kw))
    return ConstrainedCoolingAction(
        cooling_kw=applied,
        minimum_feasible_kw=minimum_kw,
        maximum_feasible_kw=maximum_kw,
        predicted_temp_c=predicted_temp,
        intervention=intervention_kw > 1e-9,
        intervention_kw=intervention_kw,
        thermal_infeasible=not physical_bounds.is_feasible,
    )
