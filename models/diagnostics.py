from __future__ import annotations

import math

import pandas as pd

from models.thermal import next_temperature_c


CHECK_COLUMNS = [
    "total_load",
    "it_power_kw",
    "heat_kw",
    "temp_c",
    "cooling_kw",
    "cooling_power_kw",
    "grid_power_kw",
    "renewable_available_kw",
    "renewable_used_kw",
    "renewable_curtailed_kw",
]


def is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def add_step_diagnostics(
    rows: list[dict],
    dc_config: dict,
    step_hours: float,
    plant_parameters: dict | None = None,
) -> list[dict]:
    plant_parameters = plant_parameters or {
        "thermal_capacity_scale": 1.0,
        "heat_transfer_scale": 1.0,
        "cooling_effectiveness_scale": 1.0,
    }
    thermal_cfg = dc_config["thermal"]
    power_cfg = dc_config["power"]
    min_temp = float(thermal_cfg["min_temp_c"])
    max_temp = float(thermal_cfg["max_temp_c"])
    p_aux_kw = float(power_cfg["p_aux_kw"])

    diagnosed: list[dict] = []
    for row in rows:
        expected_total_kw = (
            float(row["it_power_kw"])
            + float(row["cooling_power_kw"])
            + p_aux_kw
        )
        expected_grid_kw = expected_total_kw - float(row["renewable_used_kw"])
        power_balance_error = float(row["grid_power_kw"]) - expected_grid_kw
        renewable_balance_error = (
            float(row["renewable_available_kw"])
            - float(row["renewable_used_kw"])
            - float(row["renewable_curtailed_kw"])
        )

        expected_temp = next_temperature_c(
            current_temp_c=float(row["prev_temp_c"]),
            heat_kw=(
                float(row["heat_kw"]) * float(plant_parameters["heat_transfer_scale"])
            ),
            cooling_kw=(
                float(row.get("applied_cooling_kw", row["cooling_kw"]))
                * float(plant_parameters["cooling_effectiveness_scale"])
            ),
            outdoor_temp_c=float(row["outdoor_temperature_c"]),
            step_hours=step_hours,
            thermal_resistance_c_per_kw=float(thermal_cfg["thermal_resistance_c_per_kw"]),
            thermal_capacitance_kwh_per_c=(
                float(thermal_cfg["thermal_capacitance_kwh_per_c"])
                * float(plant_parameters["thermal_capacity_scale"])
            ),
        )
        thermal_balance_error = float(row["temp_c"]) - expected_temp
        below_min = float(row["temp_c"]) < min_temp
        above_max = float(row["temp_c"]) > max_temp
        temp_violation = below_min or above_max
        temp_step_change = float(row["temp_c"]) - float(row["prev_temp_c"])
        temp_deviation = abs(float(row["temp_c"]) - float(row["temp_setpoint_c"]))

        invalid_value_count = sum(not is_finite(float(row[column])) for column in CHECK_COLUMNS)
        negative_power_count = sum(
            float(row[column]) < 0.0
            for column in ["it_power_kw", "heat_kw", "cooling_kw", "cooling_power_kw", "grid_power_kw"]
            if is_finite(float(row[column]))
        )

        diagnosed.append(
            {
                **row,
                "power_balance_error": power_balance_error,
                "renewable_balance_error": renewable_balance_error,
                "thermal_balance_error": thermal_balance_error,
                "temperature_is_finite": int(is_finite(float(row["temp_c"]))),
                "temperature_within_physical_range": int(-50.0 <= float(row["temp_c"]) <= 100.0),
                "temperature_step_change": temp_step_change,
                "below_min_temperature": int(below_min),
                "above_max_temperature": int(above_max),
                "temperature_violation": int(temp_violation),
                "temperature_deviation_from_setpoint_c": temp_deviation,
                "cooling_constraint_intervention_energy_kwh": (
                    float(row["cooling_constraint_intervention_kw"]) * step_hours
                ),
                "invalid_value_count": int(invalid_value_count),
                "negative_power_count": int(negative_power_count),
            }
        )
    return diagnosed


def summarize_diagnostics(result: pd.DataFrame) -> dict[str, float]:
    below_min = result["temp_c"] < result["temp_min_c"]
    above_max = result["temp_c"] > result["temp_max_c"]
    setpoint = result["temp_setpoint_c"]
    deviation = (result["temp_c"] - setpoint).abs()
    return {
        "max_abs_power_balance_error": float(result["power_balance_error"].abs().max()),
        "mean_abs_power_balance_error": float(result["power_balance_error"].abs().mean()),
        "max_abs_renewable_balance_error": float(result["renewable_balance_error"].abs().max()),
        "mean_abs_renewable_balance_error": float(result["renewable_balance_error"].abs().mean()),
        "max_abs_thermal_balance_error": float(result["thermal_balance_error"].abs().max()),
        "mean_abs_thermal_balance_error": float(result["thermal_balance_error"].abs().mean()),
        "temperature_violation_count": float(result["temperature_violation"].sum()),
        "below_min_temperature_count": float(below_min.sum()),
        "above_max_temperature_count": float(above_max.sum()),
        "minimum_temperature_c": float(result["temp_c"].min()),
        "maximum_temperature_c": float(result["temp_c"].max()),
        "mean_temperature_deviation_from_setpoint": float(deviation.mean()),
        "max_temperature_deviation_from_setpoint": float(deviation.max()),
        "longest_below_min_streak": float(_longest_true_streak(below_min)),
        "longest_above_max_streak": float(_longest_true_streak(above_max)),
        "thermal_infeasibility_count": float(result["thermal_infeasible"].sum()),
        "cooling_constraint_intervention_count": float(
            result["cooling_constraint_intervention"].sum()
        ),
        "cooling_constraint_intervention_energy_kwh": float(
            result["cooling_constraint_intervention_energy_kwh"].sum()
        ),
        "invalid_value_count": float(result["invalid_value_count"].sum()),
        "negative_power_count": float(result["negative_power_count"].sum()),
    }


def _longest_true_streak(values: pd.Series) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
