from __future__ import annotations

from copy import deepcopy


def resolve_condition(config: dict, condition_name: str) -> dict:
    conditions = config["conditions"]
    if condition_name not in conditions:
        raise ValueError(f"unknown robustness condition: {condition_name}")
    definition = conditions[condition_name]
    actuator_name = str(definition["actuator"])
    mismatch_name = str(definition["mismatch"])
    measurement_name = str(definition["measurement"])
    return {
        "condition_name": condition_name,
        "actuator_name": actuator_name,
        "actuator": deepcopy(config["actuators"][actuator_name]),
        "mismatch_name": mismatch_name,
        "plant": deepcopy(config["plant"]),
        "prediction": deepcopy(config["mismatch_scenarios"][mismatch_name]),
        "measurement_name": measurement_name,
        "measurement": deepcopy(config["measurements"][measurement_name]),
    }
