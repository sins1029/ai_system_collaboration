from __future__ import annotations

from dataclasses import dataclass

from models.exogenous_signals import ExogenousSignals
from optimization.finite_horizon import FiniteHorizonCoolingController
from optimization.baseline import immediate_dispatch
from optimization.temporal_shift import price_aware_dispatch
from optimization.thermal_control import TemperatureFeedbackCoolingController
from models.thermal_constraints import validate_thermal_control_config


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    dispatch_strategy: str
    cooling_strategy: str
    description: str


def load_scenarios(exp_config: dict) -> dict[str, ScenarioDefinition]:
    configured = exp_config.get("scenarios", {})
    if not configured:
        configured = {
            "baseline": {
                "dispatch_strategy": "immediate",
                "cooling_strategy": "load_following",
                "description": "Process batch load immediately with load-following cooling.",
            },
            "heuristic": {
                "dispatch_strategy": "price_aware_temporal_shift",
                "cooling_strategy": "price_aware_precooling",
                "description": "Shift batch work away from high-price periods and add low-price pre-cooling.",
            },
            "finite_horizon": {
                "dispatch_strategy": "immediate",
                "cooling_strategy": "finite_horizon",
                "description": "Receding-horizon cooling optimization.",
            },
        }
    return {
        name: ScenarioDefinition(
            name=name,
            dispatch_strategy=str(values["dispatch_strategy"]),
            cooling_strategy=str(values["cooling_strategy"]),
            description=str(values.get("description", "")),
        )
        for name, values in configured.items()
    }


def build_controls(
    scenario: ScenarioDefinition,
    signals: ExogenousSignals,
    dc_config: dict,
    opt_config: dict,
    step_hours: float,
    actuator_config: dict | None = None,
):
    validate_thermal_control_config(dc_config, opt_config)
    if scenario.dispatch_strategy == "immediate":
        dispatch = immediate_dispatch(signals)
    elif scenario.dispatch_strategy == "price_aware_temporal_shift":
        dispatch = price_aware_dispatch(signals, opt_config)
    else:
        raise ValueError(f"unknown dispatch strategy: {scenario.dispatch_strategy}")

    if scenario.cooling_strategy == "finite_horizon":
        cooling = FiniteHorizonCoolingController(
            dc_config, opt_config, step_hours, actuator_config=actuator_config
        )
    else:
        cooling = TemperatureFeedbackCoolingController(
            strategy=scenario.cooling_strategy,
            dc_config=dc_config,
            opt_config=opt_config,
        )

    return dispatch, cooling
