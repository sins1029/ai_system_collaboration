from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from datacenter_env import (
    DataCenterAction,
    DataCenterSystem,
    DataCenterSystemConfig,
    ForecastWindow,
    NullRunStore,
    RunMetadata,
)
from datacenter_env.contracts import ControllerDecision, ControllerResult, DataCenterObservation
from external_signals import generate_standard_signal_dataset, provider_from_legacy_signals
from models.actuator import ActuatorState
from models.config_loader import load_simple_yaml
from models.exogenous_signals import ExogenousSignals, SignalWindow
from models.it_power import it_power_kw
from optimization.scenarios import build_controls, load_scenarios


IDEAL_SCALES = {
    "thermal_capacity_scale": 1.0,
    "heat_transfer_scale": 1.0,
    "cooling_effectiveness_scale": 1.0,
}


def load_yaml(path: str | Path) -> dict:
    return load_simple_yaml(path)


def prepare_signals(root: Path, exp_config: dict, signal_config: dict) -> ExogenousSignals:
    return generate_standard_signal_dataset(
        signal_config, root / exp_config["input_csv"]
    )


def ideal_condition(opt_config: dict) -> dict:
    return {
        "condition_name": "ideal",
        "actuator_name": "ideal",
        "actuator": {"mode": "ideal", "initial_applied_cooling_kw": 0.0},
        "mismatch_name": "perfect_model",
        "plant": deepcopy(IDEAL_SCALES),
        "prediction": deepcopy(opt_config["finite_horizon"]["prediction"]),
        "measurement_name": "none",
        "measurement": {
            "mode": "none",
            "temperature_noise_std_c": 0.0,
            "seed": 0,
        },
    }


def simulate(
    signals: ExogenousSignals,
    batch_service: pd.Series,
    cooling_control,
    dc_config: dict,
    step_hours: float,
    condition: dict | None = None,
) -> pd.DataFrame:
    """Deprecated compatibility adapter around the package-level runtime."""
    condition = condition or ideal_condition(cooling_control.opt_config)
    if cooling_control.__class__.__name__ == "FiniteHorizonCoolingController":
        controller_name = "finite_horizon"
    elif getattr(cooling_control, "strategy", "") == "price_aware_precooling":
        controller_name = "heuristic"
    else:
        controller_name = "baseline"
    opt_config = deepcopy(cooling_control.opt_config)
    opt_config["finite_horizon"]["prediction"] = deepcopy(condition["prediction"])
    config = DataCenterSystemConfig.from_dict(
        dc_config,
        opt_config,
        controller_name=controller_name,
        condition_name=str(condition.get("condition_name", "ideal")),
        step_minutes=int(round(step_hours * 60.0)),
        actuator=condition["actuator"],
        plant_parameters=condition["plant"],
        prediction_parameters=condition["prediction"],
        measurement=condition["measurement"],
        mismatch_scenario=str(condition.get("mismatch_name", "perfect_model")),
    )
    provider = provider_from_legacy_signals(signals, batch_service)
    system = DataCenterSystem.from_config(config, store=NullRunStore())
    system.controller = _LegacyControllerAdapter(
        cooling_control, signals, batch_service, dc_config
    )
    system.start_run(
        RunMetadata(
            controller_name=controller_name,
            condition_name=config.condition_name,
        )
    )
    for index, current in enumerate(provider):
        system.step(current, provider.window(index, system.controller.max_forecast_steps))
    return system.finish_run().to_frame()


class _LegacyControllerAdapter:
    """Temporary forwarding layer for callers of the deprecated simulate API."""

    def __init__(
        self,
        controller,
        signals: ExogenousSignals,
        batch_service: pd.Series,
        dc_config: dict,
    ):
        self.controller = controller
        self.signals = signals
        self.batch_service = batch_service.reset_index(drop=True)
        self.dc_config = dc_config
        self._index = 0

    @property
    def name(self) -> str:
        if self.controller.__class__.__name__ == "FiniteHorizonCoolingController":
            return "finite_horizon"
        if getattr(self.controller, "strategy", "") == "price_aware_precooling":
            return "heuristic"
        return "baseline"

    @property
    def max_forecast_steps(self) -> int:
        return int(self.controller.lookahead_steps)

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._index = 0

    def act(
        self,
        observation: DataCenterObservation,
        forecast_window: ForecastWindow | None = None,
    ) -> ControllerDecision:
        del forecast_window
        legacy_window = self.signals.window(self._index, self.max_forecast_steps)
        future_service = self.batch_service.iloc[
            self._index : self._index + len(legacy_window)
        ].astype(float).tolist()
        heat = it_power_kw(
            observation.workload_fraction,
            float(self.dc_config["capacity"]["max_load"]),
            float(self.dc_config["power"]["p_idle_kw"]),
            float(self.dc_config["power"]["p_peak_kw"]),
        )
        command = self.controller.command(
            current_temp_c=observation.measured_temperature_c,
            heat_kw=heat,
            outdoor_temp_c=observation.outdoor_temperature_c,
            electricity_price=observation.electricity_price_per_kwh,
            forecast=legacy_window,
            previous_cooling_kw=observation.previous_applied_cooling_kw,
            batch_service_forecast=future_service,
            actuator_state=ActuatorState(observation.previous_applied_cooling_kw),
        )
        self._index += 1
        return ControllerDecision(
            action=DataCenterAction(command.requested_cooling_kw),
            result=ControllerResult(
                controller_name=self.name,
                optimizer_attempted=command.optimizer_attempted,
                optimizer_success=command.optimizer_success,
                optimizer_failure=command.optimizer_failure,
                fallback_used=command.optimizer_fallback,
                prediction_infeasibility=command.prediction_infeasibility,
                actuator_infeasibility=command.actuator_infeasibility,
                fallback_reason=command.fallback_reason or None,
                candidate_evaluations=command.optimizer_candidate_evaluations,
                objective_total=command.objective_total,
                objective_energy_cost=command.objective_energy_cost,
                objective_carbon=command.objective_carbon,
                objective_temperature=command.objective_temperature,
                objective_control_movement=command.objective_control_movement,
            ),
            target_temperature_c=command.target_temp_c,
            control_floor_temperature_c=command.control_floor_temp_c,
            precooling_active=command.precooling_active,
        )


def run_mvp(root: Path) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    dc_config = load_yaml(root / "configs" / "datacenter.yaml")
    exp_config = load_yaml(root / "configs" / "experiment.yaml")
    opt_config = load_yaml(root / "configs" / "optimization.yaml")
    signal_config = load_yaml(root / "configs" / "signals.yaml")
    signals = prepare_signals(root, exp_config, signal_config)
    condition = ideal_condition(opt_config)
    frames: list[pd.DataFrame] = []
    metrics: dict[str, dict[str, float]] = {}
    for name, scenario in load_scenarios(exp_config).items():
        service, legacy_controller = build_controls(
            scenario,
            signals,
            dc_config,
            opt_config,
            float(exp_config["time_step_minutes"]) / 60.0,
            actuator_config=condition["actuator"],
        )
        frame = simulate(
            signals,
            service,
            legacy_controller,
            dc_config,
            float(exp_config["time_step_minutes"]) / 60.0,
            condition,
        )
        frame["scenario"] = name
        frames.append(frame)
        metrics[name] = _summary_from_frame(frame)
    combined = pd.concat(frames, ignore_index=True)
    output_path = root / exp_config["output_csv"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined, metrics


def _summary_from_frame(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "energy_cost": float(frame["energy_cost_step"].sum()),
        "carbon_kg": float(frame["carbon_kg_step"].sum()),
        "temperature_violation_count": float(frame["temperature_violation"].sum()),
    }
