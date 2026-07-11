from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from datacenter_env import (
    DataCenterSystem,
    DataCenterSystemConfig,
    RunMetadata,
    SQLiteRunStore,
)
from experiments.mvp import load_yaml, prepare_signals
from external_signals import build_scenario_provider
from models.conditions import resolve_condition
from optimization.scenarios import load_scenarios


def run_robustness_experiments(
    root: Path,
    condition_names: list[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    dc_config = load_yaml(root / "configs" / "datacenter.yaml")
    exp_config = load_yaml(root / "configs" / "experiment.yaml")
    base_opt_config = load_yaml(root / "configs" / "optimization.yaml")
    signal_config = load_yaml(root / "configs" / "signals.yaml")
    robustness_config = load_yaml(root / "configs" / "robustness.yaml")
    signals = prepare_signals(root, exp_config, signal_config)
    scenarios = load_scenarios(exp_config)
    selected = condition_names or list(robustness_config["conditions"])
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for condition_name in selected:
        condition = resolve_condition(robustness_config, condition_name)
        all_metrics[condition_name] = {}
        for controller_name, scenario in scenarios.items():
            optimization = deepcopy(base_opt_config)
            optimization["finite_horizon"]["prediction"] = deepcopy(
                condition["prediction"]
            )
            provider = build_scenario_provider(
                signals, scenario.dispatch_strategy, optimization
            )
            config = DataCenterSystemConfig.from_dict(
                dc_config,
                optimization,
                controller_name=controller_name,
                condition_name=condition_name,
                step_minutes=int(exp_config["time_step_minutes"]),
                actuator=condition["actuator"],
                plant_parameters=condition["plant"],
                prediction_parameters=condition["prediction"],
                measurement=condition["measurement"],
                mismatch_scenario=str(condition["mismatch_name"]),
            )
            store = SQLiteRunStore(root / exp_config["database_path"])
            system = DataCenterSystem.from_config(config, store=store)
            system.start_run(
                RunMetadata(
                    name="single_center_robustness",
                    controller_name=controller_name,
                    condition_name=condition_name,
                    dataset_name=str(exp_config["dataset_name"]),
                    seed=int(exp_config["seed"]),
                    source_path=str(root / exp_config["input_csv"]),
                    config_snapshot={
                        "datacenter": dc_config,
                        "experiment": exp_config,
                        "optimization": optimization,
                        "signals": signal_config,
                        "robustness": robustness_config,
                        "active_condition": condition,
                    },
                )
            )
            for index, current in enumerate(provider):
                window = provider.window(index, system.controller.max_forecast_steps)
                system.step(current, window)
            summary = system.finish_run()
            all_metrics[condition_name][controller_name] = dict(summary.metrics)
            store.close()
    return all_metrics
