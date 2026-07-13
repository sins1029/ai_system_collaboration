from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from datacenter_env import (
    DataCenterSystem,
    DataCenterSystemConfig,
    ExogenousInput,
    RunMetadata,
    SQLiteRunStore,
)
from experiments.mvp import load_yaml, prepare_signals
from external_signals.providers import ScenarioSignalProvider
from external_workloads import generate_standard_task_dataset
from models.conditions import resolve_condition


SCHEDULERS = (
    "fifo_immediate",
    "earliest_deadline_first",
    "energy_aware_deferral",
)


def run_task_scheduling(root: Path) -> dict[str, dict[str, float | None]]:
    dc_config = load_yaml(root / "configs" / "datacenter.yaml")
    optimization = load_yaml(root / "configs" / "optimization.yaml")
    experiment = load_yaml(root / "configs" / "experiment.yaml")
    signal_config = load_yaml(root / "configs" / "signals.yaml")
    robustness = load_yaml(root / "configs" / "robustness.yaml")
    task_config = load_yaml(root / "configs" / "tasks.yaml")
    condition = resolve_condition(robustness, "ideal")
    signals = prepare_signals(root, experiment, signal_config)
    task_provider = generate_standard_task_dataset(
        task_config, root / str(task_config["output_csv"])
    )
    environmental_steps: list[ExogenousInput] = []
    for index in range(len(signals)):
        row = signals.current(index)
        environmental_steps.append(
            ExogenousInput(
                timestamp=row["timestamp"].to_pydatetime(),
                workload_fraction=0.0,
                electricity_price_per_kwh=float(row["electricity_price"]),
                carbon_intensity_kg_per_kwh=float(
                    row["carbon_intensity_kg_per_kwh"]
                ),
                outdoor_temperature_c=float(row["outdoor_temperature_c"]),
                renewable_power_kw=float(row["renewable_power_kw"]),
            )
        )
    signal_provider = ScenarioSignalProvider(tuple(environmental_steps))
    all_metrics: dict[str, dict[str, float | None]] = {}
    for scheduler_name in SCHEDULERS:
        scheduler_config = deepcopy(task_config["scheduler"])
        scheduler_config["name"] = scheduler_name
        config = DataCenterSystemConfig.from_dict(
            dc_config,
            optimization,
            controller_name="baseline",
            condition_name="ideal",
            step_minutes=int(task_config["time_step_minutes"]),
            actuator=condition["actuator"],
            plant_parameters=condition["plant"],
            prediction_parameters=condition["prediction"],
            measurement=condition["measurement"],
            mismatch_scenario=str(condition["mismatch_name"]),
            database_path=str(root / str(task_config["database_path"])),
            workload={"mode": "task_queue"},
            capacity=task_config["capacity"],
            tasks=task_config["tasks"],
            scheduler=scheduler_config,
            cooling_controller={"name": "baseline"},
            load_aggregation=task_config["load_aggregation"],
        )
        store = SQLiteRunStore(config.database_path)
        system = DataCenterSystem.from_config(config, store=store)
        system.start_run(
            RunMetadata(
                name="single_center_task_scheduling_v0.2",
                controller_name="baseline",
                cooling_controller_name="baseline",
                scheduler_name=scheduler_name,
                condition_name="ideal",
                dataset_name=task_provider.metadata.name,
                task_dataset_id=task_provider.metadata.name,
                seed=task_provider.metadata.seed,
                source_path=str(root / str(task_config["output_csv"])),
                config_snapshot=config.to_dict(),
                dataset_metadata={
                    "seed": task_provider.metadata.seed,
                    "timezone": str(task_config["timezone"]),
                    "start_timestamp": task_provider.metadata.start_timestamp.isoformat(),
                    "end_timestamp": task_provider.metadata.end_timestamp.isoformat(),
                    "number_of_steps": task_provider.metadata.number_of_steps,
                    "task_count": task_provider.metadata.task_count,
                    "generation_parameters": dict(
                        task_provider.metadata.generation_parameters
                    ),
                },
            )
        )
        for index, current_input in enumerate(signal_provider):
            horizon = max(
                system.controller.max_forecast_steps,
                system.scheduler.max_forecast_steps if system.scheduler else 1,
            )
            window = signal_provider.window(index, horizon)
            system.step(
                current_input,
                window,
                task_arrivals=task_provider.arrivals_at(current_input.timestamp),
            )
        summary = system.finish_run()
        all_metrics[scheduler_name] = dict(summary.metrics)
        store.close()
    return all_metrics
