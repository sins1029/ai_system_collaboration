from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable

from datacenter_env import (
    DataCenterSystemConfig,
    ExogenousInput,
    GymEnvironmentConfig,
    NullRunStore,
    ObservationConfig,
    RewardConfig,
    SingleCenterTaskSchedulingEnv,
)
from datacenter_env.contracts import RunStore
from experiments.mvp import load_yaml, prepare_signals
from external_signals.providers import ScenarioSignalProvider
from external_workloads import TimelineTaskProvider, generate_standard_task_dataset
from models.conditions import resolve_condition


def build_standard_gym_environment(
    root: Path,
    *,
    scheduler_name: str = "fifo_immediate",
    store_factory: Callable[[], RunStore] | None = None,
    invalid_action_policy: str | None = None,
    candidate_order: str | None = None,
    reward_overrides: dict | None = None,
) -> SingleCenterTaskSchedulingEnv:
    config, signal_factory, task_factory = build_standard_gym_components(
        root,
        scheduler_name=scheduler_name,
        invalid_action_policy=invalid_action_policy,
        candidate_order=candidate_order,
        reward_overrides=reward_overrides,
    )
    return SingleCenterTaskSchedulingEnv(
        config,
        signal_provider_factory=signal_factory,
        task_provider_factory=task_factory,
        store_factory=store_factory or NullRunStore,
    )


def build_standard_gym_components(
    root: Path,
    *,
    scheduler_name: str = "fifo_immediate",
    invalid_action_policy: str | None = None,
    candidate_order: str | None = None,
    reward_overrides: dict | None = None,
):
    datacenter = load_yaml(root / "configs" / "datacenter.yaml")
    optimization = load_yaml(root / "configs" / "optimization.yaml")
    experiment = load_yaml(root / "configs" / "experiment.yaml")
    signal_config = load_yaml(root / "configs" / "signals.yaml")
    robustness = load_yaml(root / "configs" / "robustness.yaml")
    tasks = load_yaml(root / "configs" / "tasks.yaml")
    gym_config = load_yaml(root / "configs" / "gym.yaml")
    condition = resolve_condition(robustness, "ideal")
    signals = prepare_signals(root, experiment, signal_config)
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
    task_template = generate_standard_task_dataset(tasks)

    def signal_factory() -> ScenarioSignalProvider:
        return ScenarioSignalProvider(tuple(environmental_steps))

    def task_factory() -> TimelineTaskProvider:
        return TimelineTaskProvider(task_template.tasks, task_template.metadata)

    scheduler = deepcopy(tasks["scheduler"])
    scheduler["name"] = scheduler_name
    system_config = DataCenterSystemConfig.from_dict(
        datacenter,
        optimization,
        controller_name="baseline",
        condition_name="ideal",
        step_minutes=int(tasks["time_step_minutes"]),
        actuator=condition["actuator"],
        plant_parameters=condition["plant"],
        prediction_parameters=condition["prediction"],
        measurement=condition["measurement"],
        mismatch_scenario=str(condition["mismatch_name"]),
        workload={"mode": "task_queue"},
        capacity=tasks["capacity"],
        tasks=tasks["tasks"],
        scheduler=scheduler,
        cooling_controller={"name": "baseline"},
        load_aggregation=tasks["load_aggregation"],
    )
    gym_values = deepcopy(gym_config["gym"])
    if invalid_action_policy is not None:
        gym_values["invalid_action_policy"] = invalid_action_policy
    if candidate_order is not None:
        gym_values["candidate_order"] = candidate_order
    elif scheduler_name == "fifo_immediate":
        gym_values["candidate_order"] = "fifo"
    elif scheduler_name == "earliest_deadline_first":
        gym_values["candidate_order"] = "edf"
    elif scheduler_name == "energy_aware_deferral":
        gym_values["candidate_order"] = "energy_aware"
    reward_values = deepcopy(gym_config["reward"])
    reward_values.update(reward_overrides or {})
    config = GymEnvironmentConfig(
        system_config=system_config,
        environment_id=str(gym_values["environment_id"]),
        interaction_mode=str(gym_values["interaction_mode"]),
        candidate_order=str(gym_values["candidate_order"]),
        invalid_action_policy=str(gym_values["invalid_action_policy"]),
        max_simulation_steps=int(gym_values["max_simulation_steps"]),
        auto_advance_empty_steps=bool(gym_values["auto_advance_empty_steps"]),
        observation=ObservationConfig.coerce(gym_config["observation"]),
        reward=RewardConfig.coerce(reward_values),
    )
    return config, signal_factory, task_factory
