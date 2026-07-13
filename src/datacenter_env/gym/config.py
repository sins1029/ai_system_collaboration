from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from datacenter_env.config import DataCenterSystemConfig
from datacenter_env.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    max_wait_steps: float = 96.0
    max_queue_length: float = 512.0
    max_duration_steps: float = 16.0
    max_priority: float = 10.0
    max_slack_steps: float = 96.0
    price_scale: float = 1.5
    carbon_scale: float = 1.0
    outdoor_temperature_min_c: float = -20.0
    outdoor_temperature_max_c: float = 50.0
    measured_temperature_min_c: float = 0.0
    measured_temperature_max_c: float = 50.0
    renewable_power_scale_kw: float = 500.0
    grid_power_scale_kw: float = 1000.0

    def __post_init__(self) -> None:
        positive = (
            self.max_wait_steps,
            self.max_queue_length,
            self.max_duration_steps,
            self.max_priority,
            self.max_slack_steps,
            self.price_scale,
            self.carbon_scale,
            self.renewable_power_scale_kw,
            self.grid_power_scale_kw,
        )
        if any(float(value) <= 0 for value in positive):
            raise ConfigurationError("observation normalization scales must be positive")
        if self.outdoor_temperature_max_c <= self.outdoor_temperature_min_c:
            raise ConfigurationError("outdoor temperature bounds are invalid")
        if self.measured_temperature_max_c <= self.measured_temperature_min_c:
            raise ConfigurationError("measured temperature bounds are invalid")

    @classmethod
    def coerce(cls, value: "ObservationConfig | Mapping[str, Any] | None") -> "ObservationConfig":
        return value if isinstance(value, cls) else cls(**dict(value or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RewardConfig:
    completion_weight: float = 1.0
    energy_cost_weight: float = 0.20
    carbon_weight: float = 0.10
    waiting_weight: float = 0.05
    queue_length_weight: float = 0.02
    sla_violation_weight: float = 5.0
    invalid_action_weight: float = 1.0
    unfinished_weight: float = 2.0
    temperature_violation_weight: float = 5.0
    task_start_weight: float = 0.0
    reference_cost_per_step: float = 20.0
    reference_carbon_per_step: float = 20.0
    reference_queue_length: float = 10.0

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if any(float(value) < 0 for value in values):
            raise ConfigurationError("reward weights and references must be nonnegative")
        if min(
            self.reference_cost_per_step,
            self.reference_carbon_per_step,
            self.reference_queue_length,
        ) <= 0:
            raise ConfigurationError("reward reference scales must be positive")

    @classmethod
    def coerce(cls, value: "RewardConfig | Mapping[str, Any] | None") -> "RewardConfig":
        return value if isinstance(value, cls) else cls(**dict(value or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GymEnvironmentConfig:
    system_config: DataCenterSystemConfig
    environment_id: str = "DataCenterTaskScheduling-v0"
    interaction_mode: str = "sequential_task_decision"
    candidate_order: str = "edf"
    invalid_action_policy: str = "penalize_and_noop"
    max_simulation_steps: int = 96
    auto_advance_empty_steps: bool = True
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "system_config", DataCenterSystemConfig.coerce(self.system_config))
        object.__setattr__(self, "observation", ObservationConfig.coerce(self.observation))
        object.__setattr__(self, "reward", RewardConfig.coerce(self.reward))
        if self.system_config.workload_mode != "task_queue":
            raise ConfigurationError("Gym task scheduling requires workload.mode=task_queue")
        if self.interaction_mode != "sequential_task_decision":
            raise ConfigurationError("only sequential_task_decision is supported")
        if self.candidate_order not in {"fifo", "edf", "priority", "energy_aware"}:
            raise ConfigurationError(f"unknown candidate order: {self.candidate_order}")
        if self.invalid_action_policy not in {
            "raise",
            "penalize_and_noop",
            "project_to_legal",
        }:
            raise ConfigurationError(
                f"unknown invalid action policy: {self.invalid_action_policy}"
            )
        if int(self.max_simulation_steps) < 1:
            raise ConfigurationError("max_simulation_steps must be positive")

    @classmethod
    def coerce(
        cls, value: "GymEnvironmentConfig | Mapping[str, Any]"
    ) -> "GymEnvironmentConfig":
        if isinstance(value, cls):
            return value
        raw = dict(value)
        if "system_config" not in raw:
            raise ConfigurationError("Gym config must contain system_config")
        gym_values = dict(raw.get("gym", {}))
        return cls(
            system_config=DataCenterSystemConfig.coerce(raw["system_config"]),
            environment_id=str(
                gym_values.get(
                    "environment_id",
                    raw.get("environment_id", "DataCenterTaskScheduling-v0"),
                )
            ),
            interaction_mode=str(
                gym_values.get("interaction_mode", "sequential_task_decision")
            ),
            candidate_order=str(gym_values.get("candidate_order", "edf")),
            invalid_action_policy=str(
                gym_values.get("invalid_action_policy", "penalize_and_noop")
            ),
            max_simulation_steps=int(gym_values.get("max_simulation_steps", 96)),
            auto_advance_empty_steps=bool(
                gym_values.get("auto_advance_empty_steps", True)
            ),
            observation=ObservationConfig.coerce(raw.get("observation")),
            reward=RewardConfig.coerce(raw.get("reward")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_config": self.system_config.to_dict(),
            "gym": {
                "environment_id": self.environment_id,
                "interaction_mode": self.interaction_mode,
                "candidate_order": self.candidate_order,
                "invalid_action_policy": self.invalid_action_policy,
                "max_simulation_steps": self.max_simulation_steps,
                "auto_advance_empty_steps": self.auto_advance_empty_steps,
            },
            "observation": self.observation.to_dict(),
            "reward": self.reward.to_dict(),
        }
