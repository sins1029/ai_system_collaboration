from __future__ import annotations

from typing import Mapping, Protocol

from datacenter_env.contracts import StepResult, TaskOutcome, TaskStatus
from datacenter_env.gym.config import RewardConfig


COMPONENT_NAMES = (
    "completion",
    "energy_cost",
    "carbon",
    "waiting",
    "queue",
    "sla_violation",
    "invalid_action",
    "temperature",
    "terminal_unfinished",
)


def empty_reward_components() -> dict[str, float]:
    return {**{name: 0.0 for name in COMPONENT_NAMES}, "total": 0.0}


def combine_reward_components(
    *values: Mapping[str, float],
) -> dict[str, float]:
    result = empty_reward_components()
    for value in values:
        for name in COMPONENT_NAMES:
            result[name] += float(value.get(name, 0.0))
    result["total"] = sum(result[name] for name in COMPONENT_NAMES)
    return result


class RewardFunction(Protocol):
    def reset(self) -> None: ...

    def compute_decision_reward(
        self, *, invalid_action: bool, task_started: bool
    ) -> Mapping[str, float]: ...

    def compute_simulation_reward(self, result: StepResult) -> Mapping[str, float]: ...

    def compute_terminal_reward(
        self, outcomes: tuple[TaskOutcome, ...]
    ) -> Mapping[str, float]: ...

    def components(self) -> Mapping[str, float]: ...


class CompositeTaskSchedulingReward:
    def __init__(self, config: RewardConfig | Mapping[str, float] | None = None):
        self.config = RewardConfig.coerce(config)
        self.reset()

    def reset(self) -> None:
        self._components = empty_reward_components()

    def compute_decision_reward(
        self, *, invalid_action: bool, task_started: bool
    ) -> Mapping[str, float]:
        result = empty_reward_components()
        result["invalid_action"] = (
            -self.config.invalid_action_weight if invalid_action else 0.0
        )
        result["completion"] = (
            self.config.task_start_weight if task_started else 0.0
        )
        return self._remember(result)

    def compute_simulation_reward(self, result: StepResult) -> Mapping[str, float]:
        tasking = result.tasking
        values = empty_reward_components()
        completed = len(tasking.completed_task_ids) if tasking else 0
        violated = len(tasking.newly_sla_violated_task_ids) if tasking else 0
        waiting = tasking.waiting_count if tasking else 0
        values["completion"] = self.config.completion_weight * completed
        values["energy_cost"] = -self.config.energy_cost_weight * (
            result.accounting.energy_cost / self.config.reference_cost_per_step
        )
        values["carbon"] = -self.config.carbon_weight * (
            result.accounting.carbon_kg / self.config.reference_carbon_per_step
        )
        values["waiting"] = -self.config.waiting_weight * (
            waiting / self.config.reference_queue_length
        )
        values["queue"] = -self.config.queue_length_weight * (
            waiting / self.config.reference_queue_length
        )
        values["sla_violation"] = -self.config.sla_violation_weight * violated
        values["temperature"] = (
            -self.config.temperature_violation_weight
            if result.diagnostics.temperature_violation
            else 0.0
        )
        return self._remember(values)

    def compute_terminal_reward(
        self, outcomes: tuple[TaskOutcome, ...]
    ) -> Mapping[str, float]:
        unfinished = sum(
            outcome.final_status
            in {TaskStatus.WAITING, TaskStatus.RUNNING, TaskStatus.UNSCHEDULABLE}
            for outcome in outcomes
        )
        values = empty_reward_components()
        values["terminal_unfinished"] = -self.config.unfinished_weight * unfinished
        return self._remember(values)

    def components(self) -> Mapping[str, float]:
        return dict(self._components)

    def _remember(self, values: Mapping[str, float]) -> Mapping[str, float]:
        result = combine_reward_components(values)
        self._components = result
        return result
