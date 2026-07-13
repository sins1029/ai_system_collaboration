from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from datacenter_env.config import DataCenterSystemConfig
from datacenter_env.contracts import (
    AgentDecisionRecord,
    ExogenousInput,
    ForecastWindow,
    ResourceAvailability,
    ResourceUsage,
    RunMetadata,
    RunStore,
    StepResult,
    TaskArrivalProvider,
    TaskSchedulingDecision,
    TaskSchedulingObservation,
    TaskStatus,
    TaskView,
)
from datacenter_env.exceptions import ConfigurationError, InputValidationError
from datacenter_env.gym.action import DEFER_CURRENT_TASK, START_CURRENT_TASK
from datacenter_env.gym.config import GymEnvironmentConfig
from datacenter_env.gym.masks import order_task_candidates, task_action_mask
from datacenter_env.gym.observation import ObservationEncoder
from datacenter_env.gym.rewards import (
    CompositeTaskSchedulingReward,
    RewardFunction,
    combine_reward_components,
    empty_reward_components,
)
from datacenter_env.runtime import DataCenterSystem
from datacenter_env.storage import NullRunStore


SignalProviderFactory = Callable[[], Iterable[ExogenousInput]]
TaskProviderFactory = Callable[[], TaskArrivalProvider]
StoreFactory = Callable[[], RunStore]


class SingleCenterTaskSchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: GymEnvironmentConfig | Mapping[str, Any],
        signal_provider_factory: SignalProviderFactory,
        task_provider_factory: TaskProviderFactory,
        store_factory: StoreFactory | None = None,
        reward_function: RewardFunction | None = None,
    ):
        super().__init__()
        self.config = GymEnvironmentConfig.coerce(config)
        self._signal_provider_factory = signal_provider_factory
        self._task_provider_factory = task_provider_factory
        self._store_factory = store_factory or NullRunStore
        self.reward_function = reward_function or CompositeTaskSchedulingReward(
            self.config.reward
        )
        self.action_space = gym.spaces.Discrete(2)
        self._encoder = ObservationEncoder(
            self.config.observation,
            self.config.system_config,
            self.config.max_simulation_steps,
        )
        self.observation_space = self._encoder.space
        self._system: DataCenterSystem | None = None
        self._finished = True
        self._last_result: StepResult | None = None
        self._task_observation: TaskSchedulingObservation | None = None
        self._candidates: tuple[TaskView, ...] = ()
        self._candidate_index = 0
        self._selected_task_ids: list[str] = []
        self._deferred_task_ids: list[str] = []
        self._auto_skipped_candidate_ids: list[str] = []
        self._forced_resource_blocked_task_ids: list[str] = []
        self._recent_auto_skipped_candidate_ids: list[str] = []
        self._recent_forced_resource_blocked_task_ids: list[str] = []
        self._planned_start_task_ids: tuple[str, ...] = ()
        self._plan_declared = False
        self._last_plan_difference_task_ids: tuple[str, ...] = ()
        self._virtual_usage = ResourceUsage()
        self._virtual_availability = ResourceAvailability(0.0, 0.0, 0.0)
        self._signals: tuple[ExogenousInput, ...] = ()
        self._task_provider: TaskArrivalProvider | None = None
        self._simulation_step_index = 0
        self._decision_step_index = 0
        self._episode_reward = 0.0
        self._invalid_action_count = 0
        self._seed = 0
        self._summary = None
        self._simulation_traces: list[dict[str, Any]] = []
        self._current_gym_actions: list[tuple[str, int, bool]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._close_active_run("Gym environment reset before episode completion")
        self._seed = 0 if seed is None else int(seed)
        self._signals = tuple(self._signal_provider_factory())
        self._task_provider = self._task_provider_factory()
        if not self._signals:
            raise ConfigurationError("signal provider returned no episode inputs")
        self._system = DataCenterSystem.from_config(
            self.config.system_config, store=self._store_factory()
        )
        provider_metadata = getattr(self._task_provider, "metadata", None)
        dataset_name = getattr(provider_metadata, "name", "gym-task-provider")
        source_path = str(
            getattr(provider_metadata, "generation_parameters", {}).get(
                "source_path", "external-task-provider"
            )
        )
        self._system.start_run(
            RunMetadata(
                name="single_center_task_scheduling_gym_v0.3",
                controller_name=self.config.system_config.cooling_controller_name,
                condition_name=self.config.system_config.condition_name,
                dataset_name=dataset_name,
                task_dataset_id=dataset_name,
                source_path=source_path,
                seed=self._seed,
                config_snapshot=self.config.to_dict(),
                interface_type="gymnasium",
                environment_id=self.config.environment_id,
                reward_config=self.config.reward.to_dict(),
                observation_config=self.config.observation.to_dict(),
                invalid_action_policy=self.config.invalid_action_policy,
                candidate_order=self.config.candidate_order,
                episode_seed=self._seed,
            ),
            seed=self._seed,
        )
        self.reward_function.reset()
        self._encoder.reset()
        self._finished = False
        self._last_result = None
        self._task_observation = None
        self._candidates = ()
        self._candidate_index = 0
        self._selected_task_ids = []
        self._deferred_task_ids = []
        self._recent_auto_skipped_candidate_ids = []
        self._recent_forced_resource_blocked_task_ids = []
        self._simulation_step_index = 0
        self._decision_step_index = 0
        self._episode_reward = 0.0
        self._invalid_action_count = 0
        self._summary = None
        self._simulation_traces = []
        self._current_gym_actions = []
        observation, auto_count, components, results = self._seek_decision_state()
        self._episode_reward += components["total"]
        info = self._build_info(
            candidate=self.current_candidate,
            candidate_position=self._candidate_index,
            action_mask=tuple(int(value) for value in self.current_action_mask),
            invalid_action=False,
            invalid_reason=None,
            physical_advanced=bool(results),
            auto_advanced=auto_count,
            components=components,
            results=results,
        )
        info["reset_auto_advance_reward"] = components["total"]
        return observation, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._finished or self._system is None or self._task_observation is None:
            raise RuntimeError("step() called outside an active Gym episode")
        candidate = self.current_candidate
        if candidate is None:
            raise RuntimeError("normal decision state is missing a candidate task")
        position = self._candidate_index
        action_simulation_index = self._simulation_step_index
        self._recent_auto_skipped_candidate_ids = []
        self._recent_forced_resource_blocked_task_ids = []
        timestamp = self._task_observation.timestamp
        mask_array = self.current_action_mask
        action_mask = (int(mask_array[0]), int(mask_array[1]))
        requested_action = int(action)
        in_space = self.action_space.contains(action)
        action_legal = in_space and bool(mask_array[requested_action])
        invalid_reason = None
        effective_action: int | None = requested_action if action_legal else None
        if not action_legal:
            self._invalid_action_count += 1
            invalid_reason = (
                "action is outside Discrete(2)"
                if not in_space
                else "action is disabled by the current action mask"
            )
            if self.config.invalid_action_policy == "raise":
                raise InputValidationError(invalid_reason)
            if self.config.invalid_action_policy == "project_to_legal":
                legal = np.flatnonzero(mask_array)
                if not len(legal):
                    raise RuntimeError("cannot project action from a state with no legal action")
                effective_action = int(
                    START_CURRENT_TASK
                    if START_CURRENT_TASK in legal
                    else legal[0]
                )
        task_started = effective_action == START_CURRENT_TASK
        decision_components = self.reward_function.compute_decision_reward(
            invalid_action=not action_legal,
            task_started=task_started,
        )
        if task_started:
            self._selected_task_ids.append(candidate.spec.task_id)
            self._allocate_virtual(candidate)
        elif action_legal and effective_action == DEFER_CURRENT_TASK:
            self._deferred_task_ids.append(candidate.spec.task_id)
        self._current_gym_actions.append(
            (candidate.spec.task_id, requested_action, action_legal)
        )
        self._candidate_index += 1
        self._skip_non_decision_candidates()

        simulation_components = empty_reward_components()
        auto_components = empty_reward_components()
        terminal_components = empty_reward_components()
        results: list[StepResult] = []
        physical_advanced = False
        auto_advanced = 0
        should_advance = self._candidate_index >= len(self._candidates)
        if should_advance:
            result = self._advance_physical_step()
            results.append(result)
            physical_advanced = True
            simulation_components = self.reward_function.compute_simulation_reward(result)
            self._simulation_step_index += 1

        truncated = self._episode_limit_reached()
        terminated = False
        if physical_advanced and not truncated:
            observation, auto_advanced, auto_components, auto_results = (
                self._seek_decision_state()
            )
            results.extend(auto_results)
            truncated = self._episode_limit_reached()
        elif truncated:
            observation = self._terminal_observation()
        else:
            observation = self._encode_current_observation()

        if truncated:
            terminal_components = self.reward_function.compute_terminal_reward(
                self._system.environment.task_outcomes()
            )
        components = combine_reward_components(
            decision_components,
            simulation_components,
            auto_components,
            terminal_components,
        )
        reward = float(components["total"])
        self._episode_reward += reward
        record = AgentDecisionRecord(
            decision_step_index=self._decision_step_index,
            simulation_step_index=(
                self._simulation_step_index - len(results)
                if results
                else self._simulation_step_index
            ),
            timestamp=timestamp,
            candidate_task_id=candidate.spec.task_id,
            action=requested_action,
            action_legal=action_legal,
            action_mask=action_mask,
            decision_reward=float(decision_components["total"]),
            simulation_reward=float(
                simulation_components["total"] + auto_components["total"]
            ),
            total_reward=reward,
            reward_components=components,
        )
        self._system.append_agent_decision(record)
        info = self._build_info(
            candidate=candidate,
            candidate_position=position,
            action_mask=action_mask,
            invalid_action=not action_legal,
            invalid_reason=invalid_reason,
            physical_advanced=physical_advanced,
            auto_advanced=auto_advanced,
            components=components,
            results=results,
            timestamp=timestamp.isoformat(),
            simulation_step_index=action_simulation_index,
        )
        self._decision_step_index += 1
        if truncated:
            episode_summary = {
                "episode_reward": self._episode_reward,
                "decision_steps": self._decision_step_index,
                "simulation_steps": self._simulation_step_index,
                "invalid_actions": self._invalid_action_count,
                "terminated": terminated,
                "truncated": truncated,
            }
            self._system.append_gym_episode_summary(episode_summary)
            self._summary = self._system.finish_run()
            self._finished = True
            unfinished = tuple(
                outcome.task_id
                for outcome in self._summary.task_outcomes
                if outcome.final_status in {TaskStatus.WAITING, TaskStatus.RUNNING}
            )
            info.update(
                {
                    "final_metrics": dict(self._summary.metrics),
                    "run_id": self._summary.run_id,
                    "unfinished_task_ids": unfinished,
                    "episode_summary": episode_summary,
                }
            )
        return observation, reward, terminated, truncated, info

    @property
    def current_candidate(self) -> TaskView | None:
        if self._candidate_index >= len(self._candidates):
            return None
        return self._candidates[self._candidate_index]

    @property
    def current_action_mask(self) -> np.ndarray:
        if self._task_observation is None:
            return np.asarray([0, 0], dtype=np.int8)
        return task_action_mask(
            self.current_candidate,
            self._task_observation.timestamp,
            self._virtual_availability,
        )

    @property
    def scheduling_observation(self) -> TaskSchedulingObservation:
        if self._task_observation is None:
            raise RuntimeError("no active task scheduling observation")
        return self._task_observation

    @property
    def simulation_step_index(self) -> int:
        return self._simulation_step_index

    def forecast_for_scheduler(self, steps: int) -> ForecastWindow:
        if self._simulation_step_index >= len(self._signals):
            raise RuntimeError("episode input data is exhausted")
        return ForecastWindow(
            self._signals[
                self._simulation_step_index : self._simulation_step_index + max(1, steps)
            ]
        )

    def set_planned_start_ids(self, task_ids: tuple[str, ...]) -> None:
        self._planned_start_task_ids = tuple(task_ids)
        self._plan_declared = True

    @property
    def simulation_traces(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(trace) for trace in self._simulation_traces)

    def close(self) -> None:
        self._close_active_run("Gym environment closed before episode completion")

    def _seek_decision_state(
        self,
    ) -> tuple[dict[str, np.ndarray], int, dict[str, float], list[StepResult]]:
        auto_count = 0
        components = empty_reward_components()
        results: list[StepResult] = []
        while not self._episode_limit_reached():
            self._prepare_simulation_step()
            self._skip_non_decision_candidates()
            if self.current_candidate is not None:
                return self._encode_current_observation(), auto_count, components, results
            result = self._advance_physical_step()
            results.append(result)
            components = combine_reward_components(
                components, self.reward_function.compute_simulation_reward(result)
            )
            self._simulation_step_index += 1
            auto_count += 1
        return self._terminal_observation(), auto_count, components, results

    def _prepare_simulation_step(self) -> None:
        if self._system is None or self._task_provider is None:
            raise RuntimeError("Gym episode providers are not initialized")
        current = self._signals[self._simulation_step_index]
        arrivals = self._task_provider.arrivals_at(current.timestamp)
        self._task_observation = self._system.prepare_task_step(current, arrivals)
        self._candidates = order_task_candidates(
            self._task_observation.waiting_tasks, self.config.candidate_order
        )
        self._candidate_index = 0
        self._selected_task_ids = []
        self._deferred_task_ids = []
        self._auto_skipped_candidate_ids = []
        self._forced_resource_blocked_task_ids = []
        self._planned_start_task_ids = ()
        self._plan_declared = False
        self._current_gym_actions = []
        self._virtual_usage = self._task_observation.used_resources
        self._virtual_availability = self._task_observation.available_resources

    def _advance_physical_step(self) -> StepResult:
        if self._system is None:
            raise RuntimeError("Gym episode is not initialized")
        current = self._signals[self._simulation_step_index]
        result = self._system.step_with_task_decision(
            current,
            TaskSchedulingDecision(
                tuple(self._selected_task_ids),
                tuple(self._deferred_task_ids),
                True,
            ),
            ForecastWindow((current,)),
        )
        self._last_result = result
        executed = tuple(result.tasking.started_task_ids) if result.tasking else ()
        self._last_plan_difference_task_ids = (
            tuple(sorted(set(self._planned_start_task_ids) ^ set(executed)))
            if self._plan_declared
            else ()
        )
        self._simulation_traces.append(self._trace_record(result))
        return result

    def _trace_record(self, result: StepResult) -> dict[str, Any]:
        if self._task_observation is None or result.tasking is None:
            raise RuntimeError("task trace requires a prepared task step")
        outcomes = self._system.environment.task_outcomes() if self._system else ()
        waiting_after = tuple(
            sorted(item.task_id for item in outcomes if item.final_status is TaskStatus.WAITING)
        )
        running_after = tuple(
            sorted(item.task_id for item in outcomes if item.final_status is TaskStatus.RUNNING)
        )
        deferred = tuple(
            event.task_id
            for event in result.tasking.events
            if event.event_type.value == "deferred"
        )
        return {
            "simulation_step_index": self._simulation_step_index,
            "timestamp": self._task_observation.timestamp.isoformat(),
            "arrived_task_ids": result.tasking.arrived_task_ids,
            "waiting_task_ids_before": tuple(
                view.spec.task_id for view in self._task_observation.waiting_tasks
            ),
            "running_task_ids_before": tuple(
                view.spec.task_id for view in self._task_observation.running_tasks
            ),
            "scheduler_planned_start_ids": self._planned_start_task_ids,
            "candidate_order": tuple(view.spec.task_id for view in self._candidates),
            "gym_actions": tuple(self._current_gym_actions),
            "actual_started_task_ids": result.tasking.started_task_ids,
            "deferred_task_ids": deferred,
            "resource_blocked_task_ids": result.tasking.resource_blocked_task_ids,
            "forced_resource_blocked_task_ids": (
                result.tasking.forced_resource_blocked_task_ids
            ),
            "auto_skipped_candidate_ids": tuple(self._auto_skipped_candidate_ids),
            "available_resources": (
                self._task_observation.available_resources.cpu_cores,
                self._task_observation.available_resources.gpu_units,
                self._task_observation.available_resources.memory_gb,
            ),
            "workload_fraction": result.tasking.workload_fraction,
            "completed_task_ids": result.tasking.completed_task_ids,
            "newly_violated_task_ids": result.tasking.newly_sla_violated_task_ids,
            "waiting_task_ids_after": waiting_after,
            "running_task_ids_after": running_after,
            "energy_cost": result.accounting.energy_cost,
            "carbon_kg": result.accounting.carbon_kg,
            "grid_power_kw": result.physical.grid_power_kw,
            "temperature_c": result.physical.true_temperature_c,
        }

    def _skip_non_decision_candidates(self) -> None:
        while self._candidate_index < len(self._candidates):
            if self.current_action_mask.any():
                return
            candidate = self._candidates[self._candidate_index]
            task_id = candidate.spec.task_id
            self._auto_skipped_candidate_ids.append(task_id)
            self._forced_resource_blocked_task_ids.append(task_id)
            self._recent_auto_skipped_candidate_ids.append(task_id)
            self._recent_forced_resource_blocked_task_ids.append(task_id)
            self._candidate_index += 1

    def _allocate_virtual(self, candidate: TaskView) -> None:
        task = candidate.spec
        self._virtual_usage = ResourceUsage(
            self._virtual_usage.cpu_cores + task.cpu_cores,
            self._virtual_usage.gpu_units + task.gpu_units,
            self._virtual_usage.memory_gb + task.memory_gb,
        )
        capacity = self.scheduling_observation.total_resources
        self._virtual_availability = ResourceAvailability(
            capacity.cpu_cores - self._virtual_usage.cpu_cores,
            capacity.gpu_units - self._virtual_usage.gpu_units,
            capacity.memory_gb - self._virtual_usage.memory_gb,
        )

    def _encode_current_observation(self) -> dict[str, np.ndarray]:
        if self._system is None or self._task_observation is None:
            raise RuntimeError("Gym observation requested without active state")
        current = self._signals[self._simulation_step_index]
        domain_observation = self._system.environment.get_observation(current)
        return self._encoder.encode(
            domain_observation=domain_observation,
            task_observation=self._task_observation,
            candidate=self.current_candidate,
            waiting_tasks=self._candidates[self._candidate_index :],
            virtual_usage=self._virtual_usage,
            virtual_availability=self._virtual_availability,
            simulation_step_index=self._simulation_step_index,
            action_mask=self.current_action_mask,
            last_result=self._last_result,
        )

    def _terminal_observation(self) -> dict[str, np.ndarray]:
        if self._system is None or self._task_observation is None:
            sample = self.observation_space.sample()
            return {key: np.zeros_like(value) for key, value in sample.items()}
        current_index = min(self._simulation_step_index, len(self._signals) - 1)
        current = self._signals[current_index]
        domain_observation = self._system.environment.get_observation(current)
        return self._encoder.encode(
            domain_observation=domain_observation,
            task_observation=self._task_observation,
            candidate=None,
            waiting_tasks=(),
            virtual_usage=self._virtual_usage,
            virtual_availability=self._virtual_availability,
            simulation_step_index=min(
                self._simulation_step_index, self.config.max_simulation_steps
            ),
            action_mask=np.asarray([0, 0], dtype=np.int8),
            last_result=self._last_result,
        )

    def _build_info(
        self,
        *,
        candidate: TaskView | None,
        candidate_position: int,
        action_mask: tuple[int, int],
        invalid_action: bool,
        invalid_reason: str | None,
        physical_advanced: bool,
        auto_advanced: int,
        components: Mapping[str, float],
        results: list[StepResult],
        timestamp: str | None = None,
        simulation_step_index: int | None = None,
    ) -> dict[str, Any]:
        started = tuple(
            task_id
            for result in results
            if result.tasking is not None
            for task_id in result.tasking.started_task_ids
        )
        completed = tuple(
            task_id
            for result in results
            if result.tasking is not None
            for task_id in result.tasking.completed_task_ids
        )
        violated = tuple(
            task_id
            for result in results
            if result.tasking is not None
            for task_id in result.tasking.newly_sla_violated_task_ids
        )
        active_timestamp = timestamp or (
            self._task_observation.timestamp.isoformat()
            if self._task_observation is not None
            else None
        )
        current_metrics = (
            dict(self._system.current_metrics()) if self._system is not None else {}
        )
        return {
            "decision_step_index": self._decision_step_index,
            "simulation_step_index": (
                self._simulation_step_index
                if simulation_step_index is None
                else simulation_step_index
            ),
            "timestamp": active_timestamp,
            "candidate_task_id": candidate.spec.task_id if candidate else None,
            "candidate_position": candidate_position,
            "candidate_count": len(self._candidates),
            "candidate_order": self.config.candidate_order,
            "candidate_task_ids": tuple(view.spec.task_id for view in self._candidates),
            "action_mask": action_mask,
            "invalid_action": invalid_action,
            "invalid_action_reason": invalid_reason,
            "physical_step_advanced": physical_advanced,
            "auto_advanced_simulation_steps": auto_advanced,
            "started_task_ids": started,
            "completed_task_ids": completed,
            "newly_violated_task_ids": violated,
            "reward_components": dict(components),
            "current_metrics": current_metrics,
            "observation_clipping_count": self._encoder.clipping_count,
            "auto_skipped_candidate_ids": tuple(
                self._recent_auto_skipped_candidate_ids
            ),
            "auto_skipped_candidate_count": len(
                self._recent_auto_skipped_candidate_ids
            ),
            "forced_resource_blocked_task_ids": tuple(
                self._recent_forced_resource_blocked_task_ids
            ),
            "planned_start_task_ids": self._planned_start_task_ids,
            "executed_start_task_ids": started,
            "plan_difference_task_ids": self._last_plan_difference_task_ids,
        }

    def _episode_limit_reached(self) -> bool:
        return self._simulation_step_index >= min(
            self.config.max_simulation_steps, len(self._signals)
        )

    def _close_active_run(self, reason: str) -> None:
        if self._system is None:
            return
        if not self._finished:
            self._system.abort_run(reason)
        close = getattr(self._system.store, "close", None)
        if callable(close):
            close()
        self._system = None
        self._finished = True
