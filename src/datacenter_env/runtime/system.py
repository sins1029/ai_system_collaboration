from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from datacenter_env.config import DataCenterSystemConfig
from datacenter_env.contracts import (
    AgentDecisionRecord,
    DataCenterAction,
    DataCenterObservation,
    ExogenousInput,
    ForecastWindow,
    RunHandle,
    RunMetadata,
    RunStore,
    RunSummary,
    StepResult,
    TaskArrivalBatch,
    TaskSchedulingDecision,
    TaskSchedulingObservation,
)
from datacenter_env.control import build_controller
from datacenter_env.core import DataCenterEnvironment
from datacenter_env.evaluation import MetricAggregator
from datacenter_env.exceptions import RunStateError, TimeAlignmentError
from datacenter_env.storage import NullRunStore, SQLiteRunStore
from datacenter_env.tasking import build_task_scheduler
from datacenter_env.version import __version__


class DataCenterSystem:
    """Facade combining one environment, one controller, one store, and metrics."""

    def __init__(self, config: DataCenterSystemConfig, store: RunStore):
        self.config = config
        self.environment = DataCenterEnvironment.from_config(config)
        self.controller = build_controller(config)
        self.scheduler = (
            build_task_scheduler(config) if config.workload_mode == "task_queue" else None
        )
        self.store = store
        if hasattr(store, "bind_config"):
            store.bind_config(config)  # type: ignore[attr-defined]
        self.store.initialize()
        self.aggregator = MetricAggregator(config)
        self._run_handle: RunHandle | None = None
        self._metadata: RunMetadata | None = None
        self._steps: list[StepResult] = []
        self._default_seed: int | None = None

    @classmethod
    def from_config(
        cls,
        config: DataCenterSystemConfig | Mapping[str, Any],
        store: RunStore | None = None,
    ) -> "DataCenterSystem":
        normalized = DataCenterSystemConfig.coerce(config)
        if store is None:
            store = (
                SQLiteRunStore(normalized.database_path)
                if normalized.database_path
                else NullRunStore()
            )
        return cls(normalized, store)

    def start_run(
        self,
        metadata: RunMetadata,
        seed: int | None = None,
    ) -> RunHandle:
        if self._run_handle is not None:
            raise RunStateError("a run is already active")
        active_seed = metadata.seed if seed is None else int(seed)
        snapshot = metadata.config_snapshot or self.config.to_dict()
        normalized = replace(
            metadata,
            seed=active_seed,
            config_snapshot=snapshot,
            package_version=metadata.package_version or __version__,
            scheduler_name=metadata.scheduler_name or (
                self.scheduler.name if self.scheduler is not None else None
            ),
            cooling_controller_name=(
                metadata.cooling_controller_name or self.config.cooling_controller_name
            ),
        )
        self.reset(seed=active_seed)
        handle = self.store.create_run(normalized)
        self._run_handle = handle
        self._metadata = normalized
        return handle

    def reset(
        self,
        seed: int | None = None,
        initial_input: ExogenousInput | None = None,
    ) -> DataCenterObservation:
        if self._run_handle is not None:
            raise RunStateError("cannot reset while a run is active")
        self._default_seed = seed
        self.controller.reset(seed)
        if self.scheduler is not None:
            self.scheduler.reset(seed)
        self.aggregator.reset()
        self._steps.clear()
        return self.environment.reset(seed, initial_input)

    def step(
        self,
        current_input: ExogenousInput,
        forecast_window: ForecastWindow | None = None,
        task_arrivals: TaskArrivalBatch | None = None,
    ) -> StepResult:
        try:
            window = self._validate_forecast(current_input, forecast_window)
            task_decision = None
            if self.scheduler is not None:
                if task_arrivals is None:
                    raise TimeAlignmentError(
                        "task_queue mode requires a TaskArrivalBatch for every step"
                    )
                task_observation = self.environment.prepare_task_step(
                    current_input, task_arrivals
                )
                scheduler_window = self._slice_window(
                    window, self.scheduler.max_forecast_steps
                )
                task_decision = self.scheduler.schedule(
                    task_observation, scheduler_window
                )
            elif task_arrivals is not None:
                raise TimeAlignmentError(
                    "task arrivals are not accepted in legacy_aggregate mode"
                )
            return self._execute_prepared_task_step(
                current_input, window, task_decision
            )
        except Exception as error:
            if self._run_handle is not None:
                handle = self._run_handle
                self._run_handle = None
                self.store.fail_run(handle.run_id, error)
            raise

    def prepare_task_step(
        self,
        current_input: ExogenousInput,
        task_arrivals: TaskArrivalBatch,
    ) -> TaskSchedulingObservation:
        if self.scheduler is None:
            raise TimeAlignmentError("external task decisions require task_queue mode")
        return self.environment.prepare_task_step(current_input, task_arrivals)

    def step_with_task_decision(
        self,
        current_input: ExogenousInput,
        task_decision: TaskSchedulingDecision,
        forecast_window: ForecastWindow | None = None,
    ) -> StepResult:
        try:
            window = self._validate_forecast(current_input, forecast_window)
            return self._execute_prepared_task_step(
                current_input, window, task_decision
            )
        except Exception as error:
            if self._run_handle is not None:
                handle = self._run_handle
                self._run_handle = None
                self.store.fail_run(handle.run_id, error)
            raise

    @property
    def run_id(self) -> int | None:
        return self._run_handle.run_id if self._run_handle is not None else None

    def append_agent_decision(self, decision: AgentDecisionRecord) -> None:
        if self._run_handle is not None:
            try:
                self.store.append_agent_decision(self._run_handle.run_id, decision)
            except Exception as error:
                self._fail_active_run(error)
                raise

    def append_gym_episode_summary(self, summary: Mapping[str, object]) -> None:
        if self._run_handle is not None:
            try:
                self.store.append_gym_episode_summary(self._run_handle.run_id, summary)
            except Exception as error:
                self._fail_active_run(error)
                raise

    def current_metrics(self) -> Mapping[str, float | None]:
        return self.aggregator.summarize(self.environment.task_outcomes())

    def abort_run(self, reason: str = "run aborted before completion") -> None:
        if self._run_handle is None:
            return
        handle = self._run_handle
        self._run_handle = None
        self._metadata = None
        self.store.fail_run(handle.run_id, RunStateError(reason))

    def _fail_active_run(self, error: Exception) -> None:
        if self._run_handle is None:
            return
        handle = self._run_handle
        self._run_handle = None
        self._metadata = None
        self.store.fail_run(handle.run_id, error)

    def _execute_prepared_task_step(
        self,
        current_input: ExogenousInput,
        window: ForecastWindow,
        task_decision: TaskSchedulingDecision | None,
    ) -> StepResult:
        if task_decision is not None:
            self.environment.apply_task_decision(
                current_input.timestamp, task_decision
            )
        observation = self.environment.get_observation(current_input)
        cooling_window = self._slice_window(
            window, self.controller.max_forecast_steps
        )
        decision = self.controller.act(observation, cooling_window)
        action = DataCenterAction(
            cooling_target_kw=decision.action.cooling_target_kw,
            task_decision=task_decision,
        )
        result = self.environment.step(
            action=action,
            current_input=current_input,
            controller_decision=decision,
        )
        self.aggregator.update(result)
        self._steps.append(result)
        if self._run_handle is not None:
            self.store.append_transition(
                self._run_handle.run_id, current_input, result
            )
        return result

    def finish_run(self) -> RunSummary:
        if self._run_handle is None or self._metadata is None:
            raise RunStateError("no active run to finish")
        handle = self._run_handle
        metadata = self._metadata
        summary = RunSummary(
            status="completed",
            run_id=handle.run_id,
            controller_name=metadata.controller_name,
            condition_name=metadata.condition_name,
            metrics=self.aggregator.summarize(self.environment.task_outcomes()),
            steps=tuple(self._steps),
            records=tuple(self.aggregator.frame().to_dict(orient="records")),
            task_outcomes=self.environment.task_outcomes(),
        )
        try:
            self.store.finish_run(handle.run_id, summary)
        except Exception as error:
            try:
                self.store.fail_run(handle.run_id, error)
            finally:
                self._run_handle = None
                self._metadata = None
            raise
        self._run_handle = None
        self._metadata = None
        return summary

    def _validate_forecast(
        self,
        current_input: ExogenousInput,
        forecast_window: ForecastWindow | None,
    ) -> ForecastWindow:
        window = forecast_window or ForecastWindow((current_input,))
        if window.steps[0].timestamp != current_input.timestamp:
            raise TimeAlignmentError(
                "forecast window must begin at the current input timestamp"
            )
        max_steps = self.controller.max_forecast_steps
        if self.scheduler is not None:
            max_steps = max(max_steps, self.scheduler.max_forecast_steps)
        if len(window) > max_steps:
            raise TimeAlignmentError(
                f"active control stack permits at most {max_steps} forecast steps"
            )
        return window

    @staticmethod
    def _slice_window(window: ForecastWindow, length: int) -> ForecastWindow:
        return ForecastWindow(tuple(window.steps[:length]))
