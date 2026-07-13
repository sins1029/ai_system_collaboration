from __future__ import annotations

from typing import Mapping, Protocol

from datacenter_env.contracts.inputs import ExogenousInput, ForecastWindow
from datacenter_env.contracts.observations import DataCenterObservation
from datacenter_env.contracts.results import (
    AgentDecisionRecord,
    ControllerDecision,
    RunHandle,
    RunMetadata,
    RunSummary,
    StepResult,
)
from datacenter_env.contracts.tasks import (
    TaskArrivalBatch,
    TaskSchedulingDecision,
    TaskSchedulingObservation,
)


class DataCenterController(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def max_forecast_steps(self) -> int: ...

    def reset(self, seed: int | None = None) -> None: ...

    def act(
        self,
        observation: DataCenterObservation,
        forecast_window: ForecastWindow | None = None,
    ) -> ControllerDecision: ...


class TaskScheduler(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def max_forecast_steps(self) -> int: ...

    def reset(self, seed: int | None = None) -> None: ...

    def schedule(
        self,
        observation: TaskSchedulingObservation,
        forecast_window: ForecastWindow | None = None,
    ) -> TaskSchedulingDecision: ...


class TaskArrivalProvider(Protocol):
    def arrivals_at(self, timestamp) -> TaskArrivalBatch: ...


class RunStore(Protocol):
    def initialize(self) -> None: ...

    def create_run(self, metadata: RunMetadata) -> RunHandle: ...

    def append_input(self, run_id: int | None, external_input: ExogenousInput) -> None: ...

    def append_step(self, run_id: int | None, step_result: StepResult) -> None: ...

    def append_transition(
        self,
        run_id: int | None,
        external_input: ExogenousInput,
        step_result: StepResult,
    ) -> None: ...

    def append_agent_decision(
        self, run_id: int | None, decision: AgentDecisionRecord
    ) -> None: ...

    def append_gym_episode_summary(
        self, run_id: int | None, summary: Mapping[str, object]
    ) -> None: ...

    def finish_run(self, run_id: int | None, summary: RunSummary) -> None: ...

    def fail_run(self, run_id: int | None, error: Exception) -> None: ...
