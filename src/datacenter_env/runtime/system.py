from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from datacenter_env.config import DataCenterSystemConfig
from datacenter_env.contracts import (
    DataCenterObservation,
    ExogenousInput,
    ForecastWindow,
    RunHandle,
    RunMetadata,
    RunStore,
    RunSummary,
    StepResult,
)
from datacenter_env.control import build_controller
from datacenter_env.core import DataCenterEnvironment
from datacenter_env.evaluation import MetricAggregator
from datacenter_env.exceptions import RunStateError, TimeAlignmentError
from datacenter_env.storage import NullRunStore, SQLiteRunStore
from datacenter_env.version import __version__


class DataCenterSystem:
    """Facade combining one environment, one controller, one store, and metrics."""

    def __init__(self, config: DataCenterSystemConfig, store: RunStore):
        self.config = config
        self.environment = DataCenterEnvironment.from_config(config)
        self.controller = build_controller(config)
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
        self.aggregator.reset()
        self._steps.clear()
        return self.environment.reset(seed, initial_input)

    def step(
        self,
        current_input: ExogenousInput,
        forecast_window: ForecastWindow | None = None,
    ) -> StepResult:
        window = self._validate_forecast(current_input, forecast_window)
        observation = self.environment.get_observation(current_input)
        try:
            decision = self.controller.act(observation, window)
            result = self.environment.step(
                action=decision.action,
                current_input=current_input,
                controller_decision=decision,
            )
            self.aggregator.update(result)
            self._steps.append(result)
            if self._run_handle is not None:
                self.store.append_input(self._run_handle.run_id, current_input)
                self.store.append_step(self._run_handle.run_id, result)
            return result
        except Exception as error:
            if self._run_handle is not None:
                handle = self._run_handle
                self._run_handle = None
                self.store.fail_run(handle.run_id, error)
            raise

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
            metrics=self.aggregator.summarize(),
            steps=tuple(self._steps),
            records=tuple(self.aggregator.frame().to_dict(orient="records")),
        )
        self.store.finish_run(handle.run_id, summary)
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
        if len(window) > self.controller.max_forecast_steps:
            raise TimeAlignmentError(
                f"controller {self.controller.name} permits at most "
                f"{self.controller.max_forecast_steps} forecast steps"
            )
        return window
