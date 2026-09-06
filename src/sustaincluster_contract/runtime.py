from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Literal

import pandas as pd


InformationMode = Literal["oracle", "deployable"]
DurationEstimateMode = Literal[
    "oracle",
    "declared_or_baseline",
    "externally_supplied",
]


@dataclass(frozen=True)
class RuntimeEstimateRequest:
    """Arrival-time fields that an external runtime estimator may access."""

    source_job_name: str
    arrival_time: Any
    cores_req: float
    gpu_req: float
    mem_req: float
    bandwidth_gb: float

    def __post_init__(self) -> None:
        if not self.source_job_name:
            raise ValueError("source_job_name cannot be empty")
        for name in ("cores_req", "gpu_req", "mem_req", "bandwidth_gb"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


ExternalDurationEstimator = Callable[[RuntimeEstimateRequest], float]


@dataclass(frozen=True)
class RuntimeInformationContract:
    """Separates simulator runtime truth from controller-visible estimates."""

    information_mode: InformationMode = "oracle"
    duration_estimate_mode: DurationEstimateMode = "oracle"
    baseline_estimated_duration_minutes: float = 60.0
    external_duration_estimator: ExternalDurationEstimator | None = None

    def __post_init__(self) -> None:
        if self.information_mode not in ("oracle", "deployable"):
            raise ValueError(f"Unsupported information_mode={self.information_mode!r}")
        if self.duration_estimate_mode not in (
            "oracle",
            "declared_or_baseline",
            "externally_supplied",
        ):
            raise ValueError(
                "Unsupported duration_estimate_mode="
                f"{self.duration_estimate_mode!r}"
            )
        baseline = float(self.baseline_estimated_duration_minutes)
        if not math.isfinite(baseline) or baseline <= 0:
            raise ValueError(
                "baseline_estimated_duration_minutes must be finite and positive"
            )
        if (
            self.information_mode == "deployable"
            and self.duration_estimate_mode == "oracle"
        ):
            raise ValueError("deployable mode cannot use true duration as its estimate")
        if (
            self.duration_estimate_mode == "externally_supplied"
            and self.external_duration_estimator is None
        ):
            raise ValueError(
                "externally_supplied mode requires external_duration_estimator"
            )

    @classmethod
    def for_mode(
        cls,
        information_mode: InformationMode,
        *,
        duration_estimate_mode: DurationEstimateMode | None = None,
        baseline_estimated_duration_minutes: float = 60.0,
        external_duration_estimator: ExternalDurationEstimator | None = None,
    ) -> "RuntimeInformationContract":
        if duration_estimate_mode is None:
            duration_estimate_mode = (
                "oracle"
                if information_mode == "oracle"
                else "declared_or_baseline"
            )
        return cls(
            information_mode=information_mode,
            duration_estimate_mode=duration_estimate_mode,
            baseline_estimated_duration_minutes=baseline_estimated_duration_minutes,
            external_duration_estimator=external_duration_estimator,
        )

    @property
    def estimate_source(self) -> str:
        if self.duration_estimate_mode == "oracle":
            return "TRACE_TRUE_DURATION_ORACLE"
        if self.duration_estimate_mode == "declared_or_baseline":
            return "PLACEHOLDER_CONSTANT_BASELINE"
        return "EXTERNAL_RUNTIME_ESTIMATOR"

    def estimate_duration(
        self,
        request: RuntimeEstimateRequest,
        true_duration: float,
    ) -> float:
        if self.duration_estimate_mode == "oracle":
            value = true_duration
        elif self.duration_estimate_mode == "declared_or_baseline":
            value = self.baseline_estimated_duration_minutes
        else:
            assert self.external_duration_estimator is not None
            value = self.external_duration_estimator(request)
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                "Estimated duration for "
                f"{request.source_job_name!r} must be positive"
            )
        return value


def true_duration_minutes(task: Any) -> float:
    value = getattr(task, "true_duration", getattr(task, "duration", None))
    if value is None:
        raise AttributeError("Task has neither true_duration nor duration")
    return _positive_duration(value, "true_duration")


def estimated_duration_minutes(task: Any) -> float:
    if not hasattr(task, "estimated_duration"):
        raise AttributeError(
            "Task is missing estimated_duration; bind the information contract "
            "before using deployable controller state"
        )
    return _positive_duration(task.estimated_duration, "estimated_duration")


def controller_duration_minutes(
    task: Any,
    information_mode: InformationMode,
) -> float:
    if information_mode == "oracle":
        return true_duration_minutes(task)
    if information_mode == "deployable":
        return estimated_duration_minutes(task)
    raise ValueError(f"Unsupported information_mode={information_mode!r}")


def controller_finish_time(task: Any, information_mode: InformationMode) -> Any:
    if information_mode == "oracle":
        finish_time = getattr(task, "finish_time", None)
        if finish_time is None:
            raise ValueError(f"Task {task.job_name!r} has no finish_time")
        return finish_time
    start_time = getattr(task, "start_time", None)
    if start_time is None:
        raise ValueError(
            f"Running task {task.job_name!r} has no start_time for estimated release"
        )
    return start_time + pd.Timedelta(minutes=estimated_duration_minutes(task))


def attach_runtime_information(
    task: Any,
    *,
    source_job_name: str,
    true_duration: float,
    estimated_duration: float,
    contract: RuntimeInformationContract,
) -> Any:
    true_value = _positive_duration(true_duration, "true_duration")
    estimated_value = _positive_duration(
        estimated_duration, "estimated_duration"
    )
    task.source_job_name = str(source_job_name)
    task.true_duration = true_value
    task.estimated_duration = estimated_value
    task.duration = true_value
    task.runtime_information_mode = contract.information_mode
    task.estimated_duration_source = contract.estimate_source
    task.duration_source_type = "TRACE_DERIVED"
    deadline_duration = (
        true_value
        if contract.information_mode == "oracle"
        else estimated_value
    )
    task.sla_deadline = task.arrival_time + pd.Timedelta(
        minutes=float(task.sla_multiplier) * deadline_duration
    )
    task.sla_deadline_basis = (
        "true_duration_oracle"
        if contract.information_mode == "oracle"
        else "estimated_duration_at_arrival"
    )
    return task


def _positive_duration(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result
