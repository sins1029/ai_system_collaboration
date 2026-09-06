from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

import numpy as np
import pandas as pd


ForecastProviderMode = Literal["oracle", "persistence", "transformer"]


@dataclass(frozen=True)
class WorkloadForecastPoint:
    horizon_step: int
    horizon_minutes: int
    forecast_timestamp: str
    global_task_count: float
    global_cpu_demand: float
    global_gpu_demand: float
    global_memory_demand: float

    def resources(self) -> tuple[float, float, float]:
        return (
            self.global_cpu_demand,
            self.global_gpu_demand,
            self.global_memory_demand,
        )


@dataclass(frozen=True)
class ForecastBundle:
    provider: ForecastProviderMode
    issued_at_timestamp: str
    points: tuple[WorkloadForecastPoint, ...]
    forecast_history_available: bool
    persistence_fallback_used: bool
    inference_ms: float
    negative_prediction_clip_count: int
    clip_count_by_target: tuple[int, int, int, int]

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                (
                    point.global_task_count,
                    point.global_cpu_demand,
                    point.global_gpu_demand,
                    point.global_memory_demand,
                )
                for point in self.points
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class ForecastRequest:
    current_timestamp: pd.Timestamp
    workload_history: np.ndarray | None
    current_workload: np.ndarray
    oracle_future_workload: np.ndarray | None = None


class WorkloadForecastProvider(Protocol):
    mode: ForecastProviderMode

    def forecast(self, request: ForecastRequest) -> ForecastBundle:
        ...


class ForecastTraceSource:
    """Read-only bridge between SustainCluster's repeated outer time and Dataset v1."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        history_length: int = 96,
        horizon: int = 4,
        resolution_minutes: int = 15,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.history_length = int(history_length)
        self.horizon = int(horizon)
        self.resolution_minutes = int(resolution_minutes)
        if self.history_length <= 0 or self.horizon <= 0:
            raise ValueError("history_length and horizon must be positive")
        table = pd.read_parquet(
            self.dataset_root / "dataset/workload_15min.parquet"
        )
        table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True)
        self._table = table.set_index("timestamp", drop=False).sort_index()
        self.feature_names = (
            "new_task_count",
            "arriving_cpu_demand",
            "arriving_gpu_demand",
            "arriving_memory_demand",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        )
        self.target_names = self.feature_names[:4]
        self._dataset_start = pd.Timestamp("1970-01-26T00:00:00Z")
        self._cycle = pd.Timedelta(days=49)
        self._outer_start = pd.Timestamp("2020-01-01T00:00:00Z")

    def align_environment_timestamp(
        self, timestamp: str | pd.Timestamp
    ) -> pd.Timestamp:
        value = pd.Timestamp(timestamp)
        value = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        outer = value.replace(year=2020)
        offset = (outer - self._outer_start) % self._cycle
        aligned = self._dataset_start + offset
        if aligned.minute % self.resolution_minutes or aligned.second:
            raise ValueError("environment timestamp is not 15-minute aligned")
        return aligned

    def history(
        self, current_timestamp: str | pd.Timestamp
    ) -> tuple[np.ndarray | None, pd.Timestamp]:
        aligned = self.align_environment_timestamp(current_timestamp)
        start = aligned - pd.Timedelta(
            minutes=self.resolution_minutes * (self.history_length - 1)
        )
        expected = pd.date_range(
            start,
            aligned,
            periods=self.history_length,
            tz="UTC",
        )
        selected = self._table.reindex(expected)
        if len(selected) != self.history_length or selected["timestamp"].isna().any():
            return None, aligned
        return selected.loc[:, self.feature_names].to_numpy(np.float32), aligned

    def oracle_future(
        self, current_timestamp: str | pd.Timestamp
    ) -> tuple[np.ndarray, tuple[pd.Timestamp, ...], pd.Timestamp]:
        aligned = self.align_environment_timestamp(current_timestamp)
        timestamps = tuple(
            aligned + pd.Timedelta(minutes=self.resolution_minutes * step)
            for step in range(1, self.horizon + 1)
        )
        selected = self._table.reindex(pd.DatetimeIndex(timestamps))
        if selected["timestamp"].isna().any():
            raise ValueError("oracle future leaves the frozen unique-cycle timeline")
        values = selected.loc[:, self.target_names].to_numpy(np.float64)
        return values, timestamps, aligned

    def request(
        self,
        current_timestamp: str | pd.Timestamp,
        *,
        include_oracle_future: bool,
    ) -> ForecastRequest:
        history, aligned = self.history(current_timestamp)
        if history is None:
            raise ValueError("no legal current workload is available for this timestamp")
        oracle = None
        if include_oracle_future:
            oracle, _, _ = self.oracle_future(current_timestamp)
        return ForecastRequest(
            current_timestamp=aligned,
            workload_history=history,
            current_workload=np.asarray(history[-1, :4], dtype=np.float64),
            oracle_future_workload=oracle,
        )


def build_bundle(
    *,
    provider: ForecastProviderMode,
    current_timestamp: pd.Timestamp,
    values: np.ndarray,
    history_available: bool,
    fallback_used: bool,
    inference_ms: float = 0.0,
    clip_counts: Sequence[int] = (0, 0, 0, 0),
) -> ForecastBundle:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("forecast values must be finite [4,4]")
    timestamp = pd.Timestamp(current_timestamp)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    points = tuple(
        WorkloadForecastPoint(
            horizon_step=step,
            horizon_minutes=step * 15,
            forecast_timestamp=(timestamp + pd.Timedelta(minutes=step * 15)).isoformat(),
            global_task_count=float(matrix[step - 1, 0]),
            global_cpu_demand=float(matrix[step - 1, 1]),
            global_gpu_demand=float(matrix[step - 1, 2]),
            global_memory_demand=float(matrix[step - 1, 3]),
        )
        for step in range(1, 5)
    )
    counts = tuple(int(value) for value in clip_counts)
    if len(counts) != 4 or any(value < 0 for value in counts):
        raise ValueError("clip_counts must contain four nonnegative integers")
    return ForecastBundle(
        provider=provider,
        issued_at_timestamp=timestamp.isoformat(),
        points=points,
        forecast_history_available=history_available,
        persistence_fallback_used=fallback_used,
        inference_ms=float(inference_ms),
        negative_prediction_clip_count=sum(counts),
        clip_count_by_target=counts,
    )


def clip_nonnegative(values: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("forecast values must be finite [4,4]")
    counts = tuple(int((matrix[:, index] < 0).sum()) for index in range(4))
    return np.maximum(matrix, 0.0), counts


class PersistenceWorkloadForecastProvider:
    mode: ForecastProviderMode = "persistence"

    def forecast(self, request: ForecastRequest) -> ForecastBundle:
        current = np.asarray(request.current_workload, dtype=np.float64)
        if current.shape != (4,) or not np.isfinite(current).all():
            raise ValueError("current_workload must be finite [4]")
        values = np.repeat(np.maximum(current, 0.0)[None, :], 4, axis=0)
        return build_bundle(
            provider=self.mode,
            current_timestamp=request.current_timestamp,
            values=values,
            history_available=request.workload_history is not None,
            fallback_used=False,
        )


class OracleWorkloadForecastProvider:
    mode: ForecastProviderMode = "oracle"

    def forecast(self, request: ForecastRequest) -> ForecastBundle:
        if request.oracle_future_workload is None:
            raise ValueError("oracle provider requires true future workload")
        warnings.warn(
            "Oracle workload forecast is enabled; planning is non-deployable.",
            RuntimeWarning,
            stacklevel=2,
        )
        values, counts = clip_nonnegative(request.oracle_future_workload)
        return build_bundle(
            provider=self.mode,
            current_timestamp=request.current_timestamp,
            values=values,
            history_available=request.workload_history is not None,
            fallback_used=False,
            clip_counts=counts,
        )
