from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


H60_TARGET_NAMES = (
    "new_task_count",
    "arriving_cpu_demand",
    "arriving_gpu_demand",
    "arriving_memory_demand",
)
H60_CALENDAR_FEATURE_NAMES = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)
H60_FEATURE_NAMES = H60_TARGET_NAMES + H60_CALENDAR_FEATURE_NAMES


@dataclass(frozen=True)
class H60ForecastDatasetConfig:
    history_length: int = 96
    target_offset_steps: int = 4
    resolution_minutes: int = 15

    def __post_init__(self) -> None:
        if self.history_length <= 0:
            raise ValueError("history_length must be positive")
        if self.target_offset_steps != 4:
            raise ValueError("H60 target must be current_step + 4")
        if self.resolution_minutes != 15:
            raise ValueError("H60 dataset requires 15-minute resolution")


@dataclass(frozen=True)
class H60WindowDataset:
    X: np.ndarray
    Y: np.ndarray
    X_raw: np.ndarray
    Y_raw: np.ndarray
    current_step: np.ndarray
    target_step: np.ndarray
    history_end_timestamp: np.ndarray
    target_timestamp: np.ndarray
    split: str

    @property
    def sample_count(self) -> int:
        return int(self.X.shape[0])


def build_spot_h60_time_series(
    states: pd.DataFrame,
    tasks: pd.DataFrame,
) -> pd.DataFrame:
    required_states = {"step", "timestamp_utc", "split"}
    required_tasks = {
        "task_id",
        "arrival_step",
        "cpu_cores",
        "gpu_units",
        "memory_gb",
    }
    if not required_states.issubset(states.columns):
        missing = sorted(required_states - set(states.columns))
        raise ValueError(f"states missing columns: {missing}")
    if not required_tasks.issubset(tasks.columns):
        missing = sorted(required_tasks - set(tasks.columns))
        raise ValueError(f"tasks missing columns: {missing}")

    state_rows = states.loc[:, sorted(required_states)].copy()
    state_rows.sort_values("step", inplace=True)
    if state_rows["step"].duplicated().any():
        raise ValueError("states contain duplicate steps")
    expected_steps = np.arange(len(state_rows), dtype=np.int64)
    if not np.array_equal(state_rows["step"].to_numpy(dtype=np.int64), expected_steps):
        raise ValueError("states must cover a continuous zero-based timeline")
    if tasks["task_id"].duplicated().any():
        raise ValueError("Spot v3 task decisions must contain one row per unique task")

    grouped = (
        tasks.groupby("arrival_step", sort=True)
        .agg(
            new_task_count=("task_id", "size"),
            arriving_cpu_demand=("cpu_cores", "sum"),
            arriving_gpu_demand=("gpu_units", "sum"),
            arriving_memory_demand=("memory_gb", "sum"),
        )
        .reindex(expected_steps, fill_value=0.0)
    )
    timeline = state_rows.reset_index(drop=True)
    for name in H60_TARGET_NAMES:
        timeline[name] = grouped[name].to_numpy(dtype=np.float64)
    timeline["new_task_count"] = timeline["new_task_count"].astype(np.int64)
    timeline["timestamp"] = pd.to_datetime(timeline.pop("timestamp_utc"), utc=True)

    hour = timeline["timestamp"].dt.hour + timeline["timestamp"].dt.minute / 60.0
    day_of_week = timeline["timestamp"].dt.dayofweek
    timeline["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    timeline["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    timeline["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    timeline["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    return timeline.loc[
        :,
        ("step", "timestamp", "split") + H60_FEATURE_NAMES,
    ]


def fit_h60_train_scaler(timeline: pd.DataFrame) -> dict[str, Any]:
    train = timeline[timeline["split"] == "train"].copy()
    if train.empty:
        raise ValueError("train split is empty")
    parameters: dict[str, dict[str, Any]] = {}
    for name in H60_TARGET_NAMES:
        values = train[name].to_numpy(dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        zero_std = math.isclose(std, 0.0, abs_tol=0.0)
        parameters[name] = {
            "mean": mean,
            "std": std,
            "scale": 1.0 if zero_std else std,
            "zero_std": zero_std,
        }
    return {
        "method": "standard_score",
        "fitted_on": "TRAIN_ONLY",
        "fit_rows": int(len(train)),
        "fit_step_start": int(train["step"].iloc[0]),
        "fit_step_end": int(train["step"].iloc[-1]),
        "parameters": parameters,
    }


def build_h60_windows(
    timeline: pd.DataFrame,
    split: str,
    scaler: Mapping[str, Any],
    *,
    config: H60ForecastDatasetConfig | None = None,
) -> H60WindowDataset:
    config = config or H60ForecastDatasetConfig()
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split={split!r}")
    selected = timeline[timeline["split"] == split].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"{split} split is empty")
    steps = selected["step"].to_numpy(dtype=np.int64)
    if len(steps) > 1 and not np.all(np.diff(steps) == 1):
        raise ValueError(f"{split} split is not a continuous timeline")

    first_current = config.history_length - 1
    last_current_exclusive = len(selected) - config.target_offset_steps
    if last_current_exclusive <= first_current:
        raise ValueError(f"{split} split is too short for H60 windows")
    current_indices = np.arange(
        first_current, last_current_exclusive, dtype=np.int64
    )
    target_indices = current_indices + config.target_offset_steps

    features = selected.loc[:, H60_FEATURE_NAMES].to_numpy(dtype=np.float32)
    targets = selected.loc[:, H60_TARGET_NAMES].to_numpy(dtype=np.float32)
    X_raw = np.stack(
        [
            features[index - config.history_length + 1 : index + 1]
            for index in current_indices
        ]
    )
    Y_raw = targets[target_indices, None, :]
    X = X_raw.copy()
    Y = Y_raw.copy()
    for index, name in enumerate(H60_TARGET_NAMES):
        parameter = scaler["parameters"][name]
        mean = float(parameter["mean"])
        scale = float(parameter["scale"])
        X[:, :, index] = (X[:, :, index] - mean) / scale
        Y[:, :, index] = (Y[:, :, index] - mean) / scale

    timestamps = (
        pd.to_datetime(selected["timestamp"], utc=True)
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    current_timestamp = timestamps[current_indices]
    target_timestamp = timestamps[target_indices]
    expected_delta = np.timedelta64(
        config.target_offset_steps * config.resolution_minutes, "m"
    )
    if not np.all(target_timestamp - current_timestamp == expected_delta):
        raise ValueError("H60 target timestamp is not exactly current + 60 minutes")

    return H60WindowDataset(
        X=X,
        Y=Y,
        X_raw=X_raw,
        Y_raw=Y_raw,
        current_step=steps[current_indices],
        target_step=steps[target_indices],
        history_end_timestamp=current_timestamp,
        target_timestamp=target_timestamp,
        split=split,
    )


def persistence_h60(window: H60WindowDataset) -> np.ndarray:
    return window.X_raw[:, -1:, : len(H60_TARGET_NAMES)].copy()


def h60_mae_rows(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    split: str,
    model: str,
) -> list[dict[str, Any]]:
    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    if truth_array.shape != prediction_array.shape:
        raise ValueError("truth and prediction shapes must match")
    if truth_array.ndim != 3 or truth_array.shape[1:] != (1, 4):
        raise ValueError("H60 arrays must use [samples,1,4] layout")
    errors = np.abs(prediction_array - truth_array)
    return [
        {
            "split": split,
            "model": model,
            "target": name,
            "mae": float(errors[:, 0, index].mean()),
            "samples": int(len(errors)),
        }
        for index, name in enumerate(H60_TARGET_NAMES)
    ]


def h60_feature_schema() -> dict[str, Any]:
    return {
        "X": [
            {
                "index": index,
                "name": name,
                "role": (
                    "OBSERVED_HISTORY"
                    if name in H60_TARGET_NAMES
                    else "KNOWN_IN_ADVANCE"
                ),
                "normalization": (
                    "TRAIN_STANDARD_SCORE"
                    if name in H60_TARGET_NAMES
                    else "NONE"
                ),
            }
            for index, name in enumerate(H60_FEATURE_NAMES)
        ],
        "Y": [
            {
                "index": index,
                "name": name,
                "role": "FORECAST_TARGET_AT_EXACT_T_PLUS_60",
                "normalization": "TRAIN_STANDARD_SCORE",
            }
            for index, name in enumerate(H60_TARGET_NAMES)
        ],
        "feature_tensor_layout": "samples,history,feature",
        "target_tensor_layout": "samples,1,target",
        "history_steps": 96,
        "target_offset_steps": 4,
        "intermediate_targets_used": False,
    }
