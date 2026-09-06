from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from sustaincluster_contract.workload import WORKLOAD_FIELD_INDEX


TARGET_NAMES = (
    "new_task_count",
    "arriving_cpu_demand",
    "arriving_gpu_demand",
    "arriving_memory_demand",
)
CALENDAR_FEATURE_NAMES = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)
FEATURE_NAMES = TARGET_NAMES + CALENDAR_FEATURE_NAMES


@dataclass(frozen=True)
class ForecastDatasetConfig:
    history_length: int = 96
    horizon: int = 4
    resolution_minutes: int = 15
    task_scale: float = 5.0

    def __post_init__(self) -> None:
        for name in ("history_length", "horizon", "resolution_minutes"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(float(self.task_scale)) or self.task_scale <= 0:
            raise ValueError("task_scale must be finite and positive")


@dataclass(frozen=True)
class WindowDataset:
    X: np.ndarray
    Y: np.ndarray
    X_raw: np.ndarray
    Y_raw: np.ndarray
    history_end_timestamp: np.ndarray
    target_start_timestamp: np.ndarray
    target_end_timestamp: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.X.shape[0])


def extract_first_unique_cycle(
    full_year: pd.DataFrame,
    *,
    cycle_days: int = 49,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select and verify the first cycle from the repeated parent artifact."""
    required = {"interval_15m", "tasks_matrix"}
    if not required.issubset(full_year.columns):
        raise ValueError(f"source is missing columns {sorted(required - set(full_year))}")
    ordered = full_year.loc[:, ["interval_15m", "tasks_matrix"]].copy()
    ordered["interval_15m"] = pd.to_datetime(
        ordered["interval_15m"], utc=True
    )
    ordered.sort_values("interval_15m", inplace=True)
    if ordered["interval_15m"].duplicated().any():
        raise ValueError("full-year source contains duplicate outer timestamps")

    start = ordered["interval_15m"].iloc[0]
    cycle = pd.Timedelta(days=cycle_days)
    cycle_end = start + cycle
    first = ordered[
        (ordered["interval_15m"] >= start)
        & (ordered["interval_15m"] < cycle_end)
    ].copy()
    if first.empty:
        raise ValueError("first source cycle is empty")

    base: dict[pd.Timedelta, np.ndarray] = {}
    for row in first.itertuples(index=False):
        base[row.interval_15m - start] = np.asarray(row.tasks_matrix)

    block_results: list[dict[str, Any]] = []
    elapsed = ordered["interval_15m"] - start
    block_ids = (elapsed // cycle).astype(int)
    for block_id, block in ordered.groupby(block_ids, sort=True):
        block_start = start + int(block_id) * cycle
        exact = True
        offsets: set[pd.Timedelta] = set()
        for row in block.itertuples(index=False):
            offset = row.interval_15m - block_start
            offsets.add(offset)
            expected = base.get(offset)
            current = np.asarray(row.tasks_matrix)
            matrices_equal = expected is not None and (
                np.array_equal(current, expected)
                or pd.DataFrame(current).equals(pd.DataFrame(expected))
            )
            if not matrices_equal:
                exact = False
                break
        block_results.append(
            {
                "block_id": int(block_id),
                "rows": int(len(block)),
                "offsets_match_first": offsets == set(base),
                "matrices_match_first": bool(exact),
            }
        )
    if not all(
        item["offsets_match_first"] and item["matrices_match_first"]
        for item in block_results
    ):
        raise ValueError("parent artifact is not an exact repeated-cycle source")

    return first.reset_index(drop=True), {
        "parent_rows": int(len(ordered)),
        "selected_rows": int(len(first)),
        "outer_start": start.isoformat(),
        "outer_end_inclusive": first["interval_15m"].iloc[-1].isoformat(),
        "cycle_days": int(cycle_days),
        "block_count": len(block_results),
        "all_blocks_exact": True,
        "blocks": block_results,
    }


def recover_original_intervals(first_cycle: pd.DataFrame) -> pd.DataFrame:
    """Recover pre-repeat bins from task start_dt values retained in matrices."""
    rows: list[dict[str, Any]] = []
    outer_start = pd.to_datetime(first_cycle["interval_15m"], utc=True).iloc[0]
    recovered_start: pd.Timestamp | None = None
    for row in first_cycle.itertuples(index=False):
        matrix = np.asarray(row.tasks_matrix)
        if matrix.ndim != 2 or matrix.shape[1] != len(WORKLOAD_FIELD_INDEX):
            raise ValueError(
                f"tasks_matrix must use 12-field schema, got {matrix.shape}"
            )
        starts = pd.to_datetime(
            matrix[:, WORKLOAD_FIELD_INDEX["start_dt"]], utc=True
        ).floor("15min")
        unique = pd.Index(starts).unique()
        if len(unique) != 1:
            raise ValueError("one tasks_matrix contains multiple 15-minute bins")
        recovered = pd.Timestamp(unique[0])
        if recovered_start is None:
            recovered_start = recovered
        expected_offset = pd.Timestamp(row.interval_15m) - outer_start
        actual_offset = recovered - recovered_start
        if actual_offset != expected_offset:
            raise ValueError("inner source time does not align with outer cycle offset")
        rows.append({"interval_15m": recovered, "tasks_matrix": matrix})

    result = pd.DataFrame(rows).sort_values("interval_15m").reset_index(drop=True)
    if result["interval_15m"].duplicated().any():
        raise ValueError("recovered source contains duplicate intervals")
    return result


def aggregate_tasks_matrix(
    matrix: Any,
    *,
    task_scale: float = 5.0,
) -> dict[str, float]:
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[1] != len(WORKLOAD_FIELD_INDEX):
        raise ValueError(f"invalid tasks_matrix shape {values.shape}")
    count = int(values.shape[0])
    cpu = np.asarray(
        values[:, WORKLOAD_FIELD_INDEX["cpu_usage"]], dtype=np.float64
    )
    gpu = np.asarray(
        values[:, WORKLOAD_FIELD_INDEX["gpu_wrk_util"]], dtype=np.float64
    )
    memory = np.asarray(
        values[:, WORKLOAD_FIELD_INDEX["avg_mem"]], dtype=np.float64
    )
    return {
        "new_task_count": float(count),
        "arriving_cpu_demand": float(task_scale * cpu.sum() / 100.0),
        "arriving_gpu_demand": float(task_scale * gpu.sum() / 100.0),
        "arriving_memory_demand": float(task_scale * memory.sum()),
    }


def build_workload_time_series(
    observed: pd.DataFrame,
    *,
    interval_start: Any,
    interval_end_exclusive: Any,
    config: ForecastDatasetConfig | None = None,
) -> pd.DataFrame:
    config = config or ForecastDatasetConfig()
    start = pd.Timestamp(interval_start)
    end = pd.Timestamp(interval_end_exclusive)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    if end <= start:
        raise ValueError("interval_end_exclusive must be after interval_start")

    observed_times = pd.to_datetime(observed["interval_15m"], utc=True)
    if observed_times.duplicated().any():
        raise ValueError("observed source contains duplicate timestamps")
    aggregates = [
        aggregate_tasks_matrix(matrix, task_scale=config.task_scale)
        for matrix in observed["tasks_matrix"]
    ]
    sparse = pd.DataFrame(aggregates, index=observed_times)
    grid = pd.date_range(
        start,
        end,
        freq=f"{config.resolution_minutes}min",
        inclusive="left",
    )
    if not observed_times.isin(grid).all():
        raise ValueError("observed timestamps fall outside the continuous grid")
    table = sparse.reindex(grid, fill_value=0.0)
    table.index.name = "timestamp"
    table.reset_index(inplace=True)
    table["new_task_count"] = table["new_task_count"].astype(np.int64)

    hour = table["timestamp"].dt.hour + table["timestamp"].dt.minute / 60.0
    day_of_week = table["timestamp"].dt.dayofweek
    table["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    table["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    table["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    table["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    assert_continuous_timeline(table, config.resolution_minutes)
    return table.loc[:, ("timestamp",) + FEATURE_NAMES]


def assert_continuous_timeline(
    table: pd.DataFrame,
    resolution_minutes: int = 15,
) -> None:
    timestamps = pd.to_datetime(table["timestamp"], utc=True)
    if timestamps.duplicated().any():
        raise ValueError("timeline contains duplicate timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("timeline is not chronological")
    expected = pd.Timedelta(minutes=resolution_minutes)
    if len(timestamps) > 1 and not timestamps.diff().iloc[1:].eq(expected).all():
        raise ValueError("timeline is not strictly continuous")


def choose_split_row_counts(
    row_count: int,
    *,
    resolution_minutes: int = 15,
    preferred_days: Sequence[int] = (35, 7, 7),
) -> tuple[int, int, int]:
    rows_per_day = 24 * 60 // resolution_minutes
    preferred_rows = tuple(int(days) * rows_per_day for days in preferred_days)
    if sum(preferred_rows) == row_count:
        return preferred_rows
    full_days = row_count // rows_per_day
    if full_days >= 3 and full_days * rows_per_day == row_count:
        validation_days = max(1, int(round(full_days * 0.15)))
        test_days = max(1, int(round(full_days * 0.15)))
        train_days = full_days - validation_days - test_days
        if train_days < 1:
            raise ValueError("not enough complete days for three chronological splits")
        return tuple(
            days * rows_per_day
            for days in (train_days, validation_days, test_days)
        )
    train = int(math.floor(row_count * 0.70))
    validation = int(math.floor(row_count * 0.15))
    test = row_count - train - validation
    if min(train, validation, test) <= 0:
        raise ValueError("not enough rows for three chronological splits")
    return train, validation, test


def split_chronologically(
    table: pd.DataFrame,
    row_counts: Sequence[int],
) -> dict[str, pd.DataFrame]:
    if len(row_counts) != 3 or sum(int(value) for value in row_counts) != len(table):
        raise ValueError("row_counts must contain train/validation/test and cover table")
    train_rows, validation_rows, _ = (int(value) for value in row_counts)
    train_end = train_rows
    validation_end = train_rows + validation_rows
    result = {
        "train": table.iloc[:train_end].reset_index(drop=True),
        "val": table.iloc[train_end:validation_end].reset_index(drop=True),
        "test": table.iloc[validation_end:].reset_index(drop=True),
    }
    if not (
        result["train"]["timestamp"].iloc[-1]
        < result["val"]["timestamp"].iloc[0]
        <= result["val"]["timestamp"].iloc[-1]
        < result["test"]["timestamp"].iloc[0]
    ):
        raise ValueError("chronological split ordering failed")
    return result


def fit_train_scaler(
    train: pd.DataFrame,
    target_names: Sequence[str] = TARGET_NAMES,
) -> dict[str, Any]:
    parameters: dict[str, dict[str, Any]] = {}
    for name in target_names:
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
        "fit_time_start": pd.Timestamp(train["timestamp"].iloc[0]).isoformat(),
        "fit_time_end": pd.Timestamp(train["timestamp"].iloc[-1]).isoformat(),
        "parameters": parameters,
    }


def build_sliding_windows(
    split: pd.DataFrame,
    scaler: Mapping[str, Any],
    *,
    config: ForecastDatasetConfig | None = None,
) -> WindowDataset:
    config = config or ForecastDatasetConfig()
    assert_continuous_timeline(split, config.resolution_minutes)
    sample_count = len(split) - config.history_length - config.horizon + 1
    if sample_count <= 0:
        raise ValueError("split is too short for requested history and horizon")

    features = split.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float32)
    targets = split.loc[:, TARGET_NAMES].to_numpy(dtype=np.float32)
    X_raw = np.stack(
        [
            features[index : index + config.history_length]
            for index in range(sample_count)
        ]
    )
    Y_raw = np.stack(
        [
            targets[
                index
                + config.history_length : index
                + config.history_length
                + config.horizon
            ]
            for index in range(sample_count)
        ]
    )
    X = X_raw.copy()
    Y = Y_raw.copy()
    for index, name in enumerate(TARGET_NAMES):
        parameter = scaler["parameters"][name]
        mean = float(parameter["mean"])
        scale = float(parameter["scale"])
        X[:, :, index] = (X[:, :, index] - mean) / scale
        Y[:, :, index] = (Y[:, :, index] - mean) / scale

    timestamps = (
        pd.to_datetime(split["timestamp"], utc=True)
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    history_end = timestamps[
        np.arange(sample_count) + config.history_length - 1
    ]
    target_start = timestamps[np.arange(sample_count) + config.history_length]
    target_end = timestamps[
        np.arange(sample_count) + config.history_length + config.horizon - 1
    ]
    step = np.timedelta64(config.resolution_minutes, "m")
    if not np.all(target_start == history_end + step):
        raise ValueError("first horizon target is not history_end + one step")
    if not np.all(target_end == history_end + config.horizon * step):
        raise ValueError("last horizon target is misaligned")
    return WindowDataset(
        X=X,
        Y=Y,
        X_raw=X_raw,
        Y_raw=Y_raw,
        history_end_timestamp=history_end,
        target_start_timestamp=target_start,
        target_end_timestamp=target_end,
    )


def feature_schema() -> dict[str, Any]:
    x_features = []
    for index, name in enumerate(FEATURE_NAMES):
        is_target_history = name in TARGET_NAMES
        x_features.append(
            {
                "index": index,
                "name": name,
                "role": (
                    "OBSERVED_HISTORY" if is_target_history else "KNOWN_IN_ADVANCE"
                ),
                "normalization": "TRAIN_STANDARD_SCORE" if is_target_history else "NONE",
            }
        )
    return {
        "X": x_features,
        "Y": [
            {
                "index": index,
                "name": name,
                "role": "FORECAST_TARGET",
                "normalization": "TRAIN_STANDARD_SCORE",
            }
            for index, name in enumerate(TARGET_NAMES)
        ],
        "timestamp_timezone": "UTC",
        "target_tensor_layout": "samples,horizon,target",
        "feature_tensor_layout": "samples,history,feature",
    }


def descriptive_statistics(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in TARGET_NAMES:
        values = table[name].astype(float)
        median = float(values.median())
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        p95 = float(values.quantile(0.95))
        rows.append(
            {
                "target": name,
                "count": int(values.count()),
                "mean": mean,
                "std": std,
                "min": float(values.min()),
                "p25": float(values.quantile(0.25)),
                "median": median,
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
                "p95": p95,
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
                "zero_ratio": float((values == 0).mean()),
                "coefficient_of_variation": (
                    std / mean if not math.isclose(mean, 0.0) else None
                ),
                "p95_over_median": (
                    p95 / median if not math.isclose(median, 0.0) else None
                ),
            }
        )
    return pd.DataFrame(rows)


def autocorrelation_table(
    table: pd.DataFrame,
    lags: Sequence[int] = (1, 2, 4, 8, 24, 48, 96, 192, 672),
) -> pd.DataFrame:
    rows = []
    for name in TARGET_NAMES:
        values = table[name].astype(float)
        for lag in lags:
            available = len(values) > int(lag)
            rows.append(
                {
                    "target": name,
                    "lag_steps": int(lag),
                    "lag_minutes": int(lag) * 15,
                    "autocorrelation": (
                        float(values.autocorr(lag=int(lag))) if available else None
                    ),
                    "status": "COMPUTED" if available else "SKIPPED_TOO_SHORT",
                }
            )
    return pd.DataFrame(rows)
