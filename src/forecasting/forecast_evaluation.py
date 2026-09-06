from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def schema_names(schema: dict[str, Any], section: str) -> list[str]:
    items = sorted(schema[section], key=lambda item: int(item["index"]))
    indices = [int(item["index"]) for item in items]
    if indices != list(range(len(items))):
        raise ValueError(f"{section} schema indices must be contiguous from zero")
    return [str(item["name"]) for item in items]


def inverse_transform_targets(
    values: np.ndarray,
    scaler: dict[str, Any],
    target_names: Sequence[str],
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if result.ndim < 1 or result.shape[-1] != len(target_names):
        raise ValueError("target tensor last dimension does not match target schema")
    parameters = scaler["parameters"]
    for index, name in enumerate(target_names):
        item = parameters[name]
        result[..., index] = (
            result[..., index] * float(item["scale"]) + float(item["mean"])
        )
    return result


def normalize_history(
    history: np.ndarray,
    scaler: dict[str, Any],
    feature_schema: Sequence[dict[str, Any]],
) -> np.ndarray:
    ordered = sorted(feature_schema, key=lambda item: int(item["index"]))
    result = np.asarray(history, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != len(ordered):
        raise ValueError("history must be [history_length, input_dim]")
    for item in ordered:
        if item["normalization"] == "TRAIN_STANDARD_SCORE":
            stats = scaler["parameters"][item["name"]]
            index = int(item["index"])
            result[:, index] = (
                result[:, index] - float(stats["mean"])
            ) / float(stats["scale"])
        elif item["normalization"] != "NONE":
            raise ValueError(f"unsupported normalization={item['normalization']!r}")
    return result


def persistence_forecast(
    raw_history: np.ndarray,
    *,
    target_indices: Sequence[int],
    horizon: int,
) -> np.ndarray:
    history = np.asarray(raw_history)
    if history.ndim != 3:
        raise ValueError("raw_history must be [N,L,F]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    last = history[:, -1, list(target_indices)]
    return np.repeat(last[:, None, :], horizon, axis=1)


def forecast_timestamps(
    history_end_timestamp: np.ndarray,
    *,
    horizon: int,
    resolution_minutes: int = 15,
) -> np.ndarray:
    end = np.asarray(history_end_timestamp).astype("datetime64[ns]")
    if end.ndim != 1 or horizon <= 0 or resolution_minutes <= 0:
        raise ValueError("invalid timestamp alignment arguments")
    offsets = (
        np.arange(1, horizon + 1, dtype=np.int64)
        * np.timedelta64(resolution_minutes, "m")
    )
    return end[:, None] + offsets[None, :]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_names: Sequence[str],
    model: str,
    seed: int | str,
    resolution_minutes: int = 15,
) -> pd.DataFrame:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 3:
        raise ValueError("y_true and y_pred must share [N,H,T] shape")
    if truth.shape[2] != len(target_names):
        raise ValueError("metric target dimension does not match target schema")
    rows: list[dict[str, Any]] = []
    for horizon_index in range(truth.shape[1]):
        for target_index, target in enumerate(target_names):
            error = prediction[:, horizon_index, target_index] - truth[:, horizon_index, target_index]
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "target": target,
                    "horizon_step": horizon_index + 1,
                    "horizon_minutes": (horizon_index + 1) * resolution_minutes,
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                }
            )
    return pd.DataFrame(rows)


def prediction_long_frame(
    *,
    history_end_timestamp: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persistence_pred: np.ndarray,
    target_names: Sequence[str],
    resolution_minutes: int = 15,
) -> pd.DataFrame:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    persistence = np.asarray(persistence_pred, dtype=np.float64)
    if truth.shape != prediction.shape or truth.shape != persistence.shape:
        raise ValueError("truth, Transformer and persistence arrays must align")
    timestamps = forecast_timestamps(
        history_end_timestamp,
        horizon=truth.shape[1],
        resolution_minutes=resolution_minutes,
    )
    rows: list[dict[str, Any]] = []
    for sample in range(truth.shape[0]):
        for horizon_index in range(truth.shape[1]):
            for target_index, target in enumerate(target_names):
                rows.append(
                    {
                        "history_end_timestamp": pd.Timestamp(history_end_timestamp[sample]),
                        "forecast_timestamp": pd.Timestamp(timestamps[sample, horizon_index]),
                        "horizon_step": horizon_index + 1,
                        "horizon_minutes": (horizon_index + 1) * resolution_minutes,
                        "target": target,
                        "y_true": truth[sample, horizon_index, target_index],
                        "y_pred": prediction[sample, horizon_index, target_index],
                        "persistence_pred": persistence[sample, horizon_index, target_index],
                    }
                )
    return pd.DataFrame(rows)


def relative_improvement(reference: float, candidate: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(candidate):
        return float("nan")
    if reference == 0:
        return 0.0 if candidate == 0 else float("nan")
    return (reference - candidate) / reference


def write_json(path: str | Path, value: Any) -> None:
    import json

    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
