from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from forecasting.dataset import ForecastDatasetConfig, build_sliding_windows


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class LoadedForecastDataset:
    X: np.ndarray
    Y: np.ndarray
    history_end_timestamp: np.ndarray
    target_start_timestamp: np.ndarray
    target_end_timestamp: np.ndarray
    feature_schema: dict[str, Any]
    split: SplitName
    normalized: bool


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts/forecast_dataset_v1"


def load_forecast_dataset(
    split: SplitName = "train",
    history_length: int = 96,
    horizon: int = 4,
    normalized: bool = True,
    *,
    dataset_root: str | Path | None = None,
) -> LoadedForecastDataset:
    """Load canonical NPZ data or deterministically rebuild another history length."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"unsupported split={split!r}")
    root = Path(dataset_root) if dataset_root is not None else _default_root()
    manifest = json.loads((root / "13_dataset_manifest.json").read_text("utf-8"))
    schema = json.loads((root / "10_feature_schema.json").read_text("utf-8"))
    if horizon != int(manifest["forecast_horizon"]):
        raise ValueError(
            f"horizon must match generated dataset ({manifest['forecast_horizon']})"
        )

    canonical_history = int(manifest["history_length"])
    if history_length == canonical_history:
        with np.load(root / "dataset" / f"{split}.npz", allow_pickle=False) as data:
            return LoadedForecastDataset(
                X=np.asarray(data["X" if normalized else "X_raw"]),
                Y=np.asarray(data["Y" if normalized else "Y_raw"]),
                history_end_timestamp=np.asarray(data["history_end_timestamp"]),
                target_start_timestamp=np.asarray(data["target_start_timestamp"]),
                target_end_timestamp=np.asarray(data["target_end_timestamp"]),
                feature_schema=schema,
                split=split,
                normalized=normalized,
            )

    split_manifest = json.loads(
        (root / "08_split_manifest.json").read_text("utf-8")
    )
    scaler = json.loads((root / "09_scaler_stats.json").read_text("utf-8"))
    table = pd.read_parquet(root / "dataset/workload_15min.parquet")
    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True)
    bounds = split_manifest["splits"][split]
    selected = table[
        (table["timestamp"] >= pd.Timestamp(bounds["start"]))
        & (table["timestamp"] <= pd.Timestamp(bounds["end"]))
    ].reset_index(drop=True)
    windows = build_sliding_windows(
        selected,
        scaler,
        config=ForecastDatasetConfig(
            history_length=history_length,
            horizon=horizon,
            resolution_minutes=int(manifest["time_resolution_minutes"]),
        ),
    )
    return LoadedForecastDataset(
        X=windows.X if normalized else windows.X_raw,
        Y=windows.Y if normalized else windows.Y_raw,
        history_end_timestamp=windows.history_end_timestamp,
        target_start_timestamp=windows.target_start_timestamp,
        target_end_timestamp=windows.target_end_timestamp,
        feature_schema=schema,
        split=split,
        normalized=normalized,
    )
