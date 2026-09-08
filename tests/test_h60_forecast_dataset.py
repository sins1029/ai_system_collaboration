from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.h60_dataset import (
    H60ForecastDatasetConfig,
    build_h60_windows,
    build_spot_h60_time_series,
    fit_h60_train_scaler,
    persistence_h60,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    steps = np.arange(36)
    splits = np.repeat(["train", "validation", "test"], 12)
    states = pd.DataFrame(
        {
            "step": steps,
            "timestamp_utc": pd.date_range(
                "2026-01-01", periods=36, freq="15min", tz="UTC"
            ).astype(str),
            "split": splits,
        }
    )
    tasks = pd.DataFrame(
        {
            "task_id": [f"task-{step}" for step in steps],
            "arrival_step": steps,
            "cpu_cores": steps + 1.0,
            "gpu_units": steps + 2.0,
            "memory_gb": steps + 3.0,
        }
    )
    return states, tasks


def test_h60_windows_use_exact_current_plus_four_without_crossing_split() -> None:
    states, tasks = _inputs()
    timeline = build_spot_h60_time_series(states, tasks)
    scaler = fit_h60_train_scaler(timeline)
    config = H60ForecastDatasetConfig(history_length=4)
    train = build_h60_windows(timeline, "train", scaler, config=config)
    validation = build_h60_windows(
        timeline, "validation", scaler, config=config
    )

    assert train.sample_count == 5
    assert np.all(train.target_step - train.current_step == 4)
    assert np.all(validation.target_step - validation.current_step == 4)
    assert train.target_step.max() < validation.current_step.min()
    assert train.Y.shape == (5, 1, 4)
    assert train.Y_raw[0, 0, 1] == 8.0


def test_scaler_is_fit_on_train_only() -> None:
    states, tasks = _inputs()
    timeline = build_spot_h60_time_series(states, tasks)
    scaler = fit_h60_train_scaler(timeline)
    assert scaler["fitted_on"] == "TRAIN_ONLY"
    assert scaler["fit_step_start"] == 0
    assert scaler["fit_step_end"] == 11


def test_persistence_uses_last_observed_history_row() -> None:
    states, tasks = _inputs()
    timeline = build_spot_h60_time_series(states, tasks)
    scaler = fit_h60_train_scaler(timeline)
    window = build_h60_windows(
        timeline,
        "train",
        scaler,
        config=H60ForecastDatasetConfig(history_length=4),
    )
    prediction = persistence_h60(window)
    np.testing.assert_array_equal(prediction[:, 0], window.X_raw[:, -1, :4])


def test_duplicate_task_rows_are_rejected() -> None:
    states, tasks = _inputs()
    duplicated = pd.concat([tasks, tasks.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per unique task"):
        build_spot_h60_time_series(states, duplicated)
