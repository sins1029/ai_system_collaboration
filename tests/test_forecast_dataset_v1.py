from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from forecasting.dataset import (
    CALENDAR_FEATURE_NAMES,
    FEATURE_NAMES,
    TARGET_NAMES,
    ForecastDatasetConfig,
    aggregate_tasks_matrix,
    assert_continuous_timeline,
    build_sliding_windows,
    build_workload_time_series,
    choose_split_row_counts,
    extract_first_unique_cycle,
    feature_schema,
    fit_train_scaler,
    recover_original_intervals,
    split_chronologically,
)
from forecasting.loader import load_forecast_dataset


START = pd.Timestamp("1970-01-26T00:00:00Z")


def _task(
    timestamp: pd.Timestamp,
    *,
    name: str = "job",
    cpu: float = 100.0,
    gpu: float = 40.0,
    memory: float = 2.0,
) -> list[object]:
    return [
        name,
        0,
        1,
        timestamp + pd.Timedelta(minutes=1),
        60.0,
        cpu,
        gpu,
        memory,
        123.456,
        7.89,
        timestamp.day_name(),
        timestamp.dayofweek,
    ]


def _observed(offsets: tuple[int, ...] = (0, 2)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "interval_15m": [
                START + pd.Timedelta(minutes=15 * offset) for offset in offsets
            ],
            "tasks_matrix": [
                np.asarray(
                    [_task(START + pd.Timedelta(minutes=15 * offset))],
                    dtype=object,
                )
                for offset in offsets
            ],
        }
    )


def _continuous_table(rows: int = 600) -> pd.DataFrame:
    timestamps = pd.date_range(START, periods=rows, freq="15min")
    base = np.arange(rows, dtype=np.float64)
    table = pd.DataFrame(
        {
            "timestamp": timestamps,
            "new_task_count": (base % 11).astype(np.int64),
            "arriving_cpu_demand": base + 10.0,
            "arriving_gpu_demand": base * 0.5 + 2.0,
            "arriving_memory_demand": base * 2.0 + 3.0,
        }
    )
    hour = table["timestamp"].dt.hour + table["timestamp"].dt.minute / 60.0
    dow = table["timestamp"].dt.dayofweek
    table["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    table["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    table["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    table["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return table


def test_timeline_is_strictly_continuous_at_15_minutes() -> None:
    table = build_workload_time_series(
        _observed(),
        interval_start=START,
        interval_end_exclusive=START + pd.Timedelta(minutes=45),
    )

    assert len(table) == 3
    assert_continuous_timeline(table, 15)
    assert table["timestamp"].diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all()


def test_zero_task_interval_is_preserved() -> None:
    table = build_workload_time_series(
        _observed(),
        interval_start=START,
        interval_end_exclusive=START + pd.Timedelta(minutes=45),
    )

    zero = table.iloc[1]
    assert zero["timestamp"] == START + pd.Timedelta(minutes=15)
    assert zero.loc[list(TARGET_NAMES)].sum() == 0.0


def test_aggregate_targets_match_task_mapping() -> None:
    matrix = np.asarray(
        [
            _task(START, name="a", cpu=100.0, gpu=20.0, memory=2.0),
            _task(START, name="b", cpu=50.0, gpu=40.0, memory=3.0),
        ],
        dtype=object,
    )

    result = aggregate_tasks_matrix(matrix, task_scale=5.0)

    assert result == {
        "new_task_count": 2.0,
        "arriving_cpu_demand": 7.5,
        "arriving_gpu_demand": 3.0,
        "arriving_memory_demand": 25.0,
    }


def test_chronological_split_ordering() -> None:
    table = _continuous_table()
    splits = split_chronologically(table, (400, 100, 100))

    assert splits["train"]["timestamp"].is_monotonic_increasing
    assert splits["val"]["timestamp"].is_monotonic_increasing
    assert splits["test"]["timestamp"].is_monotonic_increasing
    assert splits["train"]["timestamp"].max() < splits["val"]["timestamp"].min()


def test_split_timestamps_never_overlap() -> None:
    splits = split_chronologically(_continuous_table(), (400, 100, 100))
    sets = {
        name: set(pd.to_datetime(frame["timestamp"], utc=True))
        for name, frame in splits.items()
    }

    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])


def test_strict_windows_stay_inside_each_split() -> None:
    splits = split_chronologically(_continuous_table(), (400, 100, 100))
    scaler = fit_train_scaler(splits["train"])
    config = ForecastDatasetConfig(history_length=24, horizon=4)

    for frame in splits.values():
        windows = build_sliding_windows(frame, scaler, config=config)
        split_start = pd.Timestamp(frame["timestamp"].min()).tz_localize(None)
        split_end = pd.Timestamp(frame["timestamp"].max()).tz_localize(None)
        assert windows.history_end_timestamp[0] >= split_start.to_datetime64()
        assert windows.target_end_timestamp[-1] <= split_end.to_datetime64()


def test_horizon_alignment_has_no_off_by_one() -> None:
    table = _continuous_table(140)
    scaler = fit_train_scaler(table)
    config = ForecastDatasetConfig(history_length=96, horizon=4)
    windows = build_sliding_windows(table, scaler, config=config)

    history_end = windows.history_end_timestamp[0]
    step = np.timedelta64(15, "m")
    assert windows.target_start_timestamp[0] == history_end + step
    assert windows.target_end_timestamp[0] == history_end + 4 * step
    np.testing.assert_allclose(
        windows.Y_raw[0],
        table.loc[96:99, TARGET_NAMES].to_numpy(dtype=np.float32),
    )


def test_scaler_is_fit_on_train_only() -> None:
    table = _continuous_table()
    splits = split_chronologically(table, (400, 100, 100))
    scaler = fit_train_scaler(splits["train"])

    assert scaler["fitted_on"] == "TRAIN_ONLY"
    assert scaler["fit_rows"] == 400
    for name in TARGET_NAMES:
        assert scaler["parameters"][name]["mean"] == float(
            splits["train"][name].mean()
        )
        assert scaler["parameters"][name]["mean"] != float(
            table[name].mean()
        )


def test_feature_schema_has_explicit_indices_and_roles() -> None:
    schema = feature_schema()

    assert [item["index"] for item in schema["X"]] == list(range(8))
    assert [item["name"] for item in schema["X"]] == list(FEATURE_NAMES)
    assert [item["name"] for item in schema["Y"]] == list(TARGET_NAMES)
    assert all(
        item["role"] == "KNOWN_IN_ADVANCE"
        for item in schema["X"]
        if item["name"] in CALENDAR_FEATURE_NAMES
    )


def test_generation_is_deterministic() -> None:
    first = build_workload_time_series(
        _observed(),
        interval_start=START,
        interval_end_exclusive=START + pd.Timedelta(minutes=45),
    )
    second = build_workload_time_series(
        _observed(),
        interval_start=START,
        interval_end_exclusive=START + pd.Timedelta(minutes=45),
    )

    pdt.assert_frame_equal(first, second, check_exact=True)
    table = _continuous_table(140)
    scaler = fit_train_scaler(table)
    config = ForecastDatasetConfig(history_length=24, horizon=4)
    left = build_sliding_windows(table, scaler, config=config)
    right = build_sliding_windows(table, scaler, config=config)
    np.testing.assert_array_equal(left.X, right.X)
    np.testing.assert_array_equal(left.Y, right.Y)


def test_feature_leakage_allowlist_excludes_future_truth() -> None:
    names = {item["name"] for item in feature_schema()["X"]}
    forbidden = {
        "future_new_task_count",
        "future_cpu_demand",
        "future_gpu_demand",
        "future_memory_demand",
        "true_duration",
        "future_task_identity",
        "future_finish_time",
    }

    assert names == set(FEATURE_NAMES)
    assert names.isdisjoint(forbidden)


def test_source_unique_cycle_excludes_later_copies() -> None:
    original_offsets = (0, 1, 3)
    first_outer = pd.Timestamp("2020-01-01T00:00:00Z")
    rows = []
    for block in range(2):
        for offset in original_offsets:
            original_time = START + pd.Timedelta(minutes=15 * offset)
            matrix = np.asarray([_task(original_time)], dtype=object)
            matrix[0, 8] = np.nan
            rows.append(
                {
                    "interval_15m": (
                        first_outer
                        + pd.Timedelta(days=block)
                        + pd.Timedelta(minutes=15 * offset)
                    ),
                    "tasks_matrix": matrix,
                }
            )
    parent = pd.DataFrame(rows)

    selected, audit = extract_first_unique_cycle(parent, cycle_days=1)
    recovered = recover_original_intervals(selected)

    assert len(selected) == len(original_offsets)
    assert audit["block_count"] == 2
    assert audit["all_blocks_exact"] is True
    assert recovered["interval_15m"].tolist() == [
        START + pd.Timedelta(minutes=15 * offset) for offset in original_offsets
    ]


def test_preferred_49_day_split_is_35_7_7() -> None:
    assert choose_split_row_counts(49 * 96) == (35 * 96, 7 * 96, 7 * 96)


def test_loader_supports_canonical_and_alternate_history(
    tmp_path: Path,
) -> None:
    root = tmp_path / "forecast_dataset_v1"
    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True)
    table = _continuous_table()
    splits = split_chronologically(table, (400, 100, 100))
    scaler = fit_train_scaler(splits["train"])
    config = ForecastDatasetConfig(history_length=24, horizon=4)
    windows = {
        name: build_sliding_windows(frame, scaler, config=config)
        for name, frame in splits.items()
    }
    table.to_parquet(dataset_dir / "workload_15min.parquet", index=False)
    for name, value in windows.items():
        np.savez_compressed(
            dataset_dir / f"{name}.npz",
            X=value.X,
            Y=value.Y,
            X_raw=value.X_raw,
            Y_raw=value.Y_raw,
            history_end_timestamp=value.history_end_timestamp,
            target_start_timestamp=value.target_start_timestamp,
            target_end_timestamp=value.target_end_timestamp,
        )
    (root / "10_feature_schema.json").write_text(
        json.dumps(feature_schema()), encoding="utf-8"
    )
    (root / "09_scaler_stats.json").write_text(
        json.dumps(scaler), encoding="utf-8"
    )
    split_manifest = {
        "splits": {
            name: {
                "start": pd.Timestamp(frame["timestamp"].iloc[0]).isoformat(),
                "end": pd.Timestamp(frame["timestamp"].iloc[-1]).isoformat(),
            }
            for name, frame in splits.items()
        }
    }
    (root / "08_split_manifest.json").write_text(
        json.dumps(split_manifest), encoding="utf-8"
    )
    (root / "13_dataset_manifest.json").write_text(
        json.dumps(
            {
                "history_length": 24,
                "forecast_horizon": 4,
                "time_resolution_minutes": 15,
            }
        ),
        encoding="utf-8",
    )

    canonical = load_forecast_dataset(
        "val", 24, 4, True, dataset_root=root
    )
    alternate = load_forecast_dataset(
        "val", 48, 4, False, dataset_root=root
    )

    assert canonical.X.shape == (73, 24, 8)
    assert alternate.X.shape == (49, 48, 8)
    assert alternate.Y.shape == (49, 4, 4)

def test_generated_artifact_passes_real_leakage_contract() -> None:
    root = Path(__file__).resolve().parents[1] / "artifacts/forecast_dataset_v1"
    if not root.exists():
        pytest.skip("Forecast Dataset v1 artifact has not been generated")

    manifest = json.loads(
        (root / "13_dataset_manifest.json").read_text(encoding="utf-8")
    )
    leakage = pd.read_csv(root / "11_leakage_audit.csv")
    table = pd.read_parquet(root / "dataset/workload_15min.parquet")
    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True)

    assert manifest["source_type"] == "FIRST_VERIFIED_UNIQUE_CYCLE"
    assert manifest["full_year_repetition_used"] is False
    assert manifest["random_split_used"] is False
    assert manifest["future_leakage_detected"] is False
    assert manifest["zero_interval_count"] == 10
    assert len(table) == 4704
    assert_continuous_timeline(table, 15)
    critical = leakage[leakage["severity"] == "CRITICAL"]
    assert not critical.empty
    assert critical["status"].eq("PASS").all()
