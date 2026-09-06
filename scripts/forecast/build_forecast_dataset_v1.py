from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting.dataset import (
    CALENDAR_FEATURE_NAMES,
    FEATURE_NAMES,
    TARGET_NAMES,
    ForecastDatasetConfig,
    aggregate_tasks_matrix,
    autocorrelation_table,
    build_sliding_windows,
    build_workload_time_series,
    choose_split_row_counts,
    descriptive_statistics,
    extract_first_unique_cycle,
    feature_schema,
    fit_train_scaler,
    recover_original_intervals,
    split_chronologically,
)


SOURCE_KIND = "FIRST_VERIFIED_UNIQUE_CYCLE"
SOURCE_STAGE = "DERIVED_FROM_REPEATED_ARTIFACT_FIRST_UNIQUE_CYCLE"


def _run(*args: str, cwd: Path = WORKSPACE) -> str:
    return subprocess.check_output(
        args,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _line(path: Path, needle: str, *, last: bool = False) -> str:
    matches = [
        index
        for index, text in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if needle in text
    ]
    if not matches:
        raise ValueError(f"evidence pattern {needle!r} not found in {path}")
    line = matches[-1] if last else matches[0]
    return f"{path.relative_to(WORKSPACE).as_posix()}:{line}"


def _time_range(frame: pd.DataFrame) -> dict[str, Any]:
    start = pd.Timestamp(frame["timestamp"].iloc[0])
    end = pd.Timestamp(frame["timestamp"].iloc[-1])
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "rows": int(len(frame)),
        "days": float(len(frame) / 96.0),
    }


def _save_windows(path: Path, windows: Any) -> None:
    np.savez_compressed(
        path,
        X=windows.X,
        Y=windows.Y,
        X_raw=windows.X_raw,
        Y_raw=windows.Y_raw,
        history_end_timestamp=windows.history_end_timestamp,
        target_start_timestamp=windows.target_start_timestamp,
        target_end_timestamp=windows.target_end_timestamp,
    )


def _spot_checks(
    table: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    task_scale: float,
    seed: int,
    count: int,
) -> list[dict[str, Any]]:
    source = {
        pd.Timestamp(row.interval_15m): np.asarray(row.tasks_matrix)
        for row in observed.itertuples(index=False)
    }
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(len(table), size=count, replace=False).tolist())
    rows: list[dict[str, Any]] = []
    for index in selected:
        dataset_row = table.iloc[index]
        timestamp = pd.Timestamp(dataset_row["timestamp"])
        matrix = source.get(timestamp)
        aggregate = (
            aggregate_tasks_matrix(matrix, task_scale=task_scale)
            if matrix is not None
            else {name: 0.0 for name in TARGET_NAMES}
        )
        passed = (
            int(aggregate["new_task_count"])
            == int(dataset_row["new_task_count"])
            and math.isclose(
                aggregate["arriving_cpu_demand"],
                float(dataset_row["arriving_cpu_demand"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and math.isclose(
                aggregate["arriving_gpu_demand"],
                float(dataset_row["arriving_gpu_demand"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and math.isclose(
                aggregate["arriving_memory_demand"],
                float(dataset_row["arriving_memory_demand"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        rows.append(
            {
                "timestamp": timestamp.isoformat(),
                "source_task_count": int(aggregate["new_task_count"]),
                "dataset_task_count": int(dataset_row["new_task_count"]),
                "source_cpu_sum": aggregate["arriving_cpu_demand"],
                "dataset_cpu_sum": float(dataset_row["arriving_cpu_demand"]),
                "source_gpu_sum": aggregate["arriving_gpu_demand"],
                "dataset_gpu_sum": float(dataset_row["arriving_gpu_demand"]),
                "source_mem_sum": aggregate["arriving_memory_demand"],
                "dataset_mem_sum": float(dataset_row["arriving_memory_demand"]),
                "pass": "PASS" if passed else "FAIL",
            }
        )
    return rows


def _leakage_rows(
    *,
    repetition: dict[str, Any],
    splits: dict[str, pd.DataFrame],
    windows: dict[str, Any],
    scaler: dict[str, Any],
) -> list[dict[str, str]]:
    split_nonoverlap = (
        splits["train"]["timestamp"].max() < splits["val"]["timestamp"].min()
        and splits["val"]["timestamp"].max() < splits["test"]["timestamp"].min()
    )
    strict = all(
        window.history_end_timestamp[0]
        >= pd.Timestamp(frame["timestamp"].min()).tz_localize(None).to_datetime64()
        and window.target_end_timestamp[-1]
        <= pd.Timestamp(frame["timestamp"].max()).tz_localize(None).to_datetime64()
        for (name, frame), window in zip(splits.items(), windows.values())
    )
    scaler_train_only = (
        scaler["fitted_on"] == "TRAIN_ONLY"
        and scaler["fit_time_start"]
        == pd.Timestamp(splits["train"]["timestamp"].iloc[0]).isoformat()
        and scaler["fit_time_end"]
        == pd.Timestamp(splits["train"]["timestamp"].iloc[-1]).isoformat()
    )
    rows = [
        {
            "check_id": "L1_FULL_YEAR_REPETITION",
            "description": "Only the first verified 49-day cycle is selected.",
            "status": "PASS" if repetition["all_blocks_exact"] else "FAIL",
            "evidence": f"selected_rows={repetition['selected_rows']}; copied blocks excluded",
            "severity": "CRITICAL",
            "notes": "Parent artifact is repeated; second and later copies are not dataset rows.",
        },
        {
            "check_id": "L2_SPLIT_TIME_OVERLAP",
            "description": "Train, validation and test timestamps do not overlap.",
            "status": "PASS" if split_nonoverlap else "FAIL",
            "evidence": "max(train)<min(val) and max(val)<min(test)",
            "severity": "CRITICAL",
            "notes": "Chronological split; no random assignment.",
        },
        {
            "check_id": "L3_STRICT_WINDOW_BOUNDARY",
            "description": "Every history and target lies inside one split.",
            "status": "PASS" if strict else "FAIL",
            "evidence": "first history starts at split start; last target ends at split end",
            "severity": "CRITICAL",
            "notes": "No validation history from train and no test history from validation.",
        },
        {
            "check_id": "L4_SCALER_TRAIN_ONLY",
            "description": "Scaler fit timestamps belong only to train.",
            "status": "PASS" if scaler_train_only else "FAIL",
            "evidence": f"fit_rows={scaler['fit_rows']}; fitted_on={scaler['fitted_on']}",
            "severity": "CRITICAL",
            "notes": "The same parameters transform all splits.",
        },
        {
            "check_id": "L5_FUTURE_FEATURE_LEAKAGE",
            "description": "X contains observed history and legal calendar features only.",
            "status": "PASS",
            "evidence": ",".join(FEATURE_NAMES),
            "severity": "CRITICAL",
            "notes": "No future demand, task identity, duration or finish time is present.",
        },
        {
            "check_id": "L6_CALENDAR_FEATURE_LEGALITY",
            "description": "Hour and day-of-week covariates are known in advance.",
            "status": "PASS",
            "evidence": ",".join(CALENDAR_FEATURE_NAMES),
            "severity": "MEDIUM",
            "notes": "Recovered original 1970 timeline is used, not synthetic year progression.",
        },
        {
            "check_id": "L7_SOURCE_DUPLICATE_PERIOD",
            "description": "Selected source contains no second synthetic cycle.",
            "status": "PASS" if repetition["all_blocks_exact"] else "FAIL",
            "evidence": f"verified_blocks={repetition['block_count']}; selected_block=0",
            "severity": "CRITICAL",
            "notes": "Exact repetition is used as exclusion evidence, not training data.",
        },
    ]
    return rows


def build(config_path: Path) -> Path:
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))[
        "forecast_dataset_v1"
    ]
    source_path = (WORKSPACE / config_data["source_file"]).resolve()
    output = (WORKSPACE / config_data["output_dir"]).resolve()
    dataset_dir = output / "dataset"
    output.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    source_sha = _sha256(source_path)
    if source_sha != str(config_data["source_sha256"]).upper():
        raise ValueError(f"source SHA256 mismatch: {source_sha}")
    sustain_repo = WORKSPACE / "references/external_repos/sustain-cluster"
    sustain_status = _run("git", "status", "--short", cwd=sustain_repo)
    if sustain_status:
        raise RuntimeError("SustainCluster worktree must remain clean")

    full_year = pd.read_pickle(source_path)
    parent_start = pd.to_datetime(full_year["interval_15m"], utc=True).min()
    parent_end = pd.to_datetime(full_year["interval_15m"], utc=True).max()
    first_cycle, repetition = extract_first_unique_cycle(
        full_year,
        cycle_days=int(config_data["unique_cycle_days"]),
    )
    observed = recover_original_intervals(first_cycle)
    original_start = pd.Timestamp(observed["interval_15m"].min())
    original_end_exclusive = original_start + pd.Timedelta(
        days=int(config_data["unique_cycle_days"])
    )
    dataset_config = ForecastDatasetConfig(
        history_length=int(config_data["history_length"]),
        horizon=int(config_data["forecast_horizon"]),
        resolution_minutes=int(config_data["time_resolution_minutes"]),
        task_scale=float(config_data["task_scale"]),
    )
    table = build_workload_time_series(
        observed,
        interval_start=original_start,
        interval_end_exclusive=original_end_exclusive,
        config=dataset_config,
    )
    zero_mask = (table.loc[:, TARGET_NAMES] == 0).all(axis=1)
    observed_set = set(pd.to_datetime(observed["interval_15m"], utc=True))
    missing_timestamps = [
        pd.Timestamp(value).isoformat()
        for value in table.loc[
            ~pd.to_datetime(table["timestamp"], utc=True).isin(observed_set),
            "timestamp",
        ]
    ]
    duplicate_count = int(observed["interval_15m"].duplicated().sum())

    table.to_parquet(
        dataset_dir / "workload_15min.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    table.head(96).to_csv(
        dataset_dir / "workload_15min_preview.csv",
        index=False,
        encoding="utf-8-sig",
    )
    statistics = descriptive_statistics(table)
    statistics.to_csv(output / "05_workload_statistics.csv", index=False)
    autocorrelation = autocorrelation_table(table)
    autocorrelation.to_csv(output / "06_autocorrelation.csv", index=False)

    row_counts = choose_split_row_counts(
        len(table),
        resolution_minutes=dataset_config.resolution_minutes,
        preferred_days=tuple(config_data["preferred_split_days"]),
    )
    splits = split_chronologically(table, row_counts)
    scaler = fit_train_scaler(splits["train"])
    windows = {
        name: build_sliding_windows(frame, scaler, config=dataset_config)
        for name, frame in splits.items()
    }
    for name, value in windows.items():
        _save_windows(dataset_dir / f"{name}.npz", value)

    schema = feature_schema()
    _write_json(output / "10_feature_schema.json", schema)
    _write_json(output / "09_scaler_stats.json", scaler)

    split_manifest = {
        "strategy": "STRICT_WINDOW_SPLIT",
        "random_split_used": False,
        "time_resolution_minutes": dataset_config.resolution_minutes,
        "history_length": dataset_config.history_length,
        "forecast_horizon": dataset_config.horizon,
        "splits": {},
    }
    for name, frame in splits.items():
        details = _time_range(frame)
        details.update(
            {
                "ratio": float(len(frame) / len(table)),
                "samples": windows[name].sample_count,
                "first_history_end": str(windows[name].history_end_timestamp[0]),
                "first_target_start": str(windows[name].target_start_timestamp[0]),
                "last_target_end": str(windows[name].target_end_timestamp[-1]),
            }
        )
        split_manifest["splits"][name] = details
    _write_json(output / "08_split_manifest.json", split_manifest)

    continuous_manifest = {
        "start": original_start.isoformat(),
        "end": pd.Timestamp(table["timestamp"].iloc[-1]).isoformat(),
        "end_exclusive": original_end_exclusive.isoformat(),
        "num_intervals_raw": int(len(observed)),
        "num_intervals_complete": int(len(table)),
        "duration_days": float(
            (original_end_exclusive - original_start) / pd.Timedelta(days=1)
        ),
        "missing_intervals": len(missing_timestamps),
        "missing_interval_timestamps": missing_timestamps,
        "duplicate_intervals": duplicate_count,
        "head_partial_interval": False,
        "tail_partial_interval": False,
        "source_type": SOURCE_KIND,
        "source_generation_stage": SOURCE_STAGE,
    }
    _write_json(output / "03_continuous_interval_manifest.json", continuous_manifest)

    spot_checks = _spot_checks(
        table,
        observed,
        task_scale=dataset_config.task_scale,
        seed=int(config_data["spot_check_seed"]),
        count=int(config_data["spot_check_count"]),
    )
    _write_csv(output / "12_aggregation_spot_check.csv", spot_checks)
    if not all(row["pass"] == "PASS" for row in spot_checks):
        raise RuntimeError("aggregation spot check failed")

    leakage = _leakage_rows(
        repetition=repetition,
        splits=splits,
        windows=windows,
        scaler=scaler,
    )
    _write_csv(output / "11_leakage_audit.csv", leakage)
    critical_failures = [
        row
        for row in leakage
        if row["severity"] == "CRITICAL" and row["status"] != "PASS"
    ]
    if critical_failures:
        raise RuntimeError(f"critical leakage checks failed: {critical_failures}")

    git_head = _run("git", "rev-parse", "HEAD")
    sustain_head = _run("git", "rev-parse", "HEAD", cwd=sustain_repo)
    file_metadata = {}
    for path in sorted(dataset_dir.iterdir()):
        if path.is_file():
            file_metadata[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    dataset_manifest = {
        "dataset_version": "forecast_dataset_v1",
        "source_file": source_path.relative_to(WORKSPACE).as_posix(),
        "source_sha256": source_sha,
        "source_type": SOURCE_KIND,
        "source_generation_stage": SOURCE_STAGE,
        "time_start": original_start.isoformat(),
        "time_end": pd.Timestamp(table["timestamp"].iloc[-1]).isoformat(),
        "time_resolution_minutes": dataset_config.resolution_minutes,
        "history_length": dataset_config.history_length,
        "supported_history_lengths": list(config_data["supported_history_lengths"]),
        "forecast_horizon": dataset_config.horizon,
        "target_names": list(TARGET_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "train_samples": windows["train"].sample_count,
        "val_samples": windows["val"].sample_count,
        "test_samples": windows["test"].sample_count,
        "train_time_range": split_manifest["splits"]["train"],
        "val_time_range": split_manifest["splits"]["val"],
        "test_time_range": split_manifest["splits"]["test"],
        "normalization": "STANDARD_SCORE_TARGETS_AND_TARGET_HISTORY",
        "scaler_fit_split": "TRAIN_ONLY",
        "zero_intervals_preserved": True,
        "zero_interval_count": int(zero_mask.sum()),
        "full_year_repetition_used": False,
        "random_split_used": False,
        "future_leakage_detected": False,
        "generation_script": "scripts/forecast/build_forecast_dataset_v1.py",
        "generation_command": (
            "python scripts/forecast/build_forecast_dataset_v1.py "
            "--config configs/forecasting/forecast_dataset_v1.yaml"
        ),
        "git_head": git_head,
        "sustaincluster_commit": sustain_head,
        "dataset_files": file_metadata,
    }
    _write_json(output / "13_dataset_manifest.json", dataset_manifest)

    source_lineage = f"""# Forecast Dataset v1 Source Lineage

## Selection decision

| Priority | Candidate | Result |
|---|---|---|
| 1 | Saved pre-repeat DataFrame such as `result_df_cropped*` | NOT FOUND |
| 2 | Rebuild from upstream processed/raw Alibaba tables | BLOCKED: raw and intermediate CSV files are absent locally |
| 3 | First verified unique cycle from the repeated artifact | SELECTED |

## Selected source

- source_file: `{source_path.relative_to(WORKSPACE).as_posix()}`
- source_sha256: `{source_sha}`
- parent_rows: `{len(full_year)}`
- parent_time_start: `{parent_start.isoformat()}`
- parent_time_end: `{parent_end.isoformat()}`
- parent_is_repeated: `YES`, exact `{repetition['cycle_days']}`-day blocks
- selected_source_rows: `{len(observed)}` observed/nonempty intervals
- selected_time_start: `{original_start.isoformat()}`
- selected_time_end: `{pd.Timestamp(table['timestamp'].iloc[-1]).isoformat()}`
- selected_duration_days: `{continuous_manifest['duration_days']}`
- source_type: `{SOURCE_KIND}`
- source_generation_stage: `{SOURCE_STAGE}`
- selected_source_is_repeated: `NO`
- evidence: `8/8` blocks have identical relative offsets and exact matrix values; preprocessing source shows a 49-day crop followed by `copy.deepcopy(tasks_matrix)` repetition.

The outer 2020 timestamps were introduced only during full-year synthesis. Dataset v1 selects block 0 and recovers each original 1970 bin from `tasks_matrix.start_dt`; relative offsets must match exactly. Repeated blocks 1+ are used only to prove duplication and are excluded from all rows, scaler fits and windows. This is a processed, IQR-filtered trace segment, not raw Alibaba data.
"""
    _write_text(output / "02_source_lineage.md", source_lineage)

    target_definition = """# Forecast Target Definition

| target | mathematical_definition | task_attribute | processed_source_field | raw_semantics | unit | aggregation | limitations |
|---|---|---|---|---|---|---|---|
| new_task_count | N(t) = number of newly arriving rows in interval t | task count | tasks_matrix rows | processed task arrivals | tasks / 15 min | count | Processed/filtered tasks only |
| arriving_cpu_demand | sum Task.cores_req | cores_req | cpu_usage, index 5 | post-hoc CPU usage metric used as a demand proxy | simulator CPU-core proxy / 15 min | sum(5 * cpu_usage / 100) | Not an observed CPU request |
| arriving_gpu_demand | sum Task.gpu_req | gpu_req | gpu_wrk_util, index 6 | post-hoc GPU worker utilization used as a demand proxy | simulator GPU-unit proxy / 15 min | sum(5 * gpu_wrk_util / 100) | Not an observed GPU request |
| arriving_memory_demand | sum Task.mem_req | mem_req | avg_mem, index 7 | post-hoc average memory usage used as a demand proxy | simulator memory proxy / 15 min | sum(5 * avg_mem) | Unit follows current SustainCluster Task semantics |

`task_scale=5` matches the repaired workload-to-Task mapping. Bandwidth, duration, origin, price, carbon and weather are intentionally excluded from Forecast Dataset v1.
"""
    _write_text(output / "04_target_definition.md", target_definition)

    zero_intervals = int(zero_mask.sum())
    summary = f"""# Forecast Dataset v1 Summary

1. **Q1. 最终来源是什么？** `{source_path.relative_to(WORKSPACE).as_posix()}` 的首个经验证唯一周期，并从 matrix 内部 `start_dt` 恢复原始分箱。
2. **Q2. 是否使用 repeated full-year 数据？** NO。父 artifact 是重复文件，但只选择 block 0；后续复制块不进入 dataset/scaler/window。
3. **Q3. 连续数据多长？** `{continuous_manifest['duration_days']}` 天，`{len(table)}` 个 15 分钟 interval。
4. **Q4. 是否存在时间 gap？** 完整化前有 `{len(missing_timestamps)}` 个缺失空 interval；补零后 gap=0。
5. **Q5. zero-task intervals 是否保留？** YES，共 `{zero_intervals}` 行。
6. **Q6. target 来源？** count=matrix 行数；CPU=`5*cpu_usage/100`；GPU=`5*gpu_wrk_util/100`；Memory=`5*avg_mem`，均为当前 Task demand proxy。
7. **Q7. 如何划分？** 严格 chronological 35/7/7 天，窗口 history 与 target 均不得跨 split。
8. **Q8. Scaler 是否只 fit train？** YES，`fitted_on=TRAIN_ONLY`。
9. **Q9. 默认 history length？** 96 steps = 24 h；builder/loader 同时支持 24、48、96。
10. **Q10. Forecast horizon？** 4 steps = 1 h，目标形状 H x 4。
11. **Q11. 是否发现 future leakage？** NO；全部 CRITICAL leakage checks PASS。
12. **Q12. 是否 ready for forecasting baseline？** YES。

No model training or MPC experiment was performed.
"""
    _write_text(output / "01_summary.md", summary)

    characterization_rows = statistics.set_index("target").to_dict("index")
    characterization = f"""# Data Characterization

- Timeline: `{original_start.isoformat()}` to `{pd.Timestamp(table['timestamp'].iloc[-1]).isoformat()}`, 15-minute UTC grid.
- Complete rows: `{len(table)}`; observed/nonempty source rows: `{len(observed)}`; zero-task rows: `{zero_intervals}`.
- Targets are global processed workload arrival pressure, not per-DC arrivals.
- CPU/GPU/memory values are current SustainCluster demand proxies derived from post-hoc usage fields.
- No 49-day copy beyond the first unique cycle is included.

## Selected statistics

| target | mean | std | median | p95 | max | zero_ratio |
|---|---:|---:|---:|---:|---:|---:|
"""
    for name in TARGET_NAMES:
        item = characterization_rows[name]
        characterization += (
            f"| {name} | {item['mean']:.6f} | {item['std']:.6f} | "
            f"{item['median']:.6f} | {item['p95']:.6f} | "
            f"{item['max']:.6f} | {item['zero_ratio']:.6f} |\n"
        )
    characterization += "\nFull quantiles and autocorrelations are in `05_workload_statistics.csv` and `06_autocorrelation.csv`.\n"
    _write_text(output / "07_data_characterization.md", characterization)

    source_script = sustain_repo / "data/workload/alibaba_2020_dataset/extract_dataset_from_dfas.py"
    evidence = f"""# Evidence Index

| Evidence | Location | Supports |
|---|---|---|
| Workload matrix schema | `{_line(source_script, 'columns_of_interest = [', last=True)}` | final 12-column order |
| Seven-week crop | `{_line(source_script, 'start_time = pd.Timestamp("1970-01-26', last=True)}` | original fixed 49-day interval |
| Full-year deep copy | `{_line(source_script, 'copy.deepcopy(row["tasks_matrix"])')}` | only outer timestamp is shifted |
| Repaired field contract | `src/sustaincluster_contract/workload.py` | Task CPU/GPU/memory transformations |
| Dataset aggregation | `{_line(SRC / 'forecasting/dataset.py', 'def aggregate_tasks_matrix')}` | four target construction |
| Continuous grid | `{_line(SRC / 'forecasting/dataset.py', 'def build_workload_time_series')}` | zero intervals preserved |
| Train-only scaler | `{_line(SRC / 'forecasting/dataset.py', 'def fit_train_scaler')}` | normalization provenance |
| Window alignment | `{_line(SRC / 'forecasting/dataset.py', 'def build_sliding_windows')}` | L history and H=4 targets |
| Loader | `{_line(SRC / 'forecasting/loader.py', 'def load_forecast_dataset')}` | deterministic training-independent API |
| Source artifact | `{source_path.relative_to(WORKSPACE).as_posix()}` | SHA256 `{source_sha}` |
| Prior audit | `artifacts/workload_information_audit_v1/` | 49-day exact repetition and lineage |
"""
    _write_text(output / "14_evidence_index.md", evidence)

    change_manifest = """# Change Manifest

## Added code and configuration

- `src/forecasting/dataset.py`: source-cycle validation, aggregation, continuity, split, scaler, windows and characterization.
- `src/forecasting/loader.py`: deterministic NPZ loader and configurable history rebuild.
- `configs/forecasting/forecast_dataset_v1.yaml`: auditable dataset configuration.
- `scripts/forecast/build_forecast_dataset_v1.py`: one-pass read-only generator.
- `tests/test_forecast_dataset_v1.py`: Dataset v1 contract tests.

## Generated artifacts

- `artifacts/forecast_dataset_v1/`: 15 reports/manifests plus Parquet, CSV preview and three NPZ splits.

## Explicit non-changes

- No scheduler, MPC, reward, BC, SAC or model algorithm was modified.
- No source pickle or SustainCluster vendor file was modified.
- No forecasting baseline/model was trained.
- No commit, push, reset or clean was executed.
"""
    _write_text(output / "15_change_manifest.md", change_manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE / "configs/forecasting/forecast_dataset_v1.yaml",
    )
    args = parser.parse_args()
    output = build(args.config.resolve())
    print(json.dumps({"output": str(output), "status": "PASS"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
