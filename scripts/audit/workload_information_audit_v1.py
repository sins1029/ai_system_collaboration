#!/usr/bin/env python3
"""Read-only workload lineage, periodicity, and information-boundary audit.

This script reads the existing SustainCluster workload and source tree. It does
not mutate the workload, run training, or import project runtime modules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUSTAIN = ROOT / "references" / "external_repos" / "sustain-cluster"
DEFAULT_WORKLOAD = (
    SUSTAIN
    / "data"
    / "workload"
    / "alibaba_2020_dataset"
    / "result_df_full_year_2020.pkl"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "workload_information_audit_v1"
PERIOD_DAYS = 49
GRID_MINUTES = 15
PERIOD_GRID_STEPS = PERIOD_DAYS * 24 * 60 // GRID_MINUTES

MATRIX_COLUMNS = [
    "job_name",
    "start_time",
    "end_time",
    "start_dt",
    "duration_min",
    "cpu_usage",
    "gpu_wrk_util",
    "avg_mem",
    "avg_gpu_wrk_mem",
    "bandwidth_gb",
    "weekday_name",
    "weekday_num",
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def git(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def stable_matrix_hash(matrix: Any) -> str:
    return hashlib.sha256(pickle.dumps(matrix, protocol=4)).hexdigest()


def as_float(values: Iterable[Any]) -> np.ndarray:
    return pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return "UNKNOWN"
        return f"{float(value):.{digits}f}"
    return str(value)


def pct(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {key: float("nan") for key in ("min", "p50", "p90", "p95", "p99", "max")}
    points = np.quantile(finite, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    return dict(zip(("min", "p50", "p90", "p95", "p99", "max"), points.tolist()))


def audit_workload(workload: Path) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    frame = pd.read_pickle(workload)
    if list(frame.columns) != ["interval_15m", "tasks_matrix"]:
        raise ValueError(f"Unexpected workload columns: {list(frame.columns)!r}")

    frame = frame.sort_values("interval_15m", kind="stable").reset_index(drop=True)
    timestamps = pd.to_datetime(frame["interval_15m"], utc=True)
    aggregates: list[dict[str, Any]] = []
    column_count_counter: Counter[int] = Counter()
    all_duration: list[np.ndarray] = []
    all_span: list[np.ndarray] = []
    internal_external_weekday_matches = 0
    internal_external_weekday_total = 0
    internal_start_min: pd.Timestamp | None = None
    internal_start_max: pd.Timestamp | None = None

    for outer_time, matrix in zip(timestamps, frame["tasks_matrix"]):
        rows = list(matrix)
        count = len(rows)
        for row in rows:
            column_count_counter[len(row)] += 1

        if count:
            duration = as_float(row[4] for row in rows)
            cpu = as_float(row[5] for row in rows)
            gpu = as_float(row[6] for row in rows)
            memory = as_float(row[7] for row in rows)
            gpu_memory = as_float(row[8] for row in rows)
            bandwidth = as_float(row[9] for row in rows)
            start_seconds = as_float(row[1] for row in rows)
            end_seconds = as_float(row[2] for row in rows)
            span_minutes = (end_seconds - start_seconds) / 60.0
            all_duration.append(duration)
            all_span.append(span_minutes)

            internal_times = pd.to_datetime([row[3] for row in rows], utc=True, errors="coerce")
            valid_times = internal_times[~pd.isna(internal_times)]
            if len(valid_times):
                row_min = valid_times.min()
                row_max = valid_times.max()
                internal_start_min = row_min if internal_start_min is None else min(internal_start_min, row_min)
                internal_start_max = row_max if internal_start_max is None else max(internal_start_max, row_max)

            outer_weekday = int(outer_time.weekday())
            internal_weekdays = pd.to_numeric(
                pd.Series([row[11] for row in rows]), errors="coerce"
            ).to_numpy(dtype=float)
            valid_weekdays = np.isfinite(internal_weekdays)
            internal_external_weekday_matches += int(
                np.sum(internal_weekdays[valid_weekdays] == outer_weekday)
            )
            internal_external_weekday_total += int(np.sum(valid_weekdays))
        else:
            duration = cpu = gpu = memory = gpu_memory = bandwidth = np.array([], dtype=float)

        aggregates.append(
            {
                "interval_15m": outer_time,
                "task_count": count,
                "cpu_total": float(np.nansum(cpu)),
                "gpu_total": float(np.nansum(gpu)),
                "memory_total": float(np.nansum(memory)),
                "duration_total": float(np.nansum(duration)),
                "duration_mean": float(np.nanmean(duration)) if count else 0.0,
                "source_bandwidth_total_col9": float(np.nansum(bandwidth)),
                "runtime_bandwidth_total_col8": float(np.nansum(gpu_memory)),
                "matrix_sha256": stable_matrix_hash(matrix),
            }
        )

    aggregate_frame = pd.DataFrame(aggregates)
    durations = np.concatenate(all_duration) if all_duration else np.array([], dtype=float)
    spans = np.concatenate(all_span) if all_span else np.array([], dtype=float)
    finite_pair = np.isfinite(durations) & np.isfinite(spans)
    duration_error = np.abs(durations[finite_pair] - spans[finite_pair])

    delta_minutes = timestamps.diff().dropna().dt.total_seconds().div(60)
    expected_grid_rows = int((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() // 900 + 1)
    block_index = ((timestamps - timestamps.iloc[0]).dt.total_seconds() // (PERIOD_DAYS * 86400)).astype(int)
    block_counts = Counter(int(item) for item in block_index)

    lagged = aggregate_frame.copy()
    lagged["interval_15m"] = lagged["interval_15m"] - pd.Timedelta(days=PERIOD_DAYS)
    pairs = aggregate_frame.merge(lagged, on="interval_15m", suffixes=("_left", "_right"))
    metrics = [
        "task_count",
        "cpu_total",
        "gpu_total",
        "memory_total",
        "duration_total",
        "duration_mean",
        "source_bandwidth_total_col9",
        "runtime_bandwidth_total_col8",
    ]
    periodicity_rows: list[dict[str, Any]] = []
    for metric in metrics:
        left = pairs[f"{metric}_left"].to_numpy(dtype=float)
        right = pairs[f"{metric}_right"].to_numpy(dtype=float)
        correlation = float(np.corrcoef(left, right)[0, 1]) if len(left) > 1 else float("nan")
        periodicity_rows.append(
            {
                "lag_days": PERIOD_DAYS,
                "lag_grid_steps": PERIOD_GRID_STEPS,
                "metric": metric,
                "paired_observed_intervals": len(left),
                "pearson_correlation": correlation,
                "exact_match_rate": float(np.mean(np.isclose(left, right, rtol=0.0, atol=0.0))),
                "mean_absolute_difference": float(np.mean(np.abs(left - right))),
                "conclusion": "EXACT_49_DAY_REPETITION" if np.array_equal(left, right) else "NOT_EXACT",
            }
        )

    signature_left = pairs["matrix_sha256_left"].astype(str).to_numpy()
    signature_right = pairs["matrix_sha256_right"].astype(str).to_numpy()
    signature_match_rate = float(np.mean(signature_left == signature_right))
    periodicity_rows.append(
        {
            "lag_days": PERIOD_DAYS,
            "lag_grid_steps": PERIOD_GRID_STEPS,
            "metric": "entire_tasks_matrix_sha256",
            "paired_observed_intervals": len(signature_left),
            "pearson_correlation": "NOT_APPLICABLE",
            "exact_match_rate": signature_match_rate,
            "mean_absolute_difference": "NOT_APPLICABLE",
            "conclusion": "EXACT_49_DAY_REPETITION" if signature_match_rate == 1.0 else "NOT_EXACT",
        }
    )

    stats = {
        "row_count": int(len(frame)),
        "timestamp_min": timestamps.iloc[0].isoformat(),
        "timestamp_max": timestamps.iloc[-1].isoformat(),
        "duplicate_timestamp_count": int(timestamps.duplicated().sum()),
        "expected_complete_grid_rows": expected_grid_rows,
        "missing_15m_grid_rows": expected_grid_rows - len(frame),
        "fifteen_minute_delta_rate": float(np.mean(delta_minutes == GRID_MINUTES)),
        "delta_minutes_counts": {str(k): int(v) for k, v in delta_minutes.value_counts().sort_index().items()},
        "rows_in_2020": int(np.sum(timestamps.dt.year == 2020)),
        "rows_after_2020": int(np.sum(timestamps.dt.year > 2020)),
        "period_days": PERIOD_DAYS,
        "period_grid_steps": PERIOD_GRID_STEPS,
        "observed_rows_per_full_block": int(block_counts.get(0, 0)),
        "block_row_counts": {str(k): int(v) for k, v in sorted(block_counts.items())},
        "paired_49_day_intervals": int(len(pairs)),
        "tasks_matrix_exact_match_rate": signature_match_rate,
        "full_year_repetition": signature_match_rate == 1.0,
        "total_task_records": int(len(durations)),
        "matrix_column_count_distribution": {str(k): int(v) for k, v in sorted(column_count_counter.items())},
        "duration_minutes_quantiles": quantiles(durations),
        "task_span_minutes_quantiles": quantiles(spans),
        "duration_vs_task_span_exact_rate": float(
            np.mean(np.isclose(durations[finite_pair], spans[finite_pair], rtol=1e-12, atol=1e-12))
        ),
        "duration_vs_task_span_mean_abs_minutes": float(np.mean(duration_error)),
        "duration_vs_task_span_median_abs_minutes": float(np.median(duration_error)),
        "internal_start_dt_min": internal_start_min.isoformat() if internal_start_min is not None else None,
        "internal_start_dt_max": internal_start_max.isoformat() if internal_start_max is not None else None,
        "internal_vs_outer_weekday_match_rate": (
            internal_external_weekday_matches / internal_external_weekday_total
            if internal_external_weekday_total
            else float("nan")
        ),
    }
    return aggregate_frame, stats, periodicity_rows


def lineage_rows() -> list[dict[str, str]]:
    return [
        {
            "stage": "L01 raw tables",
            "input": "pai_job_table/pai_task_table/pai_instance_table/pai_sensor_table + headers",
            "output": "in-memory Alibaba tables",
            "transform": "Read raw CSVs with header definitions",
            "locally_recomputable": "NO; raw CSV/header files are absent",
            "evidence": "references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/alibaba_utils.py:19",
            "audit_conclusion": "Source logic exists, raw evidence is unavailable locally",
        },
        {
            "stage": "L02 instance runtime",
            "input": "instance start_time/end_time",
            "output": "runtime_i",
            "transform": "Drop nulls/duplicates; runtime_i=end_time-start_time",
            "locally_recomputable": "NO",
            "evidence": "references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/alibaba_utils.py:35",
            "audit_conclusion": "Post-hoc realized runtime is derived from completed instances",
        },
        {
            "stage": "L03 task aggregation",
            "input": "task + job + instance + group",
            "output": "dfas task rows",
            "transform": "Task start=min(instance start), end=max(instance end), runtime=mean(runtime_i), duration_min=runtime_i/60",
            "locally_recomputable": "NO",
            "evidence": "references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/alibaba_utils.py:55",
            "audit_conclusion": "duration_min is a realized-runtime label, not a declared request",
        },
        {
            "stage": "L04 task filtering",
            "input": "raw pai task fields",
            "output": "valid task analysis rows",
            "transform": "Numeric coercion, dedupe, nonnegative time/resource filters, duration>=15 min",
            "locally_recomputable": "NO",
            "evidence": "references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/analyze_alibaba2020GPU.py:34",
            "audit_conclusion": "Filtering code is visible; exact production execution cannot be replayed",
        },
        {
            "stage": "L05 bandwidth aggregation",
            "input": "sensor read/write metrics",
            "output": "bandwidth_gb by job_name/task_name",
            "transform": "(read+write)/1024^3 then aggregate",
            "locally_recomputable": "NO",
            "evidence": "references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/alibaba_analysis.ipynb:5464",
            "audit_conclusion": "Raw sensor and extracted CSV inputs are absent",
        },
        {
            "stage": "L06 bandwidth merge",
            "input": "dfas + per-task bandwidth",
            "output": "extracted_dfas_with_bandwidth.csv",
            "transform": "Merge on job_name only; fill missing with 0; drop task_name",
            "locally_recomputable": "NO",
            "evidence": "references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/alibaba_analysis.ipynb:5520",
            "audit_conclusion": "Potential task-level duplication/misassociation; cannot quantify without intermediate CSV",
        },
        {
            "stage": "L07 timestamp normalization",
            "input": "numeric start_time",
            "output": "UTC start_dt + weekday + 15m interval",
            "transform": "Unix parse, localize Asia/Shanghai, convert UTC, floor 15m",
            "locally_recomputable": "PARTIAL; final rows only",
            "evidence": "references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/extract_dataset_from_dfas.py:222",
            "audit_conclusion": "Final task contents retain 1970 timestamps",
        },
        {
            "stage": "L08 outlier filtering",
            "input": "duration/cpu/memory/bandwidth",
            "output": "IQR-filtered rows",
            "transform": "Sequential 1.5*IQR filtering per metric",
            "locally_recomputable": "NO",
            "evidence": "references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/extract_dataset_from_dfas.py:285",
            "audit_conclusion": "Order-dependent filtering; intermediate evidence absent",
        },
        {
            "stage": "L09 task matrix",
            "input": "filtered task rows",
            "output": "12-column tasks_matrix grouped by nonempty 15m bins",
            "transform": "Group observed intervals only and sort each matrix",
            "locally_recomputable": "YES; final pickle includes matrices",
            "evidence": "references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/extract_dataset_from_dfas.py:346",
            "audit_conclusion": "Column 8 is GPU worker memory; column 9 is bandwidth",
        },
        {
            "stage": "L10 seven-week crop",
            "input": "grouped 1970 trace",
            "output": "1970-01-26 through seven weeks",
            "transform": "Inclusive start, exclusive seven-week end",
            "locally_recomputable": "YES; visible in retained inner timestamps",
            "evidence": "references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/extract_dataset_from_dfas.py:397",
            "audit_conclusion": "Base trace spans 49 days with 10 empty bins omitted",
        },
        {
            "stage": "L11 full-year synthesis",
            "input": "seven-week task matrices",
            "output": "result_df_full_year_2020.pkl",
            "transform": "Repeat entire trace blocks; shift outer interval only; stop after appending a whole block",
            "locally_recomputable": "YES",
            "evidence": "references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/extract_dataset_from_dfas.py:424",
            "audit_conclusion": "Exact 49-day repetition and rows through 2021-01-26",
        },
        {
            "stage": "L12 simulator load",
            "input": "result_df_full_year_2020.pkl",
            "output": "ClusterManager workload_df",
            "transform": "Read/unzip pickle; query first matching outer interval",
            "locally_recomputable": "YES",
            "evidence": "references/external_repos/sustain-cluster/simulation/cluster_manager.py:134",
            "audit_conclusion": "Simulator uses synthesized outer time, not retained inner start_dt",
        },
        {
            "stage": "L13 runtime Task extraction",
            "input": "12-column matrix",
            "output": "Task fields",
            "transform": "arrival=current time; duration=col4; CPU/GPU/memory=5x cols5/6/7; bandwidth=col8",
            "locally_recomputable": "YES",
            "evidence": "references/external_repos/sustain-cluster/utils/workload_utils.py:74",
            "audit_conclusion": "Bandwidth mapping is wrong: col8 is GPU memory, true bandwidth is col9",
        },
        {
            "stage": "L14 synthetic origin",
            "input": "DC population weights + local hour",
            "output": "task origin",
            "transform": "Random weighted assignment with 08:00-20:00 activity multiplier",
            "locally_recomputable": "YES with seed, but not an observed label",
            "evidence": "references/external_repos/sustain-cluster/utils/workload_utils.py:7",
            "audit_conclusion": "Origin is synthetic and seed/order dependent",
        },
        {
            "stage": "L15 runtime release",
            "input": "scheduled start + duration",
            "output": "finish_time and resource release",
            "transform": "finish_time=start_time+duration; release when finish_time<=current_time",
            "locally_recomputable": "YES",
            "evidence": "references/external_repos/sustain-cluster/envs/sustaindc/sustaindc_env.py:166",
            "audit_conclusion": "Deterministic release is oracle-contaminated because duration is realized runtime",
        },
    ]


def lineage_contract_rows() -> list[dict[str, str]]:
    functions = {
        "L01 raw tables": "get_dfs / pd.read_csv",
        "L02 instance runtime": "get_dfa",
        "L03 task aggregation": "get_dfa",
        "L04 task filtering": "analysis filtering block",
        "L05 bandwidth aggregation": "notebook sensor groupby",
        "L06 bandwidth merge": "notebook merge block",
        "L07 timestamp normalization": "process_alibaba_data",
        "L08 outlier filtering": "remove_outliers_iqr",
        "L09 task matrix": "group_tasks_by_interval",
        "L10 seven-week crop": "crop_to_seven_weeks",
        "L11 full-year synthesis": "repeat_trace_for_year",
        "L12 simulator load": "DatacenterClusterManager.get_tasks_for_timestep",
        "L13 runtime Task extraction": "extract_tasks_from_row",
        "L14 synthetic origin": "assign_task_origins",
        "L15 runtime release": "SustainDCEnv step/release block",
    }
    operation_flags = {
        "L01 raw tables": (),
        "L02 instance runtime": ("filtering", "aggregation"),
        "L03 task aggregation": ("aggregation",),
        "L04 task filtering": ("filtering",),
        "L05 bandwidth aggregation": ("aggregation", "normalization"),
        "L06 bandwidth merge": ("aggregation",),
        "L07 timestamp normalization": ("normalization",),
        "L08 outlier filtering": ("filtering", "outlier_removal"),
        "L09 task matrix": ("aggregation",),
        "L10 seven-week crop": ("filtering",),
        "L11 full-year synthesis": ("replication", "synthetic_augmentation"),
        "L12 simulator load": (),
        "L13 runtime Task extraction": ("normalization", "synthetic_augmentation"),
        "L14 synthetic origin": ("randomization", "synthetic_augmentation"),
        "L15 runtime release": (),
    }
    information_loss = {
        "L02 instance runtime": "Null/duplicate workers removed; completed execution is collapsed to runtime.",
        "L03 task aggregation": "Instance-level distribution is reduced to min start, max end, and mean runtime.",
        "L04 task filtering": "Invalid rows and tasks shorter than 15 minutes are removed.",
        "L06 bandwidth merge": "task_name key is dropped after a job_name-only merge.",
        "L08 outlier filtering": "Sequential IQR removal changes workload tails.",
        "L09 task matrix": "Empty 15-minute intervals are omitted and order is materialized.",
        "L10 seven-week crop": "All events outside the selected 49-day interval are removed.",
        "L11 full-year synthesis": "No new temporal information; seven-week content is copied.",
        "L13 runtime Task extraction": "Raw start/end/start_dt/weekday and true column-9 bandwidth are not mapped.",
        "L14 synthetic origin": "Observed geographic provenance is unavailable.",
    }
    rows = []
    for source in lineage_rows():
        stage = source["stage"]
        evidence_file, evidence_lines = source["evidence"].rsplit(":", 1)
        flags = set(operation_flags.get(stage, ()))
        rows.append(
            {
                "stage": stage,
                "source_file": evidence_file,
                "function": functions.get(stage, "code block"),
                "input_fields": source["input"],
                "output_fields": source["output"],
                "transformation": source["transform"],
                "information_loss": information_loss.get(stage, "None identified in this step."),
                "filtering": "YES" if "filtering" in flags else "NO",
                "aggregation": "YES" if "aggregation" in flags else "NO",
                "normalization": "YES" if "normalization" in flags else "NO",
                "clipping": "YES" if "clipping" in flags else "NO",
                "outlier_removal": "YES" if "outlier_removal" in flags else "NO",
                "randomization": "YES" if "randomization" in flags else "NO",
                "replication": "YES" if "replication" in flags else "NO",
                "synthetic_augmentation": "YES" if "synthetic_augmentation" in flags else "NO",
                "evidence_file": evidence_file,
                "evidence_lines": evidence_lines,
                "notes": f"{source['audit_conclusion']} Recomputable: {source['locally_recomputable']}.",
            }
        )
    return rows



def information_contract_rows() -> list[dict[str, str]]:
    fields = [
        "field", "raw_source", "processed_source", "task_object",
        "environment_observation", "mpc_available", "rl_available",
        "classification", "reason", "risk", "evidence_file", "evidence_lines",
    ]
    ext = "references/external_repos/sustain-cluster"
    ali = f"{ext}/data/workload/alibaba_2020_dataset"
    data = [
        ("job_name / task id", "Alibaba job_name", "Retained; Task adds random suffix", "job_name", "Not in native numeric obs", "Metadata only", "Not encoded by FeatureEncoder", "E. SYNTHETIC", "Source ID is historical but runtime identity is mutated randomly.", "Split leakage if source identity crosses folds.", f"{ext}/rl_components/task.py", "77-78"),
        ("arrival_time", "Task start event", "Replaced by current outer simulation time", "arrival_time", "Indirect via wait/deadline", "YES", "Indirect", "A. ONLINE_KNOWN", "A real submission time is known when the task arrives.", "Low.", f"{ext}/utils/workload_utils.py", "74-85"),
        ("start_time", "Trace execution start", "Retained in matrix; not mapped to Task", "NO", "NO", "NO", "NO", "F. DROPPED", "Historical execution start is not used directly at runtime.", "Using it as an arrival feature would leak scheduling outcome.", f"{ali}/extract_dataset_from_dfas.py", "350-363"),
        ("end_time", "Trace execution end", "Used upstream to derive realized runtime; not mapped", "NO", "NO", "NO", "NO", "F. DROPPED", "Completion timestamp is unavailable at online decision time.", "Critical if restored as an input.", f"{ali}/alibaba_utils.py", "35-43,55-83"),
        ("duration", "Mean realized instance runtime", "tasks_matrix column 4", "duration", "YES", "YES", "YES", "D. ORACLE_FUTURE", "Current code uses a completed-trace outcome, not a declared estimate.", "Critical oracle contamination.", f"{ali}/alibaba_utils.py", "55-83"),
        ("cores_req", "Post-hoc cpu_usage", "5x matrix column 5", "cores_req", "YES", "YES", "YES", "E. SYNTHETIC", "The simulator scales a utilization metric and treats it as a request.", "Deployment semantics mismatch.", f"{ext}/utils/workload_utils.py", "74-85"),
        ("gpu_req", "Post-hoc gpu_wrk_util", "5x matrix column 6", "gpu_req", "YES", "YES", "YES", "E. SYNTHETIC", "The simulator scales utilization and treats it as a request.", "Deployment semantics mismatch.", f"{ext}/utils/workload_utils.py", "74-85"),
        ("mem_req", "Post-hoc avg_mem", "5x matrix column 7", "mem_req", "Not in native per-task obs", "YES", "YES", "E. SYNTHETIC", "The simulator scales average usage and treats it as a request.", "Deployment semantics mismatch.", f"{ext}/utils/workload_utils.py", "74-85"),
        ("bandwidth_gb (source)", "Sensor read/write aggregate", "tasks_matrix column 9", "NO", "NO", "NO", "NO", "F. DROPPED", "Runtime extraction does not map column 9.", "Confirmed data-integrity loss.", f"{ali}/extract_dataset_from_dfas.py", "350-363"),
        ("bandwidth_gb (runtime)", "avg_gpu_wrk_mem", "Incorrectly reads matrix column 8", "bandwidth_gb", "Not in native obs", "YES", "YES", "E. SYNTHETIC", "Current value is GPU worker memory, not bandwidth.", "Critical transmission-cost corruption.", f"{ext}/utils/workload_utils.py", "74-85"),
        ("origin_dc_id", "No observed origin", "Random population/hour-weighted assignment", "origin", "YES", "YES", "YES", "E. SYNTHETIC", "Regional origin is simulator-generated.", "Regional labels are seed/order dependent.", f"{ext}/utils/workload_utils.py", "7-43"),
        ("dest_dc_id", "None", "Controller action", "destination after action", "Action, not state", "Decision variable", "Action output", "A. ONLINE_KNOWN", "Destination is chosen at the current timestep.", "Low.", f"{ext}/envs/task_scheduling_env.py", "255-356"),
        ("sla_deadline", "No independent SLA evidenced", "arrival + 1.5 * realized duration", "sla_deadline", "YES", "YES", "YES", "D. ORACLE_FUTURE", "Synthetic SLA directly depends on oracle duration.", "Critical.", f"{ext}/rl_components/task.py", "58-60"),
        ("time_to_deadline", "Derived", "deadline - current time", "Derived", "YES", "YES", "YES", "D. ORACLE_FUTURE", "It inherits oracle duration through the current deadline rule.", "Critical.", "src/sustaincluster_mpc/state_adapter.py", "103-191"),
        ("wait_intervals", "Scheduler history", "Current time - arrival / 15 min", "wait/deferred state", "YES", "YES", "YES", "A. ONLINE_KNOWN", "Known from past scheduler decisions.", "Low.", "src/sustaincluster_mpc/state_adapter.py", "103-191"),
        ("current available CPU", "Current DC telemetry", "Current simulator state", "N/A", "YES", "YES", "YES", "A. ONLINE_KNOWN", "Observable at decision time.", "Low.", "src/sustaincluster_mpc/state_adapter.py", "193-302"),
        ("current available GPU", "Current DC telemetry", "Current simulator state", "N/A", "YES", "YES", "YES", "A. ONLINE_KNOWN", "Observable at decision time.", "Low.", "src/sustaincluster_mpc/state_adapter.py", "193-302"),
        ("current available memory", "Current DC telemetry", "Current simulator state", "N/A", "YES", "YES", "YES", "A. ONLINE_KNOWN", "Observable at decision time.", "Low.", "src/sustaincluster_mpc/state_adapter.py", "193-302"),
        ("electricity price (current)", "Historical series at current index", "Price manager", "N/A", "YES", "YES", "YES", "A. ONLINE_KNOWN", "Current price is observable.", "Low.", f"{ext}/utils/managers.py", "499-559"),
        ("carbon intensity (current)", "Historical series at current index", "CI manager", "N/A", "YES", "YES", "YES", "A. ONLINE_KNOWN", "Current carbon intensity is observable.", "Low.", f"{ext}/utils/managers.py", "236-303"),
        ("weather (current)", "Historical weather at current index", "Weather manager", "N/A", "Current environment diagnostics", "StateAdapter diagnostics", "FeatureEncoder diagnostics where present", "A. ONLINE_KNOWN", "Current telemetry can be observed.", "Low.", f"{ext}/utils/managers.py", "337-399"),
        ("current pending tasks", "Current queue", "Environment task lists", "Task objects", "YES", "YES", "YES", "A. ONLINE_KNOWN", "Current queue contents are known.", "Low.", f"{ext}/envs/task_scheduling_env.py", "389-397"),
        ("running tasks", "Current DC state", "Environment running lists", "Task objects", "Indirect", "YES", "YES via state summaries", "A. ONLINE_KNOWN", "Running assignments are scheduler state.", "Low.", "src/sustaincluster_mpc/state_adapter.py", "193-302"),
        ("finish_time", "scheduled start + realized duration", "Stored by environment", "finish_time", "Not native numeric obs", "YES", "YES via release timeline", "D. ORACLE_FUTURE", "Deterministic only because current duration is oracle truth.", "Critical.", f"{ext}/envs/sustaindc/sustaindc_env.py", "166-220"),
        ("expected capacity release", "finish_time and task resources", "Horizon release timeline", "Derived", "NO", "YES", "YES via H4 features", "D. ORACLE_FUTURE", "Current release uses exact realized duration; deployment release should be estimated.", "Critical.", "src/sustaincluster_mpc/horizon_adapter.py", "167-193,381-447"),
        ("future task arrivals", "Future workload rows", "Oracle/noisy-oracle or causal forecast", "Future TaskSnapshot", "NO", "Mode-dependent", "Causal baseline in current runners", "C. FORECAST_REQUIRED", "Unknown at decision time; oracle modes use truth only as upper bounds.", "Critical if oracle mode is reported as deployable.", "src/sustaincluster_mpc/horizon_adapter.py", "122-321"),
        ("future CPU demand", "Future arriving tasks", "Aggregate per origin/horizon", "N/A", "NO", "Mode-dependent", "Forecast feature", "C. FORECAST_REQUIRED", "Unknown new demand must be forecast.", "High if repeated trace crosses splits.", "src/sustaincluster_imitation/forecast_baseline.py", "82-181"),
        ("future GPU demand", "Future arriving tasks", "Aggregate per origin/horizon", "N/A", "NO", "Mode-dependent", "Forecast feature", "C. FORECAST_REQUIRED", "Unknown new demand must be forecast.", "High if repeated trace crosses splits.", "src/sustaincluster_imitation/forecast_baseline.py", "82-181"),
        ("future memory demand", "Future arriving tasks", "Aggregate per origin/horizon", "N/A", "NO", "Mode-dependent", "Forecast feature", "C. FORECAST_REQUIRED", "Unknown new demand must be forecast.", "High if repeated trace crosses splits.", "src/sustaincluster_imitation/forecast_baseline.py", "82-181"),
        ("future electricity price", "Future historical price array", "Direct HorizonAdapter slice", "N/A", "NO", "YES", "YES via FeatureEncoder", "D. ORACLE_FUTURE", "Current code has no as-of forecast or schedule contract.", "High future leakage.", "src/sustaincluster_mpc/horizon_adapter.py", "448-456"),
        ("future carbon intensity", "Future historical CI array", "Direct HorizonAdapter slice", "N/A", "NO", "YES", "YES via FeatureEncoder", "D. ORACLE_FUTURE", "Current code reads future truth.", "High future leakage.", "src/sustaincluster_mpc/horizon_adapter.py", "448-456"),
        ("future weather", "Historical weather series", "No H4 future-weather feature identified", "N/A", "NO", "NO", "NO", "C. FORECAST_REQUIRED", "Would require forecasting if introduced.", "Currently not a controller leakage path.", f"{ext}/utils/managers.py", "337-399"),
        ("workload type", "No stable type field in final matrix", "Not mapped", "NO", "NO", "NO", "NO", "G. UNCLEAR", "Local final schema has job ID and resource metrics but no validated workload class.", "Do not invent training/inference labels.", f"{ali}/extract_dataset_from_dfas.py", "350-363"),
    ]
    return [dict(zip(fields, row)) for row in data]



def forecast_contract_rows() -> list[dict[str, str]]:
    fields = [
        "target", "currently_known?", "forecast_required?",
        "available_training_label?", "temporal_resolution",
        "usable_by_H4_MPC?", "risk_of_leakage", "recommended_v1?", "reason",
    ]
    data = [
        ("new task count", "NO", "YES", "YES after deduplication", "15 min x origin x horizon", "YES", "HIGH: exact 49-day copies", "YES", "Primary arrival-pressure signal and already supported by the causal baseline interface."),
        ("arriving CPU demand", "NO", "YES", "YES with semantic warning", "15 min x origin x horizon", "YES", "HIGH: repetition and usage/request mismatch", "YES_WITH_RESTRICTIONS", "Include in the v0.1 aggregate demand vector with explicit units and provenance."),
        ("arriving GPU demand", "NO", "YES", "YES with semantic warning", "15 min x origin x horizon", "YES", "HIGH: repetition and scaled utilization", "YES_WITH_RESTRICTIONS", "GPU scarcity is horizon-sensitive."),
        ("arriving memory demand", "NO", "YES", "YES with semantic warning", "15 min x origin x horizon", "YES", "HIGH: repetition and scaled average usage", "YES_WITH_RESTRICTIONS", "Completes the future capacity-demand vector."),
        ("arriving bandwidth demand", "NO", "YES", "BLOCKED", "15 min x origin x horizon", "YES", "CRITICAL: merge ambiguity and runtime off-by-one", "NO", "Defer until bandwidth lineage and mapping are repaired."),
        ("workload type", "UNKNOWN", "POSSIBLY", "NO validated label", "Task or 15 min mix", "POSSIBLY", "HIGH if inferred from IDs", "NO", "Final schema has no validated workload class."),
        ("runtime / duration", "NO; may be user-estimated", "YES for realized runtime", "YES as post-hoc label", "Task level", "YES", "CRITICAL if realized label is used as input", "NO_FOR_V0.1", "First separate realized labels from online estimates; model later with uncertainty."),
        ("capacity release", "PARTLY", "ONLY UNCERTAINTY", "Current label is oracle-derived", "DC x horizon", "YES", "CRITICAL under current duration contract", "DERIVE_NOT_PRIMARY", "Compute known release from online runtime estimates; forecast only uncertainty and new arrivals."),
        ("electricity price", "Current only", "YES unless schedule is published", "Historical label exists", "DC x horizon", "YES", "HIGH: current H4 reads future truth", "SEPARATE_PROVIDER", "Use an as-of market forecast or documented day-ahead schedule."),
        ("carbon intensity", "Current only", "YES", "Historical label exists", "DC x horizon", "YES", "HIGH: current H4 reads future truth", "SEPARATE_PROVIDER", "Preserve issue time and lead time."),
        ("weather", "Current only", "YES if used in horizon", "Historical label exists", "DC x horizon", "Not currently", "MEDIUM", "NO_FOR_V0.1", "Not in the current H4 future feature set."),
    ]
    return [dict(zip(fields, row)) for row in data]


def controller_contract_rows() -> list[dict[str, str]]:
    fields = [
        "information", "environment", "RL_actor", "RL_critic",
        "MPC_H1", "MPC_H4", "source", "online_realistic", "notes",
    ]
    data = [
        ("task identity", "Object only; not native numeric obs", "Not encoded", "Not encoded", "Metadata only", "Metadata only", "Task.job_name", "PARTIAL", "Random suffix mutates identity; source IDs repeat across blocks."),
        ("duration", "YES", "YES", "YES", "YES", "YES", "Realized trace runtime", "NO", "Shared oracle risk, not an MPC-only advantage."),
        ("deadline / time-to-deadline", "YES", "YES", "YES", "YES", "YES", "arrival + 1.5*duration", "NO", "Inherits oracle duration."),
        ("defer history / wait", "YES", "YES", "YES", "YES", "YES", "Current scheduler state", "YES", "Causal and fair."),
        ("task memory / bandwidth", "Memory omitted; bandwidth omitted", "YES", "YES", "YES", "YES", "StateAdapter/FeatureEncoder", "NO for bandwidth", "Runtime bandwidth is miswired to GPU memory."),
        ("current DC capacity", "YES", "YES", "YES", "YES", "YES", "Current simulator telemetry", "YES", "Causal and fair."),
        ("running/pending tasks", "YES", "Summaries", "Summaries", "YES", "YES", "Current environment state", "YES", "Causal."),
        ("current price/carbon", "YES", "YES", "YES", "YES", "YES", "Current manager index", "YES", "Causal."),
        ("future price", "NO in native obs", "YES", "YES", "NO multi-step", "YES", "Direct historical future array", "NO", "Actor/Critic and H4 MPC share this oracle field in current FeatureEncoder runners."),
        ("future carbon", "NO in native obs", "YES", "YES", "NO multi-step", "YES", "Direct historical future array", "NO", "Actor/Critic and H4 MPC share this oracle field."),
        ("future weather", "NO", "NO", "NO", "NO", "NO", "Not in HorizonAdapter future features", "N/A", "No current information-parity issue."),
        ("exact capacity release", "Implicit transition only", "YES via H4", "YES via H4", "Current finish metadata", "YES", "finish_time from realized duration", "NO", "Shared in FeatureEncoder H4; optimistic release timeline."),
        ("future arrivals", "NO", "Causal baseline in current runners", "Same as actor", "NO", "Mode-dependent: none/causal/oracle", "HistoricalArrivalForecaster or future workload lookup", "PARTIAL", "Oracle/noisy-oracle MPC modes are privileged upper bounds; current SAC runners use causal arrivals."),
        ("future CPU/GPU/memory demand", "NO", "Causal aggregate forecast", "Same as actor", "NO", "Mode-dependent", "Forecast baseline or future true tasks", "PARTIAL", "Fair only when both controllers receive the same causal forecast."),
        ("candidate action", "Action input", "Output distribution", "YES", "Decision variable", "Decision variable", "Actor/Critic/optimizer", "YES", "Clean SAC critic adds action selection, not privileged state."),
    ]
    return [dict(zip(fields, row)) for row in data]


def risk_contract_rows() -> list[dict[str, str]]:
    fields = [
        "risk_id", "risk", "evidence", "severity",
        "confidence", "impact", "recommended_action",
    ]
    data = [
        ("R1", "Full-year repeated workload leakage", "04_full_year_periodicity_audit.csv; extract_dataset_from_dfas.py:424-472", "CRITICAL", "HIGH", "Exact 49-day copies invalidate naive forecasting evaluation.", "Use one original block and group every source interval across splits."),
        ("R2", "Duration oracle leakage", "alibaba_utils.py:35-83; 06_duration_information_audit.md", "CRITICAL", "HIGH", "Deadline, release, MPC and RL consume realized completion information.", "Separate realized label from declared/predicted runtime."),
        ("R3", "Forecast train/test temporal leakage", "49-day tasks_matrix exact-match rate=1.0", "CRITICAL", "HIGH", "Random or chronological splits can contain identical labels and task identities.", "Split only the non-repeated base trace and group by original time/job identity."),
        ("R4", "H4 MPC using oracle future information", "horizon_adapter.py:223-321,448-456", "CRITICAL", "HIGH", "Oracle arrivals and exact future exogenous traces overstate deployable gains.", "Keep oracle modes as labeled upper bounds and add as-of forecasts."),
        ("R5", "RL/MPC unfair information access", "08_controller_information_comparison.csv", "HIGH", "HIGH", "Information parity varies by runner and HorizonAdapter mode.", "Declare a shared information contract and compare only matched causal inputs."),
        ("R6", "Synthetic origin assignment", "utils/workload_utils.py:7-43", "MEDIUM", "HIGH", "Regional labels are synthetic and seed/order dependent.", "Persist deterministic labels or use observed origins."),
        ("R7", "Dropped workload semantics", "extract_dataset_from_dfas.py:350-363; workload_utils.py:74-85", "HIGH", "HIGH", "Raw timestamps, weekday context and true bandwidth are not faithfully propagated.", "Publish a field-level schema and explicit dropped-field list."),
        ("R8", "Task identity mutation/random suffix", "rl_components/task.py:77-78", "MEDIUM", "HIGH", "Task IDs are non-stable while source job IDs repeat.", "Preserve immutable source IDs separately from runtime instance IDs."),
        ("R9", "15-min aggregation information loss", "extract_dataset_from_dfas.py:346-382", "HIGH", "HIGH", "Sub-interval arrival order and empty bins are lost.", "Retain event-level source data and materialize explicit zero-arrival bins."),
        ("R10", "Outlier removal altering workload tail behavior", "extract_dataset_from_dfas.py:285-305", "HIGH", "HIGH", "Sequential IQR filtering removes capacity-critical tail events.", "Version pre/post-filter distributions and evaluate tail-preserving alternatives."),
        ("R11", "Runtime bandwidth maps column 8 instead of column 9", "extract_dataset_from_dfas.py:350-363; workload_utils.py:74-85", "CRITICAL", "HIGH", "Transmission cost uses GPU memory as bandwidth.", "Repair only in a separately reviewed production change; invalidate bandwidth-sensitive conclusions meanwhile."),
        ("R12", "Bandwidth merge may use incomplete key", "alibaba_analysis.ipynb:5464-5548", "HIGH", "MEDIUM", "Job-only merge can duplicate or misassociate task-level bandwidth.", "Recover intermediate data and validate merge cardinality."),
        ("R13", "Raw/intermediate Alibaba evidence absent locally", "Local inventory; data/workload/README.md:17-54", "HIGH", "HIGH", "Preprocessing cannot be independently replayed.", "Create a legal checksum-addressed acquisition and preprocessing manifest."),
        ("R14", "Synthetic year extends into 2021", "05_full_year_periodicity_summary.json", "MEDIUM", "HIGH", "Dataset period contract and name are inaccurate.", "Use an explicit half-open target interval and row-level truncation in a future data-prep revision."),
        ("R15", "CPU/GPU/memory requests are scaled post-hoc usage metrics", "workload_utils.py:74-85; cluster_manager.py:156-164", "HIGH", "HIGH", "Forecast target semantics may not transfer to declared deployment requests.", "Rename metrics and units or acquire true request fields."),
        ("R16", "Alibaba 2026 compatibility is unsupported locally", "Repository search found no local 2026 dataset/schema artifact", "MEDIUM", "HIGH", "2020-to-2026 generalization cannot be claimed.", "Keep all 2026 dimensions UNKNOWN until a versioned source is acquired."),
    ]
    return [dict(zip(fields, row)) for row in data]


def write_outputs(workload: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    workload_hash = file_sha256(workload)
    aggregate_frame, stats, periodicity = audit_workload(workload)
    del aggregate_frame

    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    main_status = git("status", "--short")
    sustain_head = git("rev-parse", "HEAD", cwd=SUSTAIN)
    sustain_status = git("status", "--short", cwd=SUSTAIN)
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    summary_json = {
        "audit": "Workload & Information Audit v1",
        "generated_utc": generated,
        "git_head": head,
        "git_branch": branch,
        "git_status_short": main_status.splitlines(),
        "sustaincluster_commit": sustain_head,
        "sustaincluster_clean": sustain_status == "",
        "source_workload": rel(workload),
        "source_workload_size_bytes": workload.stat().st_size,
        "source_workload_sha256": workload_hash,
        "read_only": True,
        "training_performed": False,
        "data_profile": stats,
        "periodicity_conclusion": {
            "full_year_is_exact_repetition": stats["full_year_repetition"],
            "base_period_days": PERIOD_DAYS,
            "base_period_grid_steps": PERIOD_GRID_STEPS,
            "observed_nonempty_rows_per_full_block": stats["observed_rows_per_full_block"],
            "exact_tasks_matrix_match_rate_at_49_days": stats["tasks_matrix_exact_match_rate"],
            "leakage_risk": "HIGH",
            "direct_forecast_dataset_use": "NO",
        },
        "script": rel(Path(__file__)),
        "run_command": ".venv-sustain-cluster/Scripts/python.exe scripts/audit/workload_information_audit_v1.py",
    }
    (output / "05_full_year_periodicity_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    write_csv(
        output / "02_workload_lineage.csv",
        lineage_contract_rows(),
        ["stage", "source_file", "function", "input_fields", "output_fields", "transformation", "information_loss", "filtering", "aggregation", "normalization", "clipping", "outlier_removal", "randomization", "replication", "synthetic_augmentation", "evidence_file", "evidence_lines", "notes"],
    )
    write_csv(
        output / "03_information_boundary.csv",
        information_contract_rows(),
        ["field", "raw_source", "processed_source", "task_object", "environment_observation", "mpc_available", "rl_available", "classification", "reason", "risk", "evidence_file", "evidence_lines"],
    )
    write_csv(
        output / "04_full_year_periodicity_audit.csv",
        periodicity,
        ["lag_days", "lag_grid_steps", "metric", "paired_observed_intervals", "pearson_correlation", "exact_match_rate", "mean_absolute_difference", "conclusion"],
    )
    write_csv(
        output / "07_forecast_target_candidates.csv",
        forecast_contract_rows(),
        ["target", "currently_known?", "forecast_required?", "available_training_label?", "temporal_resolution", "usable_by_H4_MPC?", "risk_of_leakage", "recommended_v1?", "reason"],
    )
    write_csv(
        output / "08_controller_information_comparison.csv",
        controller_contract_rows(),
        ["information", "environment", "RL_actor", "RL_critic", "MPC_H1", "MPC_H4", "source", "online_realistic", "notes"],
    )
    write_csv(
        output / "09_risk_register.csv",
        risk_contract_rows(),
        ["risk_id", "risk", "evidence", "severity", "confidence", "impact", "recommended_action"],
    )

    duration_q = stats["duration_minutes_quantiles"]
    span_q = stats["task_span_minutes_quantiles"]
    duration_md = f"""# Duration Information Audit

## Conclusion

**Current duration semantics: realized-runtime oracle label.** `duration_min` is derived from completed instance `start_time`/`end_time`, aggregated as the mean instance runtime for a task. It is not evidenced as a user-declared runtime request or an arrival-time prediction.

The simulator then treats this value as known when the task arrives, constructs `sla_deadline = arrival + 1.5 * duration`, sets `finish_time = scheduled_start + duration`, and exposes duration/remaining duration/release information to MPC and FeatureEncoder-based BC/SAC. This is a **CRITICAL oracle-information risk**.

## Data Checks

| Check | Result |
|---|---:|
| Task records | {stats['total_task_records']:,} |
| Duration min / P50 / P90 / P95 / P99 / max (min) | {fmt(duration_q['min'], 3)} / {fmt(duration_q['p50'], 3)} / {fmt(duration_q['p90'], 3)} / {fmt(duration_q['p95'], 3)} / {fmt(duration_q['p99'], 3)} / {fmt(duration_q['max'], 3)} |
| Task-span min / P50 / P90 / P95 / P99 / max (min) | {fmt(span_q['min'], 3)} / {fmt(span_q['p50'], 3)} / {fmt(span_q['p90'], 3)} / {fmt(span_q['p95'], 3)} / {fmt(span_q['p99'], 3)} / {fmt(span_q['max'], 3)} |
| Exact equality: duration vs `(task end-task start)/60` | {pct(stats['duration_vs_task_span_exact_rate'])} |
| Mean absolute difference (min) | {fmt(stats['duration_vs_task_span_mean_abs_minutes'], 3)} |
| Median absolute difference (min) | {fmt(stats['duration_vs_task_span_median_abs_minutes'], 3)} |

The difference is expected because task `start/end` use extrema across instances while `duration_min` uses mean instance runtime. Both remain post-hoc completion information.

## Information Contract Required Before Deployment

1. Preserve `realized_duration` only as a supervised label and evaluation outcome.
2. Add a separate arrival-time field such as `declared_duration` or `predicted_duration_p50/p90`.
3. Derive expected release from the online estimate, not the realized label.
4. Propagate uncertainty into horizon capacity constraints and SLA risk.
5. Define SLA independently of realized duration.

## Primary Evidence

- `references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/alibaba_utils.py:35-43,55-83`
- `references/external_repos/sustain-cluster/utils/workload_utils.py:74-85`
- `references/external_repos/sustain-cluster/rl_components/task.py:48-60`
- `references/external_repos/sustain-cluster/envs/sustaindc/sustaindc_env.py:166-185,214-220`
- `src/sustaincluster_mpc/state_adapter.py:103-191,231-238`
- `src/sustaincluster_mpc/horizon_adapter.py:167-193,381-447`
- `src/sustaincluster_imitation/feature_encoder.py:82-137,167-183`
"""
    (output / "06_duration_information_audit.md").write_text(duration_md, encoding="utf-8")

    metrics_by_name = {row["metric"]: row for row in periodicity}
    task_count_period = metrics_by_name["task_count"]
    summary_md = f"""# Workload & Information Audit v1

## Audit Boundary

- Git HEAD: `{head}`
- Git branch: `{branch}`
- Main Git status entries at generation: `{len(main_status.splitlines()) if main_status else 0}`
- SustainCluster commit: `{sustain_head}` (working tree: {'clean' if sustain_status == '' else 'NOT CLEAN'})
- Workload: `{rel(workload)}`
- Size: `{workload.stat().st_size:,}` bytes
- SHA256: `{workload_hash}`
- Method: read-only source trace, file hash, full-pickle metadata/statistics, exact 49-day pair comparison
- Training/model changes: none

## Executive Answers

1. **Can the current full-year pickle be used directly as a Forecast v0.1 dataset? NO.** It is an exact repetition of one seven-week trace. A restricted recovery path is possible only after reducing to one provenance-preserving base block and establishing causal grouped splits.
2. **What is the main temporal leakage?** Every observed interval's entire `tasks_matrix` repeats after 49 days. At lag {PERIOD_DAYS} days ({PERIOD_GRID_STEPS} nominal 15-minute steps), {task_count_period['paired_observed_intervals']:,} paired intervals have {pct(stats['tasks_matrix_exact_match_rate'])} exact matrix-hash matches.
3. **Is duration online-known? NO.** It is mean realized instance runtime from completed-trace start/end fields. Treating it as arrival-time input makes deadline, release, MPC and RL state oracle-contaminated.
4. **Does capacity release require a separate workload Transformer target? Not as the first target.** Release of already running tasks should be derived from online runtime estimates. Future arrivals/demands require forecasting; release uncertainty can be a later conditional model.
5. **Recommended Forecast v0.1 target:** per-origin-DC, 15-minute, multi-horizon aggregate vector `[new_task_count, arriving_cpu_demand, arriving_gpu_demand, arriving_memory_demand]`.
6. **What privileged information enters H4?** Exact release based on realized duration, direct future price/carbon trace values, and in oracle/noisy-oracle modes direct future workload arrivals and true task attributes.
7. **What do existing H4 gains establish?** They show optimization value under known future price/release or oracle arrival pressure scenarios. They do not isolate the value of a deployable workload forecast model.
8. **What must be repaired first?** Recover/version raw lineage, remove exact repeated copies from splits, separate realized versus estimated duration, repair bandwidth lineage/mapping, qualify price/carbon forecasts by issue time, and persist deterministic origin labels.

## Full-Year Construction Audit

| Item | Result |
|---|---:|
| Rows | {stats['row_count']:,} |
| Outer timestamp range | `{stats['timestamp_min']}` to `{stats['timestamp_max']}` |
| Rows in 2020 / after 2020 | {stats['rows_in_2020']:,} / {stats['rows_after_2020']:,} |
| Expected complete 15m grid / missing rows | {stats['expected_complete_grid_rows']:,} / {stats['missing_15m_grid_rows']:,} |
| Nominal base period | {PERIOD_DAYS} days = {PERIOD_GRID_STEPS} grid steps |
| Observed nonempty rows per complete block | {stats['observed_rows_per_full_block']:,} |
| Block row counts | `{json.dumps(stats['block_row_counts'], ensure_ascii=False)}` |
| 49-day entire-matrix exact-match rate | {pct(stats['tasks_matrix_exact_match_rate'])} |
| 49-day task-count correlation | {fmt(task_count_period['pearson_correlation'])} |
| Inner task-time range | `{stats['internal_start_dt_min']}` to `{stats['internal_start_dt_max']}` |
| Inner weekday vs outer weekday match | {pct(stats['internal_vs_outer_weekday_match_rate'])} |

The generator shifts only the outer `interval_15m`; it deep-copies each complete task matrix. Inner job IDs, task start/end, `start_dt`, duration, resources, bandwidth and weekday remain unchanged. The final full block is appended without row-level year truncation, so the file extends through 2021-01-26 despite its name.

## Runtime Field Integrity

The matrix schema writes `avg_gpu_wrk_mem` at column 8 and `bandwidth_gb` at column 9. Runtime extraction assigns `Task.bandwidth_gb = task[8]`. Transmission cost and MPC therefore consume GPU worker memory as bandwidth. This audit records the issue but intentionally does not change production code.

## 2020 vs 2026 Local Evidence

| Dimension | Alibaba 2020 local project evidence | Alibaba 2026 local project evidence | Impact on our research |
|---|---|---|---|
| Dataset artifact | Synthesized pickle and zip are present; raw/intermediate tables are absent | UNKNOWN | 2020 preprocessing is only partially replayable; no 2026 claim is allowed |
| Time granularity | Raw event times are second-level; final workload contains nonempty 15-minute bins | UNKNOWN | Forecast v0.1 may use 15 minutes only after explicit zero-bin materialization |
| Workload duration information | Mean realized instance runtime derived after completion | UNKNOWN | Must be a label, not an online-known input |
| Task/job/pod hierarchy | Job/task/instance hierarchy exists upstream; final Task object is flattened | UNKNOWN | Hierarchy-aware generalization cannot be evaluated from the final pickle alone |
| Training/inference semantics | UNKNOWN; no validated workload-type field survives in the final matrix | UNKNOWN | Do not invent training/inference classes |
| Priority | UNKNOWN in the audited final schema | UNKNOWN | Priority-aware scheduling is unsupported by current workload evidence |
| GPU heterogeneity | GPU utilization/memory metrics exist; GPU model/type heterogeneity is UNKNOWN | UNKNOWN | Hardware-type transfer cannot be studied from this pickle |
| Resource request | Simulator uses 5x post-hoc CPU/GPU/memory usage metrics as requests | UNKNOWN | Forecast targets require explicit semantic and unit warnings |
| Utilization | CPU, GPU, memory, GPU memory and derived bandwidth metrics are present | UNKNOWN | Useful for characterization only after lineage repair |
| Scheduling delay | Upstream code derives wait time; final tasks use simulator wait/defer history instead | UNKNOWN | Trace delay and policy-induced delay must not be conflated |
| Suitability for 15-min forecasting | YES_WITH_RESTRICTIONS: one non-repeated base block, causal grouped split, repaired labels | UNKNOWN | Current full-year pickle is not directly usable |
| Suitability for workload characterization | YES_WITH_RESTRICTIONS: tail filtering, repetition and missing raw data must be disclosed | UNKNOWN | Descriptive results must be scoped to the processed trace |

No external 2026 data was downloaded. Compatibility claims are blocked until a versioned local source, schema and license record exist.

## Capacity Release Audit

The environment stores running-task finish times and HorizonAdapter reads them. In the current simulator the release schedule is deterministic because finish time equals scheduled start plus the realized trace duration. That is not a learned forecast; it is an oracle-derived rollout.

Conceptual decomposition:

    C_available(t+h)
      = C_current
      + C_known_release_estimate(t:t+h)
      - C_future_arrival_demand(t:t+h)
      + C_other_effects(t:t+h)

- Known release component: derive from running tasks and online declared/predicted remaining runtime; no workload Transformer is required for the deterministic bookkeeping.
- Unknown future arrival component: forecast new task count and arriving CPU/GPU/memory demand.
- Other uncertainty: runtime-estimation error, transfer delay, failures, capacity changes and forecast error.

## Forecast v0.1 Dataset Gate

**Decision: BLOCKED FOR DIRECT USE; REPAIRABLE WITH RESTRICTIONS.** Minimum gate:

1. Select one original seven-week source block; exclude all synthetic repeats.
2. Recover intermediate/raw provenance and record checksums, schema and license.
3. Create causal train/validation/test intervals grouped by original source time and job identity.
4. Rename post-hoc usage labels and preserve explicit units/scaling.
5. Separate `realized_duration` from arrival-time estimates; rebuild deadline/release contracts.
6. Repair and validate bandwidth merge cardinality and matrix-to-Task mapping.
7. Generate deterministic, persisted origin labels or use observed origin data.
8. Evaluate against forecast issue time, lead time and online-available features only.

## Critical Risks

- R1/R3: exact repeated blocks and source identities invalidate naive forecast train/test splits.
- R2: realized duration leaks completion, SLA and exact capacity-release information.
- R4/R11: H4 future truth and the confirmed bandwidth column mismatch can invalidate controller conclusions.

See `09_risk_register.csv` for the complete register and required actions.
"""
    (output / "01_audit_summary.md").write_text(summary_md, encoding="utf-8")

    evidence_md = """# Evidence Index

## Workload Acquisition and Preprocessing

| Evidence | Relevant lines | Finding |
|---|---:|---|
| `references/external_repos/sustain-cluster/data/workload/README.md` | 17-22, 31-54 | Identifies final, intermediate and raw artifacts; simulator reads full-year pickle |
| `references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/alibaba_utils.py` | 19-83 | Raw readers; instance and task runtime derivation; duration semantics |
| `references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/analyze_alibaba2020GPU.py` | 24-61 | Validation/filtering logic and duration>=15m rule |
| `references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/alibaba_analysis.ipynb` | 5414-5548 | Sensor bandwidth aggregation and job-only merge |
| `references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/extract_dataset_from_dfas.py` | 222-245, 285-305, 346-472 | Time normalization, IQR filters, matrix schema, crop and full-year repetition |
| `references/external_repos/sustain-cluster/data/workload/alibaba_2020_dataset/plot_alibaba_workload_stats.py` | 49-52 | Confirms CPU/GPU/memory/bandwidth indices 5/6/7/9 |

## Runtime and Controller Information

| Evidence | Relevant lines | Finding |
|---|---:|---|
| `references/external_repos/sustain-cluster/utils/workload_utils.py` | 7-43, 74-154 | Synthetic origin; resource scaling; confirmed bandwidth off-by-one; grouping paths |
| `references/external_repos/sustain-cluster/simulation/cluster_manager.py` | 28-76, 134-168, 286-300, 358-382 | Workload loading/query; task scaling; transmission uses Task bandwidth |
| `references/external_repos/sustain-cluster/rl_components/task.py` | 34-78, 109-119 | Task duration/deadline/ID and completion semantics |
| `references/external_repos/sustain-cluster/envs/sustaindc/sustaindc_env.py` | 166-220, 393-489 | Finish-time resource release and scheduling lifecycle |
| `references/external_repos/sustain-cluster/envs/task_scheduling_env.py` | 76-79, 185-223, 255-397 | Native observation and transition information |
| `src/sustaincluster_mpc/state_adapter.py` | 9-58, 103-302, 351-424 | MPC current task/DC state, exact release and network fields |
| `src/sustaincluster_mpc/horizon_adapter.py` | 122-321, 345-517 | Arrival forecast modes, oracle future lookup, exact release, future price/CI |
| `src/sustaincluster_mpc/rolling_horizon_optimizer.py` | 128-443 | H4 objective and horizon capacity consumption |
| `src/sustaincluster_imitation/feature_encoder.py` | 82-307 | BC/SAC feature vector and feasibility mask |
| `src/sustaincluster_imitation/forecast_baseline.py` | 35-181 | Causal historical aggregate arrival baseline |
| `references/external_repos/sustain-cluster/rl_components/agent_net.py` | 4-105 | Actor/Critic information symmetry; Critic adds action selection only |
| `references/external_repos/sustain-cluster/utils/managers.py` | 236-303, 337-399, 499-559 | Historical CI/weather/price arrays and future value access |

## Existing Evaluation Context

| Evidence | Finding |
|---|---|
| `reports/sustaincluster_mpc/rolling_horizon_report.md` | Synthetic pressure tests attribute H4 value to known low-price windows, known release, or oracle GPU bursts; oracle trace is an upper bound |
| `reports/bc_reg_sac_v0_1/run_small_experiment.py:54-58` | Current BC-regularized SAC runner uses H4 no-future-arrivals plus causal historical arrival forecast |
| `reports/sustaincluster_imitation/run_sac_warm_start.py:55-59` | SAC warm-start runner uses the same horizon/forecast composition |

## Generated Statistical Evidence

| Artifact | Purpose |
|---|---|
| `04_full_year_periodicity_audit.csv` | 49-day paired correlations, exact aggregate matches and full matrix hashes |
| `05_full_year_periodicity_summary.json` | Workload checksum, time coverage, block counts, duration and internal-time consistency statistics |
| `06_duration_information_audit.md` | Duration provenance, numerical checks and deployment information contract |

All paths are repository-relative. No raw workload, production code, model checkpoint or third-party source was modified by this audit.
"""
    (output / "10_evidence_index.md").write_text(evidence_md, encoding="utf-8")
    return summary_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    workload = args.workload.resolve()
    output = args.output.resolve()
    if not workload.is_file():
        raise FileNotFoundError(workload)
    if ROOT.resolve() not in output.parents and output != ROOT.resolve():
        raise ValueError("Output must stay inside the project workspace")

    result = write_outputs(workload, output)
    profile = result["data_profile"]
    print("WORKLOAD_INFORMATION_AUDIT_V1_COMPLETE")
    print(f"workload_sha256={result['source_workload_sha256']}")
    print(f"rows={profile['row_count']}")
    print(f"total_task_records={profile['total_task_records']}")
    print(f"period_49d_exact_matrix_match_rate={profile['tasks_matrix_exact_match_rate']:.6f}")
    print(f"output={rel(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
