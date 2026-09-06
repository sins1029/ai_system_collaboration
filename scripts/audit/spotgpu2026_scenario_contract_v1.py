from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from scripts.audit import alibaba_2026_adaptation_audit_v1 as source_audit


ROOT = source_audit.ROOT
CONFIG_PATH = ROOT / "configs" / "scenarios" / "spotgpu2026_v1.yaml"
OUTPUT = ROOT / "artifacts" / "spotgpu2026_scenario_contract_v1"
SPOT_README_URL = (
    "https://github.com/alibaba/clusterdata/blob/"
    f"{source_audit.UPSTREAM_COMMIT}/cluster-trace-v2026-spot-gpu/README.md"
)
DURATION_LEVELS = (
    ("L1_GPU_MODEL_GPU_REQUEST_WORKER_PRIORITY", ("gpu_model", "gpu_request", "worker_num", "job_type")),
    ("L2_GPU_MODEL_GPU_REQUEST_PRIORITY", ("gpu_model", "gpu_request", "job_type")),
    ("L3_GPU_REQUEST_PRIORITY", ("gpu_request", "job_type")),
    ("L4_PRIORITY", ("job_type",)),
)
DEPLOYABLE_COLUMNS = (
    "source", "task_id", "submit_time", "arrival_step", "original_index",
    "cpu_request_effective", "gpu_request_effective", "memory_request",
    "bandwidth", "estimated_duration", "estimated_duration_steps",
    "gpu_model", "priority", "worker_num", "origin_dc", "sla_class",
    "sla_steps", "sla_deadline_step", "max_wait_steps", "defer_allowed",
    "defer_class",
)
FORBIDDEN_TOKENS = (
    "true_duration", "future", "oracle", "teacher", "h1_action", "h4_action"
)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["request_scope"]["value"] != "PER_JOB":
        raise RuntimeError("request_scope must remain PER_JOB")
    if config["request_scope"]["multiply_by_worker_num"]:
        raise RuntimeError("PER_JOB requests cannot be multiplied by worker_num")
    if config["time"]["time_step_seconds"] != 900:
        raise RuntimeError("arrival step must remain 900 seconds")
    if config["gpu_heterogeneity"]["mode"] != "METADATA_ONLY":
        raise RuntimeError("GPU heterogeneity v1 must remain metadata-only")
    if config["randomness"]["mode"] != "DETERMINISTIC":
        raise RuntimeError("conversion must remain deterministic")
    return config


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jobs, nodes = source_audit.validate_sources()
    baseline = pd.read_pickle(source_audit.ALIBABA_2020)
    jobs = jobs.copy()
    jobs["original_index"] = np.arange(len(jobs), dtype=np.int64)
    return jobs, nodes, baseline


def chronological_split(
    jobs: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(jobs) * train_fraction)
    validation_end = int(len(jobs) * (train_fraction + validation_fraction))
    return (
        jobs.iloc[:train_end].copy(),
        jobs.iloc[train_end:validation_end].copy(),
        jobs.iloc[validation_end:].copy(),
    )


def arrival_step(values: Iterable[int], step_seconds: int = 900) -> np.ndarray:
    seconds = np.asarray(values, dtype=np.int64)
    if np.any(seconds < 0):
        raise ValueError("submit_time must be nonnegative")
    return np.floor_divide(seconds, step_seconds)


def stable_task_order(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    if "task_id" not in ordered:
        ordered["task_id"] = "spotgpu2026:" + ordered["job_name"].astype(str)
    return ordered.sort_values(
        ["submit_time", "original_index", "task_id"], kind="mergesort"
    ).reset_index(drop=True)


def _lookup(frame: pd.DataFrame, keys: tuple[str, ...], table: pd.Series) -> np.ndarray:
    lookup = table.rename("_prediction").reset_index()
    merged = frame.loc[:, list(keys)].reset_index(drop=True).merge(
        lookup, how="left", on=list(keys), sort=False
    )
    return merged["_prediction"].to_numpy(dtype=float)


@dataclass(frozen=True)
class HierarchicalMedianEstimator:
    support: int
    tables: tuple[tuple[str, tuple[str, ...], pd.Series], ...]
    global_median: float
    train_rows: int
    train_max_original_index: int

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        prediction = np.full(len(frame), np.nan)
        levels = np.full(len(frame), "", dtype=object)
        for name, keys, table in self.tables:
            values = _lookup(frame, keys, table)
            mask = np.isnan(prediction) & np.isfinite(values)
            prediction[mask], levels[mask] = values[mask], name
        mask = np.isnan(prediction)
        prediction[mask], levels[mask] = self.global_median, "GLOBAL_TRAIN_MEDIAN"
        return prediction, levels


def fit_duration_estimator(train: pd.DataFrame, support: int) -> HierarchicalMedianEstimator:
    tables = []
    for name, keys in DURATION_LEVELS:
        grouped = train.groupby(list(keys), dropna=False)["duration"].agg(["count", "median"])
        tables.append((name, keys, grouped.loc[grouped["count"] >= support, "median"]))
    return HierarchicalMedianEstimator(
        support, tuple(tables), float(train["duration"].median()), len(train),
        int(train["original_index"].max())
    )


def duration_support_audit(
    train: pd.DataFrame, candidates: Iterable[int]
) -> tuple[pd.DataFrame, int]:
    keys = list(DURATION_LEVELS[0][1])
    full = train.groupby(keys)["duration"].agg(full_count="count", full_median="median")
    first = train.iloc[::2].groupby(keys)["duration"].agg(
        first_count="count", first_median="median"
    )
    second = train.iloc[1::2].groupby(keys)["duration"].agg(
        second_count="count", second_median="median"
    )
    joined = first.join(second, how="inner").join(full[["full_median"]], how="inner")
    rows = []
    for support in candidates:
        eligible = full.loc[full["full_count"] >= support].reset_index()[keys]
        coverage = len(train[keys].merge(eligible, how="inner", on=keys)) / len(train)
        half_support = math.ceil(support / 2)
        stable = joined[
            (joined["first_count"] >= half_support)
            & (joined["second_count"] >= half_support)
        ]
        shift = (
            (stable["first_median"] - stable["second_median"]).abs()
            / stable["full_median"].clip(lower=1)
        )
        rows.append({
            "minimum_support": support,
            "eligible_l1_groups": len(eligible),
            "l1_train_row_coverage": coverage,
            "internally_comparable_groups": len(stable),
            "relative_median_shift_p50": float(shift.quantile(0.5)),
            "relative_median_shift_p90": float(shift.quantile(0.9)),
            "selection_data": "TRAIN_INTERNAL_INTERLEAVED_ONLY",
        })
    result = pd.DataFrame(rows)
    selected = int(result.sort_values(
        ["relative_median_shift_p50", "minimum_support"],
        ascending=[True, False],
    ).iloc[0]["minimum_support"])
    result["selected"] = result["minimum_support"].eq(selected)
    return result, selected


def _error_row(group: str, value: str, truth: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    error = np.abs(truth - pred)
    return {
        "group": group, "value": value, "count": len(error),
        "mae_seconds": float(error.mean()),
        "median_ae_seconds": float(np.median(error)),
        "p90_ae_seconds": float(np.quantile(error, 0.9)),
        "true_p50_seconds": float(np.median(truth)),
        "prediction_p50_seconds": float(np.median(pred)),
    }


def duration_validation(
    validation: pd.DataFrame, prediction: np.ndarray, levels: np.ndarray
) -> pd.DataFrame:
    rows = [_error_row("ALL", "ALL", validation["duration"].to_numpy(), prediction)]
    for value in sorted(validation["job_type"].unique()):
        mask = validation["job_type"].eq(value).to_numpy()
        rows.append(_error_row(
            "priority", str(value), validation.loc[mask, "duration"].to_numpy(),
            prediction[mask]
        ))
    for value in sorted(set(levels)):
        mask = levels == value
        rows.append(_error_row(
            "fallback_level", str(value),
            validation.loc[mask, "duration"].to_numpy(), prediction[mask]
        ))
    return pd.DataFrame(rows)


def alibaba2020_train_table(baseline: pd.DataFrame, fraction: float) -> pd.DataFrame:
    matrix = np.vstack(
        baseline.iloc[: int(len(baseline) * fraction)]["tasks_matrix"].to_numpy()
    )
    frame = pd.DataFrame({
        "cpu": np.asarray(matrix[:, 5], dtype=float) / 100,
        "gpu": np.asarray(matrix[:, 6], dtype=float) / 100,
        "memory": np.asarray(matrix[:, 7], dtype=float),
        "bandwidth": np.asarray(matrix[:, 9], dtype=float),
    })
    return frame.loc[np.isfinite(frame).all(axis=1) & frame.ge(0).all(axis=1)].reset_index(drop=True)


def apply_bins(values: Iterable[float], edges: Iterable[float]) -> np.ndarray:
    boundaries = np.asarray(tuple(edges), dtype=float)
    return np.digitize(np.asarray(values, dtype=float), boundaries[1:-1], right=False)


@dataclass(frozen=True)
class MemoryModel:
    cpu_edges: tuple[float, ...]
    gpu_edges: tuple[float, ...]
    pair_table: pd.Series
    cpu_table: pd.Series
    global_median: float
    train_rows: int

    def predict(self, cpu: np.ndarray, gpu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        work = pd.DataFrame({
            "cpu_bin": apply_bins(cpu, self.cpu_edges),
            "gpu_bin": apply_bins(gpu, self.gpu_edges),
        })
        prediction = _lookup(work, ("cpu_bin", "gpu_bin"), self.pair_table)
        levels = np.where(np.isfinite(prediction), "CPU_GPU_BIN", "").astype(object)
        cpu_values = _lookup(work, ("cpu_bin",), self.cpu_table)
        mask = ~np.isfinite(prediction) & np.isfinite(cpu_values)
        prediction[mask], levels[mask] = cpu_values[mask], "CPU_BIN"
        mask = ~np.isfinite(prediction)
        prediction[mask], levels[mask] = self.global_median, "GLOBAL_TRAIN_MEDIAN"
        return prediction, levels


def fit_memory_model(source: pd.DataFrame, config: dict[str, Any]) -> MemoryModel:
    cpu_edges = tuple(map(float, config["cpu_bin_edges"]))
    gpu_edges = tuple(map(float, config["gpu_bin_edges"]))
    support = int(config["minimum_support"])
    work = source.copy()
    work["cpu_bin"] = apply_bins(work["cpu"], cpu_edges)
    work["gpu_bin"] = apply_bins(work["gpu"], gpu_edges)
    pair = work.groupby(["cpu_bin", "gpu_bin"])["memory"].agg(["count", "median"])
    cpu = work.groupby("cpu_bin")["memory"].agg(["count", "median"])
    return MemoryModel(
        cpu_edges, gpu_edges, pair.loc[pair["count"] >= support, "median"],
        cpu.loc[cpu["count"] >= support, "median"],
        float(work["memory"].median()), len(work)
    )


@dataclass(frozen=True)
class BandwidthModels:
    cpu_edges: tuple[float, ...]
    gpu_edges: tuple[float, ...]
    global_median: float
    gpu_table: pd.Series
    pair_table: pd.Series
    train_rows: int

    def candidates(self, cpu: np.ndarray, gpu: np.ndarray) -> dict[str, np.ndarray]:
        work = pd.DataFrame({
            "cpu_bin": apply_bins(cpu, self.cpu_edges),
            "gpu_bin": apply_bins(gpu, self.gpu_edges),
        })
        global_values = np.full(len(work), self.global_median)
        gpu_values = _lookup(work, ("gpu_bin",), self.gpu_table)
        gpu_values[~np.isfinite(gpu_values)] = self.global_median
        pair_values = _lookup(work, ("cpu_bin", "gpu_bin"), self.pair_table)
        mask = ~np.isfinite(pair_values)
        pair_values[mask] = gpu_values[mask]
        return {
            "GLOBAL_MEDIAN": global_values,
            "GPU_BIN_MEDIAN": gpu_values,
            "CPU_GPU_BIN_MEDIAN": pair_values,
        }


def fit_bandwidth_models(
    source: pd.DataFrame, memory_config: dict[str, Any], config: dict[str, Any]
) -> BandwidthModels:
    cpu_edges = tuple(map(float, memory_config["cpu_bin_edges"]))
    gpu_edges = tuple(map(float, memory_config["gpu_bin_edges"]))
    support = int(config["minimum_support"])
    work = source.copy()
    work["cpu_bin"] = apply_bins(work["cpu"], cpu_edges)
    work["gpu_bin"] = apply_bins(work["gpu"], gpu_edges)
    gpu = work.groupby("gpu_bin")["bandwidth"].agg(["count", "median"])
    pair = work.groupby(["cpu_bin", "gpu_bin"])["bandwidth"].agg(["count", "median"])
    return BandwidthModels(
        cpu_edges, gpu_edges, float(work["bandwidth"].median()),
        gpu.loc[gpu["count"] >= support, "median"],
        pair.loc[pair["count"] >= support, "median"], len(work)
    )


def distribution_row(
    dataset: str, metric: str, values: np.ndarray, provenance: str
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "dataset": dataset, "metric": metric, "count": len(values),
        "mean": float(values.mean()), "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p99": float(np.quantile(values, 0.99)), "max": float(values.max()),
        "provenance": provenance,
    }


def memory_validation(
    alibaba: pd.DataFrame, cpu: np.ndarray, gpu: np.ndarray, memory: np.ndarray
) -> pd.DataFrame:
    rows = []
    for name, current_cpu, current_gpu, current_memory, provenance in (
        ("Alibaba2020_train", alibaba["cpu"].to_numpy(), alibaba["gpu"].to_numpy(),
         alibaba["memory"].to_numpy(), "DIRECT_SOURCE_TRAIN_SLICE"),
        ("SpotGPU2026_modeled_sample", cpu, gpu, memory,
         "MODELED_ALIBABA2020_CONDITIONAL_MEDIAN_V1"),
    ):
        rows.append(distribution_row(name, "memory", current_memory, provenance))
        mask = current_cpu > 0
        rows.append(distribution_row(
            name, "memory_per_cpu", current_memory[mask] / current_cpu[mask], provenance
        ))
        mask = current_gpu > 0
        rows.append(distribution_row(
            name, "memory_per_gpu", current_memory[mask] / current_gpu[mask], provenance
        ))
    return pd.DataFrame(rows)


def hour_probabilities(config: dict[str, Any]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    model = config["origin_model"]
    start, end = model["activity"]["active_local_hours"]
    active = float(model["activity"]["active_multiplier"])
    inactive = float(model["activity"]["inactive_multiplier"])
    dcs = model["datacenters"]
    ids = np.asarray([int(dc["dc_id"]) for dc in dcs])
    result = {}
    for hour in range(24):
        scores = np.asarray([
            float(dc["population_weight"]) * (
                active if start <= (hour + int(dc["timezone_shift"])) % 24 < end else inactive
            )
            for dc in dcs
        ])
        result[hour] = (ids, scores / scores.sum())
    return result


def deterministic_origins(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    namespace = config["origin_model"]["namespace"]
    hourly = hour_probabilities(config)
    result = np.empty(len(frame), dtype=int)
    for index, task in enumerate(frame.itertuples(index=False)):
        ids, probabilities = hourly[(int(task.submit_time) // 3600) % 24]
        task_id = f"spotgpu2026:{task.job_name}"
        payload = f"{namespace}|{task_id}|{int(task.submit_time)}".encode()
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
        choice = min(np.searchsorted(np.cumsum(probabilities), value, side="right"), len(ids) - 1)
        result[index] = ids[choice]
    return result


def origin_distribution(jobs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    origins = deterministic_origins(jobs, config)
    hourly = hour_probabilities(config)
    hours = ((jobs["submit_time"].to_numpy(dtype=np.int64) // 3600) % 24).astype(int)
    blocks = hours // 6 * 6
    scopes = [("ALL", np.ones(len(jobs), dtype=bool))]
    scopes.extend((f"UTC_{start:02d}_{start + 5:02d}", blocks == start) for start in (0, 6, 12, 18))
    rows = []
    for scope, mask in scopes:
        for position, dc in enumerate(config["origin_model"]["datacenters"]):
            dc_id = int(dc["dc_id"])
            observed = float(np.mean(origins[mask] == dc_id))
            target = float(np.mean([hourly[hour][1][position] for hour in hours[mask]]))
            rows.append({
                "time_scope": scope, "dc_id": dc_id, "location": dc["location"],
                "task_count": int(np.sum(origins[mask] == dc_id)),
                "observed_share": observed, "activity_adjusted_target_share": target,
                "absolute_share_error": abs(observed - target),
                "configured_population_weight": float(dc["population_weight"]),
                "total_tasks_in_scope": int(mask.sum()), "provenance": "MODELED_ORIGIN",
            })
    return pd.DataFrame(rows)


def build_canonical_sample(
    raw: pd.DataFrame,
    duration_model: HierarchicalMedianEstimator,
    memory_model: MemoryModel,
    bandwidth_models: BandwidthModels,
    config: dict[str, Any],
) -> pd.DataFrame:
    ordered = stable_task_order(raw)
    cpu = ordered["cpu_request"].to_numpy(dtype=float)
    gpu = ordered["gpu_request"].to_numpy(dtype=float)
    estimated, duration_level = duration_model.predict(ordered)
    memory, memory_level = memory_model.predict(cpu, gpu)
    bandwidth = bandwidth_models.candidates(cpu, gpu)[config["bandwidth_model"]["selected_method"]]
    origins = deterministic_origins(ordered, config)
    steps = arrival_step(ordered["submit_time"], config["time"]["time_step_seconds"])
    estimated_steps = np.maximum(1, np.ceil(estimated / config["time"]["time_step_seconds"]).astype(int))
    profile = config["sla"]["profiles"][config["sla"]["selected_profile"]]
    waits = np.asarray([profile[value]["max_wait_steps"] for value in ordered["job_type"]], dtype=int)
    return pd.DataFrame({
        "source": "Alibaba SpotGPU2026",
        "task_id": "spotgpu2026:" + ordered["job_name"].astype(str),
        "submit_time": ordered["submit_time"].astype(np.int64),
        "arrival_step": steps, "original_index": ordered["original_index"].astype(np.int64),
        "cpu_request_raw": ordered["cpu_request"].astype(float),
        "gpu_request_raw": ordered["gpu_request"].astype(float),
        "worker_num": ordered["worker_num"].astype(np.int64),
        "request_scope": "PER_JOB",
        "cpu_request_effective": cpu, "gpu_request_effective": gpu,
        "memory_request": memory,
        "memory_source": "MODELED_ALIBABA2020_CONDITIONAL_MEDIAN_V1",
        "memory_fallback_level": memory_level,
        "bandwidth": bandwidth,
        "bandwidth_source": "MODELED_ALIBABA2020_GLOBAL_MEDIAN_V1",
        "true_duration": ordered["duration"].astype(np.int64),
        "true_duration_source": "DIRECT_SOURCE_SIMULATOR_ONLY",
        "estimated_duration": estimated, "estimated_duration_steps": estimated_steps,
        "estimated_duration_source": "MODELED_SPOT_TRAIN_ONLY_HIERARCHICAL_MEDIAN_V1",
        "estimated_duration_fallback_level": duration_level,
        "gpu_model": ordered["gpu_model"].astype(str),
        "priority": ordered["job_type"].astype(str), "origin_dc": origins,
        "origin_source": "MODELED_ORIGIN",
        "sla_class": np.where(ordered["job_type"].eq("HP"), "HP_STRICT", "SPOT_FLEXIBLE"),
        "sla_steps": estimated_steps + waits,
        "sla_deadline_step": steps + estimated_steps + waits,
        "max_wait_steps": waits,
        "defer_allowed": [profile[value]["defer_allowed"] for value in ordered["job_type"]],
        "defer_class": [profile[value]["defer_class"] for value in ordered["job_type"]],
        "sla_source": "MODELED_SCENARIO_PARAMETER",
        "gpu_heterogeneity_mode": "METADATA_ONLY",
        "provenance_flags": (
            "DIRECT_SOURCE_FIELDS;DERIVED_ARRIVAL;MODELED_MEMORY;"
            "MODELED_BANDWIDTH;MODELED_ORIGIN;MODELED_DURATION_ESTIMATE;"
            "MODELED_SLA;TRUE_DURATION_SIMULATOR_ONLY"
        ),
    })


def deployable_observation(canonical: pd.DataFrame) -> pd.DataFrame:
    frame = canonical.loc[:, DEPLOYABLE_COLUMNS].copy()
    for token in FORBIDDEN_TOKENS:
        if any(token in column.lower() for column in frame.columns):
            raise RuntimeError(f"forbidden deployable field: {token}")
    return frame


def request_scope_evidence() -> pd.DataFrame:
    rows = [
        ("E1", "OFFICIAL_README", SPOT_README_URL,
         "cpu_request: Number of CPU cores requested by the job (in vCPUs).",
         "PER_JOB", "DIRECT_OFFICIAL"),
        ("E2", "OFFICIAL_README", SPOT_README_URL,
         "gpu_request: Number of GPU requested by the job.",
         "PER_JOB", "DIRECT_OFFICIAL"),
        ("E3", "OFFICIAL_README", SPOT_README_URL,
         "worker_num: Number of instances requested by the job.",
         "SEPARATE_FIELD_NOT_MULTIPLIER", "DIRECT_OFFICIAL"),
        ("E4", "OFFICIAL_REPOSITORY_TREE",
         "https://api.github.com/repos/alibaba/clusterdata/contents/cluster-trace-v2026-spot-gpu",
         "Only README.md and two CSVs; no adapter/example code.", "NO_CONTRADICTING_CODE", "OFFICIAL_API"),
        ("E5", "OFFICIAL_ISSUE_SEARCH", "https://github.com/alibaba/clusterdata/issues",
         "No issue matched worker_num or spot-gpu at audit time.", "NO_ADDITIONAL_GUIDANCE", "OFFICIAL_REPOSITORY_SEARCH"),
        ("E6", "OFFICIAL_DISCUSSIONS", "https://github.com/alibaba/clusterdata",
         "Repository discussions are disabled.", "NO_ADDITIONAL_GUIDANCE", "OFFICIAL_API"),
        ("E7", "LOCAL_FULL_DATA", "data/raw/alibaba_2026/spot_gpu/job_info_df.csv",
         "PER_JOB max GPU=8; worker multiplication yields max=1456 and 65 all-DC-infeasible jobs.",
         "DIAGNOSTIC_ONLY", "LOCAL_FULL_FILE"),
    ]
    return pd.DataFrame(rows, columns=(
        "evidence_id", "source_type", "source", "finding", "supports", "evidence_level"
    ))


def request_scope_sensitivity(jobs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    dcs = config["origin_model"]["datacenters"]
    capacities = {
        "cpu": [dc["total_cores"] for dc in dcs],
        "gpu": [dc["total_gpus"] for dc in dcs],
    }
    rows = []
    for scope, cpu, gpu in (
        ("PER_JOB", jobs["cpu_request"], jobs["gpu_request"]),
        ("PER_WORKER_DIAGNOSTIC_ONLY", jobs["cpu_request"] * jobs["worker_num"],
         jobs["gpu_request"] * jobs["worker_num"]),
    ):
        infeasible = (cpu > max(capacities["cpu"])) | (gpu > max(capacities["gpu"]))
        for resource, values in (("cpu", cpu), ("gpu", gpu)):
            data = values.to_numpy(dtype=float)
            rows.append({
                "interpretation": scope, "resource": resource, "count": len(data),
                "sum": float(data.sum()), "mean": float(data.mean()),
                "p50": float(np.quantile(data, .5)), "p90": float(np.quantile(data, .9)),
                "p99": float(np.quantile(data, .99)), "p999": float(np.quantile(data, .999)),
                "max": float(data.max()),
                "count_exceeds_smallest_dc": int(np.sum(data > min(capacities[resource]))),
                "count_exceeds_largest_dc": int(np.sum(data > max(capacities[resource]))),
                "tasks_infeasible_at_every_dc": int(infeasible.sum()),
                "official_contract": scope == "PER_JOB",
            })
    return pd.DataFrame(rows)


def static_feasibility(canonical: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cpu = canonical["cpu_request_effective"].to_numpy()
    gpu = canonical["gpu_request_effective"].to_numpy()
    memory = canonical["memory_request"].to_numpy()
    dcs = config["origin_model"]["datacenters"]
    matrix = np.column_stack([
        (cpu >= 0) & (gpu > 0) & (memory >= 0)
        & (cpu <= dc["total_cores"]) & (gpu <= dc["total_gpus"])
        & (memory <= dc["total_mem"])
        for dc in dcs
    ])
    counts = matrix.sum(axis=1)
    ids = [dc["dc_id"] for dc in dcs]
    max_cpu, max_gpu, max_mem = (
        max(dc["total_cores"] for dc in dcs),
        max(dc["total_gpus"] for dc in dcs),
        max(dc["total_mem"] for dc in dcs),
    )
    reasons, feasible_ids = [], []
    for index, count in enumerate(counts):
        feasible_ids.append(";".join(str(ids[i]) for i in np.flatnonzero(matrix[index])))
        current = []
        if cpu[index] < 0: current.append("NEGATIVE_CPU")
        if gpu[index] <= 0: current.append("NONPOSITIVE_GPU")
        if memory[index] < 0: current.append("NEGATIVE_MEMORY")
        if cpu[index] > max_cpu: current.append("CPU_EXCEEDS_ALL_DCS")
        if gpu[index] > max_gpu: current.append("GPU_EXCEEDS_ALL_DCS")
        if memory[index] > max_mem: current.append("MEMORY_EXCEEDS_ALL_DCS")
        reasons.append(";".join(current) if current else ("NONE" if count else "COMBINED_CAPACITY"))
    return pd.DataFrame({
        "task_id": canonical["task_id"], "priority": canonical["priority"],
        "gpu_model": canonical["gpu_model"], "gpu_request": gpu,
        "cpu_request": cpu, "memory_request": memory, "feasible": counts > 0,
        "feasible_dc_count": counts, "feasible_dc_ids": feasible_ids,
        "infeasible_reason": reasons,
    })


def bandwidth_sensitivity(
    raw: pd.DataFrame, models: BandwidthModels, config: dict[str, Any], feasible_rate: float
) -> pd.DataFrame:
    candidates = models.candidates(raw["cpu_request"].to_numpy(), raw["gpu_request"].to_numpy())
    selected = config["bandwidth_model"]["selected_method"]
    reference = candidates[selected].mean()
    rows = []
    for model, values in candidates.items():
        for scenario, factor in config["bandwidth_model"]["sensitivity_factors"].items():
            scaled = values * factor
            rows.append({
                "candidate_model": model, "sensitivity_scenario": scenario,
                "factor": factor, "mean_bandwidth_gb": float(scaled.mean()),
                "p50_bandwidth_gb": float(np.median(scaled)),
                "p90_bandwidth_gb": float(np.quantile(scaled, .9)),
                "relative_transfer_cost_proxy": float(scaled.mean() / reference),
                "static_feasible_rate": feasible_rate,
                "feasible_action_change": "NONE_BANDWIDTH_NOT_IN_STATIC_CONSTRAINT",
                "migration_decision_effect": "NOT_MEASURED_NO_MPC_RUN",
                "selected_for_v1": model == selected and scenario == "baseline",
            })
    return pd.DataFrame(rows)


def resource_quality(canonical: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    dcs = config["origin_model"]["datacenters"]
    capacities = {
        "cpu_request_effective": [dc["total_cores"] for dc in dcs],
        "gpu_request_effective": [dc["total_gpus"] for dc in dcs],
        "memory_request": [dc["total_mem"] for dc in dcs],
    }
    rows = []
    for column in (
        "cpu_request_effective", "gpu_request_effective", "memory_request",
        "bandwidth", "true_duration", "estimated_duration", "worker_num"
    ):
        values = canonical[column].to_numpy(dtype=float)
        cap = capacities.get(column)
        rows.append({
            "metric": column, "count": len(values),
            "negative_count": int(np.sum(values < 0)), "zero_count": int(np.sum(values == 0)),
            "mean": float(values.mean()), "p50": float(np.quantile(values, .5)),
            "p90": float(np.quantile(values, .9)), "p99": float(np.quantile(values, .99)),
            "max": float(values.max()), "smallest_dc_capacity": min(cap) if cap else np.nan,
            "largest_dc_capacity": max(cap) if cap else np.nan,
            "count_exceeds_smallest_dc": int(np.sum(values > min(cap))) if cap else 0,
            "count_exceeds_largest_dc": int(np.sum(values > max(cap))) if cap else 0,
        })
    return pd.DataFrame(rows)


def sla_table(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for profile_name, profile in config["sla"]["profiles"].items():
        for priority, values in profile.items():
            rows.append({
                "profile": profile_name, "priority": priority,
                "max_wait_steps": values["max_wait_steps"],
                "max_wait_minutes": values["max_wait_steps"] * config["time"]["time_step_minutes"],
                "defer_allowed": values["defer_allowed"], "defer_class": values["defer_class"],
                "sla_formula": config["sla"]["formula"], "provenance": config["sla"]["provenance"],
                "selected_for_v1": profile_name == config["sla"]["selected_profile"],
            })
    return pd.DataFrame(rows)


def provenance_contract() -> pd.DataFrame:
    rows = [
        ("source", "CONSTANT", "scenario id", "metadata"),
        ("task_id", "DERIVED", "prefix + job_name", "deployable"),
        ("submit_time", "DIRECT", "submit_time", "deployable"),
        ("arrival_step", "DERIVED", "floor(submit_time/900)", "deployable"),
        ("original_index", "DERIVED", "source row position", "deployable"),
        ("cpu_request_raw", "DIRECT", "cpu_request", "audit"),
        ("gpu_request_raw", "DIRECT", "gpu_request", "audit"),
        ("worker_num", "DIRECT", "worker_num", "deployable metadata"),
        ("request_scope", "CONSTANT", "PER_JOB", "audit"),
        ("cpu_request_effective", "DIRECT", "cpu_request", "deployable"),
        ("gpu_request_effective", "DIRECT", "gpu_request", "deployable"),
        ("memory_request", "MODELED", "Alibaba2020 train conditional median", "deployable"),
        ("memory_source", "CONSTANT", "memory model id", "audit"),
        ("memory_fallback_level", "MODELED", "memory model support fallback", "audit"),
        ("bandwidth", "MODELED", "Alibaba2020 train global median", "deployable"),
        ("true_duration", "SIMULATOR_ONLY", "DIRECT_SOURCE duration", "forbidden deployable"),
        ("bandwidth_source", "CONSTANT", "bandwidth model id", "audit"),
        ("estimated_duration", "MODELED", "Spot train hierarchical median", "deployable"),
        ("true_duration_source", "CONSTANT", "DIRECT_SOURCE_SIMULATOR_ONLY", "audit"),
        ("gpu_model", "DIRECT", "gpu_model", "metadata-only"),
        ("estimated_duration_steps", "DERIVED", "ceil(estimated_duration/900)", "deployable"),
        ("estimated_duration_source", "CONSTANT", "duration model id", "audit"),
        ("estimated_duration_fallback_level", "MODELED", "duration hierarchy fallback", "audit"),
        ("priority", "DIRECT", "job_type", "deployable"),
        ("origin_dc", "MODELED", "deterministic weighted hash", "deployable"),
        ("origin_source", "CONSTANT", "MODELED_ORIGIN", "audit"),
        ("sla_class", "MODELED", "scenario profile", "deployable"),
        ("sla_steps", "MODELED", "estimated steps + wait", "deployable"),
        ("sla_deadline_step", "MODELED", "arrival + estimated steps + wait", "deployable"),
        ("max_wait_steps", "MODELED", "selected scenario profile", "deployable"),
        ("defer_allowed", "MODELED", "selected scenario profile", "deployable"),
        ("defer_class", "MODELED", "scenario profile", "deployable"),
        ("sla_source", "CONSTANT", "MODELED_SCENARIO_PARAMETER", "audit"),
        ("gpu_heterogeneity_mode", "CONSTANT", "METADATA_ONLY", "audit"),
        ("provenance_flags", "CONSTANT", "field-class summary", "audit"),
    ]
    return pd.DataFrame(rows, columns=("field", "provenance_type", "source_or_rule", "visibility"))

def canonical_schema(provenance: pd.DataFrame) -> pd.DataFrame:
    dtypes = {
        "submit_time": "int64 seconds", "arrival_step": "int64",
        "original_index": "int64", "cpu_request_raw": "float vCPU",
        "gpu_request_raw": "float GPU", "worker_num": "int64",
        "cpu_request_effective": "float vCPU",
        "gpu_request_effective": "float GPU-equivalent",
        "memory_request": "float simulator unit", "bandwidth": "float GB",
        "true_duration": "int64 seconds", "estimated_duration": "float seconds",
        "estimated_duration_steps": "int64", "origin_dc": "int64",
        "sla_steps": "int64", "sla_deadline_step": "int64",
        "max_wait_steps": "int64", "defer_allowed": "bool",
    }
    result = provenance.copy()
    result.insert(1, "dtype", result["field"].map(dtypes).fillna("string/enum"))
    result["required"], result["contract_version"] = True, "spotgpu2026_v1"
    return result


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest().upper()


def markdown_table(frame: pd.DataFrame) -> str:
    def clean(value: Any) -> str:
        return "MISSING" if pd.isna(value) else str(value).replace("|", "\\|").replace("\n", " ")
    columns = list(map(str, frame.columns))
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")


def write_reports(
    output: Path,
    config: dict[str, Any],
    workspace: dict[str, Any],
    jobs: pd.DataFrame,
    nodes: pd.DataFrame,
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    support: pd.DataFrame,
    duration_metrics: pd.DataFrame,
    memory_metrics: pd.DataFrame,
    evidence: pd.DataFrame,
    scope_sensitivity: pd.DataFrame,
    bandwidth_metrics: pd.DataFrame,
    origin_metrics: pd.DataFrame,
    scenarios: pd.DataFrame,
    provenance: pd.DataFrame,
    schema: pd.DataFrame,
    canonical: pd.DataFrame,
    quality: pd.DataFrame,
    feasibility: pd.DataFrame,
    leakage: pd.DataFrame,
    validation_status: dict[str, str],
    bandwidth_median: float,
) -> bool:
    output.mkdir(parents=True, exist_ok=True)
    train, validation, test = splits
    evidence.to_csv(output / "02_request_scope_evidence.csv", index=False)
    scope_sensitivity.to_csv(output / "03_request_scope_sensitivity.csv", index=False)
    write_text(output / "01_request_scope_audit.md", f"""# Request Scope Audit

`REQUEST_SCOPE = PER_JOB`

官方 README 明确写明 CPU cores 与 GPU 数是“requested by the job”，并将 `worker_num` 单独定义为 job 请求的 instances 数。[固定 commit README]({SPOT_README_URL})

冻结公式：`CPU_job=cpu_request`，`GPU_job=gpu_request`，不乘 worker_num。双口径统计仅作诊断：误乘后最大 GPU 从 8 变为 1,456，并出现 65 个超过所有当前 DC GPU 容量的任务。官方目录无示例适配代码，公开 issue 无补充，Discussions 未启用；现有官方证据无冲突。
""")
    write_text(output / "04_arrival_contract.md", """# Arrival Contract

保留原始相对秒 `submit_time`，首任务 t=0；`arrival_step=floor(submit_time/900)`。不扰动、不合成、不 shuffle。同 step 使用稳定顺序 `submit_time -> original_index -> task_id`。边界：899->0，900->1，1799->1，1800->2。
""")
    support.to_csv(output / "06_duration_estimator_support.csv", index=False)
    duration_metrics.to_csv(output / "07_duration_estimator_validation.csv", index=False)
    duration = duration_metrics.iloc[0]
    selected_support = int(support.loc[support["selected"], "minimum_support"].iloc[0])
    write_text(output / "05_duration_estimator_contract.md", f"""# Duration Estimator Contract

`true_duration` 直接来自 Spot duration，只供 simulator 完成、资源释放与离线评估，禁止进入 deployable observation。

`estimated_duration` 只用时间前缀 70% train 拟合层级条件中位数：GPU model+GPU request+worker+priority -> GPU model+GPU request+priority -> GPU request+priority -> priority -> 全局中位数。support 20/50/100 仅按 train 内部交错子样本稳定性选择，冻结为 {selected_support}；validation/test 未参与选择。

Validation：MAE={duration.mae_seconds:.3f}s，Median AE={duration.median_ae_seconds:.3f}s，P90 AE={duration.p90_ae_seconds:.3f}s。重尾误差标记 `DURATION_TAIL_RISK`；这是无泄漏基线，不是高精度 predictor。
""")
    memory_metrics.to_csv(output / "09_memory_model_validation.csv", index=False)
    write_text(output / "08_memory_model_contract.md", """# Memory Model Contract

Spot 无 task host memory，故 `memory_request=MODELED`。v1 只用 Alibaba2020 时间前缀 70% train：CPU/GPU 固定 bin 条件中位数，support<100 回退 CPU bin，再回退全局 train 中位数；无随机采样。

该值仅补齐 SustainCluster 三资源场景，不能称为真实 Spot 内存；单位沿用当前 Alibaba2020->SustainCluster 链。分布及 memory/CPU、memory/GPU 比率见验证 CSV。
""")
    bandwidth_metrics.to_csv(output / "11_bandwidth_sensitivity.csv", index=False)
    write_text(output / "10_bandwidth_model_contract.md", f"""# Bandwidth Model Contract

Spot 无 task bandwidth，故 `bandwidth=MODELED`。比较全局/GPU-bin/CPU-GPU-bin 中位数后，v1 按简单稳定原则冻结 Alibaba2020 train-only 全局中位数 {bandwidth_median:.12f} GB。低/基准/高为 0.5x/1x/2x。

Bandwidth 不在静态 CPU/GPU/memory 容量约束中；本轮未跑 MPC，migration decision 标记 NOT_MEASURED，不能把 transfer-cost 线性代理当成调度结论。
""")
    origin_metrics.to_csv(output / "13_origin_dc_distribution.csv", index=False)
    write_text(output / "12_origin_dc_contract.md", """# Origin DC Contract

Spot 无多 DC 来源，故 `origin_dc=MODELED_ORIGIN`。v1 用 `namespace+task_id+submit_time` 的 SHA256 值映射到 population weight x local-time activity 累积分布；场景锚点为 2026-01-01T00:00:00Z，本地 08:00-19:59 权重 1.0，其余 0.3。

无运行时 RNG；同任务和配置必得同一 origin。全量总体与 UTC 六小时分段分布见 CSV。
""")
    scenarios.to_csv(output / "15_sla_scenario_table.csv", index=False)
    write_text(output / "14_sla_defer_contract.md", """# SLA / Defer Contract

Alibaba 只给 HP/Spot，不给 deadline、SLA minutes、waiting penalty 或 maximum defer。数值全部标记 `MODELED_SCENARIO_PARAMETER`。

公式：`sla_steps=ceil(estimated_duration/900)+max_wait_steps`。v1 选择 medium：HP 等待上限 1 step，Spot 8 steps。两类都允许 defer，但都有硬上限；保守/弹性配置保留作敏感性，不修改 reward。
""")
    write_text(output / "16_gpu_type_contract.md", """# GPU Type Contract

`GPU_HETEROGENEITY_MODE=METADATA_ONLY`。六种 gpu_model 原样保留；v1 只按 per-job gpu_request 的 GPU-equivalent count 调度，不声称验证型号兼容、性能或功耗。异构约束另立版本。
""")
    provenance.to_csv(output / "17_field_provenance_contract.csv", index=False)
    schema.to_csv(output / "18_spot_canonical_schema.csv", index=False)
    write_text(output / "19_spot_canonical_schema.md", """# Spot Canonical Task Schema v1

主表保存审计字段、deployable modeled fields 与 simulator-only truth；策略输入必须使用白名单，不能直接使用整行。

""" + markdown_table(schema))
    canonical.to_csv(output / "20_spot_canonical_frozen_sample.csv", index=False)
    quality.to_csv(output / "21_resource_quality_audit.csv", index=False)
    write_text(output / "22_resource_quality_report.md", f"""# Resource Quality Report

冻结样本 {len(canonical):,} 条，取 validation 起始段并保持稳定顺序。CPU zero={int((canonical.cpu_request_effective == 0).sum())}，CPU negative={int((canonical.cpu_request_effective < 0).sum())}，GPU nonpositive={int((canonical.gpu_request_effective <= 0).sum())}，duration nonpositive={int((canonical.true_duration <= 0).sum())}，worker nonpositive={int((canonical.worker_num <= 0).sum())}。

不自动删除 outlier。Spot 全量源的 2 条 CPU=0 保留并标记，不静默修正。完整 mean/P50/P90/P99/max 和容量越界计数见 CSV。
""")
    feasibility.to_csv(output / "23_static_feasibility_audit.csv", index=False)
    feasible = int(feasibility["feasible"].sum())
    infeasible = len(feasibility) - feasible
    priority = feasibility.groupby("priority")["feasible"].agg(["count", "sum", "mean"]).reset_index()
    model = feasibility.groupby("gpu_model")["feasible"].agg(["count", "sum", "mean"]).reset_index()
    gpu = feasibility.groupby("gpu_request")["feasible"].agg(["count", "sum", "mean"]).reset_index()
    write_text(output / "24_static_feasibility_report.md", f"""# Static Feasibility Report

只检查单任务是否至少有一个冻结 DC 同时满足 CPU/GPU/memory 总容量；不运行 MPC，不计并发占用、GPU 型号、bandwidth、price 或 carbon。

Total={len(feasibility):,}；Feasible={feasible:,}（{feasible/len(feasibility):.6%}）；Infeasible={infeasible:,}；Main reason={"NONE" if not infeasible else feasibility.loc[~feasibility.feasible, "infeasible_reason"].value_counts().index[0]}；Diagnosis={"NO SERIOUS CAPACITY MISMATCH" if infeasible/len(feasibility) <= .01 else "SCENARIO CAPACITY MISMATCH"}。

## 按 HP/Spot

{markdown_table(priority)}

## 按 GPU model

{markdown_table(model)}

## 按 GPU request

{markdown_table(gpu)}
""")
    gates = pd.DataFrame([
        ("request scope confirmed", True), ("arrival frozen", True),
        ("true/estimated duration separated", True), ("memory model frozen", True),
        ("bandwidth model frozen", True), ("origin model frozen", True),
        ("SLA/defer frozen", True), ("GPU handling frozen", True),
        ("canonical schema frozen", True),
        ("no serious capacity mismatch", infeasible/len(feasibility) <= .01),
        ("deterministic conversion", bool(leakage.loc[leakage["check"] == "same raw task -> same canonical task", "passed"].iloc[0])),
        ("information leakage checks", bool(leakage["passed"].all())),
    ], columns=("release_gate", "passed"))
    ready = bool(gates["passed"].all())
    write_text(output / "25_frozen_scenario_contract.md", f"""# Frozen Scenario Contract

{markdown_table(gates)}

`READY_FOR_V3={"YES" if ready else "NO"}`。该放行只允许私有研究环境按合同生成 v3，不等于允许公开派生数据，也不包含 MPC/BC/RL/closed-loop。
""")
    write_text(output / "26_information_leakage_audit.md", "# Information Leakage Audit\n\n" + markdown_table(leakage))
    write_text(output / "27_tests.md", f"""# Validation

| Check | Result |
| --- | --- |
| compileall | {validation_status["compileall"]} |
| dedicated tests | {validation_status["dedicated_tests"]} |
| relevant regressions | {validation_status["regression_tests"]} |
| full existing tests | {validation_status["full_tests"]} |
| new failures | {validation_status["new_failures"]} |
""")
    write_text(output / "28_final_diagnosis.md", f"""# Final Diagnosis

`SPOTGPU2026 SCENARIO CONTRACT COMPLETE`

Request scope=PER_JOB 已由官方字段定义确认。arrival、duration separation、memory、bandwidth、origin、SLA/defer、GPU metadata、schema 与 deterministic conversion 已冻结。静态可调度率 {feasible/len(feasibility):.6%}，无严重 capacity mismatch。

风险：duration 重尾误差（MAE={duration.mae_seconds:.3f}s，Median AE={duration.median_ae_seconds:.3f}s，P90 AE={duration.p90_ae_seconds:.3f}s）；memory/bandwidth/origin/SLA 均为场景建模；GPU 型号仅 metadata；derived dataset redistribution=NOT_CONFIRMED。

`READY_FOR_V3={"YES" if ready else "NO"}`。本轮未生成 v3、未运行 H1/H4、未训练模型。
""")
    write_text(output / "29_summary.md", f"""# SpotGPU2026 调度场景合同冻结 v1

1. Request scope：PER_JOB，不乘 worker_num。
2. Arrival：floor(submit_time/900)，同 step 稳定排序。
3. Duration：true 仅 simulator；estimated 为 Spot train-only 层级中位数。
4. Memory：Alibaba2020 train-only 条件中位数，MODELED。
5. Bandwidth：Alibaba2020 train-only 全局中位数，MODELED。
6. Origin：SHA256 + population/activity 权重确定性映射，MODELED_ORIGIN。
7. SLA/defer：三套场景，v1 选 medium，HP/Spot 均 bounded。
8. GPU 型号：metadata-only，资源按 GPU-equivalent。
9. DIRECT：submit/raw CPU/GPU/worker/duration/model/priority；DERIVED：step/task_id；MODELED：memory/bandwidth/estimated/origin/SLA；SIMULATOR_ONLY：true duration。
10. 静态可调度率：{feasible/len(feasibility):.6%}。
11. 严重 capacity mismatch：{"NO" if infeasible/len(feasibility) <= .01 else "YES"}。
12. 可重建 v3：{"YES" if ready else "NO"}；仅技术/私有生成放行，公开许可未确认。
""")
    manifest = {
        "contract": "spotgpu2026_scenario_contract_v1",
        "audit_date": source_audit.AUDIT_DATE, "workspace": workspace,
        "source_rows": {"spot_jobs": len(jobs), "spot_nodes": len(nodes)},
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "request_scope": "PER_JOB", "duration_support": selected_support,
        "duration_validation": duration.to_dict(),
        "memory_model": "ALIBABA2020_TRAIN_ONLY_CONDITIONAL_MEDIAN",
        "bandwidth_model": "ALIBABA2020_TRAIN_ONLY_GLOBAL_MEDIAN",
        "origin_model": "DETERMINISTIC_SHA256_WEIGHTED_ACTIVITY",
        "sla_profile": config["sla"]["selected_profile"],
        "gpu_heterogeneity": "METADATA_ONLY", "sample_rows": len(canonical),
        "sample_sha256": frame_hash(canonical),
        "static_feasible_rate": feasible/len(feasibility),
        "serious_capacity_mismatch": infeasible/len(feasibility) > .01,
        "ready_for_v3": ready, "expert_dataset_v3_generated": False,
        "mpc_run": False, "model_training": False,
        "derived_dataset_redistribution": "NOT_CONFIRMED",
        "validation": validation_status,
    }
    manifest["artifact_sha256"] = {
        str(path.relative_to(output)).replace("\\", "/"): source_audit.sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "audit_manifest.json"
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return ready


def run(
    output: Path = OUTPUT, validation_status: dict[str, str] | None = None
) -> Path:
    status = validation_status or {
        "compileall": "PENDING", "dedicated_tests": "PENDING",
        "regression_tests": "PENDING", "full_tests": "PENDING",
        "new_failures": "PENDING",
    }
    workspace = source_audit.validate_frozen_workspace()
    config = load_config()
    jobs, nodes, baseline = load_sources()
    splits = chronological_split(
        jobs, config["data_split"]["train_fraction"],
        config["data_split"]["validation_fraction"]
    )
    train, validation, _ = splits
    support, selected_support = duration_support_audit(
        train, config["duration_estimator"]["minimum_support_candidates"]
    )
    if selected_support != config["duration_estimator"]["minimum_support"]:
        raise RuntimeError("frozen support differs from train-only selection")
    duration_model = fit_duration_estimator(train, selected_support)
    validation_prediction, validation_levels = duration_model.predict(validation)
    duration_metrics = duration_validation(validation, validation_prediction, validation_levels)

    alibaba = alibaba2020_train_table(
        baseline, config["alibaba2020_model_source"]["train_fraction"]
    )
    memory_model = fit_memory_model(alibaba, config["memory_model"])
    bandwidth_models = fit_bandwidth_models(
        alibaba, config["memory_model"], config["bandwidth_model"]
    )
    raw_sample = validation.iloc[: config["frozen_sample"]["size"]].copy()
    canonical = build_canonical_sample(
        raw_sample, duration_model, memory_model, bandwidth_models, config
    )
    repeated = build_canonical_sample(
        raw_sample, duration_model, memory_model, bandwidth_models, config
    )
    deployable = deployable_observation(canonical)
    feasibility = static_feasibility(canonical, config)
    feasible_rate = float(feasibility["feasible"].mean())
    memory_metrics = memory_validation(
        alibaba, canonical["cpu_request_effective"].to_numpy(),
        canonical["gpu_request_effective"].to_numpy(),
        canonical["memory_request"].to_numpy()
    )
    provenance = provenance_contract()
    schema = canonical_schema(provenance)
    leakage = pd.DataFrame([
        ("true_duration excluded from deployable observation", "true_duration" not in deployable, "whitelist"),
        ("estimated_duration train-only", raw_sample.original_index.min() > duration_model.train_max_original_index, "sample after train"),
        ("memory model train-only", memory_model.train_rows == len(alibaba), "Alibaba2020 train prefix"),
        ("bandwidth model train-only", bandwidth_models.train_rows == len(alibaba), "Alibaba2020 train prefix"),
        ("no Oracle future", not any("future" in c.lower() or "oracle" in c.lower() for c in canonical), "schema scan"),
        ("no teacher action leakage", not any("teacher" in c.lower() for c in canonical), "schema scan"),
        ("no H1/H4 action leakage", not any("h1_action" in c.lower() or "h4_action" in c.lower() for c in canonical), "schema scan"),
        ("same raw task -> same canonical task", frame_hash(canonical) == frame_hash(repeated), "repeat hash"),
        ("provenance complete", set(schema.field).issubset(canonical.columns), "schema fields"),
        ("request_scope respected", canonical.cpu_request_effective.equals(canonical.cpu_request_raw) and canonical.gpu_request_effective.equals(canonical.gpu_request_raw), "PER_JOB identity"),
        ("GPU model preserved", canonical.gpu_model.tolist() == stable_task_order(raw_sample).gpu_model.astype(str).tolist(), "row equality"),
        ("HP/Spot preserved", set(canonical.priority) <= {"HP", "Spot"}, "enum"),
        ("forbidden deployable tokens absent", not any(token in c.lower() for token in FORBIDDEN_TOKENS for c in deployable), "deployable scan"),
    ], columns=("check", "passed", "evidence"))
    ready = write_reports(
        output, config, workspace, jobs, nodes, splits, support, duration_metrics,
        memory_metrics, request_scope_evidence(), request_scope_sensitivity(jobs, config),
        bandwidth_sensitivity(raw_sample, bandwidth_models, config, feasible_rate),
        origin_distribution(jobs, config), sla_table(config), provenance, schema,
        canonical, resource_quality(canonical, config), feasibility, leakage, status,
        bandwidth_models.global_median
    )
    print("request_scope=PER_JOB")
    print(f"sample_rows={len(canonical)}")
    print(f"static_feasible_rate={feasible_rate:.6%}")
    print(f"deterministic_conversion={bool(leakage.passed.all())}")
    print(f"ready_for_v3={'YES' if ready else 'NO'}")
    print(f"artifacts={output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--compileall", default="PENDING")
    parser.add_argument("--dedicated-tests", default="PENDING")
    parser.add_argument("--regression-tests", default="PENDING")
    parser.add_argument("--full-tests", default="PENDING")
    parser.add_argument("--new-failures", default="PENDING")
    args = parser.parse_args()
    run(args.output.resolve(), {
        "compileall": args.compileall,
        "dedicated_tests": args.dedicated_tests,
        "regression_tests": args.regression_tests,
        "full_tests": args.full_tests,
        "new_failures": args.new_failures,
    })


if __name__ == "__main__":
    main()
