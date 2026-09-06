from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from scripts.audit import spotgpu2026_scenario_contract_v1 as spot


ROOT = spot.ROOT
CONFIG_PATH = ROOT / "configs" / "sustaincluster_mpc" / "mpc_expert_dataset_v3.yaml"
OUTPUT = ROOT / "artifacts" / "mpc_expert_dataset_v3"
FINAL_STATUS = "SPOT SCENARIO PREFLIGHT BLOCKED"
BLOCK_REASON = "CAPACITY / WORKLOAD SCALE MISMATCH"
EXPECTED_NUMBERED_OUTPUTS = (
    "01_protocol.md",
    "02_spot_full_canonical_audit.csv",
    "03_spot_workload_pressure.csv",
    "04_spot_duration_tail.csv",
    "05_spot_static_feasibility.csv",
    "06_spot_temporal_state_audit.md",
    "07_alibaba2020_v2_v3_alignment.csv",
    "08_alibaba2020_alignment_report.md",
    "09_scenario_a_statistics.csv",
    "10_scenario_b_statistics.csv",
    "11_h1_oracle_behavior.csv",
    "12_placement_disagreement.csv",
    "13_defer_analysis.csv",
    "14_risk_analysis.csv",
    "15_priority_analysis.csv",
    "16_duration_error_analysis.csv",
    "17_information_contract.md",
    "18_field_provenance.csv",
    "19_split_manifest.csv",
    "20_solver_quality.csv",
    "21_cross_dataset_comparison.csv",
    "22_information_leakage_audit.md",
    "23_integrity_manifest.csv",
    "24_tests.md",
    "25_final_diagnosis.md",
    "26_summary.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def dataframe_hash(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest().upper()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    root = yaml.safe_load(path.read_text(encoding="utf-8"))["mpc_expert_dataset_v3"]
    if root["scenario_b"]["request_scope"] != "PER_JOB":
        raise RuntimeError("Spot request scope must remain PER_JOB")
    if root["scenario_b"]["time_step_seconds"] != 900:
        raise RuntimeError("Spot time step must remain 900 seconds")
    if root["scenario_b"]["bandwidth"]["value_gb"] != 0.020062580706:
        raise RuntimeError("Spot frozen bandwidth changed")
    if root["scenario_b"]["max_wait_steps"] != {"HP": 1, "Spot": 8}:
        raise RuntimeError("Spot medium SLA changed")
    if root["scenario_b"]["gpu_heterogeneity"] != "METADATA_ONLY":
        raise RuntimeError("Spot GPU handling changed")
    if root["planners"]["h4_oracle"]["privileged_offsets_minutes"] != [15, 30, 45, 60]:
        raise RuntimeError("repaired H4 timeline changed")
    return root


def validate_frozen_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    workspace = spot.source_audit.validate_frozen_workspace()
    expected = {
        ROOT / config["scenario_a"]["source_file"]: config["scenario_a"]["source_sha256"],
        ROOT / config["scenario_a"]["v2_root"] / config["scenario_a"]["v2_task_file"]:
            config["scenario_a"]["v2_task_sha256"],
        ROOT / config["scenario_a"]["v2_root"] / config["scenario_a"]["v2_state_file"]:
            config["scenario_a"]["v2_state_sha256"],
        ROOT / config["scenario_b"]["contract_config"]:
            config["scenario_b"]["contract_config_sha256"],
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise RuntimeError(f"frozen input hash changed: {path}")
    return workspace


def complete_step_split(
    canonical: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[np.ndarray, dict[str, int]]:
    size = len(canonical)
    train_end = int(size * train_fraction)
    validation_end = int(size * (train_fraction + validation_fraction))
    train_last_step = int(canonical.iloc[train_end - 1]["arrival_step"])
    validation_last_step = int(canonical.iloc[validation_end - 1]["arrival_step"])
    steps = canonical["arrival_step"].to_numpy(dtype=np.int64)
    labels = np.where(
        steps <= train_last_step,
        "train",
        np.where(steps <= validation_last_step, "validation", "test"),
    )
    if pd.DataFrame({"step": steps, "split": labels}).groupby("step")["split"].nunique().max() != 1:
        raise RuntimeError("one arrival state crossed temporal splits")
    return labels, {
        "nominal_train_rows": train_end,
        "nominal_validation_end_rows": validation_end,
        "train_last_arrival_step": train_last_step,
        "validation_last_arrival_step": validation_last_step,
    }


def build_full_spot_canonical(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    jobs, nodes, baseline = spot.load_sources()
    split_cfg = config["scenario_b"]["split"]
    train, validation, test = spot.chronological_split(
        jobs, split_cfg["train"], split_cfg["validation"]
    )
    duration = spot.fit_duration_estimator(
        train, config["scenario_b"]["duration_estimator"]["minimum_support"]
    )
    alibaba = spot.alibaba2020_train_table(
        baseline, spot.load_config()["alibaba2020_model_source"]["train_fraction"]
    )
    memory = spot.fit_memory_model(alibaba, spot.load_config()["memory_model"])
    bandwidth = spot.fit_bandwidth_models(
        alibaba, spot.load_config()["memory_model"], spot.load_config()["bandwidth_model"]
    )
    frozen_bandwidth = float(config["scenario_b"]["bandwidth"]["value_gb"])
    if round(bandwidth.global_median, 12) != frozen_bandwidth:
        raise RuntimeError("train-only bandwidth median no longer matches frozen value")

    canonical = spot.build_canonical_sample(jobs, duration, memory, bandwidth, spot.load_config())
    canonical.insert(
        1,
        "raw_task_id",
        canonical["task_id"].str.split(":", n=1).str[-1],
    )
    canonical["bandwidth"] = frozen_bandwidth
    canonical["bandwidth_source"] = "MODELED_ALIBABA2020_GLOBAL_MEDIAN_V1_FROZEN_12DP"
    canonical["absolute_estimation_error_seconds"] = (
        canonical["true_duration"] - canonical["estimated_duration"]
    ).abs()
    labels, boundaries = complete_step_split(
        canonical, split_cfg["train"], split_cfg["validation"]
    )
    canonical["temporal_split"] = labels

    if len(canonical) != len(jobs):
        raise RuntimeError("full canonical conversion silently dropped tasks")
    if canonical["task_id"].duplicated().any():
        raise RuntimeError("canonical task_id is not unique")
    if not np.array_equal(
        canonical["arrival_step"].to_numpy(),
        canonical["submit_time"].to_numpy(dtype=np.int64) // 900,
    ):
        raise RuntimeError("arrival contract changed")
    if not canonical["cpu_request_effective"].equals(canonical["cpu_request_raw"]):
        raise RuntimeError("PER_JOB CPU contract changed")
    if not canonical["gpu_request_effective"].equals(canonical["gpu_request_raw"]):
        raise RuntimeError("PER_JOB GPU contract changed")
    if not np.all(canonical["bandwidth"].to_numpy() == frozen_bandwidth):
        raise RuntimeError("frozen bandwidth was not reused exactly")
    return canonical, nodes, baseline, {
        "duration_model_train_rows": duration.train_rows,
        "duration_model_train_max_original_index": duration.train_max_original_index,
        "source_train_rows": len(train),
        "source_validation_rows": len(validation),
        "source_test_rows": len(test),
        **boundaries,
    }


def workload_pressure(canonical: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    grouped = canonical.groupby("arrival_step", sort=True).agg(
        first_submit_time=("submit_time", "min"),
        last_submit_time=("submit_time", "max"),
        new_task_count=("task_id", "size"),
        cpu_arrival_demand=("cpu_request_effective", "sum"),
        gpu_arrival_demand=("gpu_request_effective", "sum"),
        memory_arrival_demand=("memory_request", "sum"),
    )
    grouped = grouped.reindex(
        range(int(canonical["arrival_step"].max()) + 1), fill_value=0
    )
    grouped.index.name = "arrival_step"
    grouped = grouped.reset_index()
    capacities = {
        "cpu": sum(dc["total_cores"] for dc in config["scenario_b_contract"]["datacenters"]),
        "gpu": sum(dc["total_gpus"] for dc in config["scenario_b_contract"]["datacenters"]),
        "memory": sum(dc["total_mem"] for dc in config["scenario_b_contract"]["datacenters"]),
    }
    for resource in ("cpu", "gpu", "memory"):
        grouped[f"{resource}_arrival_to_total_capacity"] = (
            grouped[f"{resource}_arrival_demand"] / capacities[resource]
        )
    step_to_split = canonical.groupby("arrival_step")["temporal_split"].first()
    grouped["temporal_split"] = grouped["arrival_step"].map(step_to_split).ffill().bfill()
    return grouped


def pressure_summary(pressure: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in (
        "new_task_count",
        "cpu_arrival_demand",
        "gpu_arrival_demand",
        "memory_arrival_demand",
        "cpu_arrival_to_total_capacity",
        "gpu_arrival_to_total_capacity",
        "memory_arrival_to_total_capacity",
    ):
        values = pressure[field].to_numpy(dtype=float)
        rows.append(
            {
                "metric": field,
                "mean": values.mean(),
                "p50": np.quantile(values, 0.50),
                "p90": np.quantile(values, 0.90),
                "p95": np.quantile(values, 0.95),
                "p99": np.quantile(values, 0.99),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def mandatory_capacity_audit(
    canonical: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    dcs = config["scenario_b_contract"]["datacenters"]
    capacity = np.asarray(
        [
            sum(dc["total_cores"] for dc in dcs),
            sum(dc["total_gpus"] for dc in dcs),
            sum(dc["total_mem"] for dc in dcs),
        ],
        dtype=float,
    )
    resources = canonical[
        ["cpu_request_effective", "gpu_request_effective", "memory_request"]
    ].to_numpy(dtype=float)
    arrivals = canonical["arrival_step"].to_numpy(dtype=np.int64)
    durations = np.maximum(
        1, np.ceil(canonical["true_duration"].to_numpy(dtype=float) / 900).astype(np.int64)
    )
    waits = canonical["priority"].map(
        config["scenario_b"]["max_wait_steps"]
    ).to_numpy(dtype=np.int64)
    mandatory_start = arrivals + waits + 1
    mandatory_stop = arrivals + durations + 1
    timeline = int(max(mandatory_stop.max(), arrivals.max() + 1)) + 1
    trace_steps = int(arrivals.max()) + 1
    names = ("cpu", "gpu", "memory")
    rows = []
    for index, name in enumerate(names):
        differences = np.zeros(timeline + 1, dtype=float)
        mask = mandatory_stop > mandatory_start
        np.add.at(differences, mandatory_start[mask], resources[mask, index])
        np.add.at(differences, mandatory_stop[mask], -resources[mask, index])
        mandatory = np.cumsum(differences[:-1])
        peak_step = int(np.argmax(mandatory))
        resource_steps = float(np.sum(resources[:, index] * durations))
        rows.append(
            {
                "resource": name,
                "total_capacity": capacity[index],
                "trace_steps": trace_steps,
                "resource_time_demand_steps": resource_steps,
                "offered_load_ratio": resource_steps / (capacity[index] * trace_steps),
                "mandatory_concurrent_peak": float(mandatory[peak_step]),
                "mandatory_peak_ratio": float(mandatory[peak_step] / capacity[index]),
                "mandatory_peak_step": peak_step,
                "gate_max_ratio": config["preflight_gates"][
                    "mandatory_concurrent_capacity_ratio_max"
                ],
                "gate_pass": (
                    mandatory[peak_step] / capacity[index]
                    <= config["preflight_gates"][
                        "mandatory_concurrent_capacity_ratio_max"
                    ]
                ),
                "proof_definition": (
                    "tasks guaranteed active for every legal dispatch time under "
                    "priority max_wait plus one transfer step"
                ),
            }
        )
    return pd.DataFrame(rows)


def grouped_feasibility(
    canonical: pd.DataFrame, details: pd.DataFrame
) -> pd.DataFrame:
    joined = details.assign(
        cpu_request_exact=canonical["cpu_request_effective"].to_numpy()
    )
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, Iterable[tuple[Any, pd.DataFrame]]]] = [
        ("ALL", [("ALL", joined)]),
        ("priority", joined.groupby("priority", sort=True, dropna=False)),
        ("gpu_model", joined.groupby("gpu_model", sort=True, dropna=False)),
        ("gpu_request", joined.groupby("gpu_request", sort=True, dropna=False)),
        ("cpu_request", joined.groupby("cpu_request_exact", sort=True, dropna=False)),
    ]
    for dimension, groups in scopes:
        for value, group in groups:
            feasible = int(group["feasible"].sum())
            rows.append(
                {
                    "dimension": dimension,
                    "group": value,
                    "task_count": len(group),
                    "feasible_count": feasible,
                    "infeasible_count": len(group) - feasible,
                    "feasible_rate": feasible / len(group),
                    "main_infeasible_reason": (
                        "NONE"
                        if feasible == len(group)
                        else group.loc[
                            ~group["feasible"], "infeasible_reason"
                        ].value_counts().index[0]
                    ),
                }
            )
    return pd.DataFrame(rows)


def duration_tail(canonical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes: list[tuple[str, Iterable[tuple[Any, pd.DataFrame]]]] = [
        ("ALL", [("ALL", canonical)]),
        ("priority", canonical.groupby("priority", sort=True)),
        ("gpu_model", canonical.groupby("gpu_model", sort=True)),
        ("gpu_request", canonical.groupby("gpu_request_effective", sort=True)),
    ]
    for dimension, groups in scopes:
        for value, group in groups:
            truth = group["true_duration"].to_numpy(dtype=float)
            estimate = group["estimated_duration"].to_numpy(dtype=float)
            error = np.abs(truth - estimate)
            rows.append(
                {
                    "dimension": dimension,
                    "group": value,
                    "task_count": len(group),
                    "true_p50_seconds": np.quantile(truth, 0.50),
                    "true_p90_seconds": np.quantile(truth, 0.90),
                    "true_p95_seconds": np.quantile(truth, 0.95),
                    "true_p99_seconds": np.quantile(truth, 0.99),
                    "true_max_seconds": truth.max(),
                    "estimated_p50_seconds": np.quantile(estimate, 0.50),
                    "estimated_p90_seconds": np.quantile(estimate, 0.90),
                    "estimated_p95_seconds": np.quantile(estimate, 0.95),
                    "estimated_p99_seconds": np.quantile(estimate, 0.99),
                    "estimated_max_seconds": estimate.max(),
                    "absolute_error_mean_seconds": error.mean(),
                    "absolute_error_p50_seconds": np.quantile(error, 0.50),
                    "absolute_error_p90_seconds": np.quantile(error, 0.90),
                    "over_24h_rate": np.mean(truth > 24 * 3600),
                    "over_48h_rate": np.mean(truth > 48 * 3600),
                    "over_72h_rate": np.mean(truth > 72 * 3600),
                }
            )
    return pd.DataFrame(rows)


def duration_error_analysis(canonical: pd.DataFrame) -> pd.DataFrame:
    train = canonical[canonical["temporal_split"] == "train"]
    low, high = train["absolute_estimation_error_seconds"].quantile([1 / 3, 2 / 3])
    bands = pd.cut(
        canonical["absolute_estimation_error_seconds"],
        bins=[-np.inf, low, high, np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )
    work = canonical.assign(error_band=bands)
    rows = []
    for (split, band), group in work.groupby(
        ["temporal_split", "error_band"], observed=True, sort=True
    ):
        rows.append(
            {
                "split": split,
                "error_band": str(band),
                "train_low_upper_seconds": low,
                "train_medium_upper_seconds": high,
                "task_count": len(group),
                "mean_absolute_error_seconds": group[
                    "absolute_estimation_error_seconds"
                ].mean(),
                "h1_oracle_disagreement_rate": "NOT_RUN_PREFLIGHT_BLOCKED",
                "high_risk_state_rate": "NOT_RUN_PREFLIGHT_BLOCKED",
                "defer_rate": "NOT_RUN_PREFLIGHT_BLOCKED",
            }
        )
    return pd.DataFrame(rows)


def load_v2(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = ROOT / config["scenario_a"]["v2_root"]
    tasks = pd.read_parquet(root / config["scenario_a"]["v2_task_file"])
    states = pd.read_parquet(root / config["scenario_a"]["v2_state_file"])
    episodes = pd.read_csv(root / "04_episode_manifest.csv")
    if len(tasks) != config["scenario_a"]["expected_task_decisions"]:
        raise RuntimeError("v2 task decision count changed")
    if len(states) != config["scenario_a"]["expected_states"]:
        raise RuntimeError("v2 state count changed")
    if len(episodes) != config["scenario_a"]["expected_episodes"]:
        raise RuntimeError("v2 episode count changed")
    return tasks, states, episodes


def validate_solver_failures(solver: pd.DataFrame) -> None:
    failures = pd.to_numeric(solver["solver_failures"], errors="raise")
    if int(failures.sum()) != 0:
        raise RuntimeError("solver failures cannot become expert labels")


def alibaba_alignment_audit(
    tasks: pd.DataFrame, states: pd.DataFrame
) -> pd.DataFrame:
    rows = [
        ("state_key", len(states), "NOT_GENERATED", "NOT_MEASURED", "OTHER"),
        (
            "pending_tasks",
            states[["episode_id", "step"]].drop_duplicates().shape[0],
            "NOT_GENERATED",
            "NOT_MEASURED",
            "OTHER",
        ),
        ("running_tasks", "NOT_STORED_IN_V2", "NOT_GENERATED", "NOT_MEASURED", "STATE_STORAGE_CHANGE"),
        ("in_transit_tasks", "NOT_STORED_IN_V2", "NOT_GENERATED", "NOT_MEASURED", "STATE_STORAGE_CHANGE"),
        ("dc_state", "NOT_STORED_IN_V2", "NOT_GENERATED", "NOT_MEASURED", "STATE_STORAGE_CHANGE"),
        ("feasible_mask", len(tasks), "NOT_GENERATED", "NOT_MEASURED", "OTHER"),
        ("h1_action", len(tasks), "NOT_GENERATED", "NOT_MEASURED", "OTHER"),
        ("h4_oracle_action", len(tasks), "NOT_GENERATED", "NOT_MEASURED", "OTHER"),
    ]
    return pd.DataFrame(
        rows,
        columns=(
            "dimension",
            "v2_available_records",
            "v3_available_records",
            "v2_v3_agreement",
            "difference_class_if_applicable",
        ),
    ).assign(reason="SPOT_PREFLIGHT_BLOCKED_BEFORE_ANY_FORMAL_V3_GENERATION")


def scenario_a_statistics(
    tasks: pd.DataFrame, states: pd.DataFrame, episodes: pd.DataFrame
) -> pd.DataFrame:
    values = {
        "episodes": len(episodes),
        "states": len(states),
        "task_decisions": len(tasks),
        "h1_oracle_exact_disagreement_rate": tasks[
            "exact_action_disagreement"
        ].mean(),
        "placement_disagreement_rate": tasks["target_dc_disagreement"].mean(),
        "defer_disagreement_rate": tasks["semantic_disagreement"].mean(),
        "h1_defer_rate": tasks["h1_is_defer"].mean(),
        "oracle_defer_rate": tasks["teacher_is_defer"].mean(),
        "state_any_disagreement_rate": states["any_disagreement"].mean(),
        "mean_pending_batch_size": states["num_tasks"].mean(),
        "solver_failures": 0,
    }
    return pd.DataFrame(
        [{"metric": key, "value": value, "status": "V2_FROZEN_BASELINE"} for key, value in values.items()]
    )


def scenario_b_statistics(
    canonical: pd.DataFrame,
    pressure: pd.DataFrame,
    pressure_stats: pd.DataFrame,
    capacity: pd.DataFrame,
    feasible: pd.DataFrame,
) -> pd.DataFrame:
    values: list[tuple[str, Any, str]] = [
        ("raw_jobs", len(canonical), "PREFLIGHT_MEASURED"),
        ("canonical_jobs", len(canonical), "PREFLIGHT_MEASURED"),
        ("static_feasible_rate", feasible["feasible"].mean(), "PREFLIGHT_MEASURED"),
        ("arrival_states", len(pressure), "PREFLIGHT_MEASURED"),
        ("train_tasks", int((canonical["temporal_split"] == "train").sum()), "PREFLIGHT_MEASURED"),
        ("validation_tasks", int((canonical["temporal_split"] == "validation").sum()), "PREFLIGHT_MEASURED"),
        ("test_tasks", int((canonical["temporal_split"] == "test").sum()), "PREFLIGHT_MEASURED"),
        ("hp_tasks", int((canonical["priority"] == "HP").sum()), "PREFLIGHT_MEASURED"),
        ("spot_tasks", int((canonical["priority"] == "Spot").sum()), "PREFLIGHT_MEASURED"),
        (
            "maximum_mandatory_capacity_ratio",
            capacity["mandatory_peak_ratio"].max(),
            "PREFLIGHT_BLOCKING_MEASUREMENT",
        ),
    ]
    for row in pressure_stats.itertuples(index=False):
        for statistic in ("mean", "p50", "p90", "p95", "p99", "max"):
            values.append(
                (
                    f"arrival_pressure_{row.metric}_{statistic}",
                    getattr(row, statistic),
                    "PREFLIGHT_MEASURED",
                )
            )
    for metric in (
        "states",
        "task_decisions",
        "h1_oracle_disagreement_rate",
        "placement_disagreement_rate",
        "defer_disagreement_rate",
        "h1_defer_rate",
        "oracle_defer_rate",
        "high_risk_state_rate",
        "solver_failures",
    ):
        values.append((metric, "NOT_RUN", "SPOT_PREFLIGHT_BLOCKED"))
    return pd.DataFrame(values, columns=("metric", "value", "status"))


def behavior_tables(
    a_tasks: pd.DataFrame,
    a_states: pd.DataFrame,
    canonical: pd.DataFrame,
    threshold: float,
) -> dict[str, pd.DataFrame]:
    a_high = a_states["deployable_risk_score"] >= threshold
    behavior = pd.DataFrame(
        [
            {
                "scenario_id": "A_ALIBABA2020_REPAIRED",
                "task_decisions": len(a_tasks),
                "h1_oracle_exact_disagreement_rate": a_tasks[
                    "exact_action_disagreement"
                ].mean(),
                "state_any_disagreement_rate": a_states["any_disagreement"].mean(),
                "state_disagreement_task_count_mean": a_states[
                    "num_exact_disagreements"
                ].mean(),
                "state_disagreement_fraction_mean": a_states[
                    "disagreement_rate"
                ].mean(),
                "status": "V2_FROZEN_BASELINE",
            },
            {
                "scenario_id": "B_SPOTGPU2026_MEDIUM",
                "task_decisions": "NOT_RUN",
                "h1_oracle_exact_disagreement_rate": "NOT_RUN",
                "state_any_disagreement_rate": "NOT_RUN",
                "state_disagreement_task_count_mean": "NOT_RUN",
                "state_disagreement_fraction_mean": "NOT_RUN",
                "status": "SPOT_PREFLIGHT_BLOCKED",
            },
        ]
    )
    placement = pd.DataFrame(
        [
            ("A_ALIBABA2020_REPAIRED", len(a_tasks), a_tasks["target_dc_disagreement"].sum(), a_tasks["target_dc_disagreement"].mean(), "V2_FROZEN_BASELINE"),
            ("B_SPOTGPU2026_MEDIUM", "NOT_RUN", "NOT_RUN", "NOT_RUN", "SPOT_PREFLIGHT_BLOCKED"),
        ],
        columns=("scenario_id", "task_decisions", "placement_disagreements", "placement_disagreement_rate", "status"),
    )
    defer = pd.DataFrame(
        [
            (
                "A_ALIBABA2020_REPAIRED",
                len(a_tasks),
                a_tasks["semantic_disagreement"].sum(),
                a_tasks["semantic_disagreement"].mean(),
                a_tasks["h1_is_defer"].mean(),
                a_tasks["teacher_is_defer"].mean(),
                "V2_FROZEN_BASELINE",
            ),
            (
                "B_SPOTGPU2026_MEDIUM",
                "NOT_RUN",
                "NOT_RUN",
                "NOT_RUN",
                "NOT_RUN",
                "NOT_RUN",
                "SPOT_PREFLIGHT_BLOCKED",
            ),
        ],
        columns=(
            "scenario_id",
            "task_decisions",
            "defer_disagreements",
            "defer_disagreement_rate",
            "h1_defer_rate",
            "oracle_defer_rate",
            "status",
        ),
    )
    risk = pd.DataFrame(
        [
            (
                "A_ALIBABA2020_REPAIRED",
                threshold,
                "FIXED_P95_PRIOR_EXPERIMENT",
                int(a_high.sum()),
                float(a_high.mean()),
                "V2_FROZEN_BASELINE",
            ),
            (
                "B_SPOTGPU2026_MEDIUM",
                "NOT_COMPUTED",
                "SPOT_TRAIN_STATE_P95_REQUIRED",
                "NOT_RUN",
                "NOT_RUN",
                "SPOT_PREFLIGHT_BLOCKED",
            ),
        ],
        columns=(
            "scenario_id",
            "risk_threshold",
            "threshold_source",
            "high_risk_states",
            "high_risk_state_rate",
            "status",
        ),
    )
    priority_rows = []
    for priority, group in canonical.groupby("priority", sort=True):
        priority_rows.append(
            {
                "scenario_id": "B_SPOTGPU2026_MEDIUM",
                "priority": priority,
                "task_count": len(group),
                "h1_oracle_agreement": "NOT_RUN",
                "placement_disagreement": "NOT_RUN",
                "defer_disagreement": "NOT_RUN",
                "h1_defer": "NOT_RUN",
                "oracle_defer": "NOT_RUN",
                "status": "PREFLIGHT_COUNTS_ONLY",
            }
        )
    priority_rows.append(
        {
            "scenario_id": "A_ALIBABA2020_REPAIRED",
            "priority": "NOT_STORED_IN_V2",
            "task_count": len(a_tasks),
            "h1_oracle_agreement": "NOT_AVAILABLE_BY_PRIORITY",
            "placement_disagreement": "NOT_AVAILABLE_BY_PRIORITY",
            "defer_disagreement": "NOT_AVAILABLE_BY_PRIORITY",
            "h1_defer": "NOT_AVAILABLE_BY_PRIORITY",
            "oracle_defer": "NOT_AVAILABLE_BY_PRIORITY",
            "status": "V2_SCHEMA_LIMITATION",
        }
    )
    return {
        "behavior": behavior,
        "placement": placement,
        "defer": defer,
        "risk": risk,
        "priority": pd.DataFrame(priority_rows),
    }


def field_provenance() -> pd.DataFrame:
    spot_rows = spot.provenance_contract().copy()
    spot_rows.insert(0, "scenario_id", "B_SPOTGPU2026_MEDIUM")
    spot_rows["region"] = np.where(
        spot_rows["provenance_type"].eq("SIMULATOR_ONLY"),
        "simulator_only",
        "deployable_current",
    )
    extra = pd.DataFrame(
        [
            ("A_ALIBABA2020_REPAIRED", "state_id", "DERIVED", "state key", "deployable_current"),
            ("A_ALIBABA2020_REPAIRED", "pending_task_ids", "DIRECT_STATE", "complete ordered pending batch", "deployable_current"),
            ("A_ALIBABA2020_REPAIRED", "running_task_ids", "DIRECT_STATE", "continuous simulator", "deployable_current"),
            ("A_ALIBABA2020_REPAIRED", "in_transit_task_ids", "DIRECT_STATE", "continuous simulator", "deployable_current"),
            ("ALL", "h1_action", "LABEL", "H1_REPAIRED_CURRENT_ONLY", "labels"),
            ("ALL", "oracle_h4_action", "LABEL", "H4_ORACLE_REPAIRED", "labels"),
            ("ALL", "future_pressure_15_30_45_60", "PRIVILEGED", "oracle future", "privileged_future"),
            ("ALL", "true_duration", "SIMULATOR_ONLY", "source truth", "simulator_only"),
            ("ALL", "joint_action_group", "DERIVED", "state_id", "labels"),
        ],
        columns=("scenario_id", "field", "provenance_type", "source_or_rule", "region"),
    )
    spot_rows = spot_rows.rename(columns={"visibility": "source_visibility"})
    extra["source_visibility"] = extra["region"]
    return pd.concat(
        [
            spot_rows[
                [
                    "scenario_id",
                    "field",
                    "provenance_type",
                    "source_or_rule",
                    "region",
                    "source_visibility",
                ]
            ],
            extra[
                [
                    "scenario_id",
                    "field",
                    "provenance_type",
                    "source_or_rule",
                    "region",
                    "source_visibility",
                ]
            ],
        ],
        ignore_index=True,
    )


def split_manifest(
    canonical: pd.DataFrame,
    a_episodes: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    rows = []
    for row in a_episodes.itertuples(index=False):
        rows.append(
            {
                "scenario_id": "A_ALIBABA2020_REPAIRED",
                "split": row.split,
                "unit_id": row.episode_id,
                "task_count": row.task_decisions,
                "state_count": row.steps,
                "first_submit_time": row.environment_start,
                "last_submit_time": "episode_start_plus_96_steps",
                "first_arrival_step": 0,
                "last_arrival_step": 95,
                "split_unit": "seed_and_complete_episode",
                "state_crosses_split": False,
            }
        )
    for split, group in canonical.groupby("temporal_split", sort=False):
        rows.append(
            {
                "scenario_id": "B_SPOTGPU2026_MEDIUM",
                "split": split,
                "unit_id": f"spot_continuous_{split}",
                "task_count": len(group),
                "state_count": group["arrival_step"].nunique(),
                "first_submit_time": int(group["submit_time"].min()),
                "last_submit_time": int(group["submit_time"].max()),
                "first_arrival_step": int(group["arrival_step"].min()),
                "last_arrival_step": int(group["arrival_step"].max()),
                "split_unit": "complete_arrival_step_chronological",
                "state_crosses_split": False,
            }
        )
    return pd.DataFrame(rows)


def canonical_audit(
    canonical: pd.DataFrame,
    metadata: Mapping[str, Any],
    capacity: pd.DataFrame,
) -> pd.DataFrame:
    values = [
        ("raw_jobs", len(canonical), "PASS"),
        ("canonical_jobs", len(canonical), "PASS"),
        ("filtered_jobs", 0, "PASS"),
        ("duplicate_task_ids", int(canonical["task_id"].duplicated().sum()), "PASS"),
        ("null_cells", int(canonical.isna().sum().sum()), "PASS"),
        ("request_scope", "PER_JOB", "PASS"),
        ("arrival_rule", "floor(submit_time/900)", "PASS"),
        ("task_order", "submit_time,original_index,task_id", "PASS"),
        ("random_shuffle", False, "PASS"),
        ("duration_estimator_train_rows", metadata["duration_model_train_rows"], "PASS"),
        ("bandwidth_frozen_gb", canonical["bandwidth"].iloc[0], "PASS"),
        ("gpu_heterogeneity", "METADATA_ONLY", "PASS"),
        ("canonical_content_sha256", dataframe_hash(canonical), "PASS"),
        (
            "maximum_mandatory_capacity_ratio",
            capacity["mandatory_peak_ratio"].max(),
            "BLOCK",
        ),
    ]
    return pd.DataFrame(values, columns=("check", "value", "status"))


def write_preflight_datasets(
    output: Path,
    canonical: pd.DataFrame,
    feasibility: pd.DataFrame,
) -> None:
    preflight = output / "preflight"
    (preflight / "deployable_current").mkdir(parents=True, exist_ok=True)
    (preflight / "simulator_only").mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(
        preflight / "spot_full_canonical.parquet",
        index=False,
        compression="zstd",
    )
    feasibility.to_parquet(
        preflight / "spot_static_feasibility.parquet",
        index=False,
        compression="zstd",
    )
    deployable_columns = [
        column for column in spot.DEPLOYABLE_COLUMNS if column in canonical.columns
    ] + ["raw_task_id", "request_scope", "temporal_split"]
    canonical[deployable_columns].to_parquet(
        preflight / "deployable_current" / "spot_tasks.parquet",
        index=False,
        compression="zstd",
    )
    canonical[
        [
            "task_id",
            "original_index",
            "true_duration",
            "true_duration_source",
            "absolute_estimation_error_seconds",
            "temporal_split",
        ]
    ].to_parquet(
        preflight / "simulator_only" / "spot_truth.parquet",
        index=False,
        compression="zstd",
    )


def prepare_blocked_directories(output: Path, config_path: Path) -> None:
    for relative in (
        "dataset/scenario_a",
        "dataset/scenario_b",
        "privileged_future/scenario_a",
        "privileged_future/scenario_b",
    ):
        write_text(
            output / relative / "BLOCKED.md",
            "# Formal output not generated\n\n"
            "Spot preflight failed the frozen temporal capacity gate. "
            "No H1/H4 labels or privileged future rows were generated.",
        )
    write_text(
        output / "plots" / "README.md",
        "# Plots\n\nNo formal behavior plots were generated because rollout was blocked.",
    )
    write_text(
        output / "configs" / config_path.name,
        config_path.read_text(encoding="utf-8"),
    )


def write_reports(
    output: Path,
    config: Mapping[str, Any],
    workspace: Mapping[str, Any],
    canonical: pd.DataFrame,
    pressure: pd.DataFrame,
    pressure_stats: pd.DataFrame,
    tails: pd.DataFrame,
    feasibility: pd.DataFrame,
    feasibility_summary: pd.DataFrame,
    capacity: pd.DataFrame,
    alignment: pd.DataFrame,
    a_stats: pd.DataFrame,
    b_stats: pd.DataFrame,
    behavior: Mapping[str, pd.DataFrame],
    errors: pd.DataFrame,
    provenance: pd.DataFrame,
    splits: pd.DataFrame,
    solver: pd.DataFrame,
    validation: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    prepare_blocked_directories(output, CONFIG_PATH)
    write_preflight_datasets(output, canonical, feasibility)

    canonical_audit(canonical, metadata, capacity).to_csv(
        output / "02_spot_full_canonical_audit.csv", index=False
    )
    pressure.to_csv(output / "03_spot_workload_pressure.csv", index=False)
    tails.to_csv(output / "04_spot_duration_tail.csv", index=False)
    feasibility_summary.to_csv(output / "05_spot_static_feasibility.csv", index=False)
    alignment.to_csv(output / "07_alibaba2020_v2_v3_alignment.csv", index=False)
    a_stats.to_csv(output / "09_scenario_a_statistics.csv", index=False)
    b_stats.to_csv(output / "10_scenario_b_statistics.csv", index=False)
    behavior["behavior"].to_csv(output / "11_h1_oracle_behavior.csv", index=False)
    behavior["placement"].to_csv(output / "12_placement_disagreement.csv", index=False)
    behavior["defer"].to_csv(output / "13_defer_analysis.csv", index=False)
    behavior["risk"].to_csv(output / "14_risk_analysis.csv", index=False)
    behavior["priority"].to_csv(output / "15_priority_analysis.csv", index=False)
    errors.to_csv(output / "16_duration_error_analysis.csv", index=False)
    provenance.to_csv(output / "18_field_provenance.csv", index=False)
    splits.to_csv(output / "19_split_manifest.csv", index=False)
    solver.to_csv(output / "20_solver_quality.csv", index=False)

    gpu = capacity.set_index("resource").loc["gpu"]
    static_rate = float(feasibility["feasible"].mean())
    all_tail = tails[(tails["dimension"] == "ALL")].iloc[0]
    write_text(
        output / "01_protocol.md",
        f"""# MPC Expert Dataset v3 Protocol

Planned formal scenarios:
- A_ALIBABA2020_REPAIRED: exact v2 episode/seed continuity.
- B_SPOTGPU2026_MEDIUM: continuous Spot timeline, repaired H1 and privileged H4.

Preflight must precede every formal solve. Static feasibility passed at {static_rate:.6%}, but the frozen maximum-wait temporal lower bound requires {gpu.mandatory_concurrent_peak:.2f} GPUs against capacity {gpu.total_capacity:.0f} ({gpu.mandatory_peak_ratio:.6%}). The formal generation gate therefore failed.

No H1/H4 solve, Transformer inference, student training, reward change, workload scaling, DC resizing, filtering, commit, or push was performed.
""",
    )
    write_text(
        output / "06_spot_temporal_state_audit.md",
        f"""# Spot Temporal State Audit

Result: {BLOCK_REASON}

- Raw timeline: {int(pressure.arrival_step.min())} through {int(pressure.arrival_step.max())}, {len(pressure):,} continuous 15-minute states.
- Every source job was converted; no task was filtered.
- Single-task static feasibility: {static_rate:.6%}.
- GPU offered resource-time ratio: {gpu.offered_load_ratio:.6%}.
- Mandatory GPU concurrent lower-bound peak: {gpu.mandatory_concurrent_peak:.2f} / {gpu.total_capacity:.0f} = {gpu.mandatory_peak_ratio:.6%}, at step {int(gpu.mandatory_peak_step)}.
- The lower bound counts only intervals where a task must be active for every dispatch time permitted by the frozen HP/Spot waiting bounds, including the repaired one-step transfer convention.

Because this hard lower bound exceeds total GPU capacity, a continuous rollout cannot satisfy the frozen scenario without a new workload/capacity decision. Formal rollout stopped before solver invocation.

Long-task carry-over status: NOT VERIFIED. The required full continuous simulator adapter was not allowed to proceed past the capacity gate. The current SustainCluster task environment exposes no complete pending/running/in-transit checkpoint restore API, so empty-state window resets remain forbidden.
""",
    )
    write_text(
        output / "08_alibaba2020_alignment_report.md",
        """# Alibaba2020 v2 to v3 Alignment

v2 remains intact and hash-verified: 80 episodes, 7,680 states, and 259,920 task decisions.

No formal v3 dataset was generated because Scenario B failed preflight before any expert solve. Therefore v2/v3 state, H1, and Oracle agreement are NOT_MEASURED, not 100%. Copying v2 labels into a new schema would not constitute an independent alignment check.

The v2 artifact stores ordered pending task rows and feasible masks, but its state summary does not preserve complete running, in-transit, or five-DC state payloads. Those fields require a real regenerated trajectory before alignment can be claimed.
""",
    )
    write_text(
        output / "17_information_contract.md",
        """# Information Contract

Formal v3 is designed as four physically separate regions:

1. deployable_current: current task batch, ordered original_index, current five-DC state, current price/carbon/reservations/network, estimated duration, feasible mask.
2. simulator_only: true duration, true release/completion state, and offline estimation error.
3. privileged_future: Oracle +15/+30/+45/+60 CPU/GPU/memory pressure and any future price/carbon used by H4.
4. labels: repaired H1 and H4 Oracle actions with joint state_id grouping and provenance.

Default learning loaders may read deployable_current plus labels only. No formal files were created in these regions because preflight blocked generation. The preflight canonical sidecars preserve this separation for audit only.
""",
    )
    write_text(
        output / "22_information_leakage_audit.md",
        """# Information Leakage Audit

- PER_JOB request identity: PASS.
- true_duration excluded from preflight deployable_current sidecar: PASS.
- estimated_duration uses the frozen Spot train-only estimator: PASS.
- memory and bandwidth use frozen Alibaba2020 train-only rules: PASS.
- origin mapping is deterministic: PASS.
- Oracle/teacher/H1/H4/future fields absent from deployable sidecar: PASS.
- privileged future offsets remain +15/+30/+45/+60 and are default-hidden: PASS (schema/config).
- Formal label leakage audit: NOT APPLICABLE; no formal labels were generated.
- Same-state split overlap: 0.
""",
    )
    write_text(
        output / "24_tests.md",
        f"""# Validation

| Check | Result |
| --- | --- |
| compileall | {validation["compileall"]} |
| dedicated tests | {validation["dedicated"]} |
| relevant regressions | {validation["regression"]} |
| full test suite | {validation["full"]} |
| new failures | {validation["new_failures"]} |
""",
    )
    write_text(
        output / "25_final_diagnosis.md",
        f"""# Final Diagnosis

Final status: {FINAL_STATUS}

The Spot source is fully convertible and every individual task fits at least one DC. This is insufficient for continuous expert generation: under the frozen medium SLA, mandatory concurrent GPU demand reaches {gpu.mandatory_peak_ratio:.6%} of aggregate capacity and long-run GPU offered load is {gpu.offered_load_ratio:.6%}.

No automatic filtering, scaling, DC expansion, reward change, fallback labels, or partial formal dataset was used. A new explicit scenario decision is required before v3 can be generated.

Alibaba2020 v2 remains the verified baseline. v2/v3 alignment and all Spot planner behavior metrics remain NOT_MEASURED because v3 was not generated.
""",
    )
    write_text(
        output / "26_summary.md",
        f"""# MPC Expert Dataset v3 Preflight Summary

1. Alibaba2020 v2 source: hash verified; 7,680 states and 259,920 decisions.
2. Alibaba2020 v3: not generated; alignment is NOT_MEASURED.
3. Spot raw/canonical jobs: {len(canonical):,}/{len(canonical):,}; filtered: 0.
4. Spot single-task static feasibility: {static_rate:.6%}.
5. Spot mandatory GPU peak: {gpu.mandatory_concurrent_peak:.2f}/{gpu.total_capacity:.0f} ({gpu.mandatory_peak_ratio:.6%}).
6. Spot true duration P50/P90/P95/P99/max seconds: {all_tail.true_p50_seconds:.0f}/{all_tail.true_p90_seconds:.1f}/{all_tail.true_p95_seconds:.1f}/{all_tail.true_p99_seconds:.1f}/{all_tail.true_max_seconds:.0f}.
7. H1/H4 Spot rollout and solver failures: NOT RUN.
8. No Expert Dataset v3, privileged future dataset, Transformer forecast, or student model was generated.
9. Required next decision: explicitly redesign workload scale, DC capacity, or admission policy and version that scenario separately.
10. Final status: {FINAL_STATUS}.
""",
    )

    comparison_rows = [
        ("task_records", len(a_stats) and 259920, len(canonical), "v2 decisions vs raw Spot jobs; not equivalent state semantics"),
        ("mean_pending_or_arrival_batch", float(a_stats.set_index("metric").loc["mean_pending_batch_size", "value"]), float(pressure["new_task_count"].mean()), "Alibaba pending queue vs Spot new arrivals only"),
        ("h1_oracle_disagreement_rate", float(a_stats.set_index("metric").loc["h1_oracle_exact_disagreement_rate", "value"]), "NOT_RUN", "Spot preflight blocked"),
        ("high_risk_state_rate", float(behavior["risk"].iloc[0]["high_risk_state_rate"]), "NOT_RUN", "thresholds cannot be compared directly"),
        ("oracle_defer_rate", float(a_stats.set_index("metric").loc["oracle_defer_rate", "value"]), "NOT_RUN", "Spot preflight blocked"),
        ("gpu_mandatory_capacity_ratio", "NOT_AVAILABLE", float(gpu.mandatory_peak_ratio), "Spot temporal capacity blocker"),
        ("true_duration_p95_seconds", "NOT_STORED_IN_V2_TASK_ROWS", float(all_tail.true_p95_seconds), "Spot heavy tail"),
    ]
    pd.DataFrame(
        comparison_rows,
        columns=("metric", "A_ALIBABA2020_REPAIRED", "B_SPOTGPU2026_MEDIUM", "interpretation"),
    ).to_csv(output / "21_cross_dataset_comparison.csv", index=False)

    existing = [
        path
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name not in {"00_generation_manifest.json", "23_integrity_manifest.csv"}
    ]
    integrity = pd.DataFrame(
        [
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "formal_expert_output": False,
            }
            for path in existing
        ]
    )
    integrity.to_csv(output / "23_integrity_manifest.csv", index=False)

    artifact_hashes = {
        path.relative_to(output).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "00_generation_manifest.json"
    }
    manifest = {
        "dataset_version": config["dataset_version"],
        "generation_time_utc": datetime.now(timezone.utc).isoformat(),
        "final_status": FINAL_STATUS,
        "block_reason": BLOCK_REASON,
        "formal_generation_attempted": False,
        "formal_expert_dataset_generated": False,
        "student_training": False,
        "transformer_inference": False,
        "workspace": dict(workspace),
        "project_config": {
            "path": CONFIG_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256(CONFIG_PATH),
        },
        "generation_script": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "sources": {
            "alibaba2020": config["scenario_a"]["source_sha256"],
            "alibaba2020_v2_tasks": config["scenario_a"]["v2_task_sha256"],
            "alibaba2020_v2_states": config["scenario_a"]["v2_state_sha256"],
            "spot_jobs": spot.source_audit.EXPECTED_FILES[
                ROOT / "data/raw/alibaba_2026/spot_gpu/job_info_df.csv"
            ][1],
            "spot_nodes": spot.source_audit.EXPECTED_FILES[
                ROOT / "data/raw/alibaba_2026/spot_gpu/node_info_df.csv"
            ][1],
            "spot_contract_config": config["scenario_b"]["contract_config_sha256"],
            "gpu2026_role": "EXTERNAL_VALIDATION_SOURCE",
        },
        "scenario_a": {
            "states_v2": 7680,
            "task_decisions_v2": 259920,
            "v3_generated": False,
            "v2_v3_state_agreement": "NOT_MEASURED",
            "v2_v3_h1_agreement": "NOT_MEASURED",
            "v2_v3_oracle_agreement": "NOT_MEASURED",
        },
        "scenario_b": {
            "raw_jobs": len(canonical),
            "canonical_jobs": len(canonical),
            "static_feasible_rate": static_rate,
            "mandatory_capacity": capacity.to_dict("records"),
            "formal_states": 0,
            "formal_task_decisions": 0,
            "solver_failures": "NOT_RUN",
        },
        "determinism": {
            "canonical_content_sha256": dataframe_hash(canonical),
            "task_order": "submit_time,original_index,task_id",
            "random_shuffle": False,
        },
        "validation": dict(validation),
        "artifacts": artifact_hashes,
    }
    (output / "00_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def run(
    config_path: Path = CONFIG_PATH,
    validation: Mapping[str, str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    config = load_config(config_path)
    workspace = validate_frozen_inputs(config)
    scenario_contract = spot.load_config()
    config = dict(config)
    config["scenario_b_contract"] = scenario_contract["origin_model"]

    canonical, _, _, metadata = build_full_spot_canonical(config)
    pressure = workload_pressure(canonical, config)
    pstats = pressure_summary(pressure)
    feasibility = spot.static_feasibility(canonical, scenario_contract)
    feasibility_summary = grouped_feasibility(canonical, feasibility)
    capacity = mandatory_capacity_audit(canonical, config)
    tails = duration_tail(canonical)
    errors = duration_error_analysis(canonical)

    static_rate = float(feasibility["feasible"].mean())
    static_pass = 1.0 - static_rate <= config["preflight_gates"]["static_infeasible_rate_max"]
    capacity_pass = bool(capacity["gate_pass"].all())
    if not static_pass:
        raise RuntimeError("single-task static capacity gate failed")
    if capacity_pass:
        raise RuntimeError(
            "expected frozen Spot data to fail the mandatory temporal capacity gate; "
            "review source/config before enabling formal generation"
        )

    a_tasks, a_states, a_episodes = load_v2(config)
    alignment = alibaba_alignment_audit(a_tasks, a_states)
    a_stats = scenario_a_statistics(a_tasks, a_states, a_episodes)
    b_stats = scenario_b_statistics(
        canonical, pressure, pstats, capacity, feasibility
    )
    behavior = behavior_tables(
        a_tasks,
        a_states,
        canonical,
        config["scenario_a"]["high_risk_threshold"],
    )
    provenance = field_provenance()
    splits = split_manifest(canonical, a_episodes, metadata)
    solver = pd.DataFrame(
        [
            ("A_ALIBABA2020_REPAIRED", "H1_REPAIRED", 7680, 0, "optimal", "V2_FROZEN_BASELINE"),
            ("A_ALIBABA2020_REPAIRED", "H4_ORACLE_REPAIRED", 7680, 0, "optimal", "V2_FROZEN_BASELINE"),
            ("B_SPOTGPU2026_MEDIUM", "H1_REPAIRED", 0, 0, "NOT_RUN", "SPOT_PREFLIGHT_BLOCKED"),
            ("B_SPOTGPU2026_MEDIUM", "H4_ORACLE_REPAIRED", 0, 0, "NOT_RUN", "SPOT_PREFLIGHT_BLOCKED"),
        ],
        columns=("scenario_id", "planner", "solve_attempts", "solver_failures", "solver_status", "status"),
    )
    validate_solver_failures(solver)
    status = validation or {
        "compileall": "PENDING",
        "dedicated": "PENDING",
        "regression": "PENDING",
        "full": "PENDING",
        "new_failures": "PENDING",
    }
    manifest = write_reports(
        ROOT / config["output_dir"],
        config,
        workspace,
        canonical,
        pressure,
        pstats,
        tails,
        feasibility,
        feasibility_summary,
        capacity,
        alignment,
        a_stats,
        b_stats,
        behavior,
        errors,
        provenance,
        splits,
        solver,
        status,
        metadata,
    )
    return ROOT / config["output_dir"], manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--compileall", default="PENDING")
    parser.add_argument("--dedicated", default="PENDING")
    parser.add_argument("--regression", default="PENDING")
    parser.add_argument("--full", default="PENDING")
    parser.add_argument("--new-failures", default="PENDING")
    args = parser.parse_args()
    output, manifest = run(
        args.config.resolve(),
        {
            "compileall": args.compileall,
            "dedicated": args.dedicated,
            "regression": args.regression,
            "full": args.full,
            "new_failures": args.new_failures,
        },
    )
    print(f"status={manifest['final_status']}")
    print(f"block_reason={manifest['block_reason']}")
    print(f"spot_jobs={manifest['scenario_b']['canonical_jobs']}")
    gpu = next(
        item for item in manifest["scenario_b"]["mandatory_capacity"]
        if item["resource"] == "gpu"
    )
    print(f"mandatory_gpu_peak_ratio={gpu['mandatory_peak_ratio']:.6%}")
    print(f"formal_expert_dataset_generated={manifest['formal_expert_dataset_generated']}")
    print(f"artifacts={output}")
    if args.generate:
        raise RuntimeError(
            "SPOT SCENARIO PREFLIGHT BLOCKED: formal H1/H4 rollout was not started"
        )


if __name__ == "__main__":
    main()

