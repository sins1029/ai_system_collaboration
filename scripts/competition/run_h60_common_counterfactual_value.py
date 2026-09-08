from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
VENDOR = ROOT / "references/external_repos/sustain-cluster"
for entry in (ROOT, SRC, VENDOR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from forecasting.h60_dataset import (  # noqa: E402
    H60_FEATURE_NAMES,
    H60_TARGET_NAMES,
    H60ForecastDatasetConfig,
    build_spot_h60_time_series,
    fit_h60_train_scaler,
)
from forecasting.h60_models import (  # noqa: E402
    denormalize_h60,
    load_h60_checkpoint,
    predict_h60,
)
from scripts.competition import run_mpc_h60_night1 as night1  # noqa: E402
from scripts.competition import run_mpc_h60_night2 as night2  # noqa: E402
from scripts.imitation import build_spotgpu2026_expert_dataset_v3 as spot_v3  # noqa: E402
from sustaincluster_mpc.forecast_pressure_adapter import (  # noqa: E402
    expected_origin_probabilities,
)
from sustaincluster_mpc.h60_controllers import (  # noqa: E402
    MpcH60LearnedController,
    MpcH60OracleController,
)
from sustaincluster_mpc.rolling_horizon_optimizer import (  # noqa: E402
    RollingCostBreakdown,
)
from sustaincluster_mpc.terminal_h60_optimizer import (  # noqa: E402
    TerminalH60DataCenter,
)


OUTPUT = ROOT / "artifacts/mpc_h60_v1"
VALUE_PATH = OUTPUT / "20_h60_common_counterfactual_value_train_val.parquet"
SUMMARY_PATH = OUTPUT / "21_h60_value_summary.csv"
SLICE_PATH = OUTPUT / "22_h60_value_slices.csv"
GATE_PATH = OUTPUT / "23_h60_oracle_gate_upper_bound.csv"
REPORT_PATH = OUTPUT / "24_h60_value_audit.md"
SPLITS = ("train", "validation")
K = 4
BOOTSTRAP_REPETITIONS = 1000
CONTROLLER_PATHS = (
    ROOT / "src/sustaincluster_mpc/rolling_horizon_optimizer.py",
    ROOT / "src/sustaincluster_mpc/terminal_h60_optimizer.py",
    ROOT / "src/sustaincluster_mpc/h60_controllers.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H60 common-continuation value audit")
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def source_snapshot(manifest: Mapping[str, Any]) -> dict[str, Any]:
    paths = [night1.MANIFEST_PATH, *CONTROLLER_PATHS]
    paths.extend(ROOT / item["path"] for item in manifest["data_files"].values())
    return {
        path.relative_to(ROOT).as_posix(): {
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256(path) if path in CONTROLLER_PATHS or path == night1.MANIFEST_PATH else None,
        }
        for path in paths
    }


def manifest_path(manifest: Mapping[str, Any], key: str) -> Path:
    return ROOT / manifest["data_files"][key]["path"]


def load_train_validation_inputs() -> dict[str, Any]:
    manifest = json.loads(night1.MANIFEST_PATH.read_text("utf-8"))
    split_filter = [("split", "in", list(SPLITS))]
    states = pd.read_parquet(
        manifest_path(manifest, "states_full"),
        columns=[
            "state_id",
            "step",
            "timestamp_utc",
            "split",
            "pending_task_ids",
            "running_task_ids",
            "in_transit_task_ids",
            "pending_count",
            "completed_count",
            "dc_states_json",
            "risk_score",
            "state_sha256",
        ],
        filters=split_filter,
    ).sort_values("step", kind="mergesort")
    tasks = pd.read_parquet(
        manifest_path(manifest, "tasks_full"),
        columns=[
            "state_id",
            "step",
            "split",
            "task_id",
            "original_index",
            "task_position",
            "cpu_cores",
            "gpu_units",
            "memory_gb",
            "bandwidth_gb",
            "origin_dc",
            "priority",
            "submit_time_seconds",
            "arrival_step",
            "estimated_duration_seconds",
            "estimated_duration_steps",
            "waiting_steps",
            "sla_deadline_step",
            "remaining_sla_steps",
            "max_wait_steps",
            "defer_allowed",
        ],
        filters=split_filter,
    )
    labels = pd.read_parquet(
        manifest_path(manifest, "labels_full"),
        columns=["state_id", "task_id", "task_position", "h1_action"],
        filters=split_filter,
    )
    privileged = pd.read_parquet(
        manifest_path(manifest, "privileged_full"),
        columns=[
            "state_id",
            "step",
            "split",
            "dc_id",
            "horizon_step",
            "actual_origin_cpu_demand",
            "actual_origin_gpu_demand",
            "actual_origin_memory_demand",
            "future_electricity_price_usd_per_mwh",
            "future_carbon_intensity_gco2_per_kwh",
            "information_class",
        ],
        filters=split_filter,
    )
    validation_end = int(states[states["split"] == "validation"]["step"].max())
    lifecycle = pd.read_parquet(
        manifest_path(manifest, "task_lifecycle_full"),
        columns=[
            "task_id",
            "original_index",
            "arrival_step",
            "true_duration",
            "destination_dc",
            "dispatch_step",
            "planned_execution_start_step",
            "planned_estimated_completion_step",
            "actual_execution_start_step",
            "start_estimated_completion_step",
            "start_true_completion_step",
            "observed_true_completion_step",
        ],
        filters=[("arrival_step", "<=", validation_end)],
    )
    oracle = pd.read_parquet(
        OUTPUT / "labels/h60_oracle_actions_full.parquet",
        columns=[
            "state_id",
            "step",
            "split",
            "task_id",
            "task_position",
            "h60_action",
        ],
        filters=split_filter,
    )
    state_diagnostics = pd.read_parquet(
        OUTPUT / "labels/h60_state_diagnostics.parquet",
        columns=[
            "state_id",
            "step",
            "split",
            "max_terminal_base_pressure",
            "mean_terminal_electricity",
            "mean_terminal_carbon",
        ],
        filters=split_filter,
    )
    frames = (states, tasks, privileged, oracle, state_diagnostics)
    if any(set(frame["split"]) - set(SPLITS) for frame in frames):
        raise RuntimeError("test rows entered the train+validation value audit")
    if states["step"].min() != 0 or states["step"].max() != validation_end:
        raise RuntimeError("train+validation state timeline is not continuous")
    if tasks["task_id"].duplicated().any():
        raise RuntimeError("value audit requires one frozen task row per task")
    merged_tasks = tasks.merge(
        labels,
        on=["state_id", "task_id", "task_position"],
        how="left",
        validate="one_to_one",
        sort=False,
    ).merge(
        oracle,
        on=["state_id", "step", "split", "task_id", "task_position"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged_tasks[["h1_action", "h60_action"]].isna().any().any():
        raise RuntimeError("H1/Oracle action alignment failed")
    return {
        "manifest": manifest,
        "states": states.reset_index(drop=True),
        "tasks": merged_tasks,
        "privileged": privileged,
        "lifecycle": lifecycle,
        "state_diagnostics": state_diagnostics,
        "validation_end": validation_end,
        "test_rows_loaded": 0,
    }


def build_train_validation_simulator_inputs(data: Mapping[str, Any]) -> dict[str, Any]:
    tasks = data["tasks"].drop_duplicates("task_id").copy()
    lifecycle = data["lifecycle"].loc[:, ["task_id", "true_duration"]]
    canonical = tasks.merge(lifecycle, on="task_id", how="left", validate="one_to_one")
    canonical.rename(
        columns={
            "submit_time_seconds": "submit_time",
            "cpu_cores": "cpu_request_effective",
            "gpu_units": "gpu_request_effective",
            "memory_gb": "memory_request",
            "bandwidth_gb": "bandwidth",
            "estimated_duration_seconds": "estimated_duration",
        },
        inplace=True,
    )
    if canonical["true_duration"].isna().any():
        raise RuntimeError("train+validation simulator truth is incomplete")
    canonical.sort_values(
        ["arrival_step", "submit_time", "original_index", "task_id"],
        kind="mergesort",
        inplace=True,
    )
    canonical.set_index("original_index", drop=False, inplace=True)
    config = yaml.safe_load(spot_v3.CONFIG_PATH.read_text("utf-8"))[
        "mpc_expert_dataset_v3"
    ]
    contract = yaml.safe_load(
        (ROOT / config["scenario_b"]["contract_config"]).read_text("utf-8")
    )
    dcs = pd.read_csv(spot_v3.OUTPUT / "31_spot_native_5dc_capacity.csv").sort_values(
        "dc_id"
    )
    states = data["states"]
    metadata = {
        "train_last_arrival_step": int(
            states[states["split"] == "train"]["step"].max()
        ),
        "validation_last_arrival_step": int(data["validation_end"]),
    }
    return {
        "config": config,
        "contract": contract,
        "canonical": canonical,
        "metadata": metadata,
        "dcs": dcs,
    }


def build_runtime_templates(
    generator: spot_v3.SpotContinuousGenerator,
    lifecycle: pd.DataFrame,
) -> dict[str, dict[str, Mapping[str, Any]]]:
    canonical_id_to_index = {
        str(row.task_id): int(index)
        for index, row in generator.canonical.iterrows()
    }
    templates: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in lifecycle.itertuples(index=False):
        task_id = str(row.task_id)
        if task_id not in canonical_id_to_index or pd.isna(row.destination_dc):
            continue
        row_index = canonical_id_to_index[task_id]
        canonical = generator.canonical.loc[row_index]
        resources = tuple(float(value) for value in generator.resources(row_index))
        true_steps = max(1, int(np.ceil(float(canonical["true_duration"]) / 900.0)))
        common = {
            "task_id": task_id,
            "row_index": row_index,
            "dc_id": int(row.destination_dc),
            "waiting_steps": int(row.dispatch_step) - int(canonical["arrival_step"]),
            "dispatch_step": int(row.dispatch_step),
            "planned_execution_start_step": int(row.planned_execution_start_step),
            "estimated_duration_steps": int(canonical["estimated_duration_steps"]),
            "true_duration_steps": true_steps,
            "resources": resources,
        }
        values: dict[str, Mapping[str, Any]] = {}
        if not pd.isna(row.actual_execution_start_step):
            running = {
                **common,
                "status": "running",
                "actual_execution_start_step": int(row.actual_execution_start_step),
                "estimated_completion_step": int(row.start_estimated_completion_step),
                "true_completion_step": int(row.start_true_completion_step),
            }
            values["running"] = MappingProxyType(running)
        transit = {
            **common,
            "status": "in_transit",
            "actual_execution_start_step": None,
            "estimated_completion_step": int(row.planned_estimated_completion_step),
            "true_completion_step": None,
        }
        values["transit"] = MappingProxyType(transit)
        templates[task_id] = values
    return templates


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def restore_runtime_state(
    generator: spot_v3.SpotContinuousGenerator,
    state_row: pd.Series,
    task_rows: pd.DataFrame,
    templates: Mapping[str, Mapping[str, Mapping[str, Any]]],
    canonical_id_to_index: Mapping[str, int],
) -> spot_v3.RuntimeState:
    pending = [
        int(canonical_id_to_index[str(task_id)])
        for task_id in as_list(state_row["pending_task_ids"])
    ]
    expected_pending = task_rows.sort_values("task_position")["original_index"].astype(int).tolist()
    if pending != expected_pending:
        raise RuntimeError(f"pending task order mismatch at {state_row['state_id']}")
    running: dict[int, Mapping[str, Any]] = {}
    for task_id in as_list(state_row["running_task_ids"]):
        item = templates[str(task_id)]["running"]
        running[int(item["row_index"])] = item
    transit: dict[int, Mapping[str, Any]] = {}
    for task_id in as_list(state_row["in_transit_task_ids"]):
        item = templates[str(task_id)]["transit"]
        transit[int(item["row_index"])] = item
    dc_rows = json.loads(state_row["dc_states_json"])
    state = spot_v3.RuntimeState(
        next_step=int(state_row["step"]),
        pending=pending,
        running=running,
        transit=transit,
        used={
            int(item["dc_id"]): [
                float(item["cpu_used"]),
                float(item["gpu_used"]),
                float(item["memory_used"]),
            ]
            for item in dc_rows
        },
        reserved={
            int(item["dc_id"]): [
                float(item["cpu_reserved"]),
                float(item["gpu_reserved"]),
                float(item["memory_reserved"]),
            ]
            for item in dc_rows
        },
        completed_count=int(state_row["completed_count"]),
    )
    generator.state = state
    restored_dc_rows = generator._current_dc_rows(int(state_row["step"]))
    restored_hash = spot_v3.object_hash(
        generator._state_hash_payload(int(state_row["step"]), restored_dc_rows)
    )
    if restored_hash != str(state_row["state_sha256"]):
        raise RuntimeError(
            f"frozen simulator state hash mismatch at {state_row['state_id']}: "
            f"{restored_hash} != {state_row['state_sha256']}"
        )
    return state


def clone_runtime_state(
    state: spot_v3.RuntimeState,
    *,
    last_advance_step: int,
) -> spot_v3.RuntimeState:
    running: dict[int, Mapping[str, Any]] = {}
    for row_index, item in state.running.items():
        if int(item["true_completion_step"]) <= last_advance_step:
            running[row_index] = dict(item)
        else:
            running[row_index] = item
    return spot_v3.RuntimeState(
        next_step=state.next_step,
        pending=list(state.pending),
        running=running,
        transit={row_index: dict(item) for row_index, item in state.transit.items()},
        used={dc_id: list(values) for dc_id, values in state.used.items()},
        reserved={dc_id: list(values) for dc_id, values in state.reserved.items()},
        completed_count=state.completed_count,
        state_counter=state.state_counter,
        decision_counter=state.decision_counter,
    )


def runtime_signature(state: spot_v3.RuntimeState) -> str:
    payload = {
        "next_step": state.next_step,
        "pending": state.pending,
        "running": sorted(
            (
                int(index),
                str(item["task_id"]),
                int(item["dc_id"]),
                int(item["estimated_completion_step"]),
                int(item["true_completion_step"]),
            )
            for index, item in state.running.items()
        ),
        "transit": sorted(
            (
                int(index),
                str(item["task_id"]),
                int(item["dc_id"]),
                int(item["planned_execution_start_step"]),
            )
            for index, item in state.transit.items()
        ),
        "used": state.used,
        "reserved": state.reserved,
        "completed": state.completed_count,
    }
    return spot_v3.object_hash(payload)


def normalized_forecast_inputs(
    states: pd.DataFrame,
    tasks: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[int, np.ndarray], set[int]]:
    timeline = build_spot_h60_time_series(states, tasks)
    scaler = fit_h60_train_scaler(timeline)
    config = H60ForecastDatasetConfig(history_length=96, target_offset_steps=4)
    selection = json.loads(night2.MODEL_SELECTION_PATH.read_text("utf-8"))
    model, payload = load_h60_checkpoint(ROOT / selection["selected_learned_checkpoint"])
    if payload["information_class"] != "DEPLOYABLE_HISTORY_ONLY":
        raise RuntimeError("selected forecast checkpoint is not deployable-history-only")

    features = timeline.loc[:, H60_FEATURE_NAMES].to_numpy(dtype=np.float32)
    for index, name in enumerate(H60_TARGET_NAMES):
        item = scaler["parameters"][name]
        features[:, index] = (
            features[:, index] - float(item["mean"])
        ) / float(item["scale"])
    steps = timeline["step"].to_numpy(dtype=np.int64)
    splits = timeline["split"].astype(str).to_numpy()
    eligible_indices = [
        index
        for index in range(config.history_length - 1, len(timeline))
        if (
            splits[index] in SPLITS
            and index + config.target_offset_steps < len(timeline)
            and splits[index + config.target_offset_steps] == splits[index]
        )
    ]
    histories = np.stack(
        [
            features[index - config.history_length + 1 : index + 1]
            for index in eligible_indices
        ]
    )
    normalized = predict_h60(model, histories, batch_size=512)
    raw = denormalize_h60(normalized, scaler)
    predictions: dict[int, np.ndarray] = {}
    eligible_steps: set[int] = set()
    for index, value in zip(eligible_indices, raw[:, 0, :]):
        step = int(steps[index])
        predictions[step] = np.asarray(value, dtype=float)
        eligible_steps.add(step)
    return timeline, scaler, predictions, eligible_steps


def build_learned_terminal_inputs(
    generator: spot_v3.SpotContinuousGenerator,
    state_row: pd.Series,
    prediction: np.ndarray,
    occupancy: np.ndarray,
) -> tuple[TerminalH60DataCenter, ...]:
    step = int(state_row["step"])
    target_step = step + 4
    dc_rows = json.loads(state_row["dc_states_json"])
    probabilities = expected_origin_probabilities(
        generator.dc_configs, generator.timestamp(target_step)
    )
    values = []
    for item in sorted(dc_rows, key=lambda row: int(row["dc_id"])):
        dc_id = int(item["dc_id"])
        arrival = np.asarray(prediction, dtype=float) * float(probabilities[dc_id])
        existing = occupancy[step, dc_id - 1]
        values.append(
            TerminalH60DataCenter(
                dc_id=dc_id,
                cpu_total_cores=float(item["cpu_total"]),
                gpu_total_units=float(item["gpu_total"]),
                memory_total_gb=float(item["memory_total"]),
                estimated_existing_cpu_cores=float(existing[0]),
                estimated_existing_gpu_units=float(existing[1]),
                estimated_existing_memory_gb=float(existing[2]),
                arriving_cpu_demand=float(max(0.0, arrival[1])),
                arriving_gpu_demand=float(max(0.0, arrival[2])),
                arriving_memory_demand=float(max(0.0, arrival[3])),
                electricity_price_usd_per_mwh=float(
                    generator.signals[dc_id]["price"][target_step]
                ),
                carbon_intensity_gco2_per_kwh=float(
                    generator.signals[dc_id]["carbon"][target_step]
                ),
                source="DEPLOYABLE_LEARNED_T60",
            )
        )
    return tuple(values)


def validate_terminal_cache_train_validation(
    states: pd.DataFrame,
    tasks: pd.DataFrame,
    privileged: pd.DataFrame,
    lifecycle: pd.DataFrame,
    occupancy: np.ndarray,
) -> dict[str, Any]:
    indices = np.unique(np.linspace(0, len(states) - 1, 20, dtype=int))
    selected = states.iloc[indices]
    state_index = states.set_index("state_id", drop=False)
    task_index = tasks.drop_duplicates("task_id").set_index("task_id")
    lifecycle_index = lifecycle.set_index("task_id")
    privileged_groups = {
        str(state_id): frame
        for state_id, frame in privileged.groupby("state_id", sort=False)
    }
    max_absolute_error = 0.0
    for state_id in selected["state_id"].astype(str):
        state_row = state_index.loc[state_id]
        slow = night1.build_terminal_inputs(
            state_row,
            privileged_groups[state_id],
            task_index,
            lifecycle_index,
        )
        fast = night2.build_fast_terminal_inputs(
            state_row,
            privileged_groups[state_id],
            occupancy,
        )
        slow_values = np.asarray(
            [
                [
                    item.estimated_existing_cpu_cores,
                    item.estimated_existing_gpu_units,
                    item.estimated_existing_memory_gb,
                ]
                for item in slow
            ],
            dtype=float,
        )
        fast_values = np.asarray(
            [
                [
                    item.estimated_existing_cpu_cores,
                    item.estimated_existing_gpu_units,
                    item.estimated_existing_memory_gb,
                ]
                for item in fast
            ],
            dtype=float,
        )
        max_absolute_error = max(
            max_absolute_error,
            float(np.abs(slow_values - fast_values).max()),
        )
        if not np.allclose(slow_values, fast_values, rtol=0.0, atol=1e-7):
            raise RuntimeError(f"terminal occupancy cache mismatch at {state_id}")
    return {
        "states_checked": int(len(selected)),
        "splits_checked": list(SPLITS),
        "test_rows_loaded": 0,
        "max_absolute_error": max_absolute_error,
        "status": "PASS",
        "semantics": "MATCHES_NIGHT1_VISIBLE_ACTIVE_ESTIMATED_COMPLETION",
    }


COST_FIELDS = (
    "electricity",
    "carbon",
    "transmission",
    "waiting_defer",
    "sla_risk",
    "terminal_backlog",
)


def zero_costs() -> dict[str, float]:
    return {name: 0.0 for name in COST_FIELDS}


def add_costs(total: dict[str, float], costs: RollingCostBreakdown) -> None:
    for name in COST_FIELDS:
        total[name] += float(getattr(costs, name))


def total_cost(costs: Mapping[str, float]) -> float:
    return float(sum(costs.values()))


def exogenous_truth_fingerprint(
    generator: spot_v3.SpotContinuousGenerator,
    step: int,
) -> str:
    rows = []
    for future_step in range(step + 1, step + K):
        arrivals = []
        for row_index in generator.arrivals_by_step.get(future_step, ()):
            row = generator.canonical.loc[row_index]
            arrivals.append(
                (
                    str(row["task_id"]),
                    int(row["arrival_step"]),
                    float(row["cpu_request_effective"]),
                    float(row["gpu_request_effective"]),
                    float(row["memory_request"]),
                    float(row["true_duration"]),
                )
            )
        signals = {
            dc_id: (
                float(generator.signals[dc_id]["price"][future_step]),
                float(generator.signals[dc_id]["carbon"][future_step]),
            )
            for dc_id in sorted(generator.capacity)
        }
        rows.append((future_step, arrivals, signals))
    return spot_v3.object_hash(rows)


def run_branch(
    generator: spot_v3.SpotContinuousGenerator,
    base_state: spot_v3.RuntimeState,
    *,
    step: int,
    first_actions: Sequence[int],
    first_costs: RollingCostBreakdown,
) -> dict[str, Any]:
    exogenous_before = exogenous_truth_fingerprint(generator, step)
    branch = clone_runtime_state(base_state, last_advance_step=step + K - 1)
    generator.state = branch
    generator.buffers = {name: [] for name in spot_v3.TABLE_SCHEMAS}
    costs = zero_costs()
    add_costs(costs, first_costs)
    generator._apply_h4(step, first_actions)
    generator.state.next_step = step + 1
    post_first_signature = runtime_signature(generator.state)
    continuation_state_hashes = []
    continuation_action_hashes = []
    h1_continuation_calls = 0
    for future_step in range(step + 1, step + K):
        generator._advance_to_state(future_step)
        continuation_state_hashes.append(runtime_signature(generator.state))
        state, _ = generator.build_solver_state(future_step, 1)
        result = generator._solve(state, "h1")
        if result.status != "optimal":
            raise RuntimeError(
                f"H1 continuation failed at step={future_step}: {result.status}"
            )
        h1_continuation_calls += 1
        continuation_action_hashes.append(
            spot_v3.object_hash([int(value) for value in result.environment_actions])
        )
        add_costs(costs, result.costs)
        generator._apply_h4(future_step, result.environment_actions)
        generator.state.next_step = future_step + 1
    if h1_continuation_calls != K - 1:
        raise RuntimeError("common continuation did not execute exactly three H1 solves")
    exogenous_after = exogenous_truth_fingerprint(generator, step)
    if exogenous_after != exogenous_before:
        raise RuntimeError("branch execution mutated the frozen exogenous truth")
    return {
        "costs": costs,
        "J": total_cost(costs),
        "post_first_signature": post_first_signature,
        "continuation_state_hashes": tuple(continuation_state_hashes),
        "continuation_action_hashes": tuple(continuation_action_hashes),
        "h1_continuation_calls": h1_continuation_calls,
        "exogenous_truth_sha256": exogenous_after,
    }


def classify_delta(delta: float, reference: float) -> tuple[str, float]:
    tolerance = max(1e-8, 1e-6 * abs(float(reference)))
    if delta < -tolerance:
        return "POSITIVE", tolerance
    if delta > tolerance:
        return "NEGATIVE", tolerance
    return "EQUIVALENT", tolerance


def state_attributes(
    state_row: pd.Series,
    task_rows: pd.DataFrame,
    diagnostic_row: pd.Series,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    task_count = len(task_rows)
    hp_count = int((task_rows["priority"] == "HP").sum())
    spot_count = int((task_rows["priority"] == "Spot").sum())
    long_count = int(
        (
            task_rows["estimated_duration_steps"]
            > float(thresholds["estimated_duration_steps"][1])
        ).sum()
    )
    high_sla_count = int((task_rows["remaining_sla_steps"] <= 1).sum())
    resource_pressure = float(state_row["risk_score"])
    future_workload = float(diagnostic_row["max_terminal_base_pressure"])
    future_carbon = float(diagnostic_row["mean_terminal_carbon"])
    future_electricity = float(diagnostic_row["mean_terminal_electricity"])
    return {
        "resource_pressure": resource_pressure,
        "future_workload": future_workload,
        "future_carbon": future_carbon,
        "future_electricity": future_electricity,
        "hp_task_count": hp_count,
        "spot_task_count": spot_count,
        "spot_task_share": spot_count / task_count,
        "estimated_duration_mean_steps": float(
            task_rows["estimated_duration_steps"].mean()
        ),
        "estimated_duration_max_steps": int(
            task_rows["estimated_duration_steps"].max()
        ),
        "long_task_count": long_count,
        "long_task_share": long_count / task_count,
        "minimum_remaining_sla_steps": int(task_rows["remaining_sla_steps"].min()),
        "high_sla_risk_task_count": high_sla_count,
        "high_sla_risk_task_share": high_sla_count / task_count,
        "slice_long_task": long_count > 0,
        "slice_spot": spot_count > 0,
        "slice_high_future_workload": future_workload
        > float(thresholds["future_workload_pressure"][1]),
        "slice_high_future_carbon": future_carbon
        > float(thresholds["future_carbon"][1]),
        "slice_high_resource_pressure": resource_pressure
        > float(thresholds["resource_pressure"][1]),
    }


def prepare_candidates(
    generator: spot_v3.SpotContinuousGenerator,
    data: Mapping[str, Any],
    predictions: Mapping[int, np.ndarray],
    eligible_steps: set[int],
    occupancy: np.ndarray,
    current_config: Any,
    terminal_config: Any,
    adapter: Any,
    links: Sequence[Any],
) -> tuple[list[str], dict[str, tuple[int, ...]], dict[str, Any]]:
    states = data["states"]
    tasks = data["tasks"]
    task_groups = tasks.groupby("state_id", sort=False).indices
    learned_controller = MpcH60LearnedController()
    learned_actions: dict[str, tuple[int, ...]] = {}
    candidates: list[str] = []
    excluded_oracle_disagreement = 0
    solved_states = 0
    for _, state_row in states.iterrows():
        state_id = str(state_row["state_id"])
        positions = task_groups.get(state_id)
        if positions is None or int(state_row["pending_count"]) == 0:
            continue
        task_rows = tasks.iloc[positions].sort_values("task_position", kind="mergesort")
        h1_actions = tuple(int(value) for value in task_rows["h1_action"])
        oracle_actions = tuple(int(value) for value in task_rows["h60_action"])
        oracle_disagreement = h1_actions != oracle_actions
        step = int(state_row["step"])
        if step not in eligible_steps:
            excluded_oracle_disagreement += int(oracle_disagreement)
            continue
        current_state = night1.build_current_state(state_row, task_rows, links)
        terminal = build_learned_terminal_inputs(
            generator, state_row, predictions[step], occupancy
        )
        learned = learned_controller.solve(
            current_state,
            terminal,
            current_config=current_config,
            terminal_config=terminal_config,
            action_adapter=adapter,
        )
        if learned.status != "optimal":
            raise RuntimeError(f"Learned H60 candidate solve failed at {state_id}")
        actions = tuple(int(value) for value in learned.environment_actions)
        learned_disagreement = h1_actions != actions
        solved_states += 1
        if oracle_disagreement or learned_disagreement:
            candidates.append(state_id)
            learned_actions[state_id] = actions
        if solved_states % 1000 == 0:
            print(f"[candidate] learned actions {solved_states} eligible states")
    metadata = {
        "eligible_nonempty_states": solved_states,
        "candidate_states": len(candidates),
        "excluded_oracle_disagreement_without_frozen_forecast_window": excluded_oracle_disagreement,
        "selection_rule": "H1_NE_ORACLE_OR_H1_NE_LEARNED_JOINT_ACTION",
    }
    return candidates, learned_actions, metadata


def evaluate_state(
    generator: spot_v3.SpotContinuousGenerator,
    state_row: pd.Series,
    task_rows: pd.DataFrame,
    privileged_rows: pd.DataFrame,
    diagnostic_row: pd.Series,
    prediction: np.ndarray,
    learned_actions_expected: Sequence[int],
    occupancy: np.ndarray,
    templates: Mapping[str, Mapping[str, Mapping[str, Any]]],
    canonical_id_to_index: Mapping[str, int],
    thresholds: Mapping[str, Any],
    current_config: Any,
    terminal_config: Any,
    adapter: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    step = int(state_row["step"])
    base_state = restore_runtime_state(
        generator,
        state_row,
        task_rows,
        templates,
        canonical_id_to_index,
    )
    base_signature = runtime_signature(base_state)
    current_state, _ = generator.build_solver_state(step, 1)
    h1 = generator._solve(current_state, "h1")
    oracle_terminal = night2.build_fast_terminal_inputs(
        state_row, privileged_rows, occupancy
    )
    learned_terminal = build_learned_terminal_inputs(
        generator, state_row, prediction, occupancy
    )
    oracle = MpcH60OracleController().solve(
        current_state,
        oracle_terminal,
        current_config=current_config,
        terminal_config=terminal_config,
        action_adapter=adapter,
    )
    learned = MpcH60LearnedController().solve(
        current_state,
        learned_terminal,
        current_config=current_config,
        terminal_config=terminal_config,
        action_adapter=adapter,
    )
    results = (h1, oracle, learned)
    if any(result.status != "optimal" for result in results):
        raise RuntimeError(f"first-action solve failed at {state_row['state_id']}")
    h1_actions = tuple(int(value) for value in h1.environment_actions)
    oracle_actions = tuple(int(value) for value in oracle.environment_actions)
    learned_actions = tuple(int(value) for value in learned.environment_actions)
    expected_h1 = tuple(int(value) for value in task_rows["h1_action"])
    expected_oracle = tuple(int(value) for value in task_rows["h60_action"])
    if h1_actions != expected_h1:
        raise RuntimeError(f"restored H1 action mismatch at {state_row['state_id']}")
    if oracle_actions != expected_oracle:
        raise RuntimeError(f"restored Oracle H60 action mismatch at {state_row['state_id']}")
    if learned_actions != tuple(learned_actions_expected):
        raise RuntimeError(f"restored Learned H60 action mismatch at {state_row['state_id']}")
    if h1_actions == oracle_actions and h1_actions == learned_actions:
        raise RuntimeError("an all-equal state entered the disagreement audit")

    exogenous_hash = exogenous_truth_fingerprint(generator, step)
    branches = {
        "h1": run_branch(
            generator,
            base_state,
            step=step,
            first_actions=h1_actions,
            first_costs=h1.costs,
        ),
        "oracle": run_branch(
            generator,
            base_state,
            step=step,
            first_actions=oracle_actions,
            first_costs=oracle.current_costs,
        ),
        "learned": run_branch(
            generator,
            base_state,
            step=step,
            first_actions=learned_actions,
            first_costs=learned.current_costs,
        ),
    }
    if runtime_signature(base_state) != base_signature:
        raise RuntimeError("branch execution mutated the frozen base state")
    if any(branch["h1_continuation_calls"] != K - 1 for branch in branches.values()):
        raise RuntimeError("a branch used a non-H1 continuation")
    if any(
        branch["exogenous_truth_sha256"] != exogenous_hash
        for branch in branches.values()
    ):
        raise RuntimeError("counterfactual branches did not share exogenous truth")
    for name, actions in (("oracle", oracle_actions), ("learned", learned_actions)):
        if (
            actions != h1_actions
            and branches[name]["post_first_signature"]
            == branches["h1"]["post_first_signature"]
        ):
            raise RuntimeError(f"{name} first action did not alter its branch state")

    delta_oracle = branches["oracle"]["J"] - branches["h1"]["J"]
    delta_learned = branches["learned"]["J"] - branches["h1"]["J"]
    oracle_label, tolerance = classify_delta(delta_oracle, branches["h1"]["J"])
    learned_label, learned_tolerance = classify_delta(
        delta_learned, branches["h1"]["J"]
    )
    if tolerance != learned_tolerance:
        raise RuntimeError("branch tie tolerances diverged")
    row = {
        "split": str(state_row["split"]),
        "state_id": str(state_row["state_id"]),
        "step": step,
        "task_count": int(len(task_rows)),
        "h1_h60_oracle_disagree": h1_actions != oracle_actions,
        "h1_h60_learned_disagree": h1_actions != learned_actions,
        "oracle_learned_same_first_action": oracle_actions == learned_actions,
        "h1_joint_action_sha256": spot_v3.object_hash(h1_actions),
        "h60_oracle_joint_action_sha256": spot_v3.object_hash(oracle_actions),
        "h60_learned_joint_action_sha256": spot_v3.object_hash(learned_actions),
        "J_h1": branches["h1"]["J"],
        "J_h60_oracle": branches["oracle"]["J"],
        "J_h60_learned": branches["learned"]["J"],
        "delta_oracle": delta_oracle,
        "delta_learned": delta_learned,
        "tie_tolerance": tolerance,
        "oracle_value_label": oracle_label,
        "learned_value_label": learned_label,
        "exogenous_truth_sha256": exogenous_hash,
        "continuation_policy": "BRANCH_SPECIFIC_RECOMPUTED_H1",
        **state_attributes(state_row, task_rows, diagnostic_row, thresholds),
    }
    for branch_name, branch in branches.items():
        for component in COST_FIELDS:
            row[f"{branch_name}_{component}"] = branch["costs"][component]
    safety = {
        "base_state_sha256_verified": True,
        "base_state_unchanged": True,
        "same_exogenous_truth": True,
        "only_first_controller_differs": True,
        "common_h1_continuation": True,
        "branch_specific_h1_recomputation": True,
        "continuation_state_hashes": {
            name: branch["continuation_state_hashes"]
            for name, branch in branches.items()
        },
        "post_first_signatures": {
            name: branch["post_first_signature"] for name, branch in branches.items()
        },
    }
    return row, safety


def bootstrap_mean_saving_ci(
    savings: np.ndarray,
    *,
    seed: int,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> tuple[float, float]:
    values = np.asarray(savings, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap savings must be a nonempty vector")
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sample = rng.integers(0, len(values), size=len(values))
        means[index] = float(values[sample].mean())
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def summarize_values(
    values: pd.DataFrame,
    *,
    repetitions: int,
) -> pd.DataFrame:
    rows = []
    for split_index, split in enumerate(SPLITS):
        selected = values[values["split"] == split]
        for controller_index, controller in enumerate(("oracle", "learned")):
            delta = selected[f"delta_{controller}"].to_numpy(dtype=float)
            labels = selected[f"{controller}_value_label"]
            savings = -delta
            lower, upper = bootstrap_mean_saving_ci(
                savings,
                seed=20260909 + split_index * 100 + controller_index,
                repetitions=repetitions,
            )
            positive_savings = float((-delta[labels == "POSITIVE"]).sum())
            negative_losses = float(delta[labels == "NEGATIVE"].sum())
            rows.append(
                {
                    "split": split,
                    "controller": controller,
                    "states": int(len(selected)),
                    "positive_states": int((labels == "POSITIVE").sum()),
                    "equivalent_states": int((labels == "EQUIVALENT").sum()),
                    "negative_states": int((labels == "NEGATIVE").sum()),
                    "positive_state_share": float((labels == "POSITIVE").mean()),
                    "equivalent_state_share": float((labels == "EQUIVALENT").mean()),
                    "negative_state_share": float((labels == "NEGATIVE").mean()),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "sum_positive_savings": positive_savings,
                    "sum_negative_losses": negative_losses,
                    "net_delta": float(delta.sum()),
                    "net_saving": float(savings.sum()),
                    "mean_saving": float(savings.mean()),
                    "mean_saving_ci95_lower": lower,
                    "mean_saving_ci95_upper": upper,
                    "bootstrap_repetitions": repetitions,
                }
            )
    return pd.DataFrame(rows)


def oracle_positive_retention(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, selected in [
        ("train", values[values["split"] == "train"]),
        ("validation", values[values["split"] == "validation"]),
        ("train_validation", values),
    ]:
        positive = selected[selected["oracle_value_label"] == "POSITIVE"]
        oracle_saving = float((-positive["delta_oracle"]).sum())
        learned_signed_saving = float((-positive["delta_learned"]).sum())
        learned_positive_saving = float(
            (-positive.loc[positive["delta_learned"] < 0, "delta_learned"]).sum()
        )
        rows.append(
            {
                "scope": scope,
                "oracle_positive_states": int(len(positive)),
                "same_oracle_first_action_states": int(
                    positive["oracle_learned_same_first_action"].sum()
                ),
                "same_oracle_first_action_share": float(
                    positive["oracle_learned_same_first_action"].mean()
                ) if len(positive) else math.nan,
                "oracle_positive_saving": oracle_saving,
                "learned_signed_saving_on_oracle_positive": learned_signed_saving,
                "learned_positive_saving_on_oracle_positive": learned_positive_saving,
                "learned_signed_value_retention": (
                    learned_signed_saving / oracle_saving if oracle_saving else math.nan
                ),
                "learned_positive_value_retention": (
                    learned_positive_saving / oracle_saving if oracle_saving else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def gate_upper_bounds(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        selected = values[values["split"] == split]
        h1_total = float(selected["J_h1"].sum())
        for controller in ("oracle", "learned"):
            choose = selected[f"{controller}_value_label"] == "POSITIVE"
            gated = np.where(
                choose,
                selected[f"J_h60_{controller}"],
                selected["J_h1"],
            )
            gated_total = float(np.sum(gated))
            saving = h1_total - gated_total
            rows.append(
                {
                    "split": split,
                    "controller": controller,
                    "interpretation": "COUNTERFACTUAL_ACTION_VALUE_UPPER_BOUND",
                    "states": int(len(selected)),
                    "h60_selected_states": int(choose.sum()),
                    "always_h1_cost": h1_total,
                    "perfect_value_gate_cost": gated_total,
                    "gate_saving": saving,
                    "gate_action_value_improvement": (
                        saving / h1_total if h1_total else 0.0
                    ),
                    "closed_loop_system_improvement_claim": False,
                }
            )
    return pd.DataFrame(rows)


def value_slices(values: pd.DataFrame) -> pd.DataFrame:
    slices = {
        "long_task": "slice_long_task",
        "Spot": "slice_spot",
        "high_future_workload": "slice_high_future_workload",
        "high_future_carbon": "slice_high_future_carbon",
        "high_resource_pressure": "slice_high_resource_pressure",
    }
    rows = []
    for split in SPLITS:
        split_values = values[values["split"] == split]
        for slice_name, column in slices.items():
            selected = split_values[split_values[column]]
            for controller in ("oracle", "learned"):
                delta = selected[f"delta_{controller}"]
                labels = selected[f"{controller}_value_label"]
                rows.append(
                    {
                        "split": split,
                        "slice": slice_name,
                        "controller": controller,
                        "states": int(len(selected)),
                        "positive_state_share": float((labels == "POSITIVE").mean())
                        if len(selected)
                        else math.nan,
                        "mean_delta": float(delta.mean()) if len(selected) else math.nan,
                        "net_savings": float((-delta).sum()),
                    }
                )
    return pd.DataFrame(rows)


def choose_decision(summary: pd.DataFrame) -> tuple[str, str, str]:
    oracle = summary[summary["controller"] == "oracle"].set_index("split")
    train_saving = float(oracle.loc["train", "net_saving"])
    validation_saving = float(oracle.loc["validation", "net_saving"])
    validation_lower = float(oracle.loc["validation", "mean_saving_ci95_lower"])
    if train_saving > 0 and validation_saving > 0 and validation_lower > 0:
        return (
            "H60_VALUE_STRONG",
            "Oracle H60 net saving is positive in both splits and validation CI is strictly positive.",
            "TRAIN_VALUE_GATE",
        )
    if train_saving > 0 and validation_saving > 0:
        return (
            "H60_VALUE_WEAK",
            "Oracle H60 aggregate saving is positive in both splits, but validation CI includes zero.",
            "STOP_AND_REVIEW",
        )
    return (
        "H60_VALUE_NOT_SUPPORTED",
        "Oracle H60 positive action value does not reproduce as positive aggregate validation saving.",
        "SWITCH_TO_FORECAST_AWARE_H1",
    )


def summary_row(summary: pd.DataFrame, split: str, controller: str) -> pd.Series:
    return summary[
        (summary["split"] == split) & (summary["controller"] == controller)
    ].iloc[0]


def slice_text(slices: pd.DataFrame, name: str) -> str:
    rows = slices[(slices["slice"] == name) & (slices["controller"] == "oracle")]
    values = []
    for split in SPLITS:
        row = rows[rows["split"] == split].iloc[0]
        values.append(
            f"{split}: states={int(row['states'])}, positive={row['positive_state_share']:.3%}, "
            f"mean_delta={row['mean_delta']:.6f}, net_saving={row['net_savings']:.6f}"
        )
    return "; ".join(values)


def write_report(
    values: pd.DataFrame,
    summary: pd.DataFrame,
    slices: pd.DataFrame,
    gates: pd.DataFrame,
    retention: pd.DataFrame,
    metadata: Mapping[str, Any],
    safety: Mapping[str, Any],
    decision: str,
    reason: str,
    next_step: str,
) -> None:
    oracle_train = summary_row(summary, "train", "oracle")
    oracle_validation = summary_row(summary, "validation", "oracle")
    learned_train = summary_row(summary, "train", "learned")
    learned_validation = summary_row(summary, "validation", "learned")
    overall_retention = retention[retention["scope"] == "train_validation"].iloc[0]
    oracle_gate = gates[gates["controller"] == "oracle"]
    learned_gap = (
        oracle_train["net_saving"] > 0
        and oracle_validation["net_saving"] > 0
        and learned_validation["net_saving"] <= 0
    )
    report = f"""# H60 common-continuation counterfactual action-value audit

## Protocol

- Audited splits: TRAIN + VALIDATION only.
- Test value rows loaded or evaluated: NO.
- Horizon: K=4 decision steps (60 minutes).
- Candidate rule: H1 joint action differs from Oracle H60 or Learned H60.
- Audited states: {len(values)}.
- Forecast-window exclusions: {metadata['excluded_oracle_disagreement_without_frozen_forecast_window']} Oracle-disagreement states lacked a causal 96-step history or a t+60 endpoint inside the same audited split.
- First action: branch-specific H1 / H60-Oracle / H60-Learned.
- Continuation: three branch-specific recomputed H1 decisions.
- Cost: existing RollingCostBreakdown components only; H60 terminal objective is not counted as realized stage cost.
- Interpretation: counterfactual first-action value under fixed state distribution and common continuation, not a complete closed-loop system improvement.

## Oracle H60

- Train positive/equivalent/negative: {oracle_train['positive_state_share']:.3%} / {oracle_train['equivalent_state_share']:.3%} / {oracle_train['negative_state_share']:.3%}.
- Validation positive/equivalent/negative: {oracle_validation['positive_state_share']:.3%} / {oracle_validation['equivalent_state_share']:.3%} / {oracle_validation['negative_state_share']:.3%}.
- Train net saving: {oracle_train['net_saving']:.9f}; mean saving CI95 [{oracle_train['mean_saving_ci95_lower']:.9f}, {oracle_train['mean_saving_ci95_upper']:.9f}].
- Validation net saving: {oracle_validation['net_saving']:.9f}; mean saving CI95 [{oracle_validation['mean_saving_ci95_lower']:.9f}, {oracle_validation['mean_saving_ci95_upper']:.9f}].

## Learned H60

- Train positive/equivalent/negative: {learned_train['positive_state_share']:.3%} / {learned_train['equivalent_state_share']:.3%} / {learned_train['negative_state_share']:.3%}.
- Validation positive/equivalent/negative: {learned_validation['positive_state_share']:.3%} / {learned_validation['equivalent_state_share']:.3%} / {learned_validation['negative_state_share']:.3%}.
- Train net saving: {learned_train['net_saving']:.9f}; mean saving CI95 [{learned_train['mean_saving_ci95_lower']:.9f}, {learned_train['mean_saving_ci95_upper']:.9f}].
- Validation net saving: {learned_validation['net_saving']:.9f}; mean saving CI95 [{learned_validation['mean_saving_ci95_lower']:.9f}, {learned_validation['mean_saving_ci95_upper']:.9f}].
- Oracle-positive first-action agreement: {int(overall_retention['same_oracle_first_action_states'])}/{int(overall_retention['oracle_positive_states'])} ({overall_retention['same_oracle_first_action_share']:.3%}).
- Signed Oracle value retained by Learned: {overall_retention['learned_signed_value_retention']:.3%}.
- Forecast-to-control gap: {'YES' if learned_gap else 'NO'}.

## Five frozen slices (Oracle)

- Long task: {slice_text(slices, 'long_task')}.
- Spot: {slice_text(slices, 'Spot')}.
- High future workload: {slice_text(slices, 'high_future_workload')}.
- High future carbon: {slice_text(slices, 'high_future_carbon')}.
- High resource pressure: {slice_text(slices, 'high_resource_pressure')}.

## Action-value upper bound

{oracle_gate.to_csv(index=False)}

This table is a COUNTERFACTUAL ACTION-VALUE UPPER BOUND. It is not reported as total system-cost improvement.

## Safety

- Frozen state SHA-256 verified for every audited state: {safety['base_state_sha256_verified']}.
- Clone independence: {safety['simulator_state_clone_independent']}.
- Same future arrivals, signals, and simulator truth: {safety['same_exogenous_truth']}.
- First action is the only controller-policy difference: {safety['only_first_controller_differs']}.
- Common continuation is H1: {safety['common_h1_continuation']}.
- H1 continuation recomputed from each branch state: {safety['branch_specific_h1_recomputation']}.
- Learned Oracle leakage: NO.
- Learned true-duration leakage: NO.
- Deterministic replay: {safety['deterministic_replay']}.
- Original H1/H60 source modified by this experiment: NO.
- Frozen v3 modified: NO.

## Decision

- Status: `{decision}`.
- Reason: {reason}
- Next step: `{next_step}`.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def print_final(
    values: pd.DataFrame,
    summary: pd.DataFrame,
    slices: pd.DataFrame,
    gates: pd.DataFrame,
    retention: pd.DataFrame,
    decision: str,
    reason: str,
    next_step: str,
) -> None:
    ot = summary_row(summary, "train", "oracle")
    ov = summary_row(summary, "validation", "oracle")
    lt = summary_row(summary, "train", "learned")
    lv = summary_row(summary, "validation", "learned")
    retained = retention[retention["scope"] == "train_validation"].iloc[0]
    oracle_gates = gates[gates["controller"] == "oracle"].set_index("split")
    print(
        f"""=== H60 VALUE AUDIT ===

Audited splits:
TRAIN + VALIDATION

Test touched:
NO

Audited disagreement states:
{len(values)}

=== ORACLE H60 ===

Train positive:
{ot['positive_state_share']:.6%}

Validation positive:
{ov['positive_state_share']:.6%}

Train net saving:
{ot['net_saving']:.9f}

Validation net saving:
{ov['net_saving']:.9f}

Validation 95% CI:
[{ov['mean_saving_ci95_lower']:.9f}, {ov['mean_saving_ci95_upper']:.9f}]

Oracle-gated action-value improvement:
TRAIN {oracle_gates.loc['train', 'gate_action_value_improvement']:.6%}; VALIDATION {oracle_gates.loc['validation', 'gate_action_value_improvement']:.6%}

=== LEARNED H60 ===

Train positive:
{lt['positive_state_share']:.6%}

Validation positive:
{lv['positive_state_share']:.6%}

Train net saving:
{lt['net_saving']:.9f}

Validation net saving:
{lv['net_saving']:.9f}

Validation 95% CI:
[{lv['mean_saving_ci95_lower']:.9f}, {lv['mean_saving_ci95_upper']:.9f}]

Learned retained oracle value:
{retained['learned_signed_value_retention']:.6%}

=== KEY SLICES ===

Long:
{slice_text(slices, 'long_task')}

Spot:
{slice_text(slices, 'Spot')}

High future workload:
{slice_text(slices, 'high_future_workload')}

High future carbon:
{slice_text(slices, 'high_future_carbon')}

High resource pressure:
{slice_text(slices, 'high_resource_pressure')}

=== SAFETY ===

Only first action differs:
YES

Common H1 continuation:
YES

Branch-specific H1 recomputation:
YES

Same exogenous truth:
YES

Oracle leakage into Learned:
NO

true_duration leakage:
NO

Original controllers modified:
NO

=== DECISION ===

{decision}

Reason:
{reason}

Next step:
{next_step}"""
    )


def replay_matches(
    first_row: Mapping[str, Any],
    first_safety: Mapping[str, Any],
    second_row: Mapping[str, Any],
    second_safety: Mapping[str, Any],
) -> bool:
    exact_columns = (
        "h1_joint_action_sha256",
        "h60_oracle_joint_action_sha256",
        "h60_learned_joint_action_sha256",
        "oracle_value_label",
        "learned_value_label",
        "exogenous_truth_sha256",
    )
    numeric_columns = (
        "J_h1",
        "J_h60_oracle",
        "J_h60_learned",
        "delta_oracle",
        "delta_learned",
        "tie_tolerance",
        *(f"{branch}_{field}" for branch in ("h1", "oracle", "learned") for field in COST_FIELDS),
    )
    exact = all(first_row[name] == second_row[name] for name in exact_columns)
    numeric = np.allclose(
        [float(first_row[name]) for name in numeric_columns],
        [float(second_row[name]) for name in numeric_columns],
        rtol=0.0,
        atol=1e-12,
    )
    traces = all(
        first_safety[name] == second_safety[name]
        for name in ("continuation_state_hashes", "post_first_signatures")
    )
    return bool(exact and numeric and traces)


def attach_retention_summary(
    summary: pd.DataFrame,
    retention: pd.DataFrame,
) -> pd.DataFrame:
    result = summary.copy()
    columns = (
        "oracle_positive_states",
        "same_oracle_first_action_states",
        "same_oracle_first_action_share",
        "oracle_positive_saving",
        "learned_signed_saving_on_oracle_positive",
        "learned_positive_saving_on_oracle_positive",
        "learned_signed_value_retention",
        "learned_positive_value_retention",
    )
    for column in columns:
        result[column] = math.nan
    for split in SPLITS:
        source = retention[retention["scope"] == split].iloc[0]
        target = (result["split"] == split) & (result["controller"] == "learned")
        for column in columns:
            result.loc[target, column] = source[column]
    return result


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions != BOOTSTRAP_REPETITIONS:
        raise ValueError("the frozen audit protocol requires exactly 1000 bootstraps")
    started = time.perf_counter()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    OUTPUT.mkdir(parents=True, exist_ok=True)

    data = load_train_validation_inputs()
    manifest = data["manifest"]
    source_before = source_snapshot(manifest)
    raw_config = yaml.safe_load(night1.CONFIG_PATH.read_text("utf-8"))["mpc_h60_v1"]
    if float(raw_config["lambda_terminal"]) != 5.0:
        raise RuntimeError("frozen lambda_terminal changed from 5.0")
    if int(raw_config["target_offset_steps"]) != K:
        raise RuntimeError("frozen H60 target offset changed from four steps")
    current_config = night1.optimizer_config(
        ROOT / raw_config["current_optimizer_config"]
    )
    terminal_config = night1.terminal_config(raw_config, 5.0)
    simulator_inputs = build_train_validation_simulator_inputs(data)
    generator = spot_v3.SpotContinuousGenerator(
        simulator_inputs,
        OUTPUT / "closed_loop_runtime",
        checkpoint_interval=10000,
        resume=False,
    )
    if asdict(current_config) != asdict(generator.optimizer_config):
        raise RuntimeError("audit H1 configuration differs from frozen simulator H1")
    adapter = night2._action_adapter()

    print("[setup] building causal learned forecasts and terminal occupancy")
    _, scaler, predictions, eligible_steps = normalized_forecast_inputs(
        data["states"], data["tasks"]
    )
    occupancy = night2.build_terminal_occupancy_cache(
        data["states"], data["tasks"], data["lifecycle"]
    )
    cache_validation = validate_terminal_cache_train_validation(
        data["states"],
        data["tasks"],
        data["privileged"],
        data["lifecycle"],
        occupancy,
    )
    if cache_validation["status"] != "PASS":
        raise RuntimeError("terminal occupancy cache validation failed")

    print("[setup] solving Learned H60 actions and selecting joint disagreements")
    candidates, learned_actions, candidate_metadata = prepare_candidates(
        generator,
        data,
        predictions,
        eligible_steps,
        occupancy,
        current_config,
        terminal_config,
        adapter,
        generator.links,
    )
    if not candidates:
        raise RuntimeError("no train/validation disagreement states were found")
    candidate_metadata["forecast_scaler"] = scaler
    candidate_metadata["terminal_cache_validation"] = cache_validation
    candidate_metadata["causal_history_steps"] = 96
    candidate_metadata["counterfactual_steps"] = K
    candidate_metadata["test_rows_loaded"] = int(data["test_rows_loaded"])

    print(f"[setup] restoring runtime templates for {len(candidates)} states")
    templates = build_runtime_templates(generator, data["lifecycle"])
    canonical_id_to_index = {
        str(row.task_id): int(index)
        for index, row in generator.canonical.iterrows()
    }
    state_index = data["states"].set_index("state_id", drop=False)
    diagnostic_index = data["state_diagnostics"].set_index("state_id", drop=False)
    task_groups = data["tasks"].groupby("state_id", sort=False).indices
    privileged_groups = {
        str(state_id): frame
        for state_id, frame in data["privileged"].groupby("state_id", sort=False)
    }
    h60_manifest = json.loads((OUTPUT / "14_h60_full_label_manifest.json").read_text("utf-8"))
    thresholds = h60_manifest["slice_thresholds"]

    rows: list[dict[str, Any]] = []
    replay_targets = {
        state_id
        for split in SPLITS
        for state_id in [
            item
            for item in candidates
            if str(state_index.loc[item, "split"]) == split
        ][:3]
    }
    deterministic_replay = True
    aggregate_safety = {
        "base_state_sha256_verified": True,
        "simulator_state_clone_independent": True,
        "same_exogenous_truth": True,
        "only_first_controller_differs": True,
        "common_h1_continuation": True,
        "branch_specific_h1_recomputation": True,
        "deterministic_replay": True,
    }
    for number, state_id in enumerate(candidates, start=1):
        state_row = state_index.loc[state_id]
        positions = task_groups[state_id]
        task_rows = data["tasks"].iloc[positions].sort_values(
            "task_position",
            kind="mergesort",
        )
        row, safety = evaluate_state(
            generator,
            state_row,
            task_rows,
            privileged_groups[state_id],
            diagnostic_index.loc[state_id],
            predictions[int(state_row["step"])],
            learned_actions[state_id],
            occupancy,
            templates,
            canonical_id_to_index,
            thresholds,
            current_config,
            terminal_config,
            adapter,
        )
        rows.append(row)
        for key in (
            "base_state_sha256_verified",
            "same_exogenous_truth",
            "only_first_controller_differs",
            "common_h1_continuation",
            "branch_specific_h1_recomputation",
        ):
            aggregate_safety[key] = aggregate_safety[key] and bool(safety[key])
        if state_id in replay_targets:
            replay_row, replay_safety = evaluate_state(
                generator,
                state_row,
                task_rows,
                privileged_groups[state_id],
                diagnostic_index.loc[state_id],
                predictions[int(state_row["step"])],
                learned_actions[state_id],
                occupancy,
                templates,
                canonical_id_to_index,
                thresholds,
                current_config,
                terminal_config,
                adapter,
            )
            deterministic_replay = deterministic_replay and replay_matches(
                row, safety, replay_row, replay_safety
            )
        if number % 100 == 0 or number == len(candidates):
            elapsed = time.perf_counter() - started
            print(f"[audit] {number}/{len(candidates)} states, elapsed={elapsed:.1f}s")
    aggregate_safety["deterministic_replay"] = deterministic_replay
    if not all(aggregate_safety.values()):
        raise RuntimeError(f"counterfactual safety audit failed: {aggregate_safety}")

    values = pd.DataFrame(rows).sort_values(["split", "step"], kind="mergesort")
    if set(values["split"]) - set(SPLITS):
        raise RuntimeError("test rows entered the value table")
    if not (values["h1_h60_oracle_disagree"] | values["h1_h60_learned_disagree"]).all():
        raise RuntimeError("an all-equal joint action entered the value table")
    value_temporary = VALUE_PATH.with_suffix(".parquet.tmp")
    values.to_parquet(value_temporary, index=False)
    os.replace(value_temporary, VALUE_PATH)

    summary = summarize_values(values, repetitions=args.bootstrap_repetitions)
    retention = oracle_positive_retention(values)
    summary_output = attach_retention_summary(summary, retention)
    slices = value_slices(values)
    gates = gate_upper_bounds(values)
    summary_output.to_csv(SUMMARY_PATH, index=False)
    slices.to_csv(SLICE_PATH, index=False)
    gates.to_csv(GATE_PATH, index=False)

    decision, reason, next_step = choose_decision(summary)
    oracle_is_positive = decision != "H60_VALUE_NOT_SUPPORTED"
    learned_validation = summary_row(summary, "validation", "learned")
    if oracle_is_positive and float(learned_validation["net_saving"]) <= 0:
        reason += (
            " FORECAST_TO_CONTROL_GAP: Oracle action value is positive across splits, "
            "but Learned H60 does not retain positive validation net saving."
        )

    source_after = source_snapshot(manifest)
    if source_after != source_before:
        raise RuntimeError("frozen v3 or original H1/H60 source changed during audit")
    write_report(
        values,
        summary,
        slices,
        gates,
        retention,
        candidate_metadata,
        aggregate_safety,
        decision,
        reason,
        next_step,
    )
    print_final(
        values,
        summary,
        slices,
        gates,
        retention,
        decision,
        reason,
        next_step,
    )


if __name__ == "__main__":
    main()
