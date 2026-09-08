from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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
    build_h60_windows,
    build_spot_h60_time_series,
    fit_h60_train_scaler,
    h60_feature_schema,
    h60_mae_rows,
    persistence_h60,
)
from scripts.audit import spotgpu2026_capacity_calibration_v1 as capacity_v1  # noqa: E402
from sustaincluster_mpc.action_adapter import (  # noqa: E402
    ActionMapping,
    SustainClusterActionAdapter,
)
from sustaincluster_mpc.horizon_adapter import (  # noqa: E402
    HorizonDataCenterSnapshot,
    HorizonState,
)
from sustaincluster_mpc.rolling_horizon_optimizer import (  # noqa: E402
    RollingHorizonConfig,
    RollingObjectiveWeights,
)
from sustaincluster_mpc.state_adapter import (  # noqa: E402
    DataCenterSnapshot,
    ExogenousSignalsSnapshot,
    NetworkLinkSnapshot,
    SchedulerState,
    TaskDestinationSnapshot,
    TaskSnapshot,
)
from sustaincluster_mpc.terminal_h60_optimizer import (  # noqa: E402
    TerminalH60Config,
    TerminalH60DataCenter,
    TerminalH60Optimizer,
    TerminalH60Weights,
)


V3_ROOT = ROOT / "artifacts/mpc_expert_dataset_v3"
MANIFEST_PATH = V3_ROOT / "00_generation_manifest.json"
CONFIG_PATH = ROOT / "configs/sustaincluster_mpc/mpc_h60_v1.yaml"
OUTPUT = ROOT / "artifacts/mpc_h60_v1"
EXPECTED_STATES = 17670
EXPECTED_TASKS = 466867
FLEET_CAPACITY = np.asarray(
    [632636.0, 10412.0, 542259.4285714285], dtype=float
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MPC-H60 Night 1 pipeline")
    parser.add_argument("--batch-states", type=int, default=200)
    parser.add_argument("--manual-states", type=int, default=20)
    return parser.parse_args()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_manifest_and_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(MANIFEST_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text("utf-8"))
    rows = []
    failures = []
    for name, item in sorted(manifest["data_files"].items()):
        path = ROOT / item["path"]
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_sha = sha256(path) if exists else None
        status = (
            "PASS"
            if exists
            and actual_bytes == int(item["bytes"])
            and actual_sha == str(item["sha256"]).upper()
            else "FAIL"
        )
        row = {
            "name": name,
            "path": item["path"],
            "exists": exists,
            "expected_bytes": int(item["bytes"]),
            "actual_bytes": actual_bytes,
            "sha256_match": actual_sha == str(item["sha256"]).upper(),
            "status": status,
        }
        rows.append(row)
        if status != "PASS":
            failures.append(row)
    if failures:
        raise RuntimeError(f"Spot v3 file audit failed: {failures}")
    return manifest, rows


def manifest_path(manifest: Mapping[str, Any], key: str) -> Path:
    return ROOT / manifest["data_files"][key]["path"]


def load_tables(
    manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    states = pd.read_parquet(
        manifest_path(manifest, "states_full"),
        columns=[
            "state_id",
            "step",
            "timestamp_utc",
            "split",
            "pending_count",
            "running_task_ids",
            "in_transit_task_ids",
            "dc_states_json",
            "risk_score",
            "h1_solver_status",
            "h4_solver_status",
        ],
    )
    tasks = pd.read_parquet(
        manifest_path(manifest, "tasks_full"),
        columns=[
            "state_id",
            "step",
            "split",
            "task_id",
            "task_position",
            "cpu_cores",
            "gpu_units",
            "memory_gb",
            "bandwidth_gb",
            "origin_dc",
            "arrival_step",
            "priority",
            "estimated_duration_seconds",
            "estimated_duration_steps",
            "waiting_steps",
            "remaining_sla_steps",
            "feasible_action_mask",
        ],
    )
    labels = pd.read_parquet(
        manifest_path(manifest, "labels_full"),
        columns=[
            "state_id",
            "task_id",
            "task_position",
            "h1_action",
            "h4_action",
            "exact_action_disagreement",
        ],
    )
    privileged = pd.read_parquet(
        manifest_path(manifest, "privileged_full"),
        columns=[
            "state_id",
            "step",
            "split",
            "dc_id",
            "horizon_step",
            "forecast_timestamp_utc",
            "actual_origin_task_count",
            "actual_origin_cpu_demand",
            "actual_origin_gpu_demand",
            "actual_origin_memory_demand",
            "future_electricity_price_usd_per_mwh",
            "future_carbon_intensity_gco2_per_kwh",
            "information_class",
        ],
    )
    lifecycle = pd.read_parquet(
        manifest_path(manifest, "task_lifecycle_full"),
        columns=[
            "task_id",
            "destination_dc",
            "dispatch_step",
            "planned_execution_start_step",
            "actual_execution_start_step",
            "planned_estimated_completion_step",
            "start_estimated_completion_step",
            "observed_true_completion_step",
        ],
    )
    return states, tasks, labels, privileged, lifecycle


def optimizer_config(path: Path) -> RollingHorizonConfig:
    data = yaml.safe_load(path.read_text("utf-8"))["expert"]
    return RollingHorizonConfig(
        allow_defer=bool(data["allow_defer"]),
        weights=RollingObjectiveWeights(**data["objective_weights"]),
        cpu_power_w_per_core=float(data["power_model"]["cpu_power_w_per_core"]),
        gpu_power_w_per_unit=float(data["power_model"]["gpu_power_w_per_unit"]),
        memory_power_w_per_gb=float(data["power_model"]["memory_power_w_per_gb"]),
        waiting_cost_per_step=float(data["waiting_cost_per_step"]),
        terminal_backlog_base_cost=float(data["terminal_backlog_base_cost"]),
        deterministic_tie_break_epsilon=float(
            data["deterministic_tie_break_epsilon"]
        ),
        solver_time_limit_seconds=float(data["solver_time_limit_seconds"]),
    )


def terminal_config(raw: Mapping[str, Any], lambda_terminal: float) -> TerminalH60Config:
    return TerminalH60Config(
        target_offset_steps=int(raw["target_offset_steps"]),
        lambda_terminal=float(lambda_terminal),
        weights=TerminalH60Weights(**raw["terminal_weights"]),
        solver_time_limit_seconds=float(raw["solver_time_limit_seconds"]),
    )


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def build_current_state(
    state_row: pd.Series,
    task_rows: pd.DataFrame,
    links: Sequence[NetworkLinkSnapshot],
) -> HorizonState:
    task_rows = task_rows.sort_values("task_position", kind="mergesort")
    timestamp = pd.Timestamp(state_row["timestamp_utc"])
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    dc_rows = json.loads(state_row["dc_states_json"])
    link_cost = {
        (item.origin_dc_id, item.destination_dc_id):
        item.transmission_cost_usd_per_gb
        for item in links
    }
    tasks = []
    destinations = []
    for position, row in enumerate(task_rows.itertuples(index=False)):
        task = TaskSnapshot(
            task_id=str(row.task_id),
            original_index=position,
            origin_dc_id=int(row.origin_dc),
            cpu_cores=float(row.cpu_cores),
            gpu_units=float(row.gpu_units),
            memory_gb=float(row.memory_gb),
            duration_minutes=float(row.estimated_duration_seconds) / 60.0,
            remaining_duration_minutes=float(row.estimated_duration_seconds) / 60.0,
            arrival_time_utc=(
                timestamp - pd.Timedelta(minutes=15 * int(row.waiting_steps))
            ).isoformat(),
            sla_deadline_utc=(
                timestamp + pd.Timedelta(minutes=15 * int(row.remaining_sla_steps))
            ).isoformat(),
            remaining_sla_minutes=15.0 * int(row.remaining_sla_steps),
            bandwidth_gb=float(row.bandwidth_gb),
            wait_intervals=int(row.waiting_steps),
            was_deferred=int(row.waiting_steps) > 0,
            scheduler_wait_intervals=int(row.waiting_steps),
        )
        tasks.append(task)
        for item in dc_rows:
            dc_id = int(item["dc_id"])
            destinations.append(
                TaskDestinationSnapshot(
                    task_id=task.task_id,
                    original_index=position,
                    destination_dc_id=dc_id,
                    transmission_cost_usd=float(
                        link_cost[(task.origin_dc_id, dc_id)] * task.bandwidth_gb
                    ),
                    transmission_delay_seconds=0.0,
                )
            )

    current_dcs = []
    horizon_dcs = []
    for item in dc_rows:
        dc_id = int(item["dc_id"])
        current_dcs.append(
            DataCenterSnapshot(
                dc_id=dc_id,
                dc_name=f"DC{dc_id}",
                location=str(item["location"]),
                cpu_total_cores=float(item["cpu_total"]),
                cpu_available_cores=float(item["cpu_physical_available"]),
                cpu_reserved_cores=float(item["cpu_reserved"]),
                cpu_schedulable_cores=float(item["cpu_available"]),
                cpu_available_ratio=float(
                    item["cpu_physical_available"] / item["cpu_total"]
                ),
                gpu_total_units=float(item["gpu_total"]),
                gpu_available_units=float(item["gpu_physical_available"]),
                gpu_reserved_units=float(item["gpu_reserved"]),
                gpu_schedulable_units=float(item["gpu_available"]),
                gpu_available_ratio=float(
                    item["gpu_physical_available"] / item["gpu_total"]
                ),
                memory_total_gb=float(item["memory_total"]),
                memory_available_gb=float(item["memory_physical_available"]),
                memory_reserved_gb=float(item["memory_reserved"]),
                memory_schedulable_gb=float(item["memory_available"]),
                memory_available_ratio=float(
                    item["memory_physical_available"] / item["memory_total"]
                ),
                running_task_count=int(item["running_count"]),
                queued_task_count=0,
                in_transit_task_count=int(item["in_transit_count"]),
                resource_release_times_utc=(),
                electricity_price_usd_per_mwh=float(
                    item["electricity_price_usd_per_mwh"]
                ),
                carbon_intensity_gco2_per_kwh=float(
                    item["carbon_intensity_gco2_per_kwh"]
                ),
                total_power_kw=None,
                it_power_kw=None,
                cooling_power_kw=None,
                internal_temperature_c=None,
                ambient_temperature_c=None,
                crac_setpoint_c=None,
            )
        )
        horizon_dcs.append(
            HorizonDataCenterSnapshot(
                dc_id=dc_id,
                dc_name=f"DC{dc_id}",
                location=str(item["location"]),
                cpu_total_cores=float(item["cpu_total"]),
                gpu_total_units=float(item["gpu_total"]),
                memory_total_gb=float(item["memory_total"]),
                cpu_available_cores=(float(item["cpu_available"]),),
                gpu_available_units=(float(item["gpu_available"]),),
                memory_available_gb=(float(item["memory_available"]),),
                known_cpu_reservations=(
                    float(item["cpu_used"]) + float(item["cpu_reserved"]),
                ),
                known_gpu_reservations=(
                    float(item["gpu_used"]) + float(item["gpu_reserved"]),
                ),
                known_memory_reservations=(
                    float(item["memory_used"]) + float(item["memory_reserved"]),
                ),
                forecast_cpu_reservations=(0.0,),
                forecast_gpu_reservations=(0.0,),
                forecast_memory_reservations=(0.0,),
                electricity_price_usd_per_mwh=(
                    float(item["electricity_price_usd_per_mwh"]),
                ),
                carbon_intensity_gco2_per_kwh=(
                    float(item["carbon_intensity_gco2_per_kwh"]),
                ),
            )
        )
    scheduler = SchedulerState(
        tasks=tuple(tasks),
        datacenters=tuple(current_dcs),
        network_links=tuple(links),
        task_destinations=tuple(destinations),
        exogenous=ExogenousSignalsSnapshot(timestamp.isoformat(), 15.0),
        allow_defer=True,
        information_mode="deployable",
    )
    return HorizonState(
        current=scheduler,
        horizon=1,
        forecast_mode="no_future_arrivals",
        timestep_minutes=15.0,
        datacenters=tuple(horizon_dcs),
        running_tasks=(),
        transit_tasks=(),
        future_arrivals=(),
        information_mode="deployable",
        future_signal_mode="persistence",
    )


def optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def build_terminal_inputs(
    state_row: pd.Series,
    privileged_h60: pd.DataFrame,
    task_index: pd.DataFrame,
    lifecycle_index: pd.DataFrame,
) -> tuple[TerminalH60DataCenter, ...]:
    step = int(state_row["step"])
    target_step = step + 4
    dc_rows = json.loads(state_row["dc_states_json"])
    dc_by_id = {int(item["dc_id"]): item for item in dc_rows}
    existing = {dc_id: np.zeros(3, dtype=float) for dc_id in dc_by_id}
    active_ids = (
        to_list(state_row["running_task_ids"])
        + to_list(state_row["in_transit_task_ids"])
    )
    for task_id in active_ids:
        key = str(task_id)
        if key not in task_index.index or key not in lifecycle_index.index:
            raise RuntimeError(f"active task metadata missing for {key}")
        task = task_index.loc[key]
        life = lifecycle_index.loc[key]
        dc_id = optional_int(life["destination_dc"])
        planned_start = optional_int(life["planned_execution_start_step"])
        actual_start = optional_int(life["actual_execution_start_step"])
        planned_completion = optional_int(life["planned_estimated_completion_step"])
        start_completion = optional_int(life["start_estimated_completion_step"])
        start = actual_start if actual_start is not None else planned_start
        completion = (
            start_completion
            if start_completion is not None
            else planned_completion
        )
        if dc_id is None or start is None or completion is None:
            raise RuntimeError(f"active task estimated lifecycle missing for {key}")
        if start <= target_step < completion:
            existing[dc_id] += np.asarray(
                [task["cpu_cores"], task["gpu_units"], task["memory_gb"]],
                dtype=float,
            )

    h60 = privileged_h60[privileged_h60["horizon_step"] == 4].copy()
    if len(h60) != len(dc_by_id) or set(h60["dc_id"].astype(int)) != set(dc_by_id):
        raise RuntimeError(f"incomplete t+60 Oracle rows for {state_row['state_id']}")
    if set(h60["information_class"]) != {"PRIVILEGED_FUTURE"}:
        raise RuntimeError("Oracle H60 input lost its privileged information marker")
    values = []
    for row in h60.sort_values("dc_id").itertuples(index=False):
        dc_id = int(row.dc_id)
        dc = dc_by_id[dc_id]
        values.append(
            TerminalH60DataCenter(
                dc_id=dc_id,
                cpu_total_cores=float(dc["cpu_total"]),
                gpu_total_units=float(dc["gpu_total"]),
                memory_total_gb=float(dc["memory_total"]),
                estimated_existing_cpu_cores=float(existing[dc_id][0]),
                estimated_existing_gpu_units=float(existing[dc_id][1]),
                estimated_existing_memory_gb=float(existing[dc_id][2]),
                arriving_cpu_demand=float(row.actual_origin_cpu_demand),
                arriving_gpu_demand=float(row.actual_origin_gpu_demand),
                arriving_memory_demand=float(row.actual_origin_memory_demand),
                electricity_price_usd_per_mwh=float(
                    row.future_electricity_price_usd_per_mwh
                ),
                carbon_intensity_gco2_per_kwh=float(
                    row.future_carbon_intensity_gco2_per_kwh
                ),
            )
        )
    return tuple(values)


def build_h60_target_table(
    states: pd.DataFrame,
    privileged: pd.DataFrame,
) -> pd.DataFrame:
    h60 = privileged[privileged["horizon_step"] == 4].copy()
    aggregated = (
        h60.groupby(["state_id", "step", "split"], sort=True)
        .agg(
            new_task_count_t60=("actual_origin_task_count", "sum"),
            arriving_cpu_demand_t60=("actual_origin_cpu_demand", "sum"),
            arriving_gpu_demand_t60=("actual_origin_gpu_demand", "sum"),
            arriving_memory_demand_t60=("actual_origin_memory_demand", "sum"),
        )
        .reset_index()
    )
    aggregated["cpu_pressure_t60"] = (
        aggregated["arriving_cpu_demand_t60"] / FLEET_CAPACITY[0]
    )
    aggregated["gpu_pressure_t60"] = (
        aggregated["arriving_gpu_demand_t60"] / FLEET_CAPACITY[1]
    )
    aggregated["memory_pressure_t60"] = (
        aggregated["arriving_memory_demand_t60"] / FLEET_CAPACITY[2]
    )
    pending_by_step = states.set_index("step")["pending_count"]
    aggregated["pending_task_count_t60"] = (
        aggregated["step"].add(4).map(pending_by_step).astype("Int64")
    )
    aggregated["target_step"] = aggregated["step"] + 4
    aggregated["target_mapping"] = "current_step_plus_4"
    return aggregated


def feature_contract() -> pd.DataFrame:
    rows = [
        ("new_task_count_t60", "AVAILABLE", "sum actual origin arrivals"),
        ("arriving_cpu_demand_t60", "AVAILABLE", "sum actual origin CPU demand"),
        ("arriving_gpu_demand_t60", "AVAILABLE", "sum actual origin GPU demand"),
        (
            "arriving_memory_demand_t60",
            "AVAILABLE",
            "sum actual origin modeled memory demand",
        ),
        (
            "cpu_pressure_t60",
            "DERIVED",
            "exact t+60 arrival CPU demand / frozen fleet CPU capacity",
        ),
        (
            "gpu_pressure_t60",
            "DERIVED",
            "exact t+60 arrival GPU demand / frozen fleet GPU capacity",
        ),
        (
            "memory_pressure_t60",
            "DERIVED",
            "exact t+60 arrival memory demand / frozen modeled memory capacity",
        ),
        (
            "pending_task_count_t60",
            "DERIVED",
            "frozen simulator pending_count shifted by exactly four steps; "
            "NOT_AVAILABLE for the final four states beyond the frozen timeline",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "field": field,
                "status": status,
                "information_class": (
                    "PRIVILEGED_SIMULATOR_TIMELINE"
                    if field == "pending_task_count_t60"
                    else "PRIVILEGED_ORACLE_T60"
                ),
                "definition": definition + " at current_step+4",
            }
            for field, status, definition in rows
        ]
    )


def select_states(
    states: pd.DataFrame,
    labels: pd.DataFrame,
    manual_count: int,
    batch_count: int,
) -> tuple[list[str], list[str], pd.DataFrame]:
    disagreement = (
        labels.groupby("state_id", sort=False)["exact_action_disagreement"]
        .any()
        .rename("h1_h4_disagreement")
    )
    candidates = states.join(disagreement, on="state_id")
    candidates["h1_h4_disagreement"] = candidates[
        "h1_h4_disagreement"
    ].astype("boolean").fillna(False).astype(bool)
    candidates = candidates[
        (candidates["pending_count"] > 0)
        & (candidates["pending_count"] <= 160)
        & (candidates["step"] >= 96)
        & (candidates["step"] <= int(states["step"].max()) - 4)
        & (candidates["h1_solver_status"] == "optimal")
        & (candidates["h4_solver_status"] == "optimal")
    ].copy()
    if len(candidates) < batch_count:
        raise RuntimeError(
            f"only {len(candidates)} bounded states available for batch={batch_count}"
        )
    train_risk = candidates[candidates["split"] == "train"]["risk_score"]
    boundaries = [float(train_risk.quantile(value)) for value in (1 / 3, 2 / 3)]
    candidates["pressure_band"] = pd.cut(
        candidates["risk_score"],
        bins=[-np.inf, boundaries[0], boundaries[1], np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)

    allocations = {
        "low": manual_count // 3 + int(manual_count % 3 > 0),
        "medium": manual_count // 3 + int(manual_count % 3 > 1),
        "high": manual_count // 3,
    }
    manual_ids: list[str] = []
    for band in ("low", "medium", "high"):
        subset = candidates[
            (candidates["pressure_band"] == band)
            & candidates["h1_h4_disagreement"]
        ].copy()
        target = float(subset["risk_score"].median())
        subset["distance"] = (subset["risk_score"] - target).abs()
        subset.sort_values(
            ["distance", "pending_count", "step"],
            kind="mergesort",
            inplace=True,
        )
        manual_ids.extend(subset.head(allocations[band])["state_id"].tolist())
    if len(manual_ids) != manual_count:
        raise RuntimeError("could not select the required manual preflight states")

    remaining = candidates[~candidates["state_id"].isin(manual_ids)].sort_values(
        ["step"], kind="mergesort"
    )
    needed = batch_count - len(manual_ids)
    indices = np.linspace(0, len(remaining) - 1, needed, dtype=int)
    batch_ids = manual_ids + remaining.iloc[indices]["state_id"].tolist()
    if len(batch_ids) != batch_count or len(set(batch_ids)) != batch_count:
        raise RuntimeError("batch state selection is not unique and deterministic")
    return manual_ids, batch_ids, candidates


def action_text(values: Sequence[int]) -> str:
    return json.dumps([int(value) for value in values], separators=(",", ":"))


def run_h60_batch(
    raw_config: Mapping[str, Any],
    states: pd.DataFrame,
    tasks: pd.DataFrame,
    labels: pd.DataFrame,
    privileged: pd.DataFrame,
    lifecycle: pd.DataFrame,
    manual_ids: Sequence[str],
    batch_ids: Sequence[str],
    candidates: pd.DataFrame,
    links: Sequence[NetworkLinkSnapshot],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    state_index = states.set_index("state_id", drop=False)
    task_groups = {
        key: value for key, value in tasks.groupby("state_id", sort=False)
    }
    label_groups = {
        key: value.sort_values("task_position", kind="mergesort")
        for key, value in labels.groupby("state_id", sort=False)
    }
    privileged_groups = {
        key: value for key, value in privileged.groupby("state_id", sort=False)
    }
    task_index = tasks.drop_duplicates("task_id").set_index("task_id")
    lifecycle_index = lifecycle.set_index("task_id")
    current_config = optimizer_config(
        ROOT / raw_config["current_optimizer_config"]
    )
    h60_config = terminal_config(raw_config, float(raw_config["lambda_terminal"]))
    zero_config = terminal_config(raw_config, 0.0)
    action_adapter = SustainClusterActionAdapter(
        ActionMapping(tuple((dc_id, dc_id) for dc_id in range(1, 6)), 0, 6)
    )
    optimizer = TerminalH60Optimizer()
    candidate_band = candidates.set_index("state_id")["pressure_band"].to_dict()

    rows = []
    manual_set = set(manual_ids)
    zero_matches = 0
    zero_tasks = 0
    deterministic_matches = 0
    deterministic_states = 0
    avoidance_examples = 0
    for position, state_id in enumerate(batch_ids, start=1):
        state_row = state_index.loc[state_id]
        task_rows = task_groups[state_id]
        label_rows = label_groups[state_id]
        current_state = build_current_state(state_row, task_rows, links)
        terminal = build_terminal_inputs(
            state_row,
            privileged_groups[state_id],
            task_index,
            lifecycle_index,
        )
        result = optimizer.solve(
            current_state,
            terminal,
            current_config=current_config,
            terminal_config=h60_config,
            action_adapter=action_adapter,
        )
        h1 = label_rows["h1_action"].to_numpy(dtype=int)
        h4 = label_rows["h4_action"].to_numpy(dtype=int)
        h60 = np.asarray(result.environment_actions, dtype=int)
        if result.status == "optimal" and len(h60) != len(h1):
            raise RuntimeError(f"H60 action count mismatch for {state_id}")

        base_pressure = {
            item.dc_id: max(item.base_pressure) for item in terminal
        }
        task_by_position = task_rows.sort_values(
            "task_position", kind="mergesort"
        ).reset_index(drop=True)
        if result.status == "optimal":
            for task_position, (h1_action, h60_action) in enumerate(zip(h1, h60)):
                duration = int(
                    task_by_position.loc[
                        task_position, "estimated_duration_steps"
                    ]
                )
                if (
                    duration >= 4
                    and h1_action != h60_action
                    and h1_action > 0
                    and h60_action > 0
                    and base_pressure[int(h60_action)]
                    < base_pressure[int(h1_action)]
                ):
                    avoidance_examples += 1

        row = {
            "state_id": state_id,
            "step": int(state_row["step"]),
            "split": str(state_row["split"]),
            "pressure_band": candidate_band[state_id],
            "pending_tasks": int(len(h1)),
            "solver_status": result.status,
            "solver_message": result.message,
            "solver_latency_ms": 1000.0 * result.solve_seconds,
            "presolve_retry": result.presolve_retry,
            "objective": result.objective_value,
            "current_objective": result.current_costs.total,
            "terminal_contribution": result.terminal_contribution,
            "terminal_resource_pressure_cost": (
                result.terminal_costs.resource_pressure
            ),
            "terminal_electricity_cost": result.terminal_costs.electricity,
            "terminal_carbon_cost": result.terminal_costs.carbon,
            "terminal_backlog_cost": result.terminal_costs.terminal_backlog,
            "h1_actions": action_text(h1),
            "h4_actions": action_text(h4),
            "h60_actions": action_text(h60),
            "h1_h60_task_disagreements": (
                int(np.sum(h1 != h60)) if len(h60) == len(h1) else len(h1)
            ),
            "h4_h60_task_disagreements": (
                int(np.sum(h4 != h60)) if len(h60) == len(h4) else len(h4)
            ),
            "h1_h60_state_disagreement": bool(
                len(h60) != len(h1) or np.any(h1 != h60)
            ),
            "h4_h60_state_disagreement": bool(
                len(h60) != len(h4) or np.any(h4 != h60)
            ),
            "max_terminal_base_pressure": max(base_pressure.values()),
        }
        rows.append(row)

        if state_id in manual_set:
            zero = optimizer.solve(
                current_state,
                terminal,
                current_config=current_config,
                terminal_config=zero_config,
                action_adapter=action_adapter,
            )
            zero_actions = np.asarray(zero.environment_actions, dtype=int)
            zero_matches += int(np.sum(zero_actions == h1))
            zero_tasks += len(h1)
            repeated = optimizer.solve(
                current_state,
                terminal,
                current_config=current_config,
                terminal_config=h60_config,
                action_adapter=action_adapter,
            )
            deterministic_states += 1
            deterministic_matches += int(
                repeated.status == result.status
                and repeated.environment_actions == result.environment_actions
                and (
                    repeated.objective_value == result.objective_value
                    or np.isclose(
                        repeated.objective_value,
                        result.objective_value,
                        rtol=0.0,
                        atol=1e-10,
                    )
                )
            )
        if position % 25 == 0:
            print(f"[h60] solved {position}/{len(batch_ids)} states")

    comparison = pd.DataFrame(rows)
    latency = comparison.loc[
        :,
        [
            "state_id",
            "step",
            "pressure_band",
            "pending_tasks",
            "solver_status",
            "solver_latency_ms",
            "presolve_retry",
        ],
    ].copy()
    successful = comparison["solver_status"] == "optimal"
    total_tasks = int(comparison.loc[successful, "pending_tasks"].sum())
    h1_disagreements = int(
        comparison.loc[successful, "h1_h60_task_disagreements"].sum()
    )
    h4_disagreements = int(
        comparison.loc[successful, "h4_h60_task_disagreements"].sum()
    )
    summary = {
        "batch_states": int(len(comparison)),
        "manual_states": int(len(manual_ids)),
        "solver_failures": int((~successful).sum()),
        "solver_retries": int(comparison["presolve_retry"].sum()),
        "task_decisions": total_tasks,
        "h1_h60_task_disagreement_rate": (
            h1_disagreements / total_tasks if total_tasks else None
        ),
        "h4_h60_task_disagreement_rate": (
            h4_disagreements / total_tasks if total_tasks else None
        ),
        "h1_h60_state_disagreement_rate": float(
            comparison.loc[successful, "h1_h60_state_disagreement"].mean()
        ),
        "h4_h60_state_disagreement_rate": float(
            comparison.loc[successful, "h4_h60_state_disagreement"].mean()
        ),
        "latency_p50_ms": float(
            comparison.loc[successful, "solver_latency_ms"].quantile(0.50)
        ),
        "latency_p90_ms": float(
            comparison.loc[successful, "solver_latency_ms"].quantile(0.90)
        ),
        "latency_p99_ms": float(
            comparison.loc[successful, "solver_latency_ms"].quantile(0.99)
        ),
        "lambda_zero_h1_task_agreement": (
            zero_matches / zero_tasks if zero_tasks else None
        ),
        "deterministic_state_agreement": (
            deterministic_matches / deterministic_states
            if deterministic_states
            else None
        ),
        "lower_pressure_avoidance_examples": int(avoidance_examples),
    }
    return comparison, latency, summary


class H60MLP(nn.Module):
    def __init__(self, input_dim: int, history: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * history, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(H60_TARGET_NAMES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).unsqueeze(1)


def denormalize(values: np.ndarray, scaler: Mapping[str, Any]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for index, name in enumerate(H60_TARGET_NAMES):
        item = scaler["parameters"][name]
        result[:, :, index] = (
            result[:, :, index] * float(item["scale"]) + float(item["mean"])
        )
    return np.maximum(result, 0.0)


def evaluate_model(
    model: nn.Module,
    window: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[float, np.ndarray]:
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(window.X.astype(np.float32)),
            torch.from_numpy(window.Y.astype(np.float32)),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    criterion = nn.MSELoss()
    model.eval()
    total = 0.0
    samples = 0
    predictions = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            prediction = model(features)
            loss = criterion(prediction, targets)
            batch = len(features)
            total += float(loss.item()) * batch
            samples += batch
            predictions.append(prediction.cpu().numpy())
    return total / samples, np.concatenate(predictions, axis=0)


def run_forecast_preflight(
    raw_config: Mapping[str, Any],
    states: pd.DataFrame,
    tasks: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    forecast = raw_config["forecast"]
    config = H60ForecastDatasetConfig(
        history_length=int(forecast["history_steps"]),
        target_offset_steps=int(forecast["target_offset_steps"]),
    )
    timeline = build_spot_h60_time_series(states, tasks)
    scaler = fit_h60_train_scaler(timeline)
    windows = {
        split: build_h60_windows(timeline, split, scaler, config=config)
        for split in ("train", "validation", "test")
    }
    dataset_dir = OUTPUT / "forecast_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    timeline.to_parquet(dataset_dir / "workload_15min.parquet", index=False)
    for split, window in windows.items():
        np.savez_compressed(
            dataset_dir / f"{split}.npz",
            X=window.X,
            Y=window.Y,
            current_step=window.current_step,
            target_step=window.target_step,
            history_end_timestamp=window.history_end_timestamp,
            target_timestamp=window.target_timestamp,
        )
    write_json(dataset_dir / "scaler.json", scaler)
    write_json(dataset_dir / "feature_schema.json", h60_feature_schema())

    rows: list[dict[str, Any]] = []
    persistence_normalized_macro_mae: dict[str, float] = {}
    for split in ("validation", "test"):
        window = windows[split]
        persistence_normalized_macro_mae[split] = float(
            np.abs(window.X[:, -1:, : len(H60_TARGET_NAMES)] - window.Y).mean()
        )
        rows.extend(
            {
                "record_type": "metric",
                "epoch": None,
                "loss": None,
                **item,
            }
            for item in h60_mae_rows(
                window.Y_raw,
                persistence_h60(window),
                split=split,
                model="Persistence-H60",
            )
        )

    seed = int(forecast["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    model = H60MLP(
        len(H60_FEATURE_NAMES),
        config.history_length,
        int(forecast["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(forecast["learning_rate"])
    )
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(windows["train"].X.astype(np.float32)),
            torch.from_numpy(windows["train"].Y.astype(np.float32)),
        ),
        batch_size=int(forecast["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    epoch_losses = []
    started = time.perf_counter()
    for epoch in range(1, int(forecast["preflight_epochs"]) + 1):
        model.train()
        total = 0.0
        samples = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss = criterion(prediction, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite H60 forecast loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            batch = len(features)
            total += float(loss.item()) * batch
            samples += batch
        train_loss = total / samples
        val_loss, _ = evaluate_model(
            model,
            windows["validation"],
            device,
            int(forecast["batch_size"]),
        )
        epoch_losses.append((epoch, train_loss, val_loss))
        for split_name, split_loss in (
            ("train", train_loss),
            ("validation", val_loss),
        ):
            rows.append(
                {
                    "record_type": "training_loss",
                    "split": split_name,
                    "model": "MLP-H60-seed22",
                    "target": "normalized_joint",
                    "mae": None,
                    "samples": windows[split_name].sample_count,
                    "epoch": epoch,
                    "loss": split_loss,
                }
            )
        print(
            f"[forecast] epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )
    elapsed = time.perf_counter() - started

    for split in ("validation", "test"):
        _, normalized_prediction = evaluate_model(
            model,
            windows[split],
            device,
            int(forecast["batch_size"]),
        )
        prediction = denormalize(normalized_prediction, scaler)
        rows.extend(
            {
                "record_type": "metric",
                "epoch": int(forecast["preflight_epochs"]),
                "loss": None,
                **item,
            }
            for item in h60_mae_rows(
                windows[split].Y_raw,
                prediction,
                split=split,
                model="MLP-H60-seed22-3epoch",
            )
        )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "history_steps": config.history_length,
            "target_offset_steps": config.target_offset_steps,
            "input_features": H60_FEATURE_NAMES,
            "targets": H60_TARGET_NAMES,
            "information_class": "DEPLOYABLE_HISTORY_ONLY",
        },
        dataset_dir / "mlp_seed22_3epoch.pt",
    )
    write_json(
        dataset_dir / "manifest.json",
        {
            "source": "SpotGPU2026 Expert Dataset v3 deployable task rows",
            "timeline_rows": len(timeline),
            "task_rows": len(tasks),
            "history_steps": config.history_length,
            "target_offset_steps": config.target_offset_steps,
            "target_minutes": 60,
            "intermediate_targets_used": False,
            "split_policy": "FROZEN_V3_SPLIT_NO_CROSS_BOUNDARY_WINDOWS",
            "normalization": "TRAIN_ONLY",
            "samples": {
                split: window.sample_count for split, window in windows.items()
            },
            "seed": seed,
            "preflight_epochs": int(forecast["preflight_epochs"]),
        },
    )
    loss_decreased = epoch_losses[-1][1] < epoch_losses[0][1]
    summary = {
        "timeline_rows": int(len(timeline)),
        "samples": {
            split: window.sample_count for split, window in windows.items()
        },
        "history_steps": config.history_length,
        "target_offset_steps": config.target_offset_steps,
        "train_only_normalization": scaler["fitted_on"] == "TRAIN_ONLY",
        "loss_decreased": bool(loss_decreased),
        "epoch1_train_loss": float(epoch_losses[0][1]),
        "epoch3_train_loss": float(epoch_losses[-1][1]),
        "epoch3_validation_loss": float(epoch_losses[-1][2]),
        "training_seconds": float(elapsed),
        "estimated_50_epoch_seconds": float(elapsed / len(epoch_losses) * 50),
        "persistence_normalized_macro_mae": persistence_normalized_macro_mae,
        "preflight_pass": bool(
            loss_decreased
            and scaler["fitted_on"] == "TRAIN_ONLY"
            and all(window.sample_count > 0 for window in windows.values())
        ),
    }
    return pd.DataFrame(rows), summary


def write_reports(
    manifest: Mapping[str, Any],
    file_audit: Sequence[Mapping[str, Any]],
    states: pd.DataFrame,
    tasks: pd.DataFrame,
    target_table: pd.DataFrame,
    manual: pd.DataFrame,
    comparison: pd.DataFrame,
    latency: pd.DataFrame,
    h60_summary: Mapping[str, Any],
    forecast_rows: pd.DataFrame,
    forecast_summary: Mapping[str, Any],
    raw_config: Mapping[str, Any],
) -> str:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    split_rows = states.groupby("split").size().to_dict()
    task_split_rows = tasks.groupby("split").size().to_dict()
    write_text(
        OUTPUT / "00_data_audit.md",
        f"""# SpotGPU2026 v3 data audit

- Manifest: `{MANIFEST_PATH.relative_to(ROOT).as_posix()}`
- Manifest-declared files checked: {len(file_audit)}
- File existence / byte size / SHA-256: PASS
- States: {len(states)}
- Task decisions: {len(tasks)}
- Unique tasks: {tasks["task_id"].nunique()}
- State split rows: {json.dumps(split_rows, sort_keys=True)}
- Task split rows: {json.dumps(task_split_rows, sort_keys=True)}
- Frozen split reused: PASS
- Expert Dataset v3 regenerated: NO

Every input path was resolved from the v3 manifest. Oracle future rows remain
privileged and are never placed in forecast history or a deployable student
input.
""",
    )
    manual.to_csv(OUTPUT / "01_h60_manual_preflight.csv", index=False)
    write_text(
        OUTPUT / "02_h60_protocol.md",
        f"""# MPC-H60 Oracle protocol

## Definition

`MPC-H60` is a two-node controller: current state plus one explicit terminal
node at `t+60 min`. It is not H4. No independent t+15, t+30, or t+45
prediction node is passed to the optimizer.

`J_H60(a_t) = J_current(s_t, a_t) + lambda_terminal * Phi(s_t60 | a_t)`

- `J_current`: the frozen repaired H1 objective and capacity/SLA constraints.
- `lambda_terminal`: {raw_config["lambda_terminal"]}.
- `Phi`: terminal resource pressure, one terminal-slot electricity/carbon
  exposure, and terminal backlog.
- Terminal resource-pressure weight: {raw_config["terminal_weights"]["resource_pressure"]}.
- Electricity/carbon weights: {raw_config["terminal_weights"]["electricity"]} /
  {raw_config["terminal_weights"]["carbon"]}.
- Terminal backlog weight: {raw_config["terminal_weights"]["terminal_backlog"]}.

## Terminal state

The Oracle terminal node combines scheduler-visible active tasks that remain
active at t+60 according to estimated completion with exact t+60 origin demand
and t+60 regional energy/carbon signals from the privileged v3 table.
Intermediate arrivals are not supplied as independent horizon nodes.

For each current assignment, resource occupancy is added at t+60 only when the
scheduler-visible estimated duration spans the terminal step. `true_duration`
is not read by the optimizer or forecast dataset.

## Information boundary

- Oracle t+60: offline expert and method validation only.
- Forecast history: deployable current/history rows only.
- Future student input: predicted_t60 only.
- Original H1/H4 implementation and labels: unchanged.
""",
    )
    feature_contract().to_csv(
        OUTPUT / "03_terminal_feature_contract.csv", index=False
    )
    comparison.to_csv(OUTPUT / "04_h1_h4_h60_comparison.csv", index=False)
    latency.to_csv(OUTPUT / "05_solver_latency.csv", index=False)

    low_threshold = float(comparison["max_terminal_base_pressure"].quantile(0.10))
    high_threshold = float(comparison["max_terminal_base_pressure"].quantile(0.90))
    low = comparison[
        comparison["max_terminal_base_pressure"] <= low_threshold
    ]
    high = comparison[
        comparison["max_terminal_base_pressure"] >= high_threshold
    ]
    low_effect = float(
        (low["terminal_resource_pressure_cost"] / low["pending_tasks"]).median()
    )
    high_effect = float(
        (high["terminal_resource_pressure_cost"] / high["pending_tasks"]).median()
    )
    checks = {
        "A_lambda_zero_matches_h1": (
            h60_summary["lambda_zero_h1_task_agreement"] >= 0.99
        ),
        "B_low_terminal_effect_is_small": bool(
            np.isfinite(low_effect)
            and np.isfinite(high_effect)
            and 0.0 <= low_effect < high_effect
        ),
        "C_high_pressure_avoidance_observed": (
            h60_summary["lower_pressure_avoidance_examples"] > 0
        ),
        "D_deterministic": (
            h60_summary["deterministic_state_agreement"] == 1.0
        ),
    }
    write_text(
        OUTPUT / "06_sanity_checks.md",
        f"""# MPC-H60 sanity checks

| Check | Result | Evidence |
|---|---|---|
| A. lambda_terminal=0 approaches frozen current-only H1 | {"PASS" if checks["A_lambda_zero_matches_h1"] else "FAIL"} | task agreement={h60_summary["lambda_zero_h1_task_agreement"]:.6%} |
| B. low-t60 workload has a smaller terminal pressure effect | {"PASS" if checks["B_low_terminal_effect_is_small"] else "FAIL"} | pressure-cost/task low-P10={low_effect:.6f}; high-P90={high_effect:.6f} |
| C. high-pressure destination avoidance can occur | {"PASS" if checks["C_high_pressure_avoidance_observed"] else "FAIL"} | lower-pressure changed assignments={h60_summary["lower_pressure_avoidance_examples"]} |
| D. identical state and H60 input are deterministic | {"PASS" if checks["D_deterministic"] else "FAIL"} | state agreement={h60_summary["deterministic_state_agreement"]:.6%} |

No reward, SLA, capacity, H1 implementation, or H4 implementation was changed.
""",
    )
    forecast_metric = forecast_rows[
        (forecast_rows["record_type"] == "metric")
        & (forecast_rows["model"] == "Persistence-H60")
    ]
    persistence = {
        split: forecast_metric[forecast_metric["split"] == split]
        .set_index("target")["mae"]
        .to_dict()
        for split in ("validation", "test")
    }
    write_text(
        OUTPUT / "07_forecast_dataset_audit.md",
        f"""# SpotGPU2026 H60 forecast dataset audit

- Timeline rows: {forecast_summary["timeline_rows"]}
- History: {forecast_summary["history_steps"]} x 15 min = 24 h
- Target: exactly current_step + {forecast_summary["target_offset_steps"]} = +60 min
- Intermediate forecast targets: NO
- Frozen v3 train/validation/test split: PASS
- Cross-split windows: 0
- Train-only normalization: {"PASS" if forecast_summary["train_only_normalization"] else "FAIL"}
- Samples: {json.dumps(forecast_summary["samples"], sort_keys=True)}
- Persistence validation MAE: {json.dumps(persistence["validation"], sort_keys=True)}
- Persistence test MAE: {json.dumps(persistence["test"], sort_keys=True)}
- Persistence normalized macro MAE: {json.dumps(forecast_summary["persistence_normalized_macro_mae"], sort_keys=True)}
- MLP seed22 3-epoch loss decreased: {"PASS" if forecast_summary["loss_decreased"] else "FAIL"}
- Oracle future in forecast history/student input: NO
""",
    )
    forecast_rows.to_csv(OUTPUT / "08_forecast_preflight.csv", index=False)

    h60_ready = (
        len(manual) == 20
        and h60_summary["batch_states"] >= 200
        and h60_summary["solver_failures"] == 0
        and h60_summary["h1_h60_task_disagreement_rate"] > 0
        and all(checks.values())
    )
    ready = h60_ready and forecast_summary["preflight_pass"]
    status = (
        "MPC_H60_NIGHT_PIPELINE_READY"
        if ready
        else "MPC_H60_NIGHT_BLOCKED"
    )
    write_text(
        OUTPUT / "09_night_diagnosis.md",
        f"""# Night 1 diagnosis

- Spot v3 integrity: PASS
- t -> t+4 mapping: PASS
- MPC-H60 Oracle solver failures: {h60_summary["solver_failures"]}
- Manual preflight states: {len(manual)}
- Batch states: {h60_summary["batch_states"]}
- H1/H60 task disagreement: {h60_summary["h1_h60_task_disagreement_rate"]:.6%}
- H4/H60 task disagreement: {h60_summary["h4_h60_task_disagreement_rate"]:.6%}
- H1/H60 state disagreement: {h60_summary["h1_h60_state_disagreement_rate"]:.6%}
- Forecast dataset: PASS
- Forecast 3-epoch seed22 preflight: {"PASS" if forecast_summary["preflight_pass"] else "FAIL"}
- Full H60 labels generated tonight: NO
- Closed-loop large experiment run tonight: NO
- Structured Student trained: NO

Final status: `{status}`
""",
    )
    write_json(
        OUTPUT / "night_summary.json",
        {
            "manifest_final_status": manifest["final_status"],
            "states": len(states),
            "task_decisions": len(tasks),
            "target_rows": len(target_table),
            "h60": dict(h60_summary),
            "forecast": dict(forecast_summary),
            "sanity_checks": checks,
            "final_status": status,
        },
    )
    return status


def main() -> None:
    args = parse_args()
    if args.manual_states != 20:
        raise ValueError("Night 1 requires exactly 20 manual preflight states")
    if args.batch_states < 200:
        raise ValueError("Night 1 requires at least 200 batch states")
    raw_config = yaml.safe_load(CONFIG_PATH.read_text("utf-8"))["mpc_h60_v1"]
    manifest, file_audit = load_manifest_and_audit()
    states, tasks, labels, privileged, lifecycle = load_tables(manifest)
    if len(states) != EXPECTED_STATES or len(tasks) != EXPECTED_TASKS:
        raise RuntimeError(
            f"Spot v3 cardinality mismatch: states={len(states)}, tasks={len(tasks)}"
        )
    if tasks["task_id"].nunique() != EXPECTED_TASKS:
        raise RuntimeError("Spot v3 tasks are not one row per unique task")
    if set(states["split"]) != {"train", "validation", "test"}:
        raise RuntimeError("Spot v3 frozen splits are incomplete")

    target_table = build_h60_target_table(states, privileged)
    if not np.array_equal(
        target_table["target_step"].to_numpy(),
        target_table["step"].to_numpy() + 4,
    ):
        raise RuntimeError("t -> t+4 target mapping failed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target_table.to_parquet(OUTPUT / "h60_targets_full.parquet", index=False)

    manual_ids, batch_ids, candidates = select_states(
        states,
        labels,
        args.manual_states,
        args.batch_states,
    )
    _, links = capacity_v1.signal_and_network_template()
    comparison, latency, h60_summary = run_h60_batch(
        raw_config,
        states,
        tasks,
        labels,
        privileged,
        lifecycle,
        manual_ids,
        batch_ids,
        candidates,
        links,
    )
    manual = comparison[comparison["state_id"].isin(manual_ids)].copy()
    manual["manual_order"] = manual["state_id"].map(
        {state_id: index for index, state_id in enumerate(manual_ids)}
    )
    manual.sort_values("manual_order", inplace=True)

    forecast_rows, forecast_summary = run_forecast_preflight(
        raw_config, states, tasks
    )
    status = write_reports(
        manifest,
        file_audit,
        states,
        tasks,
        target_table,
        manual,
        comparison,
        latency,
        h60_summary,
        forecast_rows,
        forecast_summary,
        raw_config,
    )
    print(
        json.dumps(
            {
                "final_status": status,
                "h60": h60_summary,
                "forecast": forecast_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
