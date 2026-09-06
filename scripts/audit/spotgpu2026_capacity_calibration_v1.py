from __future__ import annotations

import argparse
from dataclasses import asdict
import heapq
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts.imitation import build_alibaba2020_expert_dataset_v3 as alibaba_v3  # noqa: E402
from scripts.imitation import build_mpc_expert_dataset_v3 as spot_v3  # noqa: E402
from sustaincluster_mpc.action_adapter import (  # noqa: E402
    ActionMapping,
    SustainClusterActionAdapter,
)
from sustaincluster_mpc.forecast_pressure_adapter import (  # noqa: E402
    expected_origin_probabilities,
)
from sustaincluster_mpc.horizon_adapter import (  # noqa: E402
    HorizonDataCenterSnapshot,
    HorizonState,
    RunningTaskHorizonSnapshot,
)
from sustaincluster_mpc.rolling_horizon_optimizer import (  # noqa: E402
    RollingHorizonOptimizer,
)
from sustaincluster_mpc.state_adapter import (  # noqa: E402
    DataCenterSnapshot,
    ExogenousSignalsSnapshot,
    NetworkLinkSnapshot,
    SchedulerState,
    TaskDestinationSnapshot,
    TaskSnapshot,
)


OUTPUT = ROOT / "artifacts/mpc_expert_dataset_v3"
NODE_FILE = ROOT / "data/raw/alibaba_2026/spot_gpu/node_info_df.csv"
MEMORY_CPU_RATIO = 330000.0 / 385000.0
PARTITION_WEIGHTS = (0.18, 0.22, 0.20, 0.25, 0.15)
WINDOW_STEPS = 16
MINIMUM_RESTORE_WARMUP_STEPS = 96
FINAL_STATUSES = {
    "SPOT NATIVE CAPACITY READY",
    "SPOT COMPARABLE CAPACITY REQUIRED",
    "SOURCE CAPACITY INCONSISTENCY",
    "CAPACITY CALIBRATION INCOMPLETE",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def load_frozen_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    v3 = spot_v3.load_config()
    contract = yaml.safe_load(
        (ROOT / v3["scenario_b"]["contract_config"]).read_text("utf-8")
    )
    if contract["sla"]["selected_profile"] != "medium":
        raise RuntimeError("frozen Spot SLA profile changed")
    medium = contract["sla"]["profiles"]["medium"]
    waits = {key: int(value["max_wait_steps"]) for key, value in medium.items()}
    if waits != {"HP": 1, "Spot": 8}:
        raise RuntimeError("frozen max-wait contract changed")
    if contract["request_scope"]["value"] != "PER_JOB":
        raise RuntimeError("Spot request scope changed")
    if contract["gpu_heterogeneity"]["mode"] != "METADATA_ONLY":
        raise RuntimeError("GPU metadata-only contract changed")
    return v3, contract


def load_nodes(contract: Mapping[str, Any]) -> pd.DataFrame:
    if spot_v3.sha256(NODE_FILE) != contract["source"]["node_sha256"]:
        raise RuntimeError("Spot node_info hash changed")
    nodes = pd.read_csv(NODE_FILE)
    expected = {"gpu_model", "gpu_capacity_num", "cpu_num", "node_name"}
    if set(nodes.columns) != expected:
        raise RuntimeError("Spot node_info schema changed")
    if nodes["node_name"].duplicated().any() or nodes.isna().any().any():
        raise RuntimeError("Spot node_info contains duplicate or missing nodes")
    if (nodes[["gpu_capacity_num", "cpu_num"]] <= 0).any().any():
        raise RuntimeError("Spot node_info contains nonpositive capacity")
    return nodes


def largest_remainder_counts(size: int, weights: Sequence[float]) -> np.ndarray:
    raw = np.asarray(weights, dtype=float) * size
    counts = np.floor(raw).astype(int)
    remainder = size - int(counts.sum())
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    if counts.sum() != size:
        raise RuntimeError("largest remainder allocation failed")
    return counts


def deterministic_partition(nodes: pd.DataFrame) -> pd.DataFrame:
    ordered = nodes.sort_values(
        ["gpu_model", "cpu_num", "gpu_capacity_num", "node_name"],
        kind="mergesort",
    )
    assignments: dict[int, int] = {}
    strata = ["gpu_model", "cpu_num", "gpu_capacity_num"]
    for _, group in ordered.groupby(strata, sort=True, dropna=False):
        counts = largest_remainder_counts(len(group), PARTITION_WEIGHTS)
        cursor = 0
        for dc_id, count in enumerate(counts, start=1):
            selected = group.iloc[cursor : cursor + count]
            assignments.update({int(index): dc_id for index in selected.index})
            cursor += int(count)
    result = nodes.copy()
    result.insert(0, "dc_id", result.index.map(assignments).astype(int))
    result["partition_method"] = (
        "DETERMINISTIC_STRATIFIED_LARGEST_REMAINDER_GPU_MODEL_CPU_GPU_NODE_NAME"
    )
    if len(assignments) != len(nodes) or result["dc_id"].isna().any():
        raise RuntimeError("not every Spot node was assigned exactly once")
    if result["node_name"].duplicated().any():
        raise RuntimeError("one Spot node appears in more than one DC")
    return result


def capacity_tables(
    nodes: pd.DataFrame, assigned: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    node_summary = pd.DataFrame(
        [
            {
                "source_file": NODE_FILE.relative_to(ROOT).as_posix(),
                "node_count": len(nodes),
                "total_cpu_cores": int(nodes["cpu_num"].sum()),
                "total_gpu_units": int(nodes["gpu_capacity_num"].sum()),
                "gpu_model_count": int(nodes["gpu_model"].nunique()),
                "node_name_unique": bool(nodes["node_name"].is_unique),
                "source_sha256": spot_v3.sha256(NODE_FILE),
            }
        ]
    )
    combinations = (
        nodes.groupby(["cpu_num", "gpu_capacity_num"], sort=True)
        .size()
        .rename("node_count")
        .reset_index()
    )
    model = (
        nodes.groupby("gpu_model", sort=True)
        .agg(
            node_count=("node_name", "size"),
            total_cpu_cores=("cpu_num", "sum"),
            total_gpu_units=("gpu_capacity_num", "sum"),
            cpu_min=("cpu_num", "min"),
            cpu_p50=("cpu_num", "median"),
            cpu_max=("cpu_num", "max"),
            gpu_min=("gpu_capacity_num", "min"),
            gpu_p50=("gpu_capacity_num", "median"),
            gpu_max=("gpu_capacity_num", "max"),
        )
        .reset_index()
    )
    model["node_share"] = model["node_count"] / len(nodes)
    model["gpu_share"] = model["total_gpu_units"] / nodes["gpu_capacity_num"].sum()
    dcs = (
        assigned.groupby("dc_id", sort=True)
        .agg(
            node_count=("node_name", "size"),
            total_cores=("cpu_num", "sum"),
            total_gpus=("gpu_capacity_num", "sum"),
        )
        .reset_index()
    )
    dcs["total_mem"] = dcs["total_cores"] * MEMORY_CPU_RATIO
    dcs["memory_provenance"] = "MODELED_DC_MEMORY"
    dcs["target_partition_weight"] = list(PARTITION_WEIGHTS)
    old_configs = yaml.safe_load(
        (ROOT / "references/external_repos/sustain-cluster/configs/env/datacenters.yaml").read_text("utf-8")
    )["datacenters"]
    old_by_id = {int(item["dc_id"]): item for item in old_configs}
    dcs["location"] = dcs["dc_id"].map(lambda value: old_by_id[int(value)]["location"])
    dcs["timezone_shift"] = dcs["dc_id"].map(
        lambda value: old_by_id[int(value)]["timezone_shift"]
    )
    if int(dcs["total_cores"].sum()) != int(nodes["cpu_num"].sum()):
        raise RuntimeError("CPU capacity was not conserved")
    if int(dcs["total_gpus"].sum()) != int(nodes["gpu_capacity_num"].sum()):
        raise RuntimeError("GPU capacity was not conserved")
    return node_summary, combinations, model, dcs


def model_by_dc(assigned: pd.DataFrame) -> pd.DataFrame:
    table = (
        assigned.groupby(["dc_id", "gpu_model"], sort=True)
        .agg(
            node_count=("node_name", "size"),
            total_cpu_cores=("cpu_num", "sum"),
            total_gpu_units=("gpu_capacity_num", "sum"),
        )
        .reset_index()
    )
    dc_nodes = assigned.groupby("dc_id")["node_name"].size()
    fleet = assigned["gpu_model"].value_counts(normalize=True)
    table["within_dc_node_share"] = table.apply(
        lambda row: row["node_count"] / dc_nodes.loc[row["dc_id"]], axis=1
    )
    table["fleet_node_share"] = table["gpu_model"].map(fleet)
    table["absolute_share_deviation"] = (
        table["within_dc_node_share"] - table["fleet_node_share"]
    ).abs()
    return table


def native_contract_dcs(dcs: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "dc_id": int(row.dc_id),
            "location": str(row.location),
            "timezone_shift": int(row.timezone_shift),
            "population_weight": float(row.target_partition_weight),
            "total_cores": float(row.total_cores),
            "total_gpus": float(row.total_gpus),
            "total_mem": float(row.total_mem),
            "memory_provenance": "MODELED_DC_MEMORY",
        }
        for row in dcs.itertuples(index=False)
    ]


def resource_arrays(canonical: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrivals = canonical["arrival_step"].to_numpy(dtype=np.int64)
    durations = np.maximum(
        1,
        np.ceil(canonical["true_duration"].to_numpy(dtype=float) / 900.0).astype(np.int64),
    )
    resources = canonical[
        ["cpu_request_effective", "gpu_request_effective", "memory_request"]
    ].to_numpy(dtype=float)
    return arrivals, durations, resources


def capacity_pressure(
    canonical: pd.DataFrame,
    capacities: Mapping[str, float],
    waits: Mapping[str, int],
    scenario_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arrivals, durations, resources = resource_arrays(canonical)
    wait_values = canonical["priority"].map(waits).to_numpy(dtype=np.int64)
    trace_steps = int(arrivals.max()) + 1
    timeline = int((arrivals + durations + 1).max()) + 2
    names = ("cpu", "gpu", "memory")
    capacity_rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        mandatory_start = arrivals + wait_values + 1
        mandatory_stop = arrivals + durations + 1
        mandatory_diff = np.zeros(timeline, dtype=float)
        mask = mandatory_stop > mandatory_start
        np.add.at(mandatory_diff, mandatory_start[mask], resources[mask, index])
        np.add.at(mandatory_diff, mandatory_stop[mask], -resources[mask, index])
        mandatory = np.cumsum(mandatory_diff[:-1])
        immediate_diff = np.zeros(timeline, dtype=float)
        np.add.at(immediate_diff, arrivals, resources[:, index])
        np.add.at(immediate_diff, arrivals + durations, -resources[:, index])
        immediate = np.cumsum(immediate_diff[:-1])
        mandatory_peak_step = int(np.argmax(mandatory))
        immediate_peak_step = int(np.argmax(immediate))
        resource_steps = float(np.sum(resources[:, index] * durations))
        capacity = float(capacities[name])
        capacity_rows.append(
            {
                "scenario": scenario_name,
                "resource": name,
                "capacity": capacity,
                "resource_time_demand_steps": resource_steps,
                "offered_load_ratio": resource_steps / (capacity * trace_steps),
                "mandatory_concurrent_peak": float(mandatory[mandatory_peak_step]),
                "mandatory_peak_step": mandatory_peak_step,
                "mandatory_peak_ratio": float(mandatory[mandatory_peak_step] / capacity),
                "immediate_start_concurrent_peak": float(immediate[immediate_peak_step]),
                "immediate_start_peak_step": immediate_peak_step,
                "immediate_start_peak_ratio": float(immediate[immediate_peak_step] / capacity),
                "mandatory_definition": "GUARANTEED_ACTIVE_UNDER_FROZEN_MAX_WAIT_PLUS_ONE_TRANSFER_STEP",
            }
        )

    grouped = (
        canonical.groupby("arrival_step", sort=True)
        .agg(
            new_task_count=("task_id", "size"),
            cpu_arrival_demand=("cpu_request_effective", "sum"),
            gpu_arrival_demand=("gpu_request_effective", "sum"),
            memory_arrival_demand=("memory_request", "sum"),
        )
        .reindex(range(trace_steps), fill_value=0)
        .rename_axis("arrival_step")
        .reset_index()
    )
    split_by_step = canonical.groupby("arrival_step")["temporal_split"].first()
    grouped["temporal_split"] = grouped["arrival_step"].map(split_by_step).ffill().bfill()
    for name in names:
        grouped[f"{name}_arrival_pressure"] = (
            grouped[f"{name}_arrival_demand"] / capacities[name]
        )
    summary_rows = []
    for column in (
        "new_task_count",
        "cpu_arrival_demand",
        "gpu_arrival_demand",
        "memory_arrival_demand",
        "cpu_arrival_pressure",
        "gpu_arrival_pressure",
        "memory_arrival_pressure",
    ):
        values = grouped[column].to_numpy(dtype=float)
        summary_rows.append(
            {
                "scenario": scenario_name,
                "metric": column,
                "mean": float(np.mean(values)),
                "p50": float(np.quantile(values, 0.50)),
                "p90": float(np.quantile(values, 0.90)),
                "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(capacity_rows), grouped, pd.DataFrame(summary_rows)


def select_windows(canonical: pd.DataFrame, train_last_step: int) -> pd.DataFrame:
    train_steps = train_last_step + 1
    arrivals = (
        canonical[canonical["arrival_step"] <= train_last_step]
        .groupby("arrival_step")
        .agg(
            tasks=("task_id", "size"),
            gpu=("gpu_request_effective", "sum"),
            long_tasks=("true_duration", lambda values: int((values > 72 * 3600).sum())),
        )
        .reindex(range(train_steps), fill_value=0)
    )
    all_starts = np.arange(0, train_steps - WINDOW_STEPS + 1, dtype=int)
    kernel = np.ones(WINDOW_STEPS)
    all_gpu_mean = np.convolve(arrivals["gpu"].to_numpy(float), kernel, mode="valid") / WINDOW_STEPS
    all_task_mean = np.convolve(arrivals["tasks"].to_numpy(float), kernel, mode="valid") / WINDOW_STEPS
    all_long_count = np.convolve(arrivals["long_tasks"].to_numpy(float), kernel, mode="valid")
    starts = all_starts[all_starts >= MINIMUM_RESTORE_WARMUP_STEPS]
    gpu_mean = all_gpu_mean[starts]
    task_mean = all_task_mean[starts]
    long_count = all_long_count[starts]
    selected: list[dict[str, Any]] = []
    used: set[int] = set()

    def add_quantile(label: str, quantile: float) -> None:
        target = float(np.quantile(gpu_mean, quantile))
        order = np.argsort(np.abs(gpu_mean - target), kind="mergesort")
        index = next(int(item) for item in order if int(starts[item]) not in used)
        start = int(starts[index])
        used.add(start)
        selected.append(
            {
                "window_type": label,
                "start_step": start,
                "end_step_exclusive": start + WINDOW_STEPS,
                "window_steps": WINDOW_STEPS,
                "gpu_arrival_mean": float(gpu_mean[index]),
                "new_task_mean": float(task_mean[index]),
                "long_task_arrivals_gt72h": int(long_count[index]),
                "selection_rule": f"TRAIN_ONLY_POST_WARMUP_GPU_WINDOW_MEAN_NEAREST_Q{int(quantile * 100):02d}",
            }
        )

    add_quantile("LOW_LOAD", 0.10)
    add_quantile("MEDIUM_LOAD", 0.50)
    add_quantile("HIGH_LOAD", 0.95)
    order = np.argsort(-long_count, kind="mergesort")
    index = next(int(item) for item in order if int(starts[item]) not in used)
    start = int(starts[index])
    selected.append(
        {
            "window_type": "LONG_TASK_DENSE",
            "start_step": start,
            "end_step_exclusive": start + WINDOW_STEPS,
            "window_steps": WINDOW_STEPS,
            "gpu_arrival_mean": float(gpu_mean[index]),
            "new_task_mean": float(task_mean[index]),
            "long_task_arrivals_gt72h": int(long_count[index]),
            "selection_rule": "TRAIN_ONLY_POST_WARMUP_MAX_GT72H_ARRIVALS_DETERMINISTIC_EARLIEST_TIE",
        }
    )
    return pd.DataFrame(selected)


def signal_and_network_template() -> tuple[dict[int, dict[str, Any]], tuple[NetworkLinkSnapshot, ...]]:
    env = alibaba_v3.build_env(pd.Timestamp("2023-02-17T00:00:00Z"), 2, 3001, 60.0)
    try:
        base = alibaba_v3.make_horizon_adapter().build_horizon_state(
            env, 5, "no_future_arrivals"
        )
        oracle_signal = alibaba_v3.apply_oracle_future_signals(base, env)
        signals = {
            dc.dc_id: {
                "price": tuple(dc.electricity_price_usd_per_mwh),
                "carbon": tuple(dc.carbon_intensity_gco2_per_kwh),
                "location": dc.location,
                "name": dc.dc_name,
            }
            for dc in oracle_signal.datacenters
        }
        return signals, tuple(base.current.network_links)
    finally:
        env.close()


def build_solver_state(
    *,
    canonical: pd.DataFrame,
    task_indices: Sequence[int],
    active: Mapping[int, Mapping[str, Any]],
    dcs: pd.DataFrame,
    step: int,
    horizon: int,
    contract: Mapping[str, Any],
    signals: Mapping[int, Mapping[str, Any]],
    links: Sequence[NetworkLinkSnapshot],
) -> HorizonState:
    anchor = pd.Timestamp(contract["time"]["anchor_timestamp_utc"])
    timestamp = anchor + pd.Timedelta(minutes=15 * step)
    tasks: list[TaskSnapshot] = []
    destinations: list[TaskDestinationSnapshot] = []
    link_cost = {(item.origin_dc_id, item.destination_dc_id): item.transmission_cost_usd_per_gb for item in links}
    for position, row_index in enumerate(task_indices):
        row = canonical.loc[row_index]
        remaining_sla_steps = int(row["sla_deadline_step"]) - step
        task = TaskSnapshot(
            task_id=str(row["task_id"]),
            original_index=position,
            origin_dc_id=int(row["origin_dc"]),
            cpu_cores=float(row["cpu_request_effective"]),
            gpu_units=float(row["gpu_request_effective"]),
            memory_gb=float(row["memory_request"]),
            duration_minutes=float(row["estimated_duration"]) / 60.0,
            remaining_duration_minutes=float(row["estimated_duration"]) / 60.0,
            arrival_time_utc=(anchor + pd.Timedelta(minutes=15 * int(row["arrival_step"]))).isoformat(),
            sla_deadline_utc=(anchor + pd.Timedelta(minutes=15 * int(row["sla_deadline_step"]))).isoformat(),
            remaining_sla_minutes=15.0 * remaining_sla_steps,
            bandwidth_gb=float(row["bandwidth"]),
            wait_intervals=max(0, step - int(row["arrival_step"])),
            was_deferred=step > int(row["arrival_step"]),
            scheduler_wait_intervals=max(0, step - int(row["arrival_step"])),
        )
        tasks.append(task)
        for dc_id in range(1, 6):
            destinations.append(
                TaskDestinationSnapshot(
                    task_id=task.task_id,
                    original_index=position,
                    destination_dc_id=dc_id,
                    transmission_cost_usd=float(link_cost[(task.origin_dc_id, dc_id)] * task.bandwidth_gb),
                    transmission_delay_seconds=0.0,
                )
            )

    used = {dc_id: np.zeros(3, dtype=float) for dc_id in range(1, 6)}
    running_count = {dc_id: 0 for dc_id in range(1, 6)}
    for item in active.values():
        dc_id = int(item["dc_id"])
        used[dc_id] += np.asarray(item["resources"], dtype=float)
        running_count[dc_id] += 1

    current_dcs: list[DataCenterSnapshot] = []
    horizon_dcs: list[HorizonDataCenterSnapshot] = []
    running: list[RunningTaskHorizonSnapshot] = []
    future_by_step = {
        future_step: canonical[canonical["arrival_step"] == step + future_step]
        for future_step in range(1, horizon)
    }
    dc_configs = native_contract_dcs(dcs)
    for item in active.values():
        running.append(
            RunningTaskHorizonSnapshot(
                task_id=str(item["task_id"]),
                dc_id=int(item["dc_id"]),
                release_step=max(0, int(item["end_step"]) - step),
                cpu_cores=float(item["resources"][0]),
                gpu_units=float(item["resources"][1]),
                memory_gb=float(item["resources"][2]),
            )
        )
    for row in dcs.sort_values("dc_id").itertuples(index=False):
        dc_id = int(row.dc_id)
        totals = np.asarray([row.total_cores, row.total_gpus, row.total_mem], dtype=float)
        available_now = np.maximum(0.0, totals - used[dc_id])
        current_dcs.append(
            DataCenterSnapshot(
                dc_id=dc_id,
                dc_name=f"DC{dc_id}",
                location=str(row.location),
                cpu_total_cores=float(totals[0]),
                cpu_available_cores=float(available_now[0]),
                cpu_reserved_cores=0.0,
                cpu_schedulable_cores=float(available_now[0]),
                cpu_available_ratio=float(available_now[0] / totals[0]),
                gpu_total_units=float(totals[1]),
                gpu_available_units=float(available_now[1]),
                gpu_reserved_units=0.0,
                gpu_schedulable_units=float(available_now[1]),
                gpu_available_ratio=float(available_now[1] / totals[1]),
                memory_total_gb=float(totals[2]),
                memory_available_gb=float(available_now[2]),
                memory_reserved_gb=0.0,
                memory_schedulable_gb=float(available_now[2]),
                memory_available_ratio=float(available_now[2] / totals[2]),
                running_task_count=running_count[dc_id],
                queued_task_count=0,
                in_transit_task_count=0,
                resource_release_times_utc=(),
                electricity_price_usd_per_mwh=float(signals[dc_id]["price"][0]),
                carbon_intensity_gco2_per_kwh=float(signals[dc_id]["carbon"][0]),
                total_power_kw=None,
                it_power_kw=None,
                cooling_power_kw=None,
                internal_temperature_c=None,
                ambient_temperature_c=None,
                crac_setpoint_c=None,
            )
        )
        available = np.zeros((horizon, 3), dtype=float)
        known = np.zeros((horizon, 3), dtype=float)
        forecast = np.zeros((horizon, 3), dtype=float)
        for offset in range(horizon):
            active_resources = np.zeros(3, dtype=float)
            for active_item in active.values():
                if int(active_item["dc_id"]) == dc_id and int(active_item["end_step"]) > step + offset:
                    active_resources += np.asarray(active_item["resources"], dtype=float)
            known[offset] = active_resources
            available[offset] = totals - active_resources
            if offset > 0 and not future_by_step[offset].empty:
                probabilities = expected_origin_probabilities(dc_configs, timestamp + pd.Timedelta(minutes=15 * offset))
                global_resources = future_by_step[offset][
                    ["cpu_request_effective", "gpu_request_effective", "memory_request"]
                ].sum().to_numpy(dtype=float)
                forecast[offset] = global_resources * probabilities[dc_id]
                available[offset] = np.maximum(0.0, available[offset] - forecast[offset])
        price = tuple(float(value) for value in signals[dc_id]["price"][:horizon])
        carbon = tuple(float(value) for value in signals[dc_id]["carbon"][:horizon])
        horizon_dcs.append(
            HorizonDataCenterSnapshot(
                dc_id=dc_id,
                dc_name=f"DC{dc_id}",
                location=str(row.location),
                cpu_total_cores=float(totals[0]),
                gpu_total_units=float(totals[1]),
                memory_total_gb=float(totals[2]),
                cpu_available_cores=tuple(available[:, 0]),
                gpu_available_units=tuple(available[:, 1]),
                memory_available_gb=tuple(available[:, 2]),
                known_cpu_reservations=tuple(known[:, 0]),
                known_gpu_reservations=tuple(known[:, 1]),
                known_memory_reservations=tuple(known[:, 2]),
                forecast_cpu_reservations=tuple(forecast[:, 0]),
                forecast_gpu_reservations=tuple(forecast[:, 1]),
                forecast_memory_reservations=tuple(forecast[:, 2]),
                electricity_price_usd_per_mwh=price,
                carbon_intensity_gco2_per_kwh=carbon,
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
        horizon=horizon,
        forecast_mode="no_future_arrivals" if horizon == 1 else "oracle",
        timestep_minutes=15.0,
        datacenters=tuple(horizon_dcs),
        running_tasks=tuple(running),
        transit_tasks=(),
        future_arrivals=(),
        information_mode="deployable" if horizon == 1 else "oracle",
        future_signal_mode="persistence" if horizon == 1 else "oracle",
    )


def dynamic_preflight(
    canonical: pd.DataFrame,
    dcs: pd.DataFrame,
    windows: pd.DataFrame,
    contract: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_steps: dict[int, str] = {}
    window_start_type: dict[int, str] = {}
    for row in windows.itertuples(index=False):
        window_start_type[int(row.start_step)] = str(row.window_type)
        for step in range(int(row.start_step), int(row.end_step_exclusive)):
            selected_steps[step] = str(row.window_type)
    maximum_step = max(selected_steps)
    arrivals_by_step = {
        int(step): list(group.index)
        for step, group in canonical[canonical["arrival_step"] <= maximum_step].groupby("arrival_step", sort=True)
    }
    capacity = {
        int(row.dc_id): np.asarray([row.total_cores, row.total_gpus, row.total_mem], dtype=float)
        for row in dcs.itertuples(index=False)
    }
    used = {dc_id: np.zeros(3, dtype=float) for dc_id in capacity}
    heaps = {dc_id: [] for dc_id in capacity}
    active: dict[int, dict[str, Any]] = {}
    pending: list[int] = []
    signals, links = signal_and_network_template()
    optimizer_config = alibaba_v3.repaired_runner._optimizer_config(
        ROOT / "configs/sustaincluster_mpc/h4_expert.yaml"
    )
    optimizer = RollingHorizonOptimizer()
    action_adapter = SustainClusterActionAdapter(
        ActionMapping(tuple((dc_id, dc_id) for dc_id in range(1, 6)), 0, 6)
    )
    step_rows: list[dict[str, Any]] = []
    start_rows: list[dict[str, Any]] = []
    hard_sla_violations = 0

    for step in range(maximum_step + 1):
        for dc_id, heap in heaps.items():
            while heap and heap[0][0] <= step:
                _, row_index = heapq.heappop(heap)
                item = active.pop(row_index)
                used[dc_id] -= np.asarray(item["resources"], dtype=float)
                used[dc_id] = np.maximum(used[dc_id], 0.0)
        new_indices = arrivals_by_step.get(step, [])
        pending.extend(new_indices)
        pending.sort(
            key=lambda index: (
                int(canonical.at[index, "arrival_step"]) + int(canonical.at[index, "max_wait_steps"]),
                int(canonical.at[index, "submit_time"]),
                int(canonical.at[index, "original_index"]),
                str(canonical.at[index, "task_id"]),
            )
        )
        preexisting = sum(
            int(item["arrival_step"] < step) for item in active.values()
        )
        preexisting_resources = np.sum(
            [np.asarray(item["resources"], dtype=float) for item in active.values() if item["arrival_step"] < step],
            axis=0,
        ) if preexisting else np.zeros(3)
        if step in window_start_type:
            start_rows.append(
                {
                    "window_type": window_start_type[step],
                    "start_step": step,
                    "restored_running_task_count": len(active),
                    "restored_pre_window_running_task_count": preexisting,
                    "restored_cpu": float(preexisting_resources[0]),
                    "restored_gpu": float(preexisting_resources[1]),
                    "restored_memory": float(preexisting_resources[2]),
                    "restored_pending_count": len(pending) - len(new_indices),
                    "empty_state_start": len(active) == 0 and len(pending) == len(new_indices),
                    "restore_method": "CONTINUOUS_PREFIX_REPLAY_FROM_TRAIN_STEP_0",
                }
            )

        solver_values: dict[str, Any] = {}
        if step in selected_steps:
            for name, horizon in (("h1", 1), ("h4", 5)):
                state = build_solver_state(
                    canonical=canonical,
                    task_indices=pending,
                    active=active,
                    dcs=dcs,
                    step=step,
                    horizon=horizon,
                    contract=contract,
                    signals=signals,
                    links=links,
                )
                result = optimizer.solve(state, optimizer_config, action_adapter)
                solver_values.update(
                    {
                        f"{name}_status": result.status,
                        f"{name}_solve_ms": 1000.0 * result.solve_seconds,
                        f"{name}_terminal_backlog": result.terminal_backlog_count,
                        f"{name}_projected_sla_violations": result.projected_sla_violations,
                        f"{name}_defer_actions": sum(action == 0 for action in result.environment_actions),
                        f"{name}_integer_variables": result.integer_variable_count,
                    }
                )

        queue_before = len(pending)
        overdue_before = sum(
            step > int(canonical.at[index, "arrival_step"]) + int(canonical.at[index, "max_wait_steps"])
            for index in pending
        )
        next_pending: list[int] = []
        for row_index in pending:
            row = canonical.loc[row_index]
            request = row[
                ["cpu_request_effective", "gpu_request_effective", "memory_request"]
            ].to_numpy(dtype=float)
            origin = int(row["origin_dc"])
            dc_order = sorted(
                capacity,
                key=lambda dc_id: (
                    0 if dc_id == origin else 1,
                    float(np.max((used[dc_id] + request) / capacity[dc_id])),
                    dc_id,
                ),
            )
            destination = next(
                (dc_id for dc_id in dc_order if np.all(used[dc_id] + request <= capacity[dc_id] + 1e-9)),
                None,
            )
            if destination is None:
                next_pending.append(row_index)
                if step >= int(row["arrival_step"]) + int(row["max_wait_steps"]):
                    hard_sla_violations += 1
                continue
            duration_steps = max(1, int(np.ceil(float(row["true_duration"]) / 900.0)))
            end_step = step + 1 + duration_steps
            item = {
                "task_id": str(row["task_id"]),
                "dc_id": destination,
                "arrival_step": int(row["arrival_step"]),
                "dispatch_step": step,
                "end_step": end_step,
                "resources": tuple(float(value) for value in request),
            }
            active[row_index] = item
            used[destination] += request
            heapq.heappush(heaps[destination], (end_step, row_index))
        pending = next_pending

        if step in selected_steps:
            step_rows.append(
                {
                    "window_type": selected_steps[step],
                    "step": step,
                    "new_task_count": len(new_indices),
                    "queue_before_dispatch": queue_before,
                    "queue_after_dispatch": len(pending),
                    "overdue_queue_before_dispatch": overdue_before,
                    "cumulative_hard_sla_violations": hard_sla_violations,
                    "running_task_count": len(active),
                    "used_cpu": float(sum(values[0] for values in used.values())),
                    "used_gpu": float(sum(values[1] for values in used.values())),
                    "used_memory": float(sum(values[2] for values in used.values())),
                    **solver_values,
                }
            )

    return pd.DataFrame(step_rows), pd.DataFrame(start_rows)


def dynamic_summary(steps: pd.DataFrame, starts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window_type, group in steps.groupby("window_type", sort=False):
        start = starts[starts["window_type"] == window_type].iloc[0]
        rows.append(
            {
                "window_type": window_type,
                "steps_checked": len(group),
                "restored_pre_window_running_task_count": int(start["restored_pre_window_running_task_count"]),
                "empty_state_start": bool(start["empty_state_start"]),
                "max_queue_before_dispatch": int(group["queue_before_dispatch"].max()),
                "max_queue_after_dispatch": int(group["queue_after_dispatch"].max()),
                "end_queue": int(group.iloc[-1]["queue_after_dispatch"]),
                "hard_sla_violations": int(group["cumulative_hard_sla_violations"].max()),
                "h1_optimal_steps": int(group["h1_status"].eq("optimal").sum()),
                "h4_optimal_steps": int(group["h4_status"].eq("optimal").sum()),
                "h1_failure_or_timeout": int(group["h1_status"].ne("optimal").sum()),
                "h4_failure_or_timeout": int(group["h4_status"].ne("optimal").sum()),
                "h1_solve_ms_p95": float(group["h1_solve_ms"].quantile(0.95)),
                "h4_solve_ms_p95": float(group["h4_solve_ms"].quantile(0.95)),
                "max_h1_projected_sla_violations": int(group["h1_projected_sla_violations"].max()),
                "max_h4_projected_sla_violations": int(group["h4_projected_sla_violations"].max()),
            }
        )
    return pd.DataFrame(rows)


def run() -> dict[str, Any]:
    v3, contract = load_frozen_contract()
    nodes = load_nodes(contract)
    assigned = deterministic_partition(nodes)
    node_summary, combinations, models, dcs = capacity_tables(nodes, assigned)
    models_dc = model_by_dc(assigned)
    native_dcs = native_contract_dcs(dcs)

    canonical, _, _, metadata = spot_v3.build_full_spot_canonical(v3)
    if len(canonical) != 466867:
        raise RuntimeError("Spot workload task count changed")
    waits = v3["scenario_b"]["max_wait_steps"]
    native_capacities = {
        "cpu": float(dcs["total_cores"].sum()),
        "gpu": float(dcs["total_gpus"].sum()),
        "memory": float(dcs["total_mem"].sum()),
    }
    old_capacities = {"cpu": 385000.0, "gpu": 2900.0, "memory": 330000.0}
    native_capacity, arrivals, arrival_summary = capacity_pressure(
        canonical, native_capacities, waits, "SPOT_NATIVE_5DC"
    )
    old_capacity, _, _ = capacity_pressure(
        canonical, old_capacities, waits, "OLD_SUSTAINCLUSTER_5DC"
    )
    comparison = pd.concat([old_capacity, native_capacity], ignore_index=True)

    single_task_feasible = np.zeros(len(canonical), dtype=bool)
    requests = canonical[
        ["cpu_request_effective", "gpu_request_effective", "memory_request"]
    ].to_numpy(float)
    for row in dcs.itertuples(index=False):
        cap = np.asarray([row.total_cores, row.total_gpus, row.total_mem], dtype=float)
        single_task_feasible |= np.all(requests <= cap + 1e-9, axis=1)
    static_feasible_rate = float(single_task_feasible.mean())

    windows = select_windows(canonical, int(metadata["train_last_arrival_step"]))
    preflight_steps, restored = dynamic_preflight(
        canonical, dcs, windows, contract
    )
    preflight_summary = dynamic_summary(preflight_steps, restored)

    native_lookup = native_capacity.set_index("resource")
    capacity_pass = bool((native_lookup["mandatory_peak_ratio"] <= 1.0).all())
    solver_pass = bool(
        preflight_summary["h1_failure_or_timeout"].sum() == 0
        and preflight_summary["h4_failure_or_timeout"].sum() == 0
    )
    queue_pass = bool(preflight_summary["end_queue"].max() == 0)
    sla_pass = bool(preflight_summary["hard_sla_violations"].max() == 0)
    gpu_row = native_lookup.loc["gpu"]
    native_too_loose = bool(
        gpu_row["offered_load_ratio"] < 0.30
        and gpu_row["mandatory_peak_ratio"] < 0.50
    )
    source_consistent = bool(
        node_summary.iloc[0]["node_count"] == len(assigned)
        and dcs["total_cores"].sum() == nodes["cpu_num"].sum()
        and dcs["total_gpus"].sum() == nodes["gpu_capacity_num"].sum()
    )
    if not source_consistent:
        final_status = "SOURCE CAPACITY INCONSISTENCY"
    elif not capacity_pass or not solver_pass or not queue_pass or not sla_pass:
        final_status = "CAPACITY CALIBRATION INCOMPLETE"
    elif native_too_loose:
        final_status = "SPOT COMPARABLE CAPACITY REQUIRED"
    else:
        final_status = "SPOT NATIVE CAPACITY READY"
    if final_status not in FINAL_STATUSES:
        raise RuntimeError("invalid final status")

    node_summary.to_csv(OUTPUT / "27_spot_node_inventory_summary.csv", index=False)
    combinations.to_csv(OUTPUT / "28_spot_node_capacity_distribution.csv", index=False)
    models.to_csv(OUTPUT / "29_spot_gpu_model_distribution.csv", index=False)
    assigned.to_parquet(OUTPUT / "30_spot_node_to_dc_assignment.parquet", index=False)
    dcs.to_csv(OUTPUT / "31_spot_native_5dc_capacity.csv", index=False)
    models_dc.to_csv(OUTPUT / "32_spot_gpu_model_by_dc.csv", index=False)
    write_text(
        OUTPUT / "33_memory_capacity_model.md",
        f"""# Spot Native 5DC Memory Capacity

- Source node_info host memory field: ABSENT.
- Mapping rule: native DC CPU capacity multiplied by Alibaba2020/SustainCluster fleet Memory/CPU ratio.
- Frozen reference ratio: 330000 / 385000 = {MEMORY_CPU_RATIO:.12f}.
- Resulting fleet memory: {native_capacities['memory']:.6f} GB.
- Provenance: `MODELED_DC_MEMORY`.
- This value is not Alibaba2026 measured host memory and must not be described as such.
""",
    )
    native_capacity.to_csv(OUTPUT / "34_spot_native_capacity_pressure.csv", index=False)
    arrivals.to_parquet(OUTPUT / "35_spot_native_arrival_pressure_by_step.parquet", index=False)
    arrival_summary.to_csv(OUTPUT / "36_spot_native_arrival_pressure_summary.csv", index=False)
    comparison.to_csv(OUTPUT / "37_old_vs_native_capacity.csv", index=False)
    windows.to_csv(OUTPUT / "38_dynamic_window_selection.csv", index=False)
    preflight_steps.to_csv(OUTPUT / "39_dynamic_preflight_steps.csv", index=False)
    preflight_summary.to_csv(OUTPUT / "40_dynamic_preflight_summary.csv", index=False)
    restored.to_csv(OUTPUT / "40a_dynamic_restore_evidence.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_status": "NOT_REQUIRED" if not native_too_loose else "REQUIRED_NEXT",
                "reason": (
                    "native GPU offered load and mandatory peak are not both below looseness thresholds"
                    if not native_too_loose
                    else "native scenario is too loose; train-only capacity scaling required"
                ),
                "gpu_offered_load_ratio": float(gpu_row["offered_load_ratio"]),
                "gpu_mandatory_peak_ratio": float(gpu_row["mandatory_peak_ratio"]),
                "loose_offered_threshold": 0.30,
                "loose_mandatory_threshold": 0.50,
                "workload_modified": False,
            }
        ]
    ).to_csv(OUTPUT / "41_comparable_capacity_decision.csv", index=False)

    old_gpu = old_capacity.set_index("resource").loc["gpu"]
    write_text(
        OUTPUT / "42_spot_capacity_calibration_report.md",
        f"""# SpotGPU2026 Capacity Calibration v1

Final status: **{final_status}**

## Source node pool

- Nodes: {len(nodes)}.
- CPU: {native_capacities['cpu']:.0f} cores.
- GPU: {native_capacities['gpu']:.0f} units.
- GPU models: {nodes['gpu_model'].nunique()}.
- Full node file was read and hashed; README aggregate numbers were not used as the measurement source.

## Capacity comparison

- Old GPU capacity: {old_capacities['gpu']:.0f}; native GPU capacity: {native_capacities['gpu']:.0f}; multiplier: {native_capacities['gpu'] / old_capacities['gpu']:.9f}x.
- Old mandatory GPU peak: {old_gpu['mandatory_concurrent_peak']:.2f} / {old_gpu['capacity']:.0f} = {old_gpu['mandatory_peak_ratio']:.6%}.
- Native mandatory GPU peak: {gpu_row['mandatory_concurrent_peak']:.2f} / {gpu_row['capacity']:.0f} = {gpu_row['mandatory_peak_ratio']:.6%}.
- Native GPU offered load: {gpu_row['offered_load_ratio']:.6%}; immediate-start peak ratio: {gpu_row['immediate_start_peak_ratio']:.6%}.
- Static single-task feasibility: {static_feasible_rate:.6%}.

## Dynamic preflight

- Selection: train-only low/medium/high GPU-arrival windows plus a >72h-task-dense window, {WINDOW_STEPS} steps each.
- State initialization: continuous deterministic prefix replay from train step 0; no empty window reset.
- H1/H4 solver failures or timeouts: {int(preflight_summary['h1_failure_or_timeout'].sum())} / {int(preflight_summary['h4_failure_or_timeout'].sum())}.
- Maximum post-dispatch queue: {int(preflight_summary['max_queue_after_dispatch'].max())}; hard SLA violations: {int(preflight_summary['hard_sla_violations'].max())}.
- Price/carbon provenance for this solvability-only check: an actual SustainCluster five-DC signal snapshot, persistence for H1 and the existing four future signal nodes for H4. No policy-quality conclusion is drawn from this proxy.

No task was removed, sampled, retimed or resized. HP max wait remains 1 step and Spot max wait remains 8 steps. GPU models remain metadata only.
""",
    )
    result = {
        "final_status": final_status,
        "native_capacity": native_capacities,
        "old_capacity": old_capacities,
        "node_count": len(nodes),
        "gpu_capacity_multiplier": native_capacities["gpu"] / old_capacities["gpu"],
        "static_feasible_rate": static_feasible_rate,
        "capacity_pass": capacity_pass,
        "solver_pass": solver_pass,
        "queue_pass": queue_pass,
        "sla_pass": sla_pass,
        "native_too_loose": native_too_loose,
        "comparable_capacity_required": native_too_loose,
        "train_last_arrival_step": int(metadata["train_last_arrival_step"]),
        "workload_task_count": len(canonical),
        "workload_modified": False,
        "transformer_used": False,
        "rl_used": False,
        "policy_training_used": False,
        "native_pressure": native_capacity.to_dict("records"),
        "dynamic_preflight": preflight_summary.to_dict("records"),
    }
    write_json(OUTPUT / "43_spot_capacity_calibration_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("use --run")
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
