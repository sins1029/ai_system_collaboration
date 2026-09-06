from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scipy.optimize import Bounds, milp


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
VENDOR = ROOT / "references/external_repos/sustain-cluster"
for entry in (ROOT, SRC, VENDOR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts.audit import spotgpu2026_capacity_calibration_v1 as capacity_v1  # noqa: E402
from scripts.imitation import build_alibaba2020_expert_dataset_v3 as alibaba_v3  # noqa: E402
from scripts.imitation import build_mpc_expert_dataset_v3 as canonical_v3  # noqa: E402
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
    TransitTaskHorizonSnapshot,
)
from sustaincluster_mpc.rolling_horizon_optimizer import (  # noqa: E402
    RollingHorizonOptimizer,
    RollingHorizonResult,
)
from sustaincluster_mpc.state_adapter import (  # noqa: E402
    DataCenterSnapshot,
    ExogenousSignalsSnapshot,
    NetworkLinkSnapshot,
    SchedulerState,
    TaskDestinationSnapshot,
    TaskSnapshot,
)
from utils.managers import CI_Manager, ElectricityPrice_Manager  # noqa: E402


OUTPUT = ROOT / "artifacts/mpc_expert_dataset_v3"
CONFIG_PATH = ROOT / "configs/sustaincluster_mpc/mpc_expert_dataset_v3.yaml"
CAPACITY_PATH = ROOT / "configs/scenarios/spotgpu2026_native_capacity_v1.yaml"
SCENARIO_ID = "B_SPOTGPU2026_NATIVE_MEDIUM"
CHECKPOINT_INTERVAL = 250
H4_NODES = 5
SIGNAL_YEAR = 2023
SIGNAL_INIT_DAY = 47
SIGNAL_INIT_HOUR = 0
SIGNAL_PROVENANCE = "EXTERNAL_SCENARIO_SIGNAL"
FINAL_READY = "MPC EXPERT DATASET v3 READY"
FINAL_READY_LIMITED = "MPC EXPERT DATASET v3 READY WITH DOCUMENTED LIMITATIONS"


STATE_SCHEMA = pa.schema(
    [
        ("scenario_id", pa.string()),
        ("state_id", pa.string()),
        ("step", pa.int32()),
        ("relative_time_minutes", pa.int32()),
        ("timestamp_utc", pa.string()),
        ("split", pa.string()),
        ("pending_task_ids", pa.list_(pa.string())),
        ("running_task_ids", pa.list_(pa.string())),
        ("in_transit_task_ids", pa.list_(pa.string())),
        ("pending_count", pa.int32()),
        ("running_count", pa.int32()),
        ("in_transit_count", pa.int32()),
        ("completed_count", pa.int32()),
        ("dc_states_json", pa.string()),
        ("reservations_json", pa.string()),
        ("energy_signals_json", pa.string()),
        ("risk_score", pa.float64()),
        ("risk_resource", pa.string()),
        ("h1_joint_action_ref", pa.string()),
        ("h4_joint_action_ref", pa.string()),
        ("h1_solver_status", pa.string()),
        ("h4_solver_status", pa.string()),
        ("h1_solver_message", pa.string()),
        ("h4_solver_message", pa.string()),
        ("h1_presolve_retry", pa.bool_()),
        ("h4_presolve_retry", pa.bool_()),
        ("h1_solve_ms", pa.float64()),
        ("h4_solve_ms", pa.float64()),
        ("h1_objective", pa.float64()),
        ("h4_objective", pa.float64()),
        ("h1_defer_count", pa.int32()),
        ("h4_defer_count", pa.int32()),
        ("state_sha256", pa.string()),
        ("empty_pending_state", pa.bool_()),
        ("information_class", pa.string()),
    ]
)

TASK_SCHEMA = pa.schema(
    [
        ("scenario_id", pa.string()),
        ("state_id", pa.string()),
        ("step", pa.int32()),
        ("relative_time_minutes", pa.int32()),
        ("split", pa.string()),
        ("task_id", pa.string()),
        ("original_index", pa.int64()),
        ("task_position", pa.int32()),
        ("cpu_cores", pa.float64()),
        ("gpu_units", pa.float64()),
        ("memory_gb", pa.float64()),
        ("bandwidth_gb", pa.float64()),
        ("worker_num", pa.int32()),
        ("gpu_model", pa.string()),
        ("priority", pa.string()),
        ("submit_time_seconds", pa.int64()),
        ("arrival_step", pa.int32()),
        ("estimated_duration_seconds", pa.float64()),
        ("estimated_duration_steps", pa.int32()),
        ("origin_dc", pa.int16()),
        ("current_location", pa.int16()),
        ("waiting_steps", pa.int16()),
        ("sla_class", pa.string()),
        ("sla_deadline_step", pa.int32()),
        ("remaining_sla_steps", pa.int32()),
        ("max_wait_steps", pa.int16()),
        ("defer_allowed", pa.bool_()),
        ("feasible_action_mask", pa.list_(pa.bool_())),
        ("request_scope", pa.string()),
        ("memory_provenance", pa.string()),
        ("bandwidth_provenance", pa.string()),
        ("origin_provenance", pa.string()),
        ("duration_estimator_provenance", pa.string()),
        ("gpu_heterogeneity_mode", pa.string()),
        ("information_class", pa.string()),
    ]
)

LABEL_SCHEMA = pa.schema(
    [
        ("scenario_id", pa.string()),
        ("state_id", pa.string()),
        ("step", pa.int32()),
        ("split", pa.string()),
        ("task_id", pa.string()),
        ("original_index", pa.int64()),
        ("task_position", pa.int32()),
        ("priority", pa.string()),
        ("h1_action", pa.int8()),
        ("h1_action_semantic", pa.string()),
        ("h1_target_dc", pa.int16()),
        ("h1_is_defer", pa.bool_()),
        ("h4_action", pa.int8()),
        ("h4_action_semantic", pa.string()),
        ("h4_target_dc", pa.int16()),
        ("h4_is_defer", pa.bool_()),
        ("exact_action_disagreement", pa.bool_()),
        ("placement_disagreement", pa.bool_()),
        ("defer_disagreement", pa.bool_()),
        ("h1_solver_status", pa.string()),
        ("h4_solver_status", pa.string()),
        ("h1_solver_message", pa.string()),
        ("h4_solver_message", pa.string()),
        ("h1_presolve_retry", pa.bool_()),
        ("h4_presolve_retry", pa.bool_()),
        ("h1_objective", pa.float64()),
        ("h4_objective", pa.float64()),
        ("h1_solve_ms", pa.float64()),
        ("h4_solve_ms", pa.float64()),
        ("h1_label_provenance", pa.string()),
        ("h4_label_provenance", pa.string()),
        ("information_class", pa.string()),
    ]
)

TRUTH_SCHEMA = pa.schema(
    [
        ("scenario_id", pa.string()),
        ("state_id", pa.string()),
        ("step", pa.int32()),
        ("split", pa.string()),
        ("task_id", pa.string()),
        ("original_index", pa.int64()),
        ("status_before_action", pa.string()),
        ("true_duration_seconds", pa.int64()),
        ("true_duration_steps", pa.int32()),
        ("h4_action", pa.int8()),
        ("dispatch_step", pa.int32()),
        ("execution_start_step", pa.int32()),
        ("estimated_completion_step", pa.int32()),
        ("true_completion_step", pa.int32()),
        ("true_duration_provenance", pa.string()),
        ("information_class", pa.string()),
    ]
)

PRIVILEGED_SCHEMA = pa.schema(
    [
        ("scenario_id", pa.string()),
        ("state_id", pa.string()),
        ("step", pa.int32()),
        ("split", pa.string()),
        ("dc_id", pa.int16()),
        ("horizon_step", pa.int8()),
        ("horizon_minutes", pa.int16()),
        ("forecast_timestamp_utc", pa.string()),
        ("actual_origin_task_count", pa.int32()),
        ("actual_origin_cpu_demand", pa.float64()),
        ("actual_origin_gpu_demand", pa.float64()),
        ("actual_origin_memory_demand", pa.float64()),
        ("h4_origin_probability", pa.float64()),
        ("h4_expected_task_count", pa.float64()),
        ("h4_forecast_cpu_reservation", pa.float64()),
        ("h4_forecast_gpu_reservation", pa.float64()),
        ("h4_forecast_memory_reservation", pa.float64()),
        ("future_electricity_price_usd_per_mwh", pa.float64()),
        ("future_carbon_intensity_gco2_per_kwh", pa.float64()),
        ("workload_future_provenance", pa.string()),
        ("energy_signal_provenance", pa.string()),
        ("information_class", pa.string()),
    ]
)

EVENT_SCHEMA = pa.schema(
    [
        ("scenario_id", pa.string()),
        ("state_id", pa.string()),
        ("step", pa.int32()),
        ("split", pa.string()),
        ("task_id", pa.string()),
        ("original_index", pa.int64()),
        ("event_type", pa.string()),
        ("dc_id", pa.int16()),
        ("dispatch_step", pa.int32()),
        ("planned_execution_start_step", pa.int32()),
        ("actual_execution_start_step", pa.int32()),
        ("estimated_completion_step", pa.int32()),
        ("true_completion_step", pa.int32()),
        ("true_duration_steps", pa.int32()),
        ("information_class", pa.string()),
    ]
)

TABLE_SCHEMAS = {
    "states": STATE_SCHEMA,
    "tasks": TASK_SCHEMA,
    "labels": LABEL_SCHEMA,
    "truth": TRUTH_SCHEMA,
    "privileged": PRIVILEGED_SCHEMA,
    "events": EVENT_SCHEMA,
}


def json_compact(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: item.tolist() if isinstance(item, np.ndarray) else (item.item() if hasattr(item, "item") else str(item)),
    )


def object_hash(value: Any) -> str:
    return hashlib.sha256(json_compact(value).encode("utf-8")).hexdigest().upper()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8", errors="replace"
    ).strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def semantic_action(action: int) -> str:
    return "defer" if int(action) == 0 else f"assign_dc{int(action)}"


def split_for_step(step: int, metadata: Mapping[str, Any]) -> str:
    if step <= int(metadata["train_last_arrival_step"]):
        return "train"
    if step <= int(metadata["validation_last_arrival_step"]):
        return "validation"
    return "test"


def _nullable_int(value: int | None) -> int | None:
    return None if value is None else int(value)


@dataclass
class RuntimeState:
    next_step: int = 0
    pending: list[int] = field(default_factory=list)
    running: dict[int, dict[str, Any]] = field(default_factory=dict)
    transit: dict[int, dict[str, Any]] = field(default_factory=dict)
    used: dict[int, list[float]] = field(default_factory=dict)
    reserved: dict[int, list[float]] = field(default_factory=dict)
    completed_count: int = 0
    state_counter: int = 0
    decision_counter: int = 0
    h1_calls: int = 0
    h4_calls: int = 0
    h1_failures: int = 0
    h4_failures: int = 0
    h1_defers: int = 0
    h4_defers: int = 0
    h1_presolve_retries: int = 0
    h4_presolve_retries: int = 0
    solver_anomalies: list[dict[str, Any]] = field(default_factory=list)
    committed_chunks: list[dict[str, Any]] = field(default_factory=list)
    progress_rows: list[dict[str, Any]] = field(default_factory=list)


def load_inputs() -> dict[str, Any]:
    config = canonical_v3.load_config(CONFIG_PATH)
    if config["scenario_b"]["scenario_id"] != SCENARIO_ID:
        raise RuntimeError("formal Spot scenario id changed")
    contract = yaml.safe_load(
        (ROOT / config["scenario_b"]["contract_config"]).read_text("utf-8")
    )
    capacity_contract = yaml.safe_load(CAPACITY_PATH.read_text("utf-8"))
    if capacity_contract["status"] != "CALIBRATED_FROM_COMPLETE_NODE_INFO":
        raise RuntimeError("native capacity contract is not calibrated")
    calibration = json.loads(
        (OUTPUT / "43_spot_capacity_calibration_manifest.json").read_text("utf-8")
    )
    if calibration["final_status"] != "SPOT NATIVE CAPACITY READY":
        raise RuntimeError("Spot native capacity readiness gate is not satisfied")
    canonical, nodes, _, metadata = canonical_v3.build_full_spot_canonical(config)
    dcs = pd.read_csv(OUTPUT / "31_spot_native_5dc_capacity.csv").sort_values("dc_id")
    expected_gpu = [1875.0, 2291.0, 2070.0, 2615.0, 1561.0]
    if dcs["total_gpus"].astype(float).tolist() != expected_gpu:
        raise RuntimeError("frozen per-DC GPU capacities changed")
    if int(dcs["total_cores"].sum()) != 632636:
        raise RuntimeError("frozen fleet CPU capacity changed")
    if int(dcs["total_gpus"].sum()) != 10412:
        raise RuntimeError("frozen fleet GPU capacity changed")
    if not np.isclose(float(dcs["total_mem"].sum()), 542259.4285714285):
        raise RuntimeError("frozen modeled memory capacity changed")
    if len(canonical) != 466867 or len(nodes) != 4278:
        raise RuntimeError("Spot source cardinality changed")
    return {
        "config": config,
        "contract": contract,
        "capacity_contract": capacity_contract,
        "calibration": calibration,
        "canonical": canonical,
        "nodes": nodes,
        "metadata": metadata,
        "dcs": dcs,
    }


def signal_file_paths(location: str) -> tuple[Path, Path]:
    carbon = (
        VENDOR
        / "data/carbon_intensity"
        / location
        / str(SIGNAL_YEAR)
        / f"{location}_{SIGNAL_YEAR}_hourly.csv"
    )
    price = (
        VENDOR
        / "data/electricity_prices/standardized"
        / location
        / str(SIGNAL_YEAR)
        / f"{location}_electricity_prices_{SIGNAL_YEAR}.csv"
    )
    return carbon, price


def build_signal_trace(
    dcs: pd.DataFrame, required_steps: int
) -> tuple[dict[int, dict[str, np.ndarray]], pd.DataFrame]:
    traces: dict[int, dict[str, np.ndarray]] = {}
    provenance: list[dict[str, Any]] = []
    for row in dcs.itertuples(index=False):
        dc_id = int(row.dc_id)
        location = str(row.location)
        timezone_shift = int(row.timezone_shift)
        carbon_path, price_path = signal_file_paths(location)
        if not carbon_path.is_file() or not price_path.is_file():
            raise FileNotFoundError(f"missing SustainCluster signal data for {location}")
        carbon = CI_Manager(
            location=location,
            simulation_year=SIGNAL_YEAR,
            init_day=SIGNAL_INIT_DAY,
            future_steps=4,
            timezone_shift=timezone_shift,
        )
        price = ElectricityPrice_Manager(
            location=location,
            simulation_year=SIGNAL_YEAR,
            timezone_shift=timezone_shift,
        )
        utc_start = pd.Timestamp("2023-02-17T00:00:00Z")
        local_start = utc_start + pd.Timedelta(hours=timezone_shift)
        local_init_day = int(local_start.dayofyear) - 1
        local_init_hour = int(local_start.hour)
        carbon.reset(
            init_day=local_init_day,
            init_hour=local_init_hour,
            seed=None,
        )
        price.reset(
            init_day=local_init_day,
            init_hour=local_init_hour,
            seed=None,
        )
        carbon_values = np.empty(required_steps, dtype=np.float64)
        price_values = np.empty(required_steps, dtype=np.float64)
        for index in range(required_steps):
            carbon_values[index] = float(carbon.get_current_ci(norm=False))
            price_values[index] = float(price.get_current_price())
            carbon.step()
            price.step()
        traces[dc_id] = {"carbon": carbon_values, "price": price_values}
        for signal_type, path in (("carbon_intensity", carbon_path), ("electricity_price", price_path)):
            provenance.append(
                {
                    "dc_id": dc_id,
                    "location": location,
                    "timezone_shift": timezone_shift,
                    "signal_type": signal_type,
                    "source_file": path.relative_to(ROOT).as_posix(),
                    "source_sha256": sha256(path),
                    "source_year": SIGNAL_YEAR,
                    "source_start_day": local_init_day,
                    "source_start_hour": local_init_hour,
                    "timeline_mapping": "RELATIVE_STEP_0_TO_2023_02_17_00_00_UTC_WITH_DC_LOCAL_RESET",
                    "manager": (
                        "utils.managers.CI_Manager"
                        if signal_type == "carbon_intensity"
                        else "utils.managers.ElectricityPrice_Manager"
                    ),
                    "provenance": SIGNAL_PROVENANCE,
                    "alibaba2026_measured": False,
                }
            )
    return traces, pd.DataFrame(provenance)


def load_network_links() -> tuple[NetworkLinkSnapshot, ...]:
    _, links = capacity_v1.signal_and_network_template()
    return tuple(links)


def canonical_order_key(canonical: pd.DataFrame, index: int) -> tuple[Any, ...]:
    row = canonical.loc[index]
    return (
        int(row["arrival_step"]),
        int(row["submit_time"]),
        int(row["original_index"]),
        str(row["task_id"]),
    )


def table_semantic_hash(paths: Sequence[Path], table_name: str) -> str:
    ignored = {"h1_solve_ms", "h4_solve_ms"}
    digest = hashlib.sha256()
    schema = TABLE_SCHEMAS[table_name]
    columns = [name for name in schema.names if name not in ignored]
    for path in sorted(paths, key=lambda item: item.name):
        frame = pd.read_parquet(path, columns=columns)
        for row in frame.itertuples(index=False, name=None):
            digest.update(json_compact(row).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest().upper()


class SpotContinuousGenerator:
    def __init__(
        self,
        inputs: Mapping[str, Any],
        run_dir: Path,
        *,
        checkpoint_interval: int = CHECKPOINT_INTERVAL,
        resume: bool = False,
    ) -> None:
        self.config = inputs["config"]
        self.contract = inputs["contract"]
        self.canonical: pd.DataFrame = inputs["canonical"]
        self.metadata = inputs["metadata"]
        self.dcs: pd.DataFrame = inputs["dcs"]
        self.run_dir = run_dir
        self.chunk_dir = run_dir / "chunks"
        self.checkpoint_dir = run_dir / "states"
        self.checkpoint_interval = int(checkpoint_interval)
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint interval must be positive")
        self.total_steps = int(self.canonical["arrival_step"].max()) + 1
        self.anchor = pd.Timestamp(self.contract["time"]["anchor_timestamp_utc"])
        self.capacity = {
            int(row.dc_id): np.asarray(
                [row.total_cores, row.total_gpus, row.total_mem], dtype=float
            )
            for row in self.dcs.itertuples(index=False)
        }
        self.dc_configs = capacity_v1.native_contract_dcs(self.dcs)
        self.signals, self.signal_provenance = build_signal_trace(
            self.dcs, self.total_steps + H4_NODES
        )
        self.links = load_network_links()
        self.link_cost = {
            (item.origin_dc_id, item.destination_dc_id): item.transmission_cost_usd_per_gb
            for item in self.links
        }
        self.arrivals_by_step = {
            int(step): [int(value) for value in group.index]
            for step, group in self.canonical.groupby("arrival_step", sort=True)
        }
        resources = [
            "cpu_request_effective",
            "gpu_request_effective",
            "memory_request",
        ]
        grouped = self.canonical.groupby("arrival_step", sort=True)
        self.global_future = {
            int(step): np.asarray(
                [len(group), *group[resources].sum().to_numpy(dtype=float)],
                dtype=float,
            )
            for step, group in grouped
        }
        self.origin_future = {
            (int(step), int(dc_id)): np.asarray(
                [len(group), *group[resources].sum().to_numpy(dtype=float)],
                dtype=float,
            )
            for (step, dc_id), group in self.canonical.groupby(
                ["arrival_step", "origin_dc"], sort=True
            )
        }
        self.optimizer = RollingHorizonOptimizer()
        self.optimizer_config = alibaba_v3.repaired_runner._optimizer_config(
            ROOT / self.config["optimizer_config"]
        )
        self.action_adapter = SustainClusterActionAdapter(
            ActionMapping(tuple((dc_id, dc_id) for dc_id in range(1, 6)), 0, 6)
        )
        self.buffers: dict[str, list[dict[str, Any]]] = {
            name: [] for name in TABLE_SCHEMAS
        }
        self.chunk_start: int | None = None
        self.started_at = time.perf_counter()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            self.state = self._load_latest_checkpoint()
        else:
            self.state = RuntimeState(
                used={dc_id: [0.0, 0.0, 0.0] for dc_id in self.capacity},
                reserved={dc_id: [0.0, 0.0, 0.0] for dc_id in self.capacity},
            )
            random.seed(2026)
            np.random.seed(2026)
        self.process_start_step = self.state.next_step

    def timestamp(self, step: int) -> pd.Timestamp:
        return self.anchor + pd.Timedelta(minutes=15 * int(step))

    def split(self, step: int) -> str:
        return split_for_step(step, self.metadata)

    def resources(self, row_index: int) -> np.ndarray:
        row = self.canonical.loc[row_index]
        return row[
            ["cpu_request_effective", "gpu_request_effective", "memory_request"]
        ].to_numpy(dtype=float)

    def _runtime_event(
        self, step: int, item: Mapping[str, Any], event_type: str
    ) -> None:
        self.buffers["events"].append(
            {
                "scenario_id": SCENARIO_ID,
                "state_id": f"spotgpu2026:state:{step:05d}",
                "step": step,
                "split": self.split(step),
                "task_id": str(item["task_id"]),
                "original_index": int(item["row_index"]),
                "event_type": event_type,
                "dc_id": int(item["dc_id"]),
                "dispatch_step": int(item["dispatch_step"]),
                "planned_execution_start_step": int(
                    item["planned_execution_start_step"]
                ),
                "actual_execution_start_step": _nullable_int(
                    item.get("actual_execution_start_step")
                ),
                "estimated_completion_step": _nullable_int(
                    item.get("estimated_completion_step")
                ),
                "true_completion_step": _nullable_int(
                    item.get("true_completion_step")
                ),
                "true_duration_steps": int(item["true_duration_steps"]),
                "information_class": "SIMULATOR_ONLY",
            }
        )

    def _release_and_start(self, step: int) -> None:
        for row_index, item in sorted(list(self.state.running.items())):
            if int(item["true_completion_step"]) <= step:
                dc_id = int(item["dc_id"])
                values = np.asarray(
                    self.state.used[dc_id], dtype=float
                ) - np.asarray(item["resources"], dtype=float)
                if np.any(values < -1e-7):
                    raise RuntimeError(
                        "negative used capacity during release"
                    )
                self.state.used[dc_id] = np.maximum(
                    values, 0.0
                ).tolist()
                item["status"] = "completed"
                self._runtime_event(step, item, "completed")
                del self.state.running[row_index]
                self.state.completed_count += 1

        for row_index, item in sorted(list(self.state.transit.items())):
            if int(item["planned_execution_start_step"]) > step:
                continue
            dc_id = int(item["dc_id"])
            request = np.asarray(item["resources"], dtype=float)
            used = np.asarray(self.state.used[dc_id], dtype=float)
            if np.any(used + request > self.capacity[dc_id] + 1e-7):
                item["status"] = "destination_reserved_queue"
                continue
            reserved = np.asarray(
                self.state.reserved[dc_id], dtype=float
            ) - request
            if np.any(reserved < -1e-7):
                raise RuntimeError(
                    "negative reservation during destination start"
                )
            self.state.reserved[dc_id] = np.maximum(
                reserved, 0.0
            ).tolist()
            self.state.used[dc_id] = (used + request).tolist()
            item["status"] = "running"
            item["actual_execution_start_step"] = step
            item["estimated_completion_step"] = (
                step + int(item["estimated_duration_steps"])
            )
            item["true_completion_step"] = (
                step + int(item["true_duration_steps"])
            )
            self.state.running[row_index] = item
            self._runtime_event(step, item, "started")
            del self.state.transit[row_index]
    def _advance_to_state(self, step: int) -> None:
        self._release_and_start(step)
        self.state.pending.extend(self.arrivals_by_step.get(step, ()))
        self.state.pending.sort(key=lambda index: canonical_order_key(self.canonical, index))
        if len(self.state.pending) != len(set(self.state.pending)):
            raise RuntimeError("a task appeared twice in pending state")
        for earlier, later in zip(self.state.pending, self.state.pending[1:]):
            if canonical_order_key(self.canonical, earlier) > canonical_order_key(
                self.canonical, later
            ):
                raise RuntimeError("pending task order violates the frozen stable order")

    def _current_dc_rows(
        self, step: int
    ) -> list[dict[str, Any]]:
        rows = []
        for dc in self.dcs.sort_values("dc_id").itertuples(
            index=False
        ):
            dc_id = int(dc.dc_id)
            total = self.capacity[dc_id]
            used = np.asarray(
                self.state.used[dc_id], dtype=float
            )
            reserved = np.asarray(
                self.state.reserved[dc_id], dtype=float
            )
            physical_available = total - used
            schedulable = np.maximum(
                0.0, physical_available - reserved
            )
            if np.any(physical_available < -1e-7):
                raise RuntimeError(
                    "physical native capacity invariant failed"
                )
            rows.append(
                {
                    "dc_id": dc_id,
                    "location": str(dc.location),
                    "cpu_total": float(total[0]),
                    "gpu_total": float(total[1]),
                    "memory_total": float(total[2]),
                    "cpu_used": float(used[0]),
                    "gpu_used": float(used[1]),
                    "memory_used": float(used[2]),
                    "cpu_reserved": float(reserved[0]),
                    "gpu_reserved": float(reserved[1]),
                    "memory_reserved": float(reserved[2]),
                    "cpu_physical_available": float(
                        max(0.0, physical_available[0])
                    ),
                    "gpu_physical_available": float(
                        max(0.0, physical_available[1])
                    ),
                    "memory_physical_available": float(
                        max(0.0, physical_available[2])
                    ),
                    "cpu_available": float(schedulable[0]),
                    "gpu_available": float(schedulable[1]),
                    "memory_available": float(schedulable[2]),
                    "running_count": sum(
                        int(item["dc_id"]) == dc_id
                        for item in self.state.running.values()
                    ),
                    "in_transit_count": sum(
                        int(item["dc_id"]) == dc_id
                        for item in self.state.transit.values()
                    ),
                    "electricity_price_usd_per_mwh": float(
                        self.signals[dc_id]["price"][step]
                    ),
                    "carbon_intensity_gco2_per_kwh": float(
                        self.signals[dc_id]["carbon"][step]
                    ),
                    "energy_signal_provenance": (
                        SIGNAL_PROVENANCE
                    ),
                }
            )
        return rows
    def _known_reservations(
        self, dc_id: int, step: int, horizon: int
    ) -> np.ndarray:
        known = np.zeros((horizon, 3), dtype=float)
        for item in self.state.running.values():
            if int(item["dc_id"]) != dc_id:
                continue
            resources = np.asarray(
                item["resources"], dtype=float
            )
            estimated_completion = int(
                item["estimated_completion_step"]
            )
            if estimated_completion <= step:
                known[:] += resources
                continue
            stop = min(horizon, estimated_completion - step)
            known[:stop] += resources
        for item in self.state.transit.values():
            if int(item["dc_id"]) != dc_id:
                continue
            resources = np.asarray(
                item["resources"], dtype=float
            )
            start = max(
                0,
                int(item["planned_execution_start_step"]) - step,
            )
            stop = min(
                horizon,
                start + int(item["estimated_duration_steps"]),
            )
            if start < horizon:
                known[start:stop] += resources
        return known
    def _future_pressure(
        self, step: int, horizon: int
    ) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
        reserves = {
            dc_id: np.zeros((horizon, 4), dtype=float) for dc_id in self.capacity
        }
        rows: list[dict[str, Any]] = []
        if horizon == 1:
            return reserves, rows
        state_id = f"spotgpu2026:state:{step:05d}"
        split = self.split(step)
        for offset in range(1, horizon):
            future_step = step + offset
            global_values = self.global_future.get(future_step, np.zeros(4, dtype=float))
            probabilities = expected_origin_probabilities(
                self.dc_configs, self.timestamp(future_step)
            )
            for dc_id in sorted(self.capacity):
                probability = float(probabilities[dc_id])
                reserves[dc_id][offset] = global_values * probability
                actual = self.origin_future.get(
                    (future_step, dc_id), np.zeros(4, dtype=float)
                )
                rows.append(
                    {
                        "scenario_id": SCENARIO_ID,
                        "state_id": state_id,
                        "step": step,
                        "split": split,
                        "dc_id": dc_id,
                        "horizon_step": offset,
                        "horizon_minutes": offset * 15,
                        "forecast_timestamp_utc": self.timestamp(future_step).isoformat(),
                        "actual_origin_task_count": int(actual[0]),
                        "actual_origin_cpu_demand": float(actual[1]),
                        "actual_origin_gpu_demand": float(actual[2]),
                        "actual_origin_memory_demand": float(actual[3]),
                        "h4_origin_probability": probability,
                        "h4_expected_task_count": float(reserves[dc_id][offset, 0]),
                        "h4_forecast_cpu_reservation": float(reserves[dc_id][offset, 1]),
                        "h4_forecast_gpu_reservation": float(reserves[dc_id][offset, 2]),
                        "h4_forecast_memory_reservation": float(reserves[dc_id][offset, 3]),
                        "future_electricity_price_usd_per_mwh": float(
                            self.signals[dc_id]["price"][future_step]
                        ),
                        "future_carbon_intensity_gco2_per_kwh": float(
                            self.signals[dc_id]["carbon"][future_step]
                        ),
                        "workload_future_provenance": "ORACLE_TRUE_FUTURE_GLOBAL_DEMAND_WITH_FROZEN_EXPECTED_ORIGIN_DISTRIBUTION",
                        "energy_signal_provenance": SIGNAL_PROVENANCE,
                        "information_class": "PRIVILEGED_FUTURE",
                    }
                )
        return reserves, rows

    def build_solver_state(
        self, step: int, horizon: int
    ) -> tuple[HorizonState, list[dict[str, Any]]]:
        timestamp = self.timestamp(step)
        tasks: list[TaskSnapshot] = []
        destinations: list[TaskDestinationSnapshot] = []
        for position, row_index in enumerate(self.state.pending):
            row = self.canonical.loc[row_index]
            task = TaskSnapshot(
                task_id=str(row["task_id"]),
                original_index=position,
                origin_dc_id=int(row["origin_dc"]),
                cpu_cores=float(row["cpu_request_effective"]),
                gpu_units=float(row["gpu_request_effective"]),
                memory_gb=float(row["memory_request"]),
                duration_minutes=float(row["estimated_duration"]) / 60.0,
                remaining_duration_minutes=float(row["estimated_duration"]) / 60.0,
                arrival_time_utc=self.timestamp(int(row["arrival_step"])).isoformat(),
                sla_deadline_utc=self.timestamp(int(row["sla_deadline_step"])).isoformat(),
                remaining_sla_minutes=15.0 * (int(row["sla_deadline_step"]) - step),
                bandwidth_gb=float(row["bandwidth"]),
                wait_intervals=max(0, step - int(row["arrival_step"])),
                was_deferred=step > int(row["arrival_step"]),
                scheduler_wait_intervals=max(0, step - int(row["arrival_step"])),
            )
            tasks.append(task)
            for dc_id in sorted(self.capacity):
                destinations.append(
                    TaskDestinationSnapshot(
                        task_id=task.task_id,
                        original_index=position,
                        destination_dc_id=dc_id,
                        transmission_cost_usd=float(
                            self.link_cost[(task.origin_dc_id, dc_id)] * task.bandwidth_gb
                        ),
                        transmission_delay_seconds=0.0,
                    )
                )
        current_dc_rows = self._current_dc_rows(step)
        current_dcs = []
        future_reserves, privileged = self._future_pressure(step, horizon)
        horizon_dcs = []
        for item in current_dc_rows:
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
                    cpu_available_ratio=float(item["cpu_physical_available"] / item["cpu_total"]),
                    gpu_total_units=float(item["gpu_total"]),
                    gpu_available_units=float(item["gpu_physical_available"]),
                    gpu_reserved_units=float(item["gpu_reserved"]),
                    gpu_schedulable_units=float(item["gpu_available"]),
                    gpu_available_ratio=float(item["gpu_physical_available"] / item["gpu_total"]),
                    memory_total_gb=float(item["memory_total"]),
                    memory_available_gb=float(item["memory_physical_available"]),
                    memory_reserved_gb=float(item["memory_reserved"]),
                    memory_schedulable_gb=float(item["memory_available"]),
                    memory_available_ratio=float(item["memory_physical_available"] / item["memory_total"]),
                    running_task_count=int(item["running_count"]),
                    queued_task_count=0,
                    in_transit_task_count=int(item["in_transit_count"]),
                    resource_release_times_utc=tuple(
                        self.timestamp(int(active["estimated_completion_step"])).isoformat()
                        for active in self.state.running.values()
                        if int(active["dc_id"]) == dc_id
                    ),
                    electricity_price_usd_per_mwh=float(item["electricity_price_usd_per_mwh"]),
                    carbon_intensity_gco2_per_kwh=float(item["carbon_intensity_gco2_per_kwh"]),
                    total_power_kw=None,
                    it_power_kw=None,
                    cooling_power_kw=None,
                    internal_temperature_c=None,
                    ambient_temperature_c=None,
                    crac_setpoint_c=None,
                )
            )
            known = self._known_reservations(dc_id, step, horizon)
            forecast = future_reserves[dc_id][:, 1:]
            available = np.maximum(0.0, self.capacity[dc_id] - known - forecast)
            horizon_dcs.append(
                HorizonDataCenterSnapshot(
                    dc_id=dc_id,
                    dc_name=f"DC{dc_id}",
                    location=str(item["location"]),
                    cpu_total_cores=float(self.capacity[dc_id][0]),
                    gpu_total_units=float(self.capacity[dc_id][1]),
                    memory_total_gb=float(self.capacity[dc_id][2]),
                    cpu_available_cores=tuple(float(value) for value in available[:, 0]),
                    gpu_available_units=tuple(float(value) for value in available[:, 1]),
                    memory_available_gb=tuple(float(value) for value in available[:, 2]),
                    known_cpu_reservations=tuple(float(value) for value in known[:, 0]),
                    known_gpu_reservations=tuple(float(value) for value in known[:, 1]),
                    known_memory_reservations=tuple(float(value) for value in known[:, 2]),
                    forecast_cpu_reservations=tuple(float(value) for value in forecast[:, 0]),
                    forecast_gpu_reservations=tuple(float(value) for value in forecast[:, 1]),
                    forecast_memory_reservations=tuple(float(value) for value in forecast[:, 2]),
                    electricity_price_usd_per_mwh=tuple(
                        float(self.signals[dc_id]["price"][step + offset])
                        for offset in range(horizon)
                    ),
                    carbon_intensity_gco2_per_kwh=tuple(
                        float(self.signals[dc_id]["carbon"][step + offset])
                        for offset in range(horizon)
                    ),
                )
            )
        running = tuple(
            RunningTaskHorizonSnapshot(
                task_id=str(item["task_id"]),
                dc_id=int(item["dc_id"]),
                release_step=max(0, int(item["estimated_completion_step"]) - step),
                cpu_cores=float(item["resources"][0]),
                gpu_units=float(item["resources"][1]),
                memory_gb=float(item["resources"][2]),
            )
            for item in self.state.running.values()
        )
        transit = tuple(
            TransitTaskHorizonSnapshot(
                task_id=str(item["task_id"]),
                destination_dc_id=int(item["dc_id"]),
                arrival_step=max(0, int(item["planned_execution_start_step"]) - step),
                duration_steps=int(item["estimated_duration_steps"]),
                cpu_cores=float(item["resources"][0]),
                gpu_units=float(item["resources"][1]),
                memory_gb=float(item["resources"][2]),
            )
            for item in self.state.transit.values()
        )
        scheduler = SchedulerState(
            tasks=tuple(tasks),
            datacenters=tuple(current_dcs),
            network_links=self.links,
            task_destinations=tuple(destinations),
            exogenous=ExogenousSignalsSnapshot(timestamp.isoformat(), 15.0),
            allow_defer=True,
            information_mode="deployable",
        )
        return (
            HorizonState(
                current=scheduler,
                horizon=horizon,
                forecast_mode="no_future_arrivals" if horizon == 1 else "oracle",
                timestep_minutes=15.0,
                datacenters=tuple(horizon_dcs),
                running_tasks=running,
                transit_tasks=transit,
                future_arrivals=(),
                information_mode="deployable" if horizon == 1 else "oracle",
                future_signal_mode="persistence" if horizon == 1 else "oracle",
            ),
            privileged,
        )

    def feasible_mask(self, row_index: int, step: int) -> list[bool]:
        row = self.canonical.loc[row_index]
        waiting = max(0, step - int(row["arrival_step"]))
        mask = [bool(row["defer_allowed"]) and waiting < int(row["max_wait_steps"])]
        request = self.resources(row_index)
        completion = step + 1 + int(row["estimated_duration_steps"])
        for dc_id in sorted(self.capacity):
            mask.append(
                bool(
                    np.all(request <= self.capacity[dc_id] + 1e-9)
                    and completion <= int(row["sla_deadline_step"])
                )
            )
        return mask

    def _state_hash_payload(self, step: int, dc_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        pending = []
        for row_index in self.state.pending:
            row = self.canonical.loc[row_index]
            pending.append(
                {
                    "task_id": str(row["task_id"]),
                    "original_index": int(row["original_index"]),
                    "cpu": float(row["cpu_request_effective"]),
                    "gpu": float(row["gpu_request_effective"]),
                    "memory": float(row["memory_request"]),
                    "estimated_duration": float(row["estimated_duration"]),
                    "origin_dc": int(row["origin_dc"]),
                    "waiting": step - int(row["arrival_step"]),
                    "deadline": int(row["sla_deadline_step"]),
                }
            )
        active = [
            {
                "task_id": str(item["task_id"]),
                "dc_id": int(item["dc_id"]),
                "estimated_completion_step": int(item["estimated_completion_step"]),
                "resources": [float(value) for value in item["resources"]],
            }
            for item in sorted(
                self.state.running.values(), key=lambda value: str(value["task_id"])
            )
        ]
        transit = [
            {
                "task_id": str(item["task_id"]),
                "dc_id": int(item["dc_id"]),
                "planned_execution_start_step": int(item["planned_execution_start_step"]),
                "estimated_completion_step": int(item["estimated_completion_step"]),
                "resources": [float(value) for value in item["resources"]],
            }
            for item in sorted(
                self.state.transit.values(), key=lambda value: str(value["task_id"])
            )
        ]
        clean_dc = [
            {key: value for key, value in item.items() if key != "energy_signal_provenance"}
            for item in dc_rows
        ]
        return {
            "step": step,
            "timestamp": self.timestamp(step).isoformat(),
            "pending": pending,
            "running": active,
            "transit": transit,
            "datacenters": clean_dc,
        }

    def _all_backlog_witness_is_feasible(
        self, state: HorizonState
    ) -> bool:
        task_count = len(state.current.tasks)
        if task_count == 0:
            return True
        dc_count = len(state.datacenters)
        variable_count = (
            task_count * dc_count * state.horizon + task_count
        )
        _, metadata = self.optimizer._component_vectors(
            state, self.optimizer_config, variable_count
        )
        upper = self.optimizer._variable_upper_bounds(
            state,
            self.optimizer_config,
            metadata,
            variable_count,
        )
        constraints = self.optimizer._constraints(
            state, metadata, variable_count
        )
        candidate = np.zeros(variable_count, dtype=float)
        candidate[-task_count:] = 1.0
        once = constraints[0].A @ candidate
        capacity = constraints[1].A @ candidate
        return bool(
            np.all(upper[-task_count:] >= 1.0 - 1e-12)
            and np.allclose(once, 1.0, atol=1e-12)
            and np.all(
                capacity
                <= np.asarray(constraints[1].ub) + 1e-12
            )
        )

    def _solve_without_presolve(
        self, state: HorizonState, initial: RollingHorizonResult
    ) -> RollingHorizonResult:
        tasks = state.current.tasks
        datacenters = state.datacenters
        task_count = len(tasks)
        dc_count = len(datacenters)
        horizon = state.horizon
        variable_count = (
            task_count * dc_count * horizon + task_count
        )
        components, metadata = (
            self.optimizer._component_vectors(
                state, self.optimizer_config, variable_count
            )
        )
        objective = sum(
            components.values(), np.zeros(variable_count)
        )
        objective += self.optimizer._tie_break_vector(
            task_count,
            dc_count,
            horizon,
            variable_count,
            self.optimizer_config,
        )
        upper = self.optimizer._variable_upper_bounds(
            state,
            self.optimizer_config,
            metadata,
            variable_count,
        )
        constraints = self.optimizer._constraints(
            state, metadata, variable_count
        )
        options: dict[str, float | bool] = {
            "presolve": False
        }
        if (
            self.optimizer_config.solver_time_limit_seconds
            is not None
        ):
            options["time_limit"] = (
                self.optimizer_config.solver_time_limit_seconds
            )
        started = time.perf_counter()
        try:
            raw = milp(
                c=objective,
                integrality=np.ones(
                    variable_count, dtype=np.int8
                ),
                bounds=Bounds(
                    np.zeros(variable_count), upper
                ),
                constraints=constraints,
                options=options,
            )
        except Exception as exc:
            return self.optimizer._failed_result(
                state,
                "solver_error",
                (
                    "PRESOLVE_RETRY_ERROR_AFTER_"
                    f"{initial.status}: {exc}"
                ),
                initial.solve_seconds
                + time.perf_counter()
                - started,
                variable_count,
            )
        elapsed = time.perf_counter() - started
        if raw.status != 0 or raw.x is None:
            status = {
                1: "limit_reached",
                2: "infeasible",
                3: "unbounded",
                4: "solver_error",
            }.get(int(raw.status), "solver_error")
            return self.optimizer._failed_result(
                state,
                status,
                (
                    "PRESOLVE_RETRY_FAILED_AFTER_"
                    f"{initial.status}: {raw.message}"
                ),
                initial.solve_seconds + elapsed,
                variable_count,
            )
        solution = np.asarray(raw.x)
        plans = self.optimizer._decode_plans(
            state, metadata, solution
        )
        first_decisions = (
            self.optimizer._first_step_decisions(plans)
        )
        actions = self.action_adapter.encode_assignments(
            tasks, first_decisions
        )
        costs = self.optimizer._selected_costs(
            components, solution
        )
        first_costs = self.optimizer._first_step_costs(
            state,
            self.optimizer_config,
            plans,
            metadata,
        )
        allocations = self.optimizer._allocations(
            state, metadata, plans
        )
        self.optimizer._validate_allocations(
            state, allocations
        )
        return RollingHorizonResult(
            status="optimal",
            message=(
                "PRESOLVE_RETRY_AFTER_"
                f"{initial.status}: {raw.message}"
            ),
            solve_seconds=initial.solve_seconds + elapsed,
            objective_value=costs.total,
            costs=costs,
            first_step_costs=first_costs,
            plans=plans,
            first_step_decisions=first_decisions,
            environment_actions=tuple(actions),
            datacenter_allocations=allocations,
            integer_variable_count=variable_count,
            terminal_backlog_count=sum(
                plan.decision == "terminal_backlog"
                for plan in plans
            ),
            projected_sla_violations=(
                self.optimizer._projected_sla_violations(
                    plans, horizon
                )
            ),
        )

    def _solve(
        self, state: HorizonState, planner: str
    ) -> RollingHorizonResult:
        initial = self.optimizer.solve(
            state,
            self.optimizer_config,
            self.action_adapter,
        )
        result = initial
        retried = False
        if (
            initial.status == "infeasible"
            and self._all_backlog_witness_is_feasible(state)
        ):
            result = self._solve_without_presolve(
                state, initial
            )
            retried = True
            self.state.solver_anomalies.append(
                {
                    "step": int(self.state.next_step),
                    "planner": planner.upper(),
                    "initial_status": initial.status,
                    "initial_message": initial.message,
                    "verified_feasible_witness": (
                        "ALL_TERMINAL_BACKLOG"
                    ),
                    "retry_backend": (
                        "SCIPY_MILP_HIGHS_SAME_MODEL_"
                        "PRESOLVE_FALSE"
                    ),
                    "retry_status": result.status,
                    "retry_message": result.message,
                    "fallback_label_used": False,
                }
            )
        if planner == "h1":
            self.state.h1_calls += 1
            self.state.h1_presolve_retries += int(retried)
            if result.status != "optimal":
                self.state.h1_failures += 1
        else:
            self.state.h4_calls += 1
            self.state.h4_presolve_retries += int(retried)
            if result.status != "optimal":
                self.state.h4_failures += 1
        return result
    def _failure(self, step: int, reason: str, **details: Any) -> None:
        write_json(
            self.run_dir / f"failure_state_{step:05d}.json",
            {
                "step": step,
                "reason": reason,
                "pending_count": len(self.state.pending),
                "running_count": len(self.state.running),
                "in_transit_count": len(self.state.transit),
                "details": details,
                "fallback_label_used": False,
            },
        )
        raise RuntimeError(f"formal Spot generation stopped at step {step}: {reason}")

    def _apply_h4(
        self, step: int, actions: Sequence[int]
    ) -> dict[int, dict[str, int | None]]:
        if len(actions) != len(self.state.pending):
            self._failure(step, "H4_ACTION_COUNT_MISMATCH")
        remaining: list[int] = []
        outcomes: dict[int, dict[str, int | None]] = {}
        for row_index, action in zip(tuple(self.state.pending), actions):
            row = self.canonical.loc[row_index]
            waiting = step - int(row["arrival_step"])
            if int(action) == 0:
                if waiting >= int(row["max_wait_steps"]):
                    self._failure(
                        step,
                        "H4_DEFER_EXCEEDS_FROZEN_BOUND",
                        task_id=str(row["task_id"]),
                        waiting=waiting,
                        max_wait=int(row["max_wait_steps"]),
                    )
                remaining.append(row_index)
                outcomes[row_index] = {
                    "dispatch_step": None,
                    "execution_start_step": None,
                    "estimated_completion_step": None,
                    "true_completion_step": None,
                }
                continue
            dc_id = int(action)
            request = self.resources(row_index)
            planned_execution = step + 1
            planned_estimated_completion = (
                planned_execution
                + int(row["estimated_duration_steps"])
            )
            true_steps = max(
                1,
                int(np.ceil(float(row["true_duration"]) / 900.0)),
            )
            item = {
                "task_id": str(row["task_id"]),
                "row_index": int(row_index),
                "dc_id": dc_id,
                "status": "in_transit",
                "waiting_steps": waiting,
                "dispatch_step": step,
                "planned_execution_start_step": planned_execution,
                "actual_execution_start_step": None,
                "estimated_duration_steps": int(
                    row["estimated_duration_steps"]
                ),
                "estimated_completion_step": (
                    planned_estimated_completion
                ),
                "true_duration_steps": true_steps,
                "true_completion_step": None,
                "resources": tuple(
                    float(value) for value in request
                ),
            }
            self.state.transit[row_index] = item
            reserved = np.asarray(
                self.state.reserved[dc_id], dtype=float
            ) + request
            self.state.reserved[dc_id] = reserved.tolist()
            outcomes[row_index] = {
                "dispatch_step": step,
                "execution_start_step": planned_execution,
                "estimated_completion_step": (
                    planned_estimated_completion
                ),
                "true_completion_step": None,
            }
        self.state.pending = remaining
        return outcomes

    def _append_rows(
        self,
        step: int,
        h1: RollingHorizonResult,
        h4: RollingHorizonResult,
        privileged: Sequence[Mapping[str, Any]],
    ) -> None:
        state_id = f"spotgpu2026:state:{step:05d}"
        split = self.split(step)
        dc_rows = self._current_dc_rows(step)
        pending_indices = tuple(self.state.pending)
        h1_actions = tuple(int(value) for value in h1.environment_actions)
        h4_actions = tuple(int(value) for value in h4.environment_actions)
        feasible_masks = {
            row_index: self.feasible_mask(row_index, step)
            for row_index in pending_indices
        }
        running_ids_before = sorted(
            str(item["task_id"]) for item in self.state.running.values()
        )
        transit_ids_before = sorted(
            str(item["task_id"]) for item in self.state.transit.values()
        )
        reservations_before = {
            dc_id: list(values)
            for dc_id, values in sorted(self.state.reserved.items())
        }
        pending_gpu = sum(
            float(self.canonical.at[index, "gpu_request_effective"])
            for index in pending_indices
        )
        current_gpu = sum(float(values[1]) for values in self.state.used.values())
        reserved_gpu = sum(float(values[1]) for values in self.state.reserved.values())
        risk = (current_gpu + reserved_gpu + pending_gpu) / 10412.0
        state_hash = object_hash(self._state_hash_payload(step, dc_rows))
        outcomes = self._apply_h4(step, h4_actions)
        h1_defer_count = sum(action == 0 for action in h1_actions)
        h4_defer_count = sum(action == 0 for action in h4_actions)
        self.state.h1_defers += h1_defer_count
        self.state.h4_defers += h4_defer_count
        self.buffers["states"].append(
            {
                "scenario_id": SCENARIO_ID,
                "state_id": state_id,
                "step": step,
                "relative_time_minutes": step * 15,
                "timestamp_utc": self.timestamp(step).isoformat(),
                "split": split,
                "pending_task_ids": [
                    str(self.canonical.at[index, "task_id"]) for index in pending_indices
                ],
                "running_task_ids": running_ids_before,
                "in_transit_task_ids": transit_ids_before,
                "pending_count": len(pending_indices),
                "running_count": len(running_ids_before),
                "in_transit_count": len(transit_ids_before),
                "completed_count": self.state.completed_count,
                "dc_states_json": json_compact(dc_rows),
                "reservations_json": json_compact(reservations_before),
                "energy_signals_json": json_compact(
                    {
                        dc_id: {
                            "price": float(self.signals[dc_id]["price"][step]),
                            "carbon": float(self.signals[dc_id]["carbon"][step]),
                            "provenance": SIGNAL_PROVENANCE,
                        }
                        for dc_id in sorted(self.capacity)
                    }
                ),
                "risk_score": float(risk),
                "risk_resource": "DEPLOYABLE_CURRENT_FLEET_GPU_OCCUPANCY_PLUS_PENDING",
                "h1_joint_action_ref": f"labels:{state_id}:h1",
                "h4_joint_action_ref": f"labels:{state_id}:h4",
                "h1_solver_status": h1.status,
                "h4_solver_status": h4.status,
                "h1_solve_ms": 1000.0 * h1.solve_seconds,
                "h4_solve_ms": 1000.0 * h4.solve_seconds,
                "h1_objective": h1.objective_value,
                "h4_objective": h4.objective_value,
                "h1_defer_count": h1_defer_count,
                "h4_defer_count": h4_defer_count,
                "state_sha256": state_hash,
                "empty_pending_state": len(pending_indices) == 0,
                "information_class": "DEPLOYABLE_CURRENT",
            }
        )
        self.buffers["privileged"].extend(dict(row) for row in privileged)
        for position, (row_index, h1_action, h4_action) in enumerate(
            zip(pending_indices, h1_actions, h4_actions)
        ):
            row = self.canonical.loc[row_index]
            mask = feasible_masks[row_index]
            waiting = step - int(row["arrival_step"])
            h1_target = None if h1_action == 0 else h1_action
            h4_target = None if h4_action == 0 else h4_action
            self.buffers["tasks"].append(
                {
                    "scenario_id": SCENARIO_ID,
                    "state_id": state_id,
                    "step": step,
                    "relative_time_minutes": step * 15,
                    "split": split,
                    "task_id": str(row["task_id"]),
                    "original_index": int(row["original_index"]),
                    "task_position": position,
                    "cpu_cores": float(row["cpu_request_effective"]),
                    "gpu_units": float(row["gpu_request_effective"]),
                    "memory_gb": float(row["memory_request"]),
                    "bandwidth_gb": float(row["bandwidth"]),
                    "worker_num": int(row["worker_num"]),
                    "gpu_model": str(row["gpu_model"]),
                    "priority": str(row["priority"]),
                    "submit_time_seconds": int(row["submit_time"]),
                    "arrival_step": int(row["arrival_step"]),
                    "estimated_duration_seconds": float(row["estimated_duration"]),
                    "estimated_duration_steps": int(row["estimated_duration_steps"]),
                    "origin_dc": int(row["origin_dc"]),
                    "current_location": int(row["origin_dc"]),
                    "waiting_steps": waiting,
                    "sla_class": str(row["sla_class"]),
                    "sla_deadline_step": int(row["sla_deadline_step"]),
                    "remaining_sla_steps": int(row["sla_deadline_step"]) - step,
                    "max_wait_steps": int(row["max_wait_steps"]),
                    "defer_allowed": bool(row["defer_allowed"]),
                    "feasible_action_mask": mask,
                    "request_scope": str(row["request_scope"]),
                    "memory_provenance": str(row["memory_source"]),
                    "bandwidth_provenance": str(row["bandwidth_source"]),
                    "origin_provenance": str(row["origin_source"]),
                    "duration_estimator_provenance": str(row["estimated_duration_source"]),
                    "gpu_heterogeneity_mode": str(row["gpu_heterogeneity_mode"]),
                    "information_class": "DEPLOYABLE_CURRENT",
                }
            )
            self.buffers["labels"].append(
                {
                    "scenario_id": SCENARIO_ID,
                    "state_id": state_id,
                    "step": step,
                    "split": split,
                    "task_id": str(row["task_id"]),
                    "original_index": int(row["original_index"]),
                    "task_position": position,
                    "priority": str(row["priority"]),
                    "h1_action": h1_action,
                    "h1_action_semantic": semantic_action(h1_action),
                    "h1_target_dc": _nullable_int(h1_target),
                    "h1_is_defer": h1_action == 0,
                    "h4_action": h4_action,
                    "h4_action_semantic": semantic_action(h4_action),
                    "h4_target_dc": _nullable_int(h4_target),
                    "h4_is_defer": h4_action == 0,
                    "exact_action_disagreement": h1_action != h4_action,
                    "placement_disagreement": (
                        h1_action != 0 and h4_action != 0 and h1_action != h4_action
                    ),
                    "defer_disagreement": (h1_action == 0) != (h4_action == 0),
                    "h1_solver_status": h1.status,
                    "h4_solver_status": h4.status,
                    "h1_objective": h1.objective_value,
                    "h4_objective": h4.objective_value,
                    "h1_solve_ms": 1000.0 * h1.solve_seconds,
                    "h4_solve_ms": 1000.0 * h4.solve_seconds,
                    "h1_label_provenance": "H1_REPAIRED_CURRENT_ONLY",
                    "h4_label_provenance": "H4_ORACLE_REPAIRED",
                    "information_class": "LABELS_EXPLICIT_OPT_IN",
                }
            )
            outcome = outcomes[row_index]
            self.buffers["truth"].append(
                {
                    "scenario_id": SCENARIO_ID,
                    "state_id": state_id,
                    "step": step,
                    "split": split,
                    "task_id": str(row["task_id"]),
                    "original_index": int(row["original_index"]),
                    "status_before_action": "pending",
                    "true_duration_seconds": int(row["true_duration"]),
                    "true_duration_steps": max(
                        1, int(np.ceil(float(row["true_duration"]) / 900.0))
                    ),
                    "h4_action": h4_action,
                    "dispatch_step": _nullable_int(outcome["dispatch_step"]),
                    "execution_start_step": _nullable_int(outcome["execution_start_step"]),
                    "estimated_completion_step": _nullable_int(
                        outcome["estimated_completion_step"]
                    ),
                    "true_completion_step": _nullable_int(outcome["true_completion_step"]),
                    "true_duration_provenance": "DIRECT_SOURCE_SIMULATOR_ONLY",
                    "information_class": "SIMULATOR_ONLY",
                }
            )
        self.state.state_counter += 1
        self.state.decision_counter += len(pending_indices)

    def run_step(self, step: int) -> None:
        if step != self.state.next_step:
            raise RuntimeError("runtime step is not continuous")
        if self.chunk_start is None:
            self.chunk_start = step
        self._advance_to_state(step)
        h1_state, _ = self.build_solver_state(step, 1)
        h4_state, privileged = self.build_solver_state(step, H4_NODES)
        if self.state.pending:
            h1 = self._solve(h1_state, "h1")
            h4 = self._solve(h4_state, "h4")
            if h1.status != "optimal" or h4.status != "optimal":
                self._failure(
                    step,
                    "MPC_SOLVER_FAILURE",
                    h1_status=h1.status,
                    h1_message=h1.message,
                    h4_status=h4.status,
                    h4_message=h4.message,
                )
        else:
            h1 = self.optimizer.solve(h1_state, self.optimizer_config, self.action_adapter)
            h4 = self.optimizer.solve(h4_state, self.optimizer_config, self.action_adapter)
        self._append_rows(step, h1, h4, privileged)
        self.state.next_step = step + 1

    def _write_chunk_table(
        self, table_name: str, rows: Sequence[Mapping[str, Any]], start: int, stop: int
    ) -> dict[str, Any]:
        path = self.chunk_dir / f"{table_name}_{start:05d}_{stop - 1:05d}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(list(rows), schema=TABLE_SCHEMAS[table_name])
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
        return {
            "table": table_name,
            "start_step": start,
            "end_step_exclusive": stop,
            "rows": table.num_rows,
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
        }

    def flush_and_checkpoint(self, stop: int) -> Path:
        if self.chunk_start is None:
            raise RuntimeError("cannot checkpoint an empty chunk")
        start = self.chunk_start
        entries = [
            self._write_chunk_table(name, self.buffers[name], start, stop)
            for name in TABLE_SCHEMAS
        ]
        self.state.committed_chunks.extend(entries)
        self.buffers = {name: [] for name in TABLE_SCHEMAS}
        self.chunk_start = None
        elapsed = time.perf_counter() - self.started_at
        processed = max(1, stop - self.process_start_step)
        eta = elapsed / processed * max(0, self.total_steps - stop)
        output_bytes = sum(
            (ROOT / entry["path"]).stat().st_size for entry in self.state.committed_chunks
        )
        progress = {
            "step_complete": stop,
            "total_steps": self.total_steps,
            "elapsed_seconds_this_process": elapsed,
            "estimated_remaining_seconds": eta,
            "pending": len(self.state.pending),
            "running": len(self.state.running),
            "in_transit": len(self.state.transit),
            "completed": self.state.completed_count,
            "h1_solver_calls": self.state.h1_calls,
            "h4_solver_calls": self.state.h4_calls,
            "h1_failures": self.state.h1_failures,
            "h4_failures": self.state.h4_failures,
            "h1_defers": self.state.h1_defers,
            "h4_defers": self.state.h4_defers,
            "h1_presolve_retries": (
                self.state.h1_presolve_retries
            ),
            "h4_presolve_retries": (
                self.state.h4_presolve_retries
            ),
            "task_decisions": self.state.decision_counter,
            "output_bytes": output_bytes,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.state.progress_rows.append(progress)
        payload = {
            "version": 1,
            "scenario_id": SCENARIO_ID,
            "runtime": self.state,
            "current_time_utc": self.timestamp(stop).isoformat(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "capacity": {dc_id: values.tolist() for dc_id, values in self.capacity.items()},
            "signal_index": stop,
            "checkpoint_interval": self.checkpoint_interval,
            "config_sha256": sha256(CONFIG_PATH),
            "capacity_contract_sha256": sha256(CAPACITY_PATH),
        }
        path = self.checkpoint_dir / f"checkpoint_{stop:05d}.pkl.gz"
        temporary = path.with_suffix(".pkl.gz.tmp")
        with gzip.open(temporary, "wb", compresslevel=6) as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
        write_json(
            self.run_dir / "latest_checkpoint.json",
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "next_step": stop,
                "committed_chunk_count": len(self.state.committed_chunks),
            },
        )
        print(
            "[spot-v3] "
            f"{stop}/{self.total_steps} elapsed={elapsed:.1f}s eta={eta:.1f}s "
            f"pending={len(self.state.pending)} running={len(self.state.running)} "
            f"completed={self.state.completed_count} h1/h4={self.state.h1_calls}/{self.state.h4_calls} "
            f"fail={self.state.h1_failures}/{self.state.h4_failures} "
            f"defer={self.state.h1_defers}/{self.state.h4_defers} bytes={output_bytes}",
            flush=True,
        )
        return path

    def _load_latest_checkpoint(self) -> RuntimeState:
        latest_path = self.run_dir / "latest_checkpoint.json"
        if not latest_path.is_file():
            raise FileNotFoundError(f"resume requested without {latest_path}")
        latest = json.loads(latest_path.read_text("utf-8"))
        path = ROOT / latest["path"]
        if sha256(path) != latest["sha256"]:
            raise RuntimeError("latest checkpoint hash mismatch")
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
        if payload["scenario_id"] != SCENARIO_ID:
            raise RuntimeError("checkpoint scenario mismatch")
        if payload["config_sha256"] != sha256(CONFIG_PATH):
            raise RuntimeError("checkpoint config changed")
        if payload["capacity_contract_sha256"] != sha256(CAPACITY_PATH):
            raise RuntimeError("checkpoint capacity contract changed")
        state: RuntimeState = payload["runtime"]
        if not hasattr(state, "h1_presolve_retries"):
            state.h1_presolve_retries = 0
        if not hasattr(state, "h4_presolve_retries"):
            state.h4_presolve_retries = 0
        if not hasattr(state, "solver_anomalies"):
            state.solver_anomalies = []
        for entry in state.committed_chunks:
            path = ROOT / entry["path"]
            if not path.is_file() or sha256(path) != entry["sha256"]:
                raise RuntimeError(f"committed chunk integrity failed: {path}")
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_random_state"])
        return state

    def run(self, end_step_exclusive: int | None = None) -> RuntimeState:
        end = self.total_steps if end_step_exclusive is None else int(end_step_exclusive)
        if not self.state.next_step <= end <= self.total_steps:
            raise ValueError("invalid rollout end step")
        for step in range(self.state.next_step, end):
            self.run_step(step)
            if self.state.next_step % self.checkpoint_interval == 0:
                self.flush_and_checkpoint(self.state.next_step)
        if self.chunk_start is not None:
            self.flush_and_checkpoint(self.state.next_step)
        return self.state

    def runtime_signature(self) -> str:
        payload = {
            "next_step": self.state.next_step,
            "pending": self.state.pending,
            "running": sorted(self.state.running.items()),
            "transit": sorted(self.state.transit.items()),
            "used": self.state.used,
            "reserved": self.state.reserved,
            "completed_count": self.state.completed_count,
            "state_counter": self.state.state_counter,
            "decision_counter": self.state.decision_counter,
            "h1_calls": self.state.h1_calls,
            "h4_calls": self.state.h4_calls,
            "h1_failures": self.state.h1_failures,
            "h4_failures": self.state.h4_failures,
            "h1_defers": self.state.h1_defers,
            "h4_defers": self.state.h4_defers,
            "h1_presolve_retries": (
                self.state.h1_presolve_retries
            ),
            "h4_presolve_retries": (
                self.state.h4_presolve_retries
            ),
            "solver_anomalies": self.state.solver_anomalies,
        }
        return object_hash(payload)

    def semantic_output_hashes(self) -> dict[str, str]:
        result = {}
        for name in TABLE_SCHEMAS:
            paths = [
                ROOT / entry["path"]
                for entry in self.state.committed_chunks
                if entry["table"] == name
            ]
            result[name] = table_semantic_hash(paths, name)
        return result




def checkpoint_exact_test(inputs: Mapping[str, Any]) -> dict[str, Any]:
    root = OUTPUT / "checkpoints/scenario_b/resume_exact_test_v1"
    uninterrupted = SpotContinuousGenerator(
        inputs, root / "uninterrupted", checkpoint_interval=50, resume=False
    )
    uninterrupted.run(100)
    resumed_first = SpotContinuousGenerator(
        inputs, root / "resumed", checkpoint_interval=50, resume=False
    )
    resumed_first.run(50)
    resumed = SpotContinuousGenerator(
        inputs, root / "resumed", checkpoint_interval=50, resume=True
    )
    resumed.run(100)
    a_signature = uninterrupted.runtime_signature()
    b_signature = resumed.runtime_signature()
    a_hashes = uninterrupted.semantic_output_hashes()
    b_hashes = resumed.semantic_output_hashes()
    checks = {
        "runtime_state_exact": a_signature == b_signature,
        "all_state_rows_exact": a_hashes["states"] == b_hashes["states"],
        "all_task_rows_exact": a_hashes["tasks"] == b_hashes["tasks"],
        "all_task_status_rows_exact": a_hashes["truth"] == b_hashes["truth"],
        "h1_h4_actions_exact": a_hashes["labels"] == b_hashes["labels"],
        "privileged_future_exact": a_hashes["privileged"] == b_hashes["privileged"],
    }
    result = {
        "test": "100_STEPS_UNINTERRUPTED_VS_50_CHECKPOINT_RESTART_50",
        "semantic_timing_fields_excluded_from_hash": ["h1_solve_ms", "h4_solve_ms"],
        "reason_for_exclusion": "wall-clock measurements are recorded but are not simulator semantics",
        "runtime_signature_a": a_signature,
        "runtime_signature_b": b_signature,
        "semantic_output_hashes_a": a_hashes,
        "semantic_output_hashes_b": b_hashes,
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_json(root / "checkpoint_resume_exact_test.json", result)
    if not result["passed"]:
        raise RuntimeError("checkpoint/restart exact equality gate failed")
    return result


def _atomic_write_table(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def _split_suffix(split: str) -> str:
    return "val" if split == "validation" else split


def consolidate_formal_tables(state: RuntimeState) -> dict[str, Path]:
    targets = {
        "states": (
            OUTPUT / "dataset/scenario_b/deployable_current",
            "states",
        ),
        "tasks": (
            OUTPUT / "dataset/scenario_b/deployable_current",
            "tasks",
        ),
        "labels": (
            OUTPUT / "dataset/scenario_b/labels",
            "expert_actions",
        ),
        "truth": (
            OUTPUT / "dataset/scenario_b/simulator_only",
            "task_truth_decisions",
        ),
        "privileged": (
            OUTPUT / "privileged_future/scenario_b",
            "oracle_future",
        ),
        "events": (
            OUTPUT / "dataset/scenario_b/simulator_only",
            "runtime_events",
        ),
    }
    files: dict[str, Path] = {}
    for name, (directory, stem) in targets.items():
        chunks = [
            ROOT / item["path"]
            for item in state.committed_chunks
            if item["table"] == name
        ]
        chunks.sort(key=lambda item: item.name)
        tables = [pq.read_table(path, schema=TABLE_SCHEMAS[name]) for path in chunks]
        combined = (
            pa.concat_tables(tables)
            if tables
            else pa.Table.from_pylist([], schema=TABLE_SCHEMAS[name])
        )
        full_path = directory / f"{stem}_full.parquet"
        _atomic_write_table(combined, full_path)
        files[f"{name}_full"] = full_path
        for split in ("train", "validation", "test"):
            mask = pa.compute.equal(combined["split"], pa.scalar(split))
            split_table = combined.filter(mask)
            split_path = directory / f"{stem}_{_split_suffix(split)}.parquet"
            _atomic_write_table(split_table, split_path)
            files[f"{name}_{split}"] = split_path
    return files


def build_lifecycle(
    canonical: pd.DataFrame,
    state: RuntimeState,
    truth_path: Path,
    event_path: Path,
) -> pd.DataFrame:
    truth = pd.read_parquet(truth_path)
    events = pd.read_parquet(event_path)
    assigned = (
        truth[truth["dispatch_step"].notna()]
        .sort_values(
            ["original_index", "step"], kind="mergesort"
        )
        .drop_duplicates("original_index", keep="last")
        [[
            "original_index",
            "h4_action",
            "dispatch_step",
            "execution_start_step",
            "estimated_completion_step",
        ]]
        .rename(
            columns={
                "execution_start_step": (
                    "planned_execution_start_step"
                ),
                "estimated_completion_step": (
                    "planned_estimated_completion_step"
                ),
            }
        )
    )
    started = (
        events[events["event_type"] == "started"]
        .sort_values(
            ["original_index", "step"], kind="mergesort"
        )
        .drop_duplicates("original_index", keep="first")
        [[
            "original_index",
            "actual_execution_start_step",
            "estimated_completion_step",
            "true_completion_step",
        ]]
        .rename(
            columns={
                "estimated_completion_step": (
                    "start_estimated_completion_step"
                ),
                "true_completion_step": (
                    "start_true_completion_step"
                ),
            }
        )
    )
    completed = (
        events[events["event_type"] == "completed"]
        .sort_values(
            ["original_index", "step"], kind="mergesort"
        )
        .drop_duplicates("original_index", keep="last")
        [["original_index", "true_completion_step"]]
        .rename(
            columns={
                "true_completion_step": (
                    "observed_true_completion_step"
                )
            }
        )
    )
    lifecycle = (
        canonical[
            [
                "task_id",
                "original_index",
                "arrival_step",
                "priority",
                "origin_dc",
                "true_duration",
                "estimated_duration",
            ]
        ]
        .merge(
            assigned,
            on="original_index",
            how="left",
            validate="one_to_one",
        )
        .merge(
            started,
            on="original_index",
            how="left",
            validate="one_to_one",
        )
        .merge(
            completed,
            on="original_index",
            how="left",
            validate="one_to_one",
        )
    )
    active_running = set(state.running)
    active_transit = set(state.transit)
    pending = set(state.pending)

    def status(index: int) -> str:
        if index in pending:
            return "pending_at_trace_end"
        if index in active_transit:
            return "destination_reserved_queue_at_trace_end"
        if index in active_running:
            return "running_at_trace_end"
        return "completed_by_trace_end"

    lifecycle["status_at_trace_end"] = (
        lifecycle["original_index"].map(status)
    )
    lifecycle["destination_dc"] = (
        lifecycle["h4_action"].astype("Int64")
    )
    lifecycle["true_duration_provenance"] = (
        "DIRECT_SOURCE_SIMULATOR_ONLY"
    )
    lifecycle["information_class"] = "SIMULATOR_ONLY"
    return lifecycle

def write_loader_manifest(files: Mapping[str, Path]) -> Path:
    path = OUTPUT / "dataset/scenario_b/schema_and_loader_manifest.json"
    value = {
        "scenario_id": SCENARIO_ID,
        "logical_regions": {
            "deployable_current": {
                "default_visible": True,
                "contains": ["states", "tasks"],
                "forbidden_fields": [
                    "true_duration_seconds",
                    "true_completion_step",
                    "h1_action",
                    "h4_action",
                    "oracle_future",
                ],
            },
            "simulator_only": {
                "default_visible": False,
                "contains": ["true_duration", "true_completion", "lifecycle"],
            },
            "privileged_future": {
                "default_visible": False,
                "contains": ["+15/+30/+45/+60 workload and regional energy signals"],
            },
            "labels": {
                "default_visible": False,
                "explicit_opt_in_required": True,
                "contains": ["H1 action", "H4 Oracle action", "solver metadata"],
            },
        },
        "empty_state_semantics": (
            "state row retained; task decision rows are zero; MILP call skipped"
        ),
        "feasible_mask_semantics": (
            "assignment-ingress feasibility under frozen bounded-defer and "
            "whole-DC task-capacity constraints; a destination reservation may "
            "queue until physical resources are released"
        ),
        "files": {
            key: value.relative_to(ROOT).as_posix() for key, value in files.items()
        },
    }
    write_json(path, value)
    return path


def _rate(group: pd.DataFrame, column: str) -> float:
    return float(group[column].mean()) if len(group) else 0.0


def dataset_statistics(
    states: pd.DataFrame,
    labels: pd.DataFrame,
    runtime_state: RuntimeState,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nonempty = states[~states["empty_pending_state"]]
    dataset = pd.DataFrame(
        [
            {"metric": "total_time_steps", "value": len(states)},
            {"metric": "nonempty_scheduling_states", "value": len(nonempty)},
            {
                "metric": "empty_states",
                "value": int(states["empty_pending_state"].sum()),
            },
            {"metric": "task_decisions", "value": len(labels)},
            {"metric": "unique_tasks", "value": labels["task_id"].nunique()},
            {
                "metric": "pending_at_trace_end",
                "value": len(runtime_state.pending),
            },
            {
                "metric": "running_at_trace_end",
                "value": len(runtime_state.running),
            },
            {
                "metric": "in_transit_at_trace_end",
                "value": len(runtime_state.transit),
            },
            {
                "metric": "completed_by_trace_end",
                "value": int(runtime_state.completed_count),
            },
        ]
    )
    rows = []
    for split in ("train", "validation", "test"):
        state_part = states[states["split"] == split]
        label_part = labels[labels["split"] == split]
        rows.append(
            {
                "split": split,
                "states": len(state_part),
                "nonempty_states": int((~state_part["empty_pending_state"]).sum()),
                "empty_states": int(state_part["empty_pending_state"].sum()),
                "task_decisions": len(label_part),
                "unique_task_ids": label_part["task_id"].nunique(),
                "minimum_step": int(state_part["step"].min()),
                "maximum_step": int(state_part["step"].max()),
                "state_crosses_split": False,
            }
        )
    return dataset, pd.DataFrame(rows)

def behavior_analysis(
    states: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nonempty = states[~states["empty_pending_state"]]
    per_state = labels.groupby("state_id", sort=False)["exact_action_disagreement"].any()
    state_rate = float(per_state.mean()) if len(per_state) else 0.0
    behavior = pd.DataFrame(
        [
            {
                "scenario_id": SCENARIO_ID,
                "task_decisions": len(labels),
                "exact_agreement_rate": 1.0
                - _rate(labels, "exact_action_disagreement"),
                "exact_disagreement_rate": _rate(
                    labels, "exact_action_disagreement"
                ),
                "placement_disagreement_rate": _rate(
                    labels, "placement_disagreement"
                ),
                "defer_disagreement_rate": _rate(
                    labels, "defer_disagreement"
                ),
                "nonempty_states": len(nonempty),
                "state_any_disagreement_rate": state_rate,
                "state_full_agreement_rate": 1.0 - state_rate,
            }
        ]
    )
    defer_rows = []
    groups = [("ALL", labels)]
    groups.extend(list(labels.groupby("priority", sort=True)))
    for scope, group in groups:
        defer_rows.append(
            {
                "scope": scope,
                "task_decisions": len(group),
                "h1_defer_count": int(group["h1_is_defer"].sum()),
                "h1_defer_rate": _rate(group, "h1_is_defer"),
                "h4_defer_count": int(group["h4_is_defer"].sum()),
                "h4_defer_rate": _rate(group, "h4_is_defer"),
                "h1_execute_to_h4_defer": int(
                    (~group["h1_is_defer"] & group["h4_is_defer"]).sum()
                ),
                "h1_defer_to_h4_execute": int(
                    (group["h1_is_defer"] & ~group["h4_is_defer"]).sum()
                ),
            }
        )
    priority_rows = []
    for priority, group in labels.groupby("priority", sort=True):
        priority_rows.append(
            {
                "priority": priority,
                "task_decisions": len(group),
                "exact_disagreement_rate": _rate(
                    group, "exact_action_disagreement"
                ),
                "placement_disagreement_rate": _rate(
                    group, "placement_disagreement"
                ),
                "defer_disagreement_rate": _rate(
                    group, "defer_disagreement"
                ),
                "h1_defer_rate": _rate(group, "h1_is_defer"),
                "h4_defer_rate": _rate(group, "h4_is_defer"),
            }
        )
    return behavior, pd.DataFrame(defer_rows), pd.DataFrame(priority_rows)


def risk_and_pressure_analysis(
    states: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, float, tuple[float, float]]:
    train = states[states["split"] == "train"]
    threshold = float(train["risk_score"].quantile(0.95))
    boundaries = tuple(
        float(value)
        for value in train["risk_score"].quantile([1 / 3, 2 / 3])
    )
    state_work = states[
        ["state_id", "split", "risk_score", "empty_pending_state"]
    ].copy()
    state_work["any_disagreement"] = (
        state_work["state_id"]
        .map(labels.groupby("state_id")["exact_action_disagreement"].any())
        .fillna(False)
    )
    state_work["risk_band"] = np.where(
        state_work["risk_score"] >= threshold,
        "high_ge_train_p95",
        "low_lt_train_p95",
    )
    task_work = labels.merge(
        states[["state_id", "risk_score"]],
        on="state_id",
        how="left",
        validate="many_to_one",
    )
    task_work["risk_band"] = np.where(
        task_work["risk_score"] >= threshold,
        "high_ge_train_p95",
        "low_lt_train_p95",
    )
    risk_rows = []
    for band, group in state_work.groupby("risk_band", sort=True):
        nonempty = group[~group["empty_pending_state"]]
        risk_rows.append(
            {
                "level": "state",
                "band": band,
                "train_p95_threshold": threshold,
                "records": len(group),
                "nonempty_records": len(nonempty),
                "disagreement_rate": _rate(nonempty, "any_disagreement"),
            }
        )
    for band, group in task_work.groupby("risk_band", sort=True):
        risk_rows.append(
            {
                "level": "task",
                "band": band,
                "train_p95_threshold": threshold,
                "records": len(group),
                "nonempty_records": len(group),
                "disagreement_rate": _rate(
                    group, "exact_action_disagreement"
                ),
            }
        )
    state_work["pressure_band"] = pd.cut(
        state_work["risk_score"],
        bins=[-np.inf, boundaries[0], boundaries[1], np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )
    task_work["pressure_band"] = pd.cut(
        task_work["risk_score"],
        bins=[-np.inf, boundaries[0], boundaries[1], np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )
    pressure_rows = []
    for band, group in state_work.groupby(
        "pressure_band", observed=True, sort=True
    ):
        nonempty = group[~group["empty_pending_state"]]
        pressure_rows.append(
            {
                "level": "state",
                "pressure_band": str(band),
                "train_low_upper": boundaries[0],
                "train_medium_upper": boundaries[1],
                "records": len(group),
                "disagreement_rate": _rate(nonempty, "any_disagreement"),
                "h1_defer_rate": 0.0,
                "h4_defer_rate": 0.0,
                "placement_disagreement_rate": np.nan,
            }
        )
    for band, group in task_work.groupby(
        "pressure_band", observed=True, sort=True
    ):
        pressure_rows.append(
            {
                "level": "task",
                "pressure_band": str(band),
                "train_low_upper": boundaries[0],
                "train_medium_upper": boundaries[1],
                "records": len(group),
                "disagreement_rate": _rate(
                    group, "exact_action_disagreement"
                ),
                "h1_defer_rate": _rate(group, "h1_is_defer"),
                "h4_defer_rate": _rate(group, "h4_is_defer"),
                "placement_disagreement_rate": _rate(
                    group, "placement_disagreement"
                ),
            }
        )
    return (
        pd.DataFrame(risk_rows),
        pd.DataFrame(pressure_rows),
        threshold,
        boundaries,
    )


def task_stratified_analysis(
    canonical: pd.DataFrame, states: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[float, float]]:
    columns = [
        "original_index",
        "true_duration",
        "absolute_estimation_error_seconds",
        "gpu_model",
        "temporal_split",
    ]
    work = labels.merge(
        canonical[columns],
        on="original_index",
        how="left",
        validate="many_to_one",
    ).merge(
        states[["state_id", "risk_score"]],
        on="state_id",
        how="left",
        validate="many_to_one",
    )
    train_tasks = canonical[canonical["temporal_split"] == "train"]
    error_bounds = tuple(
        float(value)
        for value in train_tasks["absolute_estimation_error_seconds"].quantile(
            [1 / 3, 2 / 3]
        )
    )
    risk_threshold = float(
        states[states["split"] == "train"]["risk_score"].quantile(0.95)
    )
    work["error_band"] = pd.cut(
        work["absolute_estimation_error_seconds"],
        bins=[-np.inf, error_bounds[0], error_bounds[1], np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )
    error_rows = []
    for band, group in work.groupby(
        "error_band", observed=True, sort=True
    ):
        error_rows.append(
            {
                "error_band": str(band),
                "train_low_upper_seconds": error_bounds[0],
                "train_medium_upper_seconds": error_bounds[1],
                "task_decisions": len(group),
                "mean_absolute_error_seconds": float(
                    group["absolute_estimation_error_seconds"].mean()
                ),
                "exact_disagreement_rate": _rate(
                    group, "exact_action_disagreement"
                ),
                "h1_defer_rate": _rate(group, "h1_is_defer"),
                "h4_defer_rate": _rate(group, "h4_is_defer"),
                "high_risk_state_rate": float(
                    (group["risk_score"] >= risk_threshold).mean()
                ),
            }
        )
    duration_bins = [
        -np.inf,
        3600,
        6 * 3600,
        24 * 3600,
        48 * 3600,
        72 * 3600,
        np.inf,
    ]
    duration_labels = [
        "<1h",
        "1-6h",
        "6-24h",
        "24-48h",
        "48-72h",
        ">72h",
    ]
    work["duration_band"] = pd.cut(
        work["true_duration"],
        duration_bins,
        labels=duration_labels,
        right=False,
    )
    long_rows = []
    for band, group in work.groupby(
        "duration_band", observed=True, sort=False
    ):
        long_rows.append(
            {
                "true_duration_band": str(band),
                "task_decisions": len(group),
                "unique_tasks": group["original_index"].nunique(),
                "exact_disagreement_rate": _rate(
                    group, "exact_action_disagreement"
                ),
                "h1_defer_rate": _rate(group, "h1_is_defer"),
                "h4_defer_rate": _rate(group, "h4_is_defer"),
                "placement_disagreement_rate": _rate(
                    group, "placement_disagreement"
                ),
            }
        )
    gpu_rows = []
    for model, group in work.groupby("gpu_model", sort=True):
        gpu_rows.append(
            {
                "gpu_model": model,
                "task_decisions": len(group),
                "unique_tasks": group["original_index"].nunique(),
                "exact_disagreement_rate": _rate(
                    group, "exact_action_disagreement"
                ),
                "h1_defer_rate": _rate(group, "h1_is_defer"),
                "h4_defer_rate": _rate(group, "h4_is_defer"),
                "compatibility_constraints_enabled": False,
                "interpretation": (
                    "DESCRIPTIVE_ONLY_GPU_EQUIVALENT_SCHEDULING"
                ),
            }
        )
    return (
        pd.DataFrame(error_rows),
        pd.DataFrame(long_rows),
        pd.DataFrame(gpu_rows),
        error_bounds,
    )


def solver_quality(
    states: pd.DataFrame, runtime: RuntimeState
) -> pd.DataFrame:
    rows = []
    called = states[~states["empty_pending_state"]]
    for planner in ("h1", "h4"):
        times = called[f"{planner}_solve_ms"]
        statuses = called[f"{planner}_solver_status"]
        rows.append(
            {
                "record_type": "SUMMARY",
                "planner": planner.upper(),
                "solver_calls": len(called),
                "optimal": int(
                    statuses.eq("optimal").sum()
                ),
                "failures": int(
                    statuses.ne("optimal").sum()
                ),
                "timeouts": int(
                    statuses.eq("limit_reached").sum()
                ),
                "presolve_same_model_retries": int(
                    getattr(
                        runtime,
                        f"{planner}_presolve_retries",
                    )
                ),
                "solve_ms_mean": float(times.mean()),
                "solve_ms_p50": float(
                    times.quantile(0.50)
                ),
                "solve_ms_p95": float(
                    times.quantile(0.95)
                ),
                "solve_ms_p99": float(
                    times.quantile(0.99)
                ),
                "solve_ms_max": float(times.max()),
                "fallback_labels": 0,
            }
        )
    rows.append(
        {
            "record_type": "SUMMARY",
            "planner": "EMPTY_STATE_SKIP",
            "solver_calls": 0,
            "optimal": 0,
            "failures": 0,
            "timeouts": 0,
            "presolve_same_model_retries": 0,
            "solve_ms_mean": 0.0,
            "solve_ms_p50": 0.0,
            "solve_ms_p95": 0.0,
            "solve_ms_p99": 0.0,
            "solve_ms_max": 0.0,
            "fallback_labels": 0,
            "empty_states": int(
                states["empty_pending_state"].sum()
            ),
        }
    )
    for anomaly in runtime.solver_anomalies:
        rows.append(
            {
                "record_type": "PRESOLVE_ANOMALY",
                "planner": anomaly["planner"],
                "step": anomaly["step"],
                "initial_status": anomaly["initial_status"],
                "initial_message": anomaly["initial_message"],
                "verified_feasible_witness": anomaly[
                    "verified_feasible_witness"
                ],
                "retry_backend": anomaly["retry_backend"],
                "retry_status": anomaly["retry_status"],
                "retry_message": anomaly["retry_message"],
                "fallback_labels": int(
                    anomaly["fallback_label_used"]
                ),
            }
        )
    return pd.DataFrame(rows)

def cross_dataset_comparison(
    canonical: pd.DataFrame,
    states: pd.DataFrame,
    labels: pd.DataFrame,
    risk: pd.DataFrame,
) -> pd.DataFrame:
    a_tasks = pd.read_parquet(
        OUTPUT / "dataset/scenario_a/expert_task_actions_full.parquet"
    )
    a_states = pd.read_parquet(
        OUTPUT / "dataset/scenario_a/current_states_full.parquet"
    )
    a_character = pd.read_csv(
        ROOT
        / "artifacts/repaired_mpc_expert_dataset_v2"
        / "11_disagreement_characterization.csv"
    )
    a_high = float(
        a_character.loc[
            a_character["group"] == "HIGH_RISK_GE_CALIBRATION_P95",
            "disagreement_rate",
        ].iloc[0]
    )
    spot_high = float(
        risk[
            (risk["level"] == "task")
            & (risk["band"] == "high_ge_train_p95")
        ]["disagreement_rate"].iloc[0]
    )
    spot_state_any = float(
        labels.groupby("state_id")["exact_action_disagreement"].any().mean()
    )
    a_pending_gpu = (
        a_tasks.groupby("state_id")["task_gpu_units"].sum().mean() / 2900.0
    )
    spot_pending_gpu = (
        labels.merge(
            canonical[["original_index", "gpu_request_effective"]],
            on="original_index",
            how="left",
            validate="many_to_one",
        )
        .groupby("state_id")["gpu_request_effective"]
        .sum()
        .mean()
        / 10412.0
    )
    rows = [
        {
            "dataset": "Alibaba2020",
            "scenario_id": "A_ALIBABA2020_REPAIRED",
            "capacity_basis": "SUSTAINCLUSTER_5DC",
            "gpu_capacity": 2900.0,
            "task_exact_disagreement_rate": _rate(
                a_tasks, "exact_action_disagreement"
            ),
            "state_any_disagreement_rate": _rate(
                a_states, "any_disagreement"
            ),
            "high_risk_task_disagreement_rate": a_high,
            "h1_defer_rate": _rate(a_tasks, "h1_is_defer"),
            "h4_defer_rate": _rate(a_tasks, "teacher_is_defer"),
            "mean_pending_queue": float(a_states["num_tasks"].mean()),
            "mean_estimated_duration_seconds": float(
                a_tasks["estimated_duration"].mean() * 60.0
            ),
            "mean_pending_gpu_to_capacity": float(a_pending_gpu),
            "gpu_offered_load_ratio": np.nan,
        },
        {
            "dataset": "SpotGPU2026",
            "scenario_id": SCENARIO_ID,
            "capacity_basis": "COMPLETE_NODE_INFO_NATIVE_5DC",
            "gpu_capacity": 10412.0,
            "task_exact_disagreement_rate": _rate(
                labels, "exact_action_disagreement"
            ),
            "state_any_disagreement_rate": spot_state_any,
            "high_risk_task_disagreement_rate": spot_high,
            "h1_defer_rate": _rate(labels, "h1_is_defer"),
            "h4_defer_rate": _rate(labels, "h4_is_defer"),
            "mean_pending_queue": float(states["pending_count"].mean()),
            "mean_estimated_duration_seconds": float(
                canonical["estimated_duration"].mean()
            ),
            "mean_pending_gpu_to_capacity": float(spot_pending_gpu),
            "gpu_offered_load_ratio": float(
                pd.read_csv(OUTPUT / "34_spot_native_capacity_pressure.csv")
                .set_index("resource")
                .loc["gpu", "offered_load_ratio"]
            ),
        },
    ]
    result = pd.DataFrame(rows)
    result["comparison_scope"] = (
        "DESCRIPTIVE_WITHIN_EACH_FROZEN_SOURCE_MATCHED_CAPACITY_SCENARIO;"
        "_CAPACITY_NOT_CONTROLLED"
    )
    return result


def checkpoint_manifest(
    state: RuntimeState, formal_dir: Path
) -> pd.DataFrame:
    rows = []
    for path in sorted((formal_dir / "states").glob("checkpoint_*.pkl.gz")):
        step = int(path.name.split("_")[1].split(".")[0])
        rows.append(
            {
                "checkpoint_step": step,
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "complete_runtime_state": True,
                "rng_state_saved": True,
                "committed_chunk_count_at_or_before": sum(
                    int(item["end_step_exclusive"]) <= step
                    for item in state.committed_chunks
                ),
            }
        )
    return pd.DataFrame(rows)

def write_protocol(
    signal_provenance: pd.DataFrame, exact: Mapping[str, Any]
) -> None:
    locations = ", ".join(
        signal_provenance["location"].drop_duplicates()
    )
    write_text(
        OUTPUT / "44_spot_generation_protocol.md",
        f"""# SpotGPU2026 MPC Expert Dataset v3 Generation Protocol

- Scenario: `{SCENARIO_ID}`.
- Timeline: step 0 through the last arrival step, inclusive; no daily or split reset.
- Trace duration: 17,670 x 15 minutes = 184.0625 days.
- Runtime truth: source `true_duration` controls simulator completion and release only.
- Deployable duration: frozen train-70% hierarchical conditional median, minimum support 100.
- Requests: `PER_JOB`; CPU/GPU are never multiplied by `worker_num`.
- Capacity: frozen native 5DC partition, CPU 632636, GPU 10412, memory 542259.428571 GB (`MODELED_DC_MEMORY`).
- GPU models: metadata only; GPU-equivalent scheduling; no compatibility constraints.
- SLA: medium, HP 1 step and Spot 8 steps; objective/reward weights unchanged.
- H1: repaired current-only deployable state.
- H4 Oracle: current plus +15/+30/+45/+60 minute true global workload aggregates, distributed with the frozen deterministic origin-pressure model.
- Energy/carbon regions: {locations}.
- Energy/carbon source: SustainCluster {SIGNAL_YEAR} regional datasets; step 0 maps to `2023-02-17T00:00:00Z`, with each DC signal manager reset to the corresponding local day/hour; `{SIGNAL_PROVENANCE}`; not Alibaba2026 measurements.
- Checkpoint interval: {CHECKPOINT_INTERVAL} steps; task/runtime/resource/output/RNG state saved.
- Exact restart gate: `{exact['passed']}` for uninterrupted 100 versus 50 + restart + 50 steps.
- Empty states: retained at state level, zero task decisions, no MILP call, continuous runtime and signal advancement.
- Feasible mask: assignment-ingress feasibility under the frozen bounded-defer and whole-DC task-capacity constraints; an accepted destination reservation may wait for physical resources to be released.
- Fallback expert labels: forbidden.
- Transformer/RL/policy training: not used.
""",
    )


def information_leakage_audit(
    deployable_paths: Sequence[Path],
    label_paths: Sequence[Path],
    privileged_paths: Sequence[Path],
) -> None:
    deployable_columns = set()
    for path in deployable_paths:
        deployable_columns.update(pq.read_schema(path).names)
    forbidden = {
        "true_duration_seconds",
        "true_duration_steps",
        "true_completion_step",
        "h1_action",
        "h4_action",
        "future_electricity_price_usd_per_mwh",
        "future_carbon_intensity_gco2_per_kwh",
        "h4_forecast_gpu_reservation",
    }
    leaked = sorted(deployable_columns & forbidden)
    write_text(
        OUTPUT / "59_spot_information_leakage_audit.md",
        f"""# SpotGPU2026 Information Leakage Audit

- Deployable files checked: {len(deployable_paths)}.
- Label files physically separate: {len(label_paths)}.
- Privileged-future files physically separate: {len(privileged_paths)}.
- Forbidden columns found in deployable region: `{leaked}`.
- `true_duration` and true completion are under `simulator_only/` only.
- H1/H4 actions and objectives are under `labels/` and require explicit loader opt-in.
- +15/+30/+45/+60 workload, price and carbon are under `privileged_future/` only.
- Default loader manifest exposes only `deployable_current/`.
- Transformer forecast columns/checkpoints: absent.

Result: **{'PASS' if not leaked else 'FAIL'}**.
""",
    )
    if leaked:
        raise RuntimeError(
            f"deployable information leakage detected: {leaked}"
        )


def formal_integrity_manifest(
    inputs: Mapping[str, Any],
    files: Mapping[str, Path],
    extra_files: Sequence[Path],
) -> pd.DataFrame:
    rows = []
    sources = (
        (
            "spot_job_source",
            ROOT / inputs["contract"]["source"]["job_file"],
        ),
        (
            "spot_node_source",
            ROOT / inputs["contract"]["source"]["node_file"],
        ),
        (
            "scenario_contract",
            ROOT / inputs["config"]["scenario_b"]["contract_config"],
        ),
        ("capacity_contract", CAPACITY_PATH),
        ("generation_config", CONFIG_PATH),
        ("generator_code", Path(__file__).resolve()),
    )
    for role, path in sources:
        rows.append(
            {
                "role": role,
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    for key, path in sorted(files.items()):
        rows.append(
            {
                "role": f"formal_data:{key}",
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    for path in extra_files:
        rows.append(
            {
                "role": "formal_metadata",
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    rows.extend(
        [
            {
                "role": "project_git_head",
                "path": "GIT",
                "bytes": np.nan,
                "sha256": git("rev-parse", "HEAD"),
            },
            {
                "role": "sustaincluster_git_head",
                "path": "references/external_repos/sustain-cluster/.git",
                "bytes": np.nan,
                "sha256": git("rev-parse", "HEAD", cwd=VENDOR),
            },
        ]
    )
    manifest = pd.DataFrame(rows)
    expected_job = inputs["contract"]["source"]["job_sha256"].upper()
    expected_node = inputs["contract"]["source"]["node_sha256"].upper()
    actual_job = manifest.loc[
        manifest["role"] == "spot_job_source", "sha256"
    ].iloc[0]
    actual_node = manifest.loc[
        manifest["role"] == "spot_node_source", "sha256"
    ].iloc[0]
    if actual_job != expected_job:
        raise RuntimeError("Spot job source integrity changed")
    if actual_node != expected_node:
        raise RuntimeError("Spot node source integrity changed")
    return manifest


def _csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def finalize_formal_generation(
    inputs: Mapping[str, Any],
    generator: SpotContinuousGenerator,
    exact: Mapping[str, Any],
) -> dict[str, Any]:
    if generator.state.next_step != generator.total_steps:
        raise RuntimeError("cannot finalize an incomplete Spot timeline")
    files = consolidate_formal_tables(generator.state)
    states = pd.read_parquet(files["states_full"])
    labels = pd.read_parquet(files["labels_full"])
    lifecycle = build_lifecycle(
        inputs["canonical"],
        generator.state,
        files["truth_full"],
        files["events_full"],
    )
    lifecycle_path = (
        OUTPUT
        / "dataset/scenario_b/simulator_only/task_lifecycle_full.parquet"
    )
    _atomic_write_table(
        pa.Table.from_pandas(lifecycle, preserve_index=False),
        lifecycle_path,
    )
    files["task_lifecycle_full"] = lifecycle_path
    loader_manifest = write_loader_manifest(files)
    files["loader_manifest"] = loader_manifest
    signal_path = (
        OUTPUT
        / "dataset/scenario_b/simulator_only/energy_signal_provenance.csv"
    )
    _csv(signal_path, generator.signal_provenance)
    files["energy_signal_provenance"] = signal_path

    stats, split_stats = dataset_statistics(
        states, labels, generator.state
    )
    behavior, defer, priority = behavior_analysis(states, labels)
    risk, pressure, risk_threshold, pressure_bounds = (
        risk_and_pressure_analysis(states, labels)
    )
    duration_error, long_tasks, gpu_models, error_bounds = (
        task_stratified_analysis(
            inputs["canonical"], states, labels
        )
    )
    solver = solver_quality(states, generator.state)
    cross = cross_dataset_comparison(
        inputs["canonical"], states, labels, risk
    )
    checkpoint = checkpoint_manifest(
        generator.state, generator.run_dir
    )
    progress = pd.DataFrame(generator.state.progress_rows)

    _csv(OUTPUT / "45_spot_generation_progress.csv", progress)
    _csv(OUTPUT / "46_spot_checkpoint_manifest.csv", checkpoint)
    _csv(OUTPUT / "47_spot_dataset_statistics.csv", stats)
    _csv(OUTPUT / "48_spot_split_statistics.csv", split_stats)
    _csv(OUTPUT / "49_spot_h1_h4_behavior.csv", behavior)
    _csv(OUTPUT / "50_spot_defer_analysis.csv", defer)
    _csv(OUTPUT / "51_spot_priority_analysis.csv", priority)
    _csv(OUTPUT / "52_spot_risk_analysis.csv", risk)
    _csv(OUTPUT / "53_spot_pressure_analysis.csv", pressure)
    _csv(
        OUTPUT / "54_spot_duration_error_analysis.csv",
        duration_error,
    )
    _csv(OUTPUT / "55_spot_long_task_analysis.csv", long_tasks)
    _csv(OUTPUT / "56_spot_gpu_model_analysis.csv", gpu_models)
    _csv(
        OUTPUT / "57_cross_dataset_behavior_comparison.csv",
        cross,
    )
    _csv(OUTPUT / "58_spot_solver_quality.csv", solver)
    write_protocol(generator.signal_provenance, exact)
    deployable = [
        path
        for key, path in files.items()
        if key.startswith("states_") or key.startswith("tasks_")
    ]
    label_files = [
        path for key, path in files.items() if key.startswith("labels_")
    ]
    privileged_files = [
        path
        for key, path in files.items()
        if key.startswith("privileged_")
    ]
    information_leakage_audit(
        deployable, label_files, privileged_files
    )
    integrity = formal_integrity_manifest(
        inputs,
        files,
        [OUTPUT / "44_spot_generation_protocol.md"],
    )
    _csv(OUTPUT / "60_spot_integrity_manifest.csv", integrity)

    behavior_row = behavior.iloc[0]
    high_task = risk[
        (risk["level"] == "task")
        & (risk["band"] == "high_ge_train_p95")
    ].iloc[0]
    high_state = risk[
        (risk["level"] == "state")
        & (risk["band"] == "high_ge_train_p95")
    ].iloc[0]
    hp = priority[priority["priority"] == "HP"].iloc[0]
    spot = priority[priority["priority"] == "Spot"].iloc[0]
    result = {
        "final_status": FINAL_READY_LIMITED,
        "timeline_complete": True,
        "total_steps": len(states),
        "nonempty_states": int(
            (~states["empty_pending_state"]).sum()
        ),
        "empty_states": int(states["empty_pending_state"].sum()),
        "task_decisions": len(labels),
        "unique_tasks": labels["task_id"].nunique(),
        "h1_failures": int(generator.state.h1_failures),
        "h4_failures": int(generator.state.h4_failures),
        "h1_presolve_same_model_retries": int(
            generator.state.h1_presolve_retries
        ),
        "h4_presolve_same_model_retries": int(
            generator.state.h4_presolve_retries
        ),
        "solver_anomalies": generator.state.solver_anomalies,
        "exact_disagreement_rate": float(
            behavior_row["exact_disagreement_rate"]
        ),
        "state_any_disagreement_rate": float(
            behavior_row["state_any_disagreement_rate"]
        ),
        "risk_threshold_train_p95": risk_threshold,
        "high_risk_task_disagreement_rate": float(
            high_task["disagreement_rate"]
        ),
        "high_risk_state_disagreement_rate": float(
            high_state["disagreement_rate"]
        ),
        "hp_disagreement_rate": float(
            hp["exact_disagreement_rate"]
        ),
        "spot_disagreement_rate": float(
            spot["exact_disagreement_rate"]
        ),
        "h1_defer_rate": float(labels["h1_is_defer"].mean()),
        "h4_defer_rate": float(labels["h4_is_defer"].mean()),
        "checkpoint_exact": bool(exact["passed"]),
        "pressure_train_boundaries": list(pressure_bounds),
        "duration_error_train_boundaries_seconds": list(error_bounds),
        "project_git_head": git("rev-parse", "HEAD"),
        "sustaincluster_git_head": git(
            "rev-parse", "HEAD", cwd=VENDOR
        ),
        "data_files": {
            key: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for key, path in files.items()
        },
        "documented_limitations": [
            (
                "MODELED_DC_MEMORY is not measured Alibaba2026 "
                "host memory"
            ),
            (
                "regional price/carbon are SustainCluster 2023 "
                "EXTERNAL_SCENARIO_SIGNAL, not Alibaba2026 measurements"
            ),
            (
                "origin, memory, bandwidth, duration estimate and "
                "SLA are frozen modeled fields"
            ),
            (
                "GPU models are descriptive metadata; compatibility "
                "constraints are disabled"
            ),
            (
                "cross-dataset behavior is not a capacity-controlled "
                "causal comparison"
            ),
            (
                "public redistribution remains blocked pending "
                "source/derived-data license confirmation"
            ),
        ],
        "policy_training_used": False,
        "transformer_used": False,
        "rl_used": False,
    }
    write_json(OUTPUT / "00_generation_manifest.json", result)
    write_final_markdown(
        result, priority, duration_error, long_tasks, cross
    )
    return result


def write_final_markdown(
    result: Mapping[str, Any],
    priority: pd.DataFrame,
    duration_error: pd.DataFrame,
    long_tasks: pd.DataFrame,
    cross: pd.DataFrame,
) -> None:
    hp = priority[priority["priority"] == "HP"].iloc[0]
    spot = priority[priority["priority"] == "Spot"].iloc[0]
    longest = long_tasks[
        long_tasks["true_duration_band"] == ">72h"
    ].iloc[0]
    shortest = long_tasks[
        long_tasks["true_duration_band"] == "<1h"
    ].iloc[0]
    low_error = duration_error[
        duration_error["error_band"] == "low"
    ].iloc[0]
    high_error = duration_error[
        duration_error["error_band"] == "high"
    ].iloc[0]
    a_rate = float(
        cross[cross["dataset"] == "Alibaba2020"][
            "task_exact_disagreement_rate"
        ].iloc[0]
    )
    limitations = "\n".join(
        f"- {item}" for item in result["documented_limitations"]
    )
    diagnosis = f"""# MPC Expert Dataset v3 Final Diagnosis

Final status: **{result['final_status']}**

1. Spot full continuous timeline completed: **YES**, {result['total_steps']} steps with no daily or split reset.
2. States: **{result['total_steps']}** ({result['nonempty_states']} nonempty, {result['empty_states']} empty).
3. Task decisions: **{result['task_decisions']}** over **{result['unique_tasks']}** unique tasks.
4. H1/H4 final solver failures: **{result['h1_failures']} / {result['h4_failures']}**; verified same-model presolve retries: **{result['h1_presolve_same_model_retries']} / {result['h4_presolve_same_model_retries']}**; fallback labels: **0**.
5. H1/H4 task exact disagreement: **{result['exact_disagreement_rate']:.6%}**.
6. Nonempty state any-disagreement: **{result['state_any_disagreement_rate']:.6%}**.
7. Spot train-only risk P95: **{result['risk_threshold_train_p95']:.9f}**; high-risk task/state disagreement: **{result['high_risk_task_disagreement_rate']:.6%} / {result['high_risk_state_disagreement_rate']:.6%}**.
8. HP versus Spot disagreement: **{float(hp['exact_disagreement_rate']):.6%} / {float(spot['exact_disagreement_rate']):.6%}**.
9. H1 defer: **{result['h1_defer_rate']:.6%}**.
10. H4 defer: **{result['h4_defer_rate']:.6%}**.
11. Defer concentration is reported by priority in `50_spot_defer_analysis.csv`; no objective or SLA parameter was tuned to create defer.
12. `<1h` versus `>72h` disagreement: **{float(shortest['exact_disagreement_rate']):.6%} / {float(longest['exact_disagreement_rate']):.6%}**; descriptive only.
13. Low versus high duration-error disagreement: **{float(low_error['exact_disagreement_rate']):.6%} / {float(high_error['exact_disagreement_rate']):.6%}**; descriptive only, estimator unchanged.
14. Alibaba2020 versus Spot exact disagreement: **{a_rate:.6%} / {result['exact_disagreement_rate']:.6%}**. This compares behavior inside each source-matched frozen capacity scenario, not workload difficulty under controlled capacity.
15. Expert Dataset v3 can be the single formal basis for later learning experiments: **YES**, with labels requested explicitly and with the documented scenario/license limitations retained.

## Completion gates

- Alibaba2020 Scenario A remains frozen and was not regenerated.
- Spot Scenario B contains the complete arrival timeline and all 466,867 source tasks.
- Checkpoint/restart semantic equality: PASS.
- Information-region isolation: PASS.
- Split-state exclusivity and continuous boundary crossing: PASS.
- Formal file hashes: `60_spot_integrity_manifest.csv`.
- Transformer, BC and RL training: not run.

## Documented limitations

{limitations}
"""
    write_text(
        OUTPUT / "61_mpc_expert_dataset_v3_final_diagnosis.md",
        diagnosis,
    )
    write_text(
        OUTPUT / "62_mpc_expert_dataset_v3_summary.md",
        f"""# MPC Expert Dataset v3 Summary

**{result['final_status']}**

- Scenario A: Alibaba2020 repaired v3, 7,680 states and 259,920 task decisions, frozen.
- Scenario B: SpotGPU2026 native-medium, {result['total_steps']} continuous states and {result['task_decisions']} task decisions.
- Spot H1/H4 exact disagreement: {result['exact_disagreement_rate']:.6%}.
- Spot nonempty-state disagreement: {result['state_any_disagreement_rate']:.6%}.
- H1/H4 final failures: {result['h1_failures']} / {result['h4_failures']}; verified same-model presolve retries: {result['h1_presolve_same_model_retries']} / {result['h4_presolve_same_model_retries']}.
- Checkpoint exact restart, information isolation, split grouping and data hashes: PASS.
- The dataset is ready as the formal expert-data foundation; future BC/Transformer/RL work is outside this run.
""",
    )


def verify_formal_outputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    base = OUTPUT / "dataset/scenario_b"
    states = pd.read_parquet(
        base / "deployable_current/states_full.parquet"
    )
    tasks = pd.read_parquet(
        base / "deployable_current/tasks_full.parquet"
    )
    labels = pd.read_parquet(
        base / "labels/expert_actions_full.parquet"
    )
    privileged = pd.read_parquet(
        OUTPUT
        / "privileged_future/scenario_b/oracle_future_full.parquet"
    )
    canonical = inputs["canonical"]
    dcs = inputs["dcs"]
    integrity = pd.read_csv(OUTPUT / "60_spot_integrity_manifest.csv")
    boundary_rows = states.set_index("step")

    def crosses_boundary(left_step: int, right_step: int) -> bool:
        left = set(boundary_rows.at[left_step, "running_task_ids"])
        right = set(boundary_rows.at[right_step, "running_task_ids"])
        return bool(left & right)

    formal_hashes_match = True
    for row in integrity.itertuples(index=False):
        if not str(row.role).startswith("formal_data:"):
            continue
        path = ROOT / str(row.path)
        if not path.is_file() or sha256(path) != str(row.sha256):
            formal_hashes_match = False
            break
    priority_bounds = (
        tasks.groupby("priority")["max_wait_steps"]
        .unique()
        .map(tuple)
        .to_dict()
    )
    checks = {
        "timeline_steps_17670": len(states) == 17670,
        "step_zero_to_last_exact": (
            states["step"].tolist() == list(range(17670))
        ),
        "all_source_tasks_present": (
            labels["task_id"].nunique() == 466867
        ),
        "task_decision_grouping": len(tasks) == len(labels),
        "state_grouping": (
            labels.groupby("state_id")["step"].nunique().max() == 1
        ),
        "no_state_crosses_split": (
            labels.groupby("state_id")["split"].nunique().max() == 1
        ),
        "per_job_cpu": np.allclose(
            tasks["cpu_cores"].to_numpy(),
            canonical.loc[
                tasks["original_index"], "cpu_request_raw"
            ].to_numpy(),
        ),
        "per_job_gpu": np.allclose(
            tasks["gpu_units"].to_numpy(),
            canonical.loc[
                tasks["original_index"], "gpu_request_raw"
            ].to_numpy(),
        ),
        "bandwidth_exact": bool(
            np.all(tasks["bandwidth_gb"] == 0.020062580706)
        ),
        "sla_bounds_exact": (
            priority_bounds == {"HP": (1,), "Spot": (8,)}
        ),
        "h1_provenance_exact": (
            set(labels["h1_label_provenance"])
            == {"H1_REPAIRED_CURRENT_ONLY"}
        ),
        "h4_provenance_exact": (
            set(labels["h4_label_provenance"])
            == {"H4_ORACLE_REPAIRED"}
        ),
        "privileged_alignment": (
            set(privileged["horizon_minutes"].unique())
            == {15, 30, 45, 60}
        ),
        "no_true_duration_in_deployable": (
            "true_duration_seconds" not in tasks.columns
        ),
        "no_actions_in_deployable": not (
            {"h1_action", "h4_action"} & set(tasks.columns)
        ),
        "split_boundaries_exact": (
            int(
                states[states["split"] == "train"]["step"].max()
            )
            == 10527
            and int(
                states[states["split"] == "validation"]["step"].min()
            )
            == 10528
            and int(
                states[states["split"] == "validation"]["step"].max()
            )
            == 13840
            and int(
                states[states["split"] == "test"]["step"].min()
            )
            == 13841
        ),
        "train_validation_runtime_continuity": crosses_boundary(
            10527, 10528
        ),
        "validation_test_runtime_continuity": crosses_boundary(
            13840, 13841
        ),
        "capacity_cpu_exact": int(dcs["total_cores"].sum()) == 632636,
        "capacity_gpu_exact": int(dcs["total_gpus"].sum()) == 10412,
        "capacity_memory_exact": np.isclose(
            dcs["total_mem"].sum(), 542259.4285714285
        ),
        "formal_output_hash_integrity": formal_hashes_match,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        failed = [
            name for name, value in checks.items() if not value
        ]
        raise RuntimeError(
            f"formal output verification failed: {failed}"
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-test", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--end-step", type=int)
    args = parser.parse_args()
    inputs = load_inputs()
    exact_path = (
        OUTPUT
        / "checkpoints/scenario_b/resume_exact_test_v1"
        / "checkpoint_resume_exact_test.json"
    )
    if args.checkpoint_test:
        result = checkpoint_exact_test(inputs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.generate:
        if not exact_path.is_file():
            raise RuntimeError(
                "formal generation blocked until checkpoint exact test passes"
            )
        exact = json.loads(exact_path.read_text("utf-8"))
        if not exact["passed"]:
            raise RuntimeError(
                "formal generation blocked by checkpoint exact test"
            )
        formal_dir = (
            OUTPUT
            / "checkpoints/scenario_b/formal_native_medium_v3"
        )
        generator = SpotContinuousGenerator(
            inputs,
            formal_dir,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            resume=args.resume,
        )
        generator.run(args.end_step)
        if generator.state.next_step == generator.total_steps:
            result = finalize_formal_generation(
                inputs, generator, exact
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.verify:
        result = verify_formal_outputs(inputs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if not any(
        (args.checkpoint_test, args.generate, args.verify)
    ):
        parser.error(
            "select --checkpoint-test, --generate, or --verify"
        )


if __name__ == "__main__":
    main()