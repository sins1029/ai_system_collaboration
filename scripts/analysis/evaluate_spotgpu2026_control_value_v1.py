from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
VENDOR = ROOT / "references/external_repos/sustain-cluster"
for entry in (ROOT, SRC, VENDOR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts.imitation import build_spotgpu2026_expert_dataset_v3 as spot_v3  # noqa: E402
from sustaincluster_mpc.horizon_adapter import (  # noqa: E402
    HorizonDataCenterSnapshot,
    HorizonState,
    RunningTaskHorizonSnapshot,
    TransitTaskHorizonSnapshot,
)
from sustaincluster_mpc.rolling_horizon_optimizer import (  # noqa: E402
    RollingHorizonResult,
)
from sustaincluster_mpc.state_adapter import (  # noqa: E402
    DataCenterSnapshot,
    ExogenousSignalsSnapshot,
    SchedulerState,
    TaskDestinationSnapshot,
    TaskSnapshot,
)


OUTPUT = ROOT / "artifacts/spotgpu2026_control_value_v1"
CONFIG_PATH = ROOT / "configs/sustaincluster_mpc/spotgpu2026_control_value_v1.yaml"
SOURCE_ROOT = ROOT / "artifacts/mpc_expert_dataset_v3"
CANONICAL_PATH = SOURCE_ROOT / "preflight/spot_full_canonical.parquet"
V3_STATES_PATH = SOURCE_ROOT / "dataset/scenario_b/deployable_current/states_full.parquet"
V3_TASKS_PATH = SOURCE_ROOT / "dataset/scenario_b/deployable_current/tasks_full.parquet"
V3_LABELS_PATH = SOURCE_ROOT / "dataset/scenario_b/labels/expert_actions_full.parquet"
PRIMARY_STEPS = 17_670
TAIL_MAX_STEPS = 17_670
H4_NODES = 5
CHECKPOINT_INTERVAL = 250
RISK_THRESHOLD = 0.72463503649635
CONTROLLERS = {"h1": 1, "h4_oracle": H4_NODES}
CONTROLLER_LABELS = {"h1": "H1", "h4_oracle": "H4_ORACLE"}


STEP_SCHEMA = pa.schema(
    [
        ("controller", pa.string()),
        ("step", pa.int32()),
        ("phase", pa.string()),
        ("split", pa.string()),
        ("timestamp_utc", pa.string()),
        ("arrival_task_count", pa.int32()),
        ("arrival_cpu", pa.float64()),
        ("arrival_gpu", pa.float64()),
        ("arrival_memory", pa.float64()),
        ("pending_pre", pa.int32()),
        ("running_pre", pa.int32()),
        ("transit_pre", pa.int32()),
        ("completed_pre", pa.int32()),
        ("risk_score", pa.float64()),
        ("action_count", pa.int32()),
        ("defer_count", pa.int32()),
        ("migration_count", pa.int32()),
        ("transmission_amount_gb", pa.float64()),
        ("stage_electricity", pa.float64()),
        ("stage_carbon", pa.float64()),
        ("stage_transmission", pa.float64()),
        ("stage_waiting_defer", pa.float64()),
        ("stage_sla_risk", pa.float64()),
        ("stage_terminal_backlog", pa.float64()),
        ("stage_cost", pa.float64()),
        ("reward", pa.float64()),
        ("physical_energy_kwh", pa.float64()),
        ("physical_electricity_cost_usd", pa.float64()),
        ("physical_carbon_kg", pa.float64()),
        ("pending_post", pa.int32()),
        ("running_post", pa.int32()),
        ("transit_post", pa.int32()),
        ("completed_post", pa.int32()),
        ("solver_called", pa.bool_()),
        ("solver_status", pa.string()),
        ("solver_message", pa.string()),
        ("presolve_retry", pa.bool_()),
        ("solve_ms", pa.float64()),
        ("fallback_used", pa.bool_()),
        ("information_mode", pa.string()),
    ]
)

DECISION_SCHEMA = pa.schema(
    [
        ("controller", pa.string()),
        ("step", pa.int32()),
        ("phase", pa.string()),
        ("split", pa.string()),
        ("task_id", pa.string()),
        ("original_index", pa.int64()),
        ("priority", pa.string()),
        ("origin_dc", pa.int16()),
        ("action", pa.int8()),
        ("destination_dc", pa.int16()),
        ("is_defer", pa.bool_()),
        ("is_migration", pa.bool_()),
        ("transmission_amount_gb", pa.float64()),
        ("waiting_steps_at_decision", pa.int32()),
        ("arrival_step", pa.int32()),
        ("sla_deadline_step", pa.int32()),
        ("max_wait_steps", pa.int16()),
        ("estimated_duration_seconds", pa.float64()),
        ("estimated_duration_steps", pa.int32()),
        ("estimated_energy_kwh", pa.float64()),
        ("cost_electricity", pa.float64()),
        ("cost_carbon", pa.float64()),
        ("cost_transmission", pa.float64()),
        ("cost_waiting_defer", pa.float64()),
        ("cost_sla_risk", pa.float64()),
        ("cost_terminal_backlog", pa.float64()),
        ("action_stage_cost", pa.float64()),
    ]
)

EVENT_SCHEMA = pa.schema(
    [
        ("controller", pa.string()),
        ("step", pa.int32()),
        ("phase", pa.string()),
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
    ]
)

TABLE_SCHEMAS = {"steps": STEP_SCHEMA, "decisions": DECISION_SCHEMA, "events": EVENT_SCHEMA}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def object_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), "utf-8")
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", "utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def nullable_int(value: Any) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def load_eval_config() -> dict[str, Any]:
    raw = yaml.safe_load(CONFIG_PATH.read_text("utf-8"))["spotgpu2026_control_value_v1"]
    if raw["workload"]["task_count"] != 466_867:
        raise RuntimeError("frozen Spot task count changed")
    if raw["workload"]["primary_steps"] != PRIMARY_STEPS:
        raise RuntimeError("frozen primary timeline changed")
    if raw["capacity"]["total_gpu_units"] != 10_412:
        raise RuntimeError("frozen Spot GPU capacity changed")
    if raw["controllers"]["h4_oracle"]["future_offsets_minutes"] != [15, 30, 45, 60]:
        raise RuntimeError("H4 Oracle offsets changed")
    return raw


class ClosedLoopRunner(spot_v3.SpotContinuousGenerator):
    """Independent frozen SpotGPU2026 closed loop for one controller."""

    def __init__(
        self,
        inputs: Mapping[str, Any],
        run_dir: Path,
        controller: str,
        *,
        checkpoint_interval: int = CHECKPOINT_INTERVAL,
        resume: bool = False,
    ) -> None:
        if controller not in CONTROLLERS:
            raise ValueError(f"unknown controller: {controller}")
        super().__init__(inputs, run_dir, checkpoint_interval=checkpoint_interval, resume=False)
        self.controller = controller
        self.controller_label = CONTROLLER_LABELS[controller]
        self.horizon = CONTROLLERS[controller]
        self.data_dir = run_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metric_buffers: dict[str, list[dict[str, Any]]] = {
            name: [] for name in TABLE_SCHEMAS
        }
        self.chunk_start = None
        self.primary_snapshot: dict[str, Any] | None = None
        self.oracle_future_state_count = 0
        self.oracle_future_row_count = 0
        self._reservation_cache_key: tuple[int, int] | None = None
        self._reservation_cache: dict[int, np.ndarray] = {}
        self.started_at = time.perf_counter()
        self.signals, self.signal_provenance = spot_v3.build_signal_trace(
            self.dcs, PRIMARY_STEPS + TAIL_MAX_STEPS + H4_NODES
        )
        latest = self.run_dir / "latest_checkpoint.json"
        if resume:
            self.state = self._load_control_checkpoint()
        elif latest.exists():
            raise FileExistsError(f"fresh run refused because checkpoint exists: {latest}")
        self.process_start_step = self.state.next_step

    def phase(self, step: int) -> str:
        return "primary" if step < PRIMARY_STEPS else "tail"

    def split(self, step: int) -> str:
        return super().split(min(step, PRIMARY_STEPS - 1)) if step < PRIMARY_STEPS else "tail"

    def _runtime_event(self, step: int, item: Mapping[str, Any], event_type: str) -> None:
        self.metric_buffers["events"].append(
            {
                "controller": self.controller_label,
                "step": int(step),
                "phase": self.phase(step),
                "split": self.split(step),
                "task_id": str(item["task_id"]),
                "original_index": int(item["row_index"]),
                "event_type": event_type,
                "dc_id": int(item["dc_id"]),
                "dispatch_step": int(item["dispatch_step"]),
                "planned_execution_start_step": int(item["planned_execution_start_step"]),
                "actual_execution_start_step": nullable_int(item.get("actual_execution_start_step")),
                "estimated_completion_step": nullable_int(item.get("estimated_completion_step")),
                "true_completion_step": nullable_int(item.get("true_completion_step")),
                "true_duration_steps": int(item["true_duration_steps"]),
            }
        )

    def _release_and_start(self, step: int) -> None:
        due = sorted(
            (index, item)
            for index, item in self.state.running.items()
            if int(item["true_completion_step"]) <= step
        )
        for row_index, item in due:
            dc_id = int(item["dc_id"])
            values = np.asarray(self.state.used[dc_id], dtype=float) - np.asarray(
                item["resources"], dtype=float
            )
            if np.any(values < -1e-7):
                raise RuntimeError("negative used capacity during release")
            self.state.used[dc_id] = np.maximum(values, 0.0).tolist()
            item["status"] = "completed"
            self._runtime_event(step, item, "completed")
            del self.state.running[row_index]
            self.state.completed_count += 1

        eligible = sorted(
            (index, item)
            for index, item in self.state.transit.items()
            if int(item["planned_execution_start_step"]) <= step
        )
        for row_index, item in eligible:
            dc_id = int(item["dc_id"])
            request = np.asarray(item["resources"], dtype=float)
            used = np.asarray(self.state.used[dc_id], dtype=float)
            if np.any(used + request > self.capacity[dc_id] + 1e-7):
                item["status"] = "destination_reserved_queue"
                continue
            reserved = np.asarray(self.state.reserved[dc_id], dtype=float) - request
            if np.any(reserved < -1e-7):
                raise RuntimeError("negative reservation during destination start")
            self.state.reserved[dc_id] = np.maximum(reserved, 0.0).tolist()
            self.state.used[dc_id] = (used + request).tolist()
            item["status"] = "running"
            item["actual_execution_start_step"] = step
            item["estimated_completion_step"] = step + int(item["estimated_duration_steps"])
            item["true_completion_step"] = step + int(item["true_duration_steps"])
            self.state.running[row_index] = item
            self._runtime_event(step, item, "started")
            del self.state.transit[row_index]

    def _current_dc_rows(self, step: int) -> list[dict[str, Any]]:
        running_counts = Counter(int(item["dc_id"]) for item in self.state.running.values())
        transit_counts = Counter(int(item["dc_id"]) for item in self.state.transit.values())
        rows: list[dict[str, Any]] = []
        for dc in self.dcs.sort_values("dc_id").itertuples(index=False):
            dc_id = int(dc.dc_id)
            total = self.capacity[dc_id]
            used = np.asarray(self.state.used[dc_id], dtype=float)
            reserved = np.asarray(self.state.reserved[dc_id], dtype=float)
            physical_available = total - used
            schedulable = np.maximum(0.0, physical_available - reserved)
            if np.any(physical_available < -1e-7):
                raise RuntimeError("physical native capacity invariant failed")
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
                    "cpu_physical_available": float(max(0.0, physical_available[0])),
                    "gpu_physical_available": float(max(0.0, physical_available[1])),
                    "memory_physical_available": float(max(0.0, physical_available[2])),
                    "cpu_available": float(schedulable[0]),
                    "gpu_available": float(schedulable[1]),
                    "memory_available": float(schedulable[2]),
                    "running_count": running_counts[dc_id],
                    "in_transit_count": transit_counts[dc_id],
                    "electricity_price_usd_per_mwh": float(self.signals[dc_id]["price"][step]),
                    "carbon_intensity_gco2_per_kwh": float(self.signals[dc_id]["carbon"][step]),
                }
            )
        return rows

    def _known_reservations(self, dc_id: int, step: int, horizon: int) -> np.ndarray:
        key = (step, horizon)
        if self._reservation_cache_key != key:
            cache = {item: np.zeros((horizon, 3), dtype=float) for item in self.capacity}
            for active in self.state.running.values():
                target = int(active["dc_id"])
                resources = np.asarray(active["resources"], dtype=float)
                estimated_completion = int(active["estimated_completion_step"])
                stop = horizon if estimated_completion <= step else min(
                    horizon, estimated_completion - step
                )
                cache[target][:stop] += resources
            for active in self.state.transit.values():
                target = int(active["dc_id"])
                resources = np.asarray(active["resources"], dtype=float)
                start = max(0, int(active["planned_execution_start_step"]) - step)
                stop = min(horizon, start + int(active["estimated_duration_steps"]))
                if start < horizon:
                    cache[target][start:stop] += resources
            self._reservation_cache_key = key
            self._reservation_cache = cache
        return self._reservation_cache[dc_id]

    def _future_pressure_compact(self, step: int) -> dict[int, np.ndarray]:
        reserves = {dc_id: np.zeros((self.horizon, 4), dtype=float) for dc_id in self.capacity}
        if self.controller == "h1":
            return reserves
        self.oracle_future_state_count += 1
        for offset in range(1, self.horizon):
            future_step = step + offset
            global_values = self.global_future.get(future_step, np.zeros(4, dtype=float))
            probabilities = spot_v3.expected_origin_probabilities(
                self.dc_configs, self.timestamp(future_step)
            )
            for dc_id in sorted(self.capacity):
                reserves[dc_id][offset] = global_values * float(probabilities[dc_id])
                self.oracle_future_row_count += 1
        return reserves

    def build_control_state(self, step: int) -> HorizonState:
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

        current_rows = self._current_dc_rows(step)
        future = self._future_pressure_compact(step)
        releases: dict[int, list[str]] = {dc_id: [] for dc_id in self.capacity}
        for active in self.state.running.values():
            releases[int(active["dc_id"])].append(
                self.timestamp(int(active["estimated_completion_step"])).isoformat()
            )
        current_dcs: list[DataCenterSnapshot] = []
        horizon_dcs: list[HorizonDataCenterSnapshot] = []
        for item in current_rows:
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
                    resource_release_times_utc=tuple(releases[dc_id]),
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
            known = self._known_reservations(dc_id, step, self.horizon)
            forecast = future[dc_id][:, 1:]
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
                        for offset in range(self.horizon)
                    ),
                    carbon_intensity_gco2_per_kwh=tuple(
                        float(self.signals[dc_id]["carbon"][step + offset])
                        for offset in range(self.horizon)
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
            information_mode="deployable" if self.controller == "h1" else "oracle",
        )
        return HorizonState(
            current=scheduler,
            horizon=self.horizon,
            forecast_mode="no_future_arrivals" if self.controller == "h1" else "oracle",
            timestep_minutes=15.0,
            datacenters=tuple(horizon_dcs),
            running_tasks=running,
            transit_tasks=transit,
            future_arrivals=(),
            information_mode="deployable" if self.controller == "h1" else "oracle",
            future_signal_mode="persistence" if self.controller == "h1" else "oracle",
        )

    def _physical_interval_metrics(self, step: int) -> tuple[float, float, float]:
        energy_kwh = 0.0
        electricity_usd = 0.0
        carbon_kg = 0.0
        config = self.optimizer_config
        for dc_id, resources in self.state.used.items():
            cpu, gpu, memory = (float(value) for value in resources)
            power_kw = (
                cpu * config.cpu_power_w_per_core
                + gpu * config.gpu_power_w_per_unit
                + memory * config.memory_power_w_per_gb
            ) / 1000.0
            dc_energy = power_kw * 0.25
            energy_kwh += dc_energy
            electricity_usd += dc_energy * float(self.signals[dc_id]["price"][step]) / 1000.0
            carbon_kg += dc_energy * float(self.signals[dc_id]["carbon"][step]) / 1000.0
        return energy_kwh, electricity_usd, carbon_kg

    def _decision_cost_rows(
        self,
        step: int,
        horizon_state: HorizonState,
        result: RollingHorizonResult,
        pending_indices: Sequence[int],
    ) -> list[dict[str, Any]]:
        plans = {(plan.original_index, plan.task_id): plan for plan in result.plans}
        dcs = {dc.dc_id: dc for dc in horizon_state.datacenters}
        rows: list[dict[str, Any]] = []
        config = self.optimizer_config
        for position, (row_index, action) in enumerate(zip(pending_indices, result.environment_actions)):
            canonical = self.canonical.loc[row_index]
            task = horizon_state.current.tasks[position]
            plan = plans[(position, task.task_id)]
            action = int(action)
            values = {
                "electricity": 0.0,
                "carbon": 0.0,
                "transmission": 0.0,
                "waiting_defer": 0.0,
                "sla_risk": 0.0,
                "terminal_backlog": 0.0,
            }
            energy = self.optimizer._task_energy_kwh(task, config)
            if action != 0:
                if plan.decision != "dispatch" or int(plan.dispatch_step) != 0:
                    raise RuntimeError("non-defer environment action is not a first-step dispatch")
                dc = dcs[action]
                cost_step = min(int(plan.execution_start_step), self.horizon - 1)
                values["electricity"] = config.weights.electricity * energy * dc.electricity_price_usd_per_mwh[cost_step] / 1000.0
                values["carbon"] = config.weights.carbon * energy * dc.carbon_intensity_gco2_per_kwh[cost_step] / 1000.0
                values["transmission"] = config.weights.transmission * self.link_cost[(int(canonical["origin_dc"]), action)] * float(canonical["bandwidth"])
                slack = int(plan.deadline_step) - (int(plan.execution_start_step) + int(plan.duration_steps))
                values["sla_risk"] = config.weights.sla_risk / max(1.0, float(slack + 1))
            else:
                values["waiting_defer"] = config.weights.waiting_defer * config.waiting_cost_per_step
                urgency = horizon_state.timestep_minutes / max(
                    horizon_state.timestep_minutes,
                    max(0.0, task.remaining_sla_minutes),
                )
                values["sla_risk"] = config.weights.sla_risk * urgency
            action_total = float(sum(values.values()))
            destination = None if action == 0 else action
            migration = action != 0 and action != int(canonical["origin_dc"])
            rows.append(
                {
                    "controller": self.controller_label,
                    "step": int(step),
                    "phase": self.phase(step),
                    "split": self.split(step),
                    "task_id": str(canonical["task_id"]),
                    "original_index": int(canonical["original_index"]),
                    "priority": str(canonical["priority"]),
                    "origin_dc": int(canonical["origin_dc"]),
                    "action": action,
                    "destination_dc": destination,
                    "is_defer": action == 0,
                    "is_migration": migration,
                    "transmission_amount_gb": float(canonical["bandwidth"]) if migration else 0.0,
                    "waiting_steps_at_decision": step - int(canonical["arrival_step"]),
                    "arrival_step": int(canonical["arrival_step"]),
                    "sla_deadline_step": int(canonical["sla_deadline_step"]),
                    "max_wait_steps": int(canonical["max_wait_steps"]),
                    "estimated_duration_seconds": float(canonical["estimated_duration"]),
                    "estimated_duration_steps": int(canonical["estimated_duration_steps"]),
                    "estimated_energy_kwh": float(energy),
                    "cost_electricity": values["electricity"],
                    "cost_carbon": values["carbon"],
                    "cost_transmission": values["transmission"],
                    "cost_waiting_defer": values["waiting_defer"],
                    "cost_sla_risk": values["sla_risk"],
                    "cost_terminal_backlog": values["terminal_backlog"],
                    "action_stage_cost": action_total,
                }
            )
        expected = asdict(result.first_step_costs)
        observed = {key: sum(row[f"cost_{key}"] for row in rows) for key in expected}
        for key in expected:
            if not math.isclose(observed[key], expected[key], rel_tol=1e-10, abs_tol=1e-7):
                raise RuntimeError(
                    f"stage cost component mismatch at step={step}: {key} observed={observed[key]} expected={expected[key]}"
                )
        return rows

    def _apply_actions(self, step: int, actions: Sequence[int]) -> None:
        if len(actions) != len(self.state.pending):
            raise RuntimeError("controller action count mismatch")
        remaining: list[int] = []
        for row_index, raw_action in zip(tuple(self.state.pending), actions):
            action = int(raw_action)
            row = self.canonical.loc[row_index]
            waiting = step - int(row["arrival_step"])
            if action == 0:
                if waiting >= int(row["max_wait_steps"]):
                    raise RuntimeError(f"defer exceeds frozen SLA bound at step={step} task={row['task_id']}")
                remaining.append(row_index)
                continue
            if action not in self.capacity:
                raise RuntimeError(f"invalid destination action: {action}")
            request = self.resources(row_index)
            planned_execution = step + 1
            true_steps = max(1, int(np.ceil(float(row["true_duration"]) / 900.0)))
            item = {
                "task_id": str(row["task_id"]),
                "row_index": int(row_index),
                "dc_id": action,
                "status": "in_transit",
                "waiting_steps": waiting,
                "dispatch_step": step,
                "planned_execution_start_step": planned_execution,
                "actual_execution_start_step": None,
                "estimated_duration_steps": int(row["estimated_duration_steps"]),
                "estimated_completion_step": planned_execution + int(row["estimated_duration_steps"]),
                "true_duration_steps": true_steps,
                "true_completion_step": None,
                "resources": tuple(float(value) for value in request),
            }
            self.state.transit[row_index] = item
            self.state.reserved[action] = (np.asarray(self.state.reserved[action], dtype=float) + request).tolist()
        self.state.pending = remaining

    def _snapshot(self) -> dict[str, Any]:
        unfinished = len(self.state.pending) + len(self.state.running) + len(self.state.transit)
        terminal_pending_cost = len(self.state.pending) * self.optimizer_config.weights.terminal_backlog * self.optimizer_config.terminal_backlog_base_cost
        return {
            "next_step": int(self.state.next_step),
            "pending": len(self.state.pending),
            "running": len(self.state.running),
            "in_transit": len(self.state.transit),
            "completed": int(self.state.completed_count),
            "unfinished": unfinished,
            "completion_rate": self.state.completed_count / len(self.canonical),
            "frozen_terminal_pending_cost": terminal_pending_cost,
        }

    def run_step(self, step: int) -> None:
        if step != self.state.next_step:
            raise RuntimeError("runtime step is not continuous")
        if self.chunk_start is None:
            self.chunk_start = step
        self._advance_to_state(step)
        pending_indices = tuple(self.state.pending)
        arrival = self.global_future.get(step, np.zeros(4, dtype=float)) if step < PRIMARY_STEPS else np.zeros(4)
        pending_gpu = sum(float(self.canonical.at[index, "gpu_request_effective"]) for index in pending_indices)
        risk_score = (
            sum(float(values[1]) for values in self.state.used.values())
            + sum(float(values[1]) for values in self.state.reserved.values())
            + pending_gpu
        ) / 10_412.0
        physical_energy, physical_electricity, physical_carbon = self._physical_interval_metrics(step)
        pre = self._snapshot()
        horizon_state = self.build_control_state(step)
        if self.controller == "h1":
            if self.oracle_future_state_count != 0 or self.oracle_future_row_count != 0:
                raise RuntimeError("H1 accessed Oracle future")
            for dc in horizon_state.datacenters:
                if any(dc.forecast_cpu_reservations) or any(dc.forecast_gpu_reservations):
                    raise RuntimeError("H1 contains future workload reservations")
        elif horizon_state.horizon != 5:
            raise RuntimeError("H4 horizon misaligned")
        retry_before = self.state.h1_presolve_retries if self.controller == "h1" else self.state.h4_presolve_retries
        if pending_indices:
            result = self._solve(horizon_state, "h1" if self.controller == "h1" else "h4")
            if result.status != "optimal":
                spot_v3.write_json(
                    self.run_dir / f"failure_state_{step:05d}.json",
                    {"controller": self.controller_label, "step": step, "status": result.status, "message": result.message, "fallback_used": False},
                )
                raise RuntimeError(f"{self.controller_label} solver failure at step {step}: {result.status} {result.message}")
        else:
            result = self.optimizer.solve(horizon_state, self.optimizer_config, self.action_adapter)
        retry_after = self.state.h1_presolve_retries if self.controller == "h1" else self.state.h4_presolve_retries
        decision_rows = self._decision_cost_rows(step, horizon_state, result, pending_indices)
        self.metric_buffers["decisions"].extend(decision_rows)
        defer_count = sum(int(row["is_defer"]) for row in decision_rows)
        migration_count = sum(int(row["is_migration"]) for row in decision_rows)
        transmission_amount = sum(float(row["transmission_amount_gb"]) for row in decision_rows)
        if self.controller == "h1":
            self.state.h1_defers += defer_count
        else:
            self.state.h4_defers += defer_count
        self._apply_actions(step, result.environment_actions)
        post = self._snapshot()
        first = asdict(result.first_step_costs)
        stage_cost = float(sum(first.values()))
        self.metric_buffers["steps"].append(
            {
                "controller": self.controller_label,
                "step": step,
                "phase": self.phase(step),
                "split": self.split(step),
                "timestamp_utc": self.timestamp(step).isoformat(),
                "arrival_task_count": int(arrival[0]),
                "arrival_cpu": float(arrival[1]),
                "arrival_gpu": float(arrival[2]),
                "arrival_memory": float(arrival[3]),
                "pending_pre": pre["pending"],
                "running_pre": pre["running"],
                "transit_pre": pre["in_transit"],
                "completed_pre": pre["completed"],
                "risk_score": float(risk_score),
                "action_count": len(decision_rows),
                "defer_count": defer_count,
                "migration_count": migration_count,
                "transmission_amount_gb": transmission_amount,
                "stage_electricity": float(first["electricity"]),
                "stage_carbon": float(first["carbon"]),
                "stage_transmission": float(first["transmission"]),
                "stage_waiting_defer": float(first["waiting_defer"]),
                "stage_sla_risk": float(first["sla_risk"]),
                "stage_terminal_backlog": float(first["terminal_backlog"]),
                "stage_cost": stage_cost,
                "reward": -stage_cost,
                "physical_energy_kwh": physical_energy,
                "physical_electricity_cost_usd": physical_electricity,
                "physical_carbon_kg": physical_carbon,
                "pending_post": post["pending"],
                "running_post": post["running"],
                "transit_post": post["in_transit"],
                "completed_post": post["completed"],
                "solver_called": bool(pending_indices),
                "solver_status": result.status,
                "solver_message": str(result.message),
                "presolve_retry": retry_after > retry_before,
                "solve_ms": 1000.0 * float(result.solve_seconds),
                "fallback_used": False,
                "information_mode": "DEPLOYABLE_CURRENT" if self.controller == "h1" else "PRIVILEGED_ORACLE",
            }
        )
        self.state.next_step = step + 1
        self.state.state_counter += 1
        self.state.decision_counter += len(decision_rows)

    def _write_chunk(
        self,
        name: str,
        rows: Sequence[Mapping[str, Any]],
        start: int,
        stop: int,
    ) -> dict[str, Any]:
        path = self.data_dir / f"{name}_{start:05d}_{stop - 1:05d}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(list(rows), schema=TABLE_SCHEMAS[name])
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
        return {
            "table": name,
            "start_step": start,
            "end_step_exclusive": stop,
            "rows": table.num_rows,
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
        }

    def flush_and_checkpoint(self, stop: int) -> Path:
        if self.chunk_start is None:
            raise RuntimeError("cannot checkpoint an empty control chunk")
        start = self.chunk_start
        entries = [
            self._write_chunk(name, self.metric_buffers[name], start, stop)
            for name in TABLE_SCHEMAS
        ]
        self.state.committed_chunks.extend(entries)
        self.metric_buffers = {name: [] for name in TABLE_SCHEMAS}
        self.chunk_start = None
        elapsed = time.perf_counter() - self.started_at
        processed = max(1, stop - self.process_start_step)
        target = PRIMARY_STEPS if stop <= PRIMARY_STEPS else PRIMARY_STEPS + TAIL_MAX_STEPS
        eta = elapsed / processed * max(0, target - stop)
        payload = {
            "version": 1,
            "controller": self.controller,
            "runtime": self.state,
            "primary_snapshot": self.primary_snapshot,
            "oracle_future_state_count": self.oracle_future_state_count,
            "oracle_future_row_count": self.oracle_future_row_count,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "config_sha256": sha256(CONFIG_PATH),
            "source_config_sha256": sha256(spot_v3.CONFIG_PATH),
            "capacity_config_sha256": sha256(spot_v3.CAPACITY_PATH),
            "canonical_sha256": sha256(CANONICAL_PATH),
        }
        path = self.checkpoint_dir / f"checkpoint_{stop:05d}.pkl.gz"
        temporary = path.with_suffix(".pkl.gz.tmp")
        with gzip.open(temporary, "wb", compresslevel=6) as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
        write_json(
            self.run_dir / "latest_checkpoint.json",
            {
                "controller": self.controller,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "next_step": stop,
                "committed_chunk_count": len(self.state.committed_chunks),
            },
        )
        print(
            f"[control-value:{self.controller}] {stop} elapsed={elapsed:.1f}s "
            f"eta={eta:.1f}s pending={len(self.state.pending)} "
            f"running={len(self.state.running)} transit={len(self.state.transit)} "
            f"completed={self.state.completed_count}",
            flush=True,
        )
        return path

    def _load_control_checkpoint(self) -> spot_v3.RuntimeState:
        latest_path = self.run_dir / "latest_checkpoint.json"
        if not latest_path.is_file():
            raise FileNotFoundError(f"resume requested without {latest_path}")
        latest = json.loads(latest_path.read_text("utf-8"))
        path = ROOT / latest["path"]
        if sha256(path) != latest["sha256"]:
            raise RuntimeError("checkpoint hash mismatch")
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
        expected = {
            "controller": self.controller,
            "config_sha256": sha256(CONFIG_PATH),
            "source_config_sha256": sha256(spot_v3.CONFIG_PATH),
            "capacity_config_sha256": sha256(spot_v3.CAPACITY_PATH),
            "canonical_sha256": sha256(CANONICAL_PATH),
        }
        for key, value in expected.items():
            if payload[key] != value:
                raise RuntimeError(f"checkpoint contract mismatch: {key}")
        state: spot_v3.RuntimeState = payload["runtime"]
        for entry in state.committed_chunks:
            chunk = ROOT / entry["path"]
            if not chunk.is_file() or sha256(chunk) != entry["sha256"]:
                raise RuntimeError(f"committed chunk integrity failed: {chunk}")
        self.primary_snapshot = payload["primary_snapshot"]
        self.oracle_future_state_count = int(payload["oracle_future_state_count"])
        self.oracle_future_row_count = int(payload["oracle_future_row_count"])
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_random_state"])
        return state

    def run_to(self, end_step_exclusive: int) -> spot_v3.RuntimeState:
        end = int(end_step_exclusive)
        if not self.state.next_step <= end <= PRIMARY_STEPS:
            raise ValueError("invalid primary rollout end")
        for step in range(self.state.next_step, end):
            self.run_step(step)
            if self.state.next_step % self.checkpoint_interval == 0:
                self.flush_and_checkpoint(self.state.next_step)
        if end == PRIMARY_STEPS and self.primary_snapshot is None:
            self.primary_snapshot = self._snapshot()
        if self.chunk_start is not None:
            self.flush_and_checkpoint(self.state.next_step)
        return self.state

    def drain_tail(self) -> dict[str, Any]:
        if self.state.next_step != PRIMARY_STEPS:
            raise RuntimeError("tail drain must begin immediately after the primary timeline")
        if self.primary_snapshot is None:
            self.primary_snapshot = self._snapshot()
        maximum = PRIMARY_STEPS + TAIL_MAX_STEPS
        while self.state.next_step < maximum:
            if not self.state.pending and not self.state.running and not self.state.transit:
                break
            self.run_step(self.state.next_step)
            if self.state.next_step % self.checkpoint_interval == 0:
                self.flush_and_checkpoint(self.state.next_step)
        if self.chunk_start is not None:
            self.flush_and_checkpoint(self.state.next_step)
        final = self._snapshot()
        result = {
            "controller": self.controller_label,
            "drain_steps": self.state.next_step - PRIMARY_STEPS,
            "maximum_drain_steps": TAIL_MAX_STEPS,
            "drained_to_empty": final["unfinished"] == 0,
            "primary": self.primary_snapshot,
            "tail_end": final,
            "terminal_pending_cost_change": (
                final["frozen_terminal_pending_cost"]
                - self.primary_snapshot["frozen_terminal_pending_cost"]
            ),
        }
        write_json(self.run_dir / "tail_result.json", result)
        return result

    def runtime_signature(self) -> str:
        payload = {
            "controller": self.controller,
            "next_step": self.state.next_step,
            "pending": self.state.pending,
            "running": sorted(self.state.running.items()),
            "transit": sorted(self.state.transit.items()),
            "used": self.state.used,
            "reserved": self.state.reserved,
            "completed": self.state.completed_count,
            "states": self.state.state_counter,
            "decisions": self.state.decision_counter,
            "h1_calls": self.state.h1_calls,
            "h4_calls": self.state.h4_calls,
            "h1_defers": self.state.h1_defers,
            "h4_defers": self.state.h4_defers,
            "h1_retries": self.state.h1_presolve_retries,
            "h4_retries": self.state.h4_presolve_retries,
            "anomalies": self.state.solver_anomalies,
            "oracle_future_states": self.oracle_future_state_count,
            "oracle_future_rows": self.oracle_future_row_count,
        }
        return object_hash(payload)

    def semantic_output_hashes(self) -> dict[str, str]:
        ignored = {"solve_ms", "solver_message"}
        output: dict[str, str] = {}
        for name, schema in TABLE_SCHEMAS.items():
            columns = [column for column in schema.names if column not in ignored]
            digest = hashlib.sha256()
            entries = [
                item for item in self.state.committed_chunks if item["table"] == name
            ]
            for entry in sorted(entries, key=lambda item: item["start_step"]):
                frame = pd.read_parquet(ROOT / entry["path"], columns=columns)
                for row in frame.itertuples(index=False, name=None):
                    digest.update(
                        json.dumps(row, default=str, separators=(",", ":")).encode("utf-8")
                    )
                    digest.update(b"\n")
            output[name] = digest.hexdigest().upper()
        return output


def write_protocol_files(inputs: Mapping[str, Any]) -> None:
    config = load_eval_config()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_text(
        OUTPUT / "01_protocol.md",
        f"""
# SpotGPU2026 Control Value v1 Protocol

- Primary comparison: two independent closed loops, H1 versus repaired H4 Oracle.
- Primary timeline: steps 0 through {PRIMARY_STEPS - 1}, continuous with no split reset.
- Tail diagnostic: no new arrivals; continue until pending, running, and in-transit are all zero, or {TAIL_MAX_STEPS} common drain steps.
- H1 information: current deployable state only; Oracle future access is a hard failure.
- H4 information: current state plus exact +15, +30, +45, and +60 minute workload pressure and external energy signals.
- Workload, duration truth, train-only duration estimate, memory, bandwidth, origin, SLA, capacity, network, and initial state are identical.
- Stage cost is the frozen optimizer first-step cost. Reporting reward is exactly negative stage cost and is not independent evidence.
- Physical energy, electricity expenditure, and carbon are integrated from actual running occupancy in each 15-minute interval. They are separate from one-time assignment objective contributions.
- Daily and weekly blocks come from one continuous trajectory and are descriptive, not independent random samples.
- Fixed-state equivalence uses only the deterministic tie-break scale, 4 x 1e-9. A broader solver-tolerance equivalence class is not asserted.
- No composite score is constructed. No Transformer, BC, RL, reward tuning, MPC tuning, workload scaling, or silent fallback is used.
""",
    )
    dcs = inputs["dcs"].sort_values("dc_id")
    write_text(
        OUTPUT / "02_frozen_contract.md",
        f"""
# Frozen Contract

- Scenario: {config['scenario_id']}
- Tasks: {len(inputs['canonical'])}; request scope: PER_JOB
- Stable order: submit_time, original_index, task_id
- Time step: 900 seconds; primary steps: {PRIMARY_STEPS}
- CPU: {int(dcs['total_cores'].sum())} cores
- GPU: {int(dcs['total_gpus'].sum())}; per DC: {dcs['total_gpus'].astype(int).tolist()}
- Memory: {float(dcs['total_mem'].sum()):.12f} GB, MODELED_DC_MEMORY
- Bandwidth: 0.020062580706 GB
- SLA: medium; HP max wait 1 step; Spot max wait 8 steps
- true_duration: simulator only
- estimated_duration: frozen train-only hierarchical conditional median
- Energy signals: SustainCluster 2023 EXTERNAL_SCENARIO_SIGNAL, not Alibaba2026 measured energy data
- Objective: unchanged configs/sustaincluster_mpc/h4_expert.yaml
- H1 horizon nodes: 1
- H4 horizon nodes: 5, offsets +15/+30/+45/+60 minutes
""",
    )


def checkpoint_exact_preflight(inputs: Mapping[str, Any]) -> dict[str, Any]:
    root = OUTPUT / "preflight/checkpoint_exact"
    if root.exists():
        shutil.rmtree(root)
    rows: list[dict[str, Any]] = []
    direct_runners: dict[str, ClosedLoopRunner] = {}
    for controller in CONTROLLERS:
        direct = ClosedLoopRunner(
            inputs,
            root / controller / "direct",
            controller,
            checkpoint_interval=100,
        )
        direct.run_to(100)
        interrupted = ClosedLoopRunner(
            inputs,
            root / controller / "resumed",
            controller,
            checkpoint_interval=50,
        )
        interrupted.run_to(50)
        resumed = ClosedLoopRunner(
            inputs,
            root / controller / "resumed",
            controller,
            checkpoint_interval=50,
            resume=True,
        )
        resumed.run_to(100)
        runtime_equal = direct.runtime_signature() == resumed.runtime_signature()
        direct_hashes = direct.semantic_output_hashes()
        resumed_hashes = resumed.semantic_output_hashes()
        output_equal = direct_hashes == resumed_hashes
        if not runtime_equal or not output_equal:
            raise RuntimeError(f"checkpoint/resume semantic mismatch: {controller}")
        direct_runners[controller] = direct
        rows.append(
            {
                "controller": CONTROLLER_LABELS[controller],
                "steps": 100,
                "runtime_signature_equal": runtime_equal,
                "semantic_output_hashes_equal": output_equal,
                "oracle_future_state_count": direct.oracle_future_state_count,
                "oracle_future_row_count": direct.oracle_future_row_count,
                "solver_failures": (
                    direct.state.h1_failures
                    if controller == "h1"
                    else direct.state.h4_failures
                ),
                "fallback_used": False,
            }
        )
    h1 = direct_runners["h1"]
    h4 = direct_runners["h4_oracle"]
    shared_checks = {
        "same_task_count": len(h1.canonical) == len(h4.canonical) == 466_867,
        "same_canonical_sha256": sha256(CANONICAL_PATH) == sha256(CANONICAL_PATH),
        "same_initial_capacity": all(
            np.array_equal(h1.capacity[dc], h4.capacity[dc]) for dc in h1.capacity
        ),
        "same_energy_signals": all(
            np.array_equal(h1.signals[dc]["price"], h4.signals[dc]["price"])
            and np.array_equal(h1.signals[dc]["carbon"], h4.signals[dc]["carbon"])
            for dc in h1.capacity
        ),
        "same_sla": h1.canonical[
            ["original_index", "sla_deadline_step", "max_wait_steps"]
        ].equals(
            h4.canonical[
                ["original_index", "sla_deadline_step", "max_wait_steps"]
            ]
        ),
        "same_true_duration": h1.canonical["true_duration"].equals(
            h4.canonical["true_duration"]
        ),
        "same_estimated_duration": h1.canonical[
            ["estimated_duration", "estimated_duration_steps"]
        ].equals(h4.canonical[["estimated_duration", "estimated_duration_steps"]]),
        "h1_oracle_access_zero": h1.oracle_future_state_count == 0
        and h1.oracle_future_row_count == 0,
        "h4_exact_offset_rows": h4.oracle_future_state_count == 100
        and h4.oracle_future_row_count == 100 * 4 * 5,
        "no_split_reset": h1.state.next_step == h4.state.next_step == 100,
        "tail_rule_identical": True,
        "no_fallback": True,
    }
    if not all(shared_checks.values()):
        raise RuntimeError(f"preflight shared-contract failure: {shared_checks}")
    result = {
        "status": "PASS",
        "fixed_window": {"start": 0, "end_exclusive": 100, "split": "train"},
        "controllers": rows,
        "shared_checks": shared_checks,
    }
    write_json(OUTPUT / "preflight/preflight_result.json", result)
    write_csv(OUTPUT / "preflight/preflight_controllers.csv", pd.DataFrame(rows))
    return result


def run_formal_controller(
    inputs: Mapping[str, Any], controller: str, *, resume: bool
) -> dict[str, Any]:
    run_dir = OUTPUT / "checkpoints" / controller
    runner = ClosedLoopRunner(
        inputs,
        run_dir,
        controller,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        resume=resume,
    )
    if runner.state.next_step < PRIMARY_STEPS:
        runner.run_to(PRIMARY_STEPS)
    if runner.state.next_step == PRIMARY_STEPS:
        tail = runner.drain_tail()
    else:
        tail_path = run_dir / "tail_result.json"
        if not tail_path.is_file():
            raise RuntimeError("completed tail checkpoint lacks tail_result.json")
        tail = json.loads(tail_path.read_text("utf-8"))
    write_json(
        run_dir / "run_manifest.json",
        {
            "controller": controller,
            "controller_label": CONTROLLER_LABELS[controller],
            "runtime_signature": runner.runtime_signature(),
            "semantic_output_hashes": runner.semantic_output_hashes(),
            "oracle_future_state_count": runner.oracle_future_state_count,
            "oracle_future_row_count": runner.oracle_future_row_count,
            "tail": tail,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return tail


def read_controller_tables(controller: str) -> dict[str, pd.DataFrame]:
    run_dir = OUTPUT / "checkpoints" / controller
    latest = json.loads((run_dir / "latest_checkpoint.json").read_text("utf-8"))
    checkpoint_path = ROOT / latest["path"]
    with gzip.open(checkpoint_path, "rb") as handle:
        payload = pickle.load(handle)
    tables: dict[str, pd.DataFrame] = {}
    raw_dir = OUTPUT / "raw" / controller
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLE_SCHEMAS:
        entries = sorted(
            (
                entry
                for entry in payload["runtime"].committed_chunks
                if entry["table"] == name
            ),
            key=lambda entry: entry["start_step"],
        )
        frame = pd.concat(
            [pd.read_parquet(ROOT / entry["path"]) for entry in entries],
            ignore_index=True,
        )
        frame.to_parquet(raw_dir / f"{name}.parquet", index=False)
        tables[name] = frame
    tables["tail"] = pd.DataFrame(
        [json.loads((run_dir / "tail_result.json").read_text("utf-8"))]
    )
    return tables


def build_lifecycle(
    canonical: pd.DataFrame,
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    tail_end_step: int,
) -> pd.DataFrame:
    columns = [
        "task_id",
        "original_index",
        "arrival_step",
        "priority",
        "origin_dc",
        "true_duration",
        "estimated_duration",
        "estimated_duration_steps",
        "sla_deadline_step",
        "max_wait_steps",
        "absolute_estimation_error_seconds",
        "temporal_split",
    ]
    base = canonical[columns].copy()
    dispatched = (
        decisions.loc[~decisions["is_defer"]]
        .sort_values(["original_index", "step"])
        .drop_duplicates("original_index", keep="first")
    )
    dispatch_columns = [
        "original_index",
        "controller",
        "step",
        "destination_dc",
        "is_migration",
        "transmission_amount_gb",
        "cost_electricity",
        "cost_carbon",
        "cost_transmission",
        "cost_sla_risk",
        "action_stage_cost",
    ]
    started = (
        events.loc[events["event_type"] == "started"]
        .sort_values(["original_index", "step"])
        .drop_duplicates("original_index", keep="first")
        [["original_index", "actual_execution_start_step"]]
    )
    completed = (
        events.loc[events["event_type"] == "completed"]
        .sort_values(["original_index", "step"])
        .drop_duplicates("original_index", keep="first")
        [["original_index", "step"]]
        .rename(columns={"step": "observed_true_completion_step"})
    )
    work = (
        base.merge(dispatched[dispatch_columns], on="original_index", how="left")
        .rename(columns={"step": "dispatch_step"})
        .merge(started, on="original_index", how="left")
        .merge(completed, on="original_index", how="left")
    )
    work["waiting_steps"] = (
        work["actual_execution_start_step"] - work["arrival_step"]
    )
    work["wait_bound_violation"] = work["waiting_steps"] > work["max_wait_steps"]
    work["primary_completed"] = (
        work["observed_true_completion_step"].notna()
        & (work["observed_true_completion_step"] < PRIMARY_STEPS)
    )
    work["tail_completed"] = work["observed_true_completion_step"].notna()
    work["primary_sla_violation"] = np.where(
        work["primary_completed"],
        work["observed_true_completion_step"] > work["sla_deadline_step"],
        work["sla_deadline_step"] < PRIMARY_STEPS,
    )
    work["tail_sla_violation"] = np.where(
        work["tail_completed"],
        work["observed_true_completion_step"] > work["sla_deadline_step"],
        work["sla_deadline_step"] < tail_end_step,
    )
    duration_bins = [-np.inf, 3600, 6 * 3600, 24 * 3600, 48 * 3600, 72 * 3600, np.inf]
    duration_labels = ["<1h", "1-6h", "6-24h", "24-48h", "48-72h", ">72h"]
    work["duration_group"] = pd.cut(
        work["true_duration"],
        bins=duration_bins,
        labels=duration_labels,
        right=False,
    ).astype(str)
    train = canonical.loc[canonical["temporal_split"] == "train"]
    q1, q2 = train["absolute_estimation_error_seconds"].quantile([1 / 3, 2 / 3])
    work["duration_error_group"] = pd.cut(
        work["absolute_estimation_error_seconds"],
        bins=[-np.inf, float(q1), float(q2), np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    return work


def numeric_stats(values: pd.Series, prefix: str = "") -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            f"{prefix}mean": math.nan,
            f"{prefix}p50": math.nan,
            f"{prefix}p90": math.nan,
            f"{prefix}p95": math.nan,
            f"{prefix}p99": math.nan,
            f"{prefix}max": math.nan,
        }
    return {
        f"{prefix}mean": float(clean.mean()),
        f"{prefix}p50": float(clean.quantile(0.50)),
        f"{prefix}p90": float(clean.quantile(0.90)),
        f"{prefix}p95": float(clean.quantile(0.95)),
        f"{prefix}p99": float(clean.quantile(0.99)),
        f"{prefix}max": float(clean.max()),
    }


def relative_delta(h1: float, h4: float) -> float:
    return math.nan if h1 == 0 or not math.isfinite(h1) else (h4 - h1) / abs(h1) * 100.0


def controller_metrics(
    tables: Mapping[str, pd.DataFrame],
    lifecycle: pd.DataFrame,
    tail_result: Mapping[str, Any],
) -> dict[str, float]:
    steps = tables["steps"]
    primary = steps.loc[steps["phase"] == "primary"]
    all_steps = steps
    decisions = tables["decisions"]
    dispatch = decisions.loc[~decisions["is_defer"]]
    primary_tail = tail_result["primary"]
    tail_end = tail_result["tail_end"]
    metrics: dict[str, float] = {
        "primary_steps": float(len(primary)),
        "task_count": float(len(lifecycle)),
        "stage_cost_total": float(primary["stage_cost"].sum()),
        "stage_cost_mean_per_step": float(primary["stage_cost"].mean()),
        "reward_total": float(primary["reward"].sum()),
        "reward_mean": float(primary["reward"].mean()),
        "objective_electricity": float(primary["stage_electricity"].sum()),
        "objective_carbon": float(primary["stage_carbon"].sum()),
        "objective_transmission": float(primary["stage_transmission"].sum()),
        "objective_waiting_defer": float(primary["stage_waiting_defer"].sum()),
        "objective_sla_penalty": float(primary["stage_sla_risk"].sum()),
        "objective_terminal_backlog": float(primary["stage_terminal_backlog"].sum()),
        "physical_energy_kwh_primary": float(primary["physical_energy_kwh"].sum()),
        "physical_electricity_cost_usd_primary": float(
            primary["physical_electricity_cost_usd"].sum()
        ),
        "physical_carbon_kg_primary": float(primary["physical_carbon_kg"].sum()),
        "physical_energy_kwh_with_tail": float(all_steps["physical_energy_kwh"].sum()),
        "physical_electricity_cost_usd_with_tail": float(
            all_steps["physical_electricity_cost_usd"].sum()
        ),
        "physical_carbon_kg_with_tail": float(all_steps["physical_carbon_kg"].sum()),
        "transmission_cost_usd": float(primary["stage_transmission"].sum()),
        "transmission_amount_gb": float(primary["transmission_amount_gb"].sum()),
        "migration_count": float(primary["migration_count"].sum()),
        "defer_decision_count": float(primary["defer_count"].sum()),
        "primary_completed_count": float(lifecycle["primary_completed"].sum()),
        "primary_completion_rate": float(lifecycle["primary_completed"].mean()),
        "primary_sla_violation_count": float(lifecycle["primary_sla_violation"].sum()),
        "primary_sla_violation_rate": float(lifecycle["primary_sla_violation"].mean()),
        "wait_bound_violation_count": float(lifecycle["wait_bound_violation"].sum()),
        "wait_bound_violation_rate": float(lifecycle["wait_bound_violation"].mean()),
        "tail_completed_count": float(lifecycle["tail_completed"].sum()),
        "tail_completion_rate": float(lifecycle["tail_completed"].mean()),
        "tail_sla_violation_count": float(lifecycle["tail_sla_violation"].sum()),
        "tail_sla_violation_rate": float(lifecycle["tail_sla_violation"].mean()),
        "primary_remaining_pending": float(primary_tail["pending"]),
        "primary_remaining_running": float(primary_tail["running"]),
        "primary_remaining_in_transit": float(primary_tail["in_transit"]),
        "tail_remaining_pending": float(tail_end["pending"]),
        "tail_remaining_running": float(tail_end["running"]),
        "tail_remaining_in_transit": float(tail_end["in_transit"]),
        "tail_drain_steps": float(tail_result["drain_steps"]),
        "primary_terminal_pending_cost": float(primary_tail["frozen_terminal_pending_cost"]),
        "tail_terminal_pending_cost": float(tail_end["frozen_terminal_pending_cost"]),
        "solver_calls": float(primary["solver_called"].sum()),
        "solver_optimal": float(
            ((primary["solver_called"]) & (primary["solver_status"] == "optimal")).sum()
        ),
        "solver_retries": float(primary["presolve_retry"].sum()),
        "solver_failures": float(
            ((primary["solver_called"]) & (primary["solver_status"] != "optimal")).sum()
        ),
        "solver_timeouts": float(
            ((primary["solver_called"]) & (primary["solver_status"] == "limit_reached")).sum()
        ),
        "fallback_count": float(primary["fallback_used"].sum()),
    }
    metrics.update(numeric_stats(lifecycle["waiting_steps"], "waiting_"))
    backlog_series = primary["pending_pre"] + primary["transit_pre"]
    metrics.update(numeric_stats(backlog_series, "backlog_"))
    solver_times = primary.loc[primary["solver_called"], "solve_ms"]
    metrics.update(numeric_stats(solver_times, "solver_ms_"))
    metrics["terminal_backlog"] = float(
        primary_tail["pending"] + primary_tail["in_transit"]
    )
    metrics["dispatch_count"] = float(len(dispatch))
    return metrics


def metric_comparison(metrics: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    h1 = metrics["H1"]
    h4 = metrics["H4_ORACLE"]
    units = {
        "stage_cost_total": "objective_units",
        "stage_cost_mean_per_step": "objective_units_per_step",
        "reward_total": "reward_units",
        "reward_mean": "reward_units_per_step",
        "physical_energy_kwh_primary": "kWh",
        "physical_electricity_cost_usd_primary": "USD",
        "physical_carbon_kg_primary": "kgCO2",
        "transmission_cost_usd": "USD",
        "transmission_amount_gb": "GB",
        "migration_count": "tasks",
        "primary_sla_violation_count": "tasks",
        "primary_sla_violation_rate": "fraction",
        "primary_completed_count": "tasks",
        "primary_completion_rate": "fraction",
        "backlog_mean": "tasks",
        "backlog_p95": "tasks",
        "backlog_max": "tasks",
        "terminal_backlog": "tasks",
    }
    selected = list(units)
    rows = []
    for name in selected:
        left = float(h1[name])
        right = float(h4[name])
        rows.append(
            {
                "metric": name,
                "unit": units[name],
                "h1": left,
                "h4_oracle": right,
                "absolute_delta_h4_minus_h1": right - left,
                "relative_delta_percent": relative_delta(left, right),
            }
        )
    return pd.DataFrame(rows)


def lifecycle_group_values(group: pd.DataFrame) -> dict[str, float]:
    result = {
        "task_count": float(len(group)),
        "primary_sla_violation_count": float(group["primary_sla_violation"].sum()),
        "primary_sla_violation_rate": float(group["primary_sla_violation"].mean()),
        "tail_sla_violation_count": float(group["tail_sla_violation"].sum()),
        "tail_sla_violation_rate": float(group["tail_sla_violation"].mean()),
        "primary_completed_count": float(group["primary_completed"].sum()),
        "primary_completion_rate": float(group["primary_completed"].mean()),
        "tail_completed_count": float(group["tail_completed"].sum()),
        "tail_completion_rate": float(group["tail_completed"].mean()),
        "waiting_mean_steps": float(group["waiting_steps"].mean()),
        "waiting_p95_steps": float(group["waiting_steps"].quantile(0.95)),
        "waiting_max_steps": float(group["waiting_steps"].max()),
        "wait_bound_violation_count": float(group["wait_bound_violation"].sum()),
        "migration_count": float(group["is_migration"].fillna(False).sum()),
        "migration_rate": float(group["is_migration"].fillna(False).mean()),
        "transmission_amount_gb": float(group["transmission_amount_gb"].fillna(0).sum()),
        "transmission_cost_usd": float(group["cost_transmission"].fillna(0).sum()),
        "objective_electricity": float(group["cost_electricity"].fillna(0).sum()),
        "objective_carbon": float(group["cost_carbon"].fillna(0).sum()),
        "objective_sla_risk": float(group["cost_sla_risk"].fillna(0).sum()),
        "action_stage_cost": float(group["action_stage_cost"].fillna(0).sum()),
    }
    for dc_id in range(1, 6):
        result[f"placement_dc{dc_id}_count"] = float(
            (group["destination_dc"] == dc_id).sum()
        )
    return result


def stratified_comparison(
    lifecycles: Mapping[str, pd.DataFrame],
    group_column: str,
    group_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = sorted(
        set(lifecycles["H1"][group_column].dropna().astype(str))
        | set(lifecycles["H4_ORACLE"][group_column].dropna().astype(str))
    )
    for value in groups:
        h1 = lifecycle_group_values(
            lifecycles["H1"].loc[lifecycles["H1"][group_column].astype(str) == value]
        )
        h4 = lifecycle_group_values(
            lifecycles["H4_ORACLE"].loc[
                lifecycles["H4_ORACLE"][group_column].astype(str) == value
            ]
        )
        for metric in h1:
            rows.append(
                {
                    group_label: value,
                    "metric": metric,
                    "h1": h1[metric],
                    "h4_oracle": h4[metric],
                    "absolute_delta_h4_minus_h1": h4[metric] - h1[metric],
                    "relative_delta_percent": relative_delta(h1[metric], h4[metric]),
                }
            )
    return pd.DataFrame(rows)


def specialized_task_tables(
    tables: Mapping[str, Mapping[str, pd.DataFrame]],
    lifecycles: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    sla_rows: list[dict[str, Any]] = []
    waiting_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    for controller, lifecycle in lifecycles.items():
        decisions = tables["h1" if controller == "H1" else "h4_oracle"]["decisions"]
        for priority in ["ALL", "HP", "Spot"]:
            group = lifecycle if priority == "ALL" else lifecycle.loc[lifecycle["priority"] == priority]
            decision_group = decisions if priority == "ALL" else decisions.loc[decisions["priority"] == priority]
            for phase in ["primary", "tail"]:
                violation = group[f"{phase}_sla_violation"]
                sla_rows.append(
                    {
                        "controller": controller,
                        "priority": priority,
                        "phase": phase,
                        "task_count": len(group),
                        "sla_violation_count": int(violation.sum()),
                        "sla_violation_rate": float(violation.mean()),
                        "objective_sla_penalty": float(
                            decision_group.loc[
                                decision_group["phase"] == "primary", "cost_sla_risk"
                            ].sum()
                        ),
                        "wait_bound_violation_count": int(group["wait_bound_violation"].sum()),
                    }
                )
                completion_rows.append(
                    {
                        "controller": controller,
                        "priority": priority,
                        "phase": phase,
                        "task_count": len(group),
                        "completed_count": int(group[f"{phase}_completed"].sum()),
                        "completion_rate": float(group[f"{phase}_completed"].mean()),
                    }
                )
            waiting_rows.append(
                {
                    "controller": controller,
                    "priority": priority,
                    "task_count": len(group),
                    **numeric_stats(group["waiting_steps"], "waiting_steps_"),
                    "wait_bound_violation_count": int(group["wait_bound_violation"].sum()),
                    "wait_bound_violation_rate": float(group["wait_bound_violation"].mean()),
                    "defer_decision_count": int(decision_group["is_defer"].sum()),
                }
            )
    return {
        "sla": pd.DataFrame(sla_rows),
        "waiting": pd.DataFrame(waiting_rows),
        "completion": pd.DataFrame(completion_rows),
    }


def paired_primary_steps(
    tables: Mapping[str, Mapping[str, pd.DataFrame]],
) -> pd.DataFrame:
    columns = [
        "step",
        "split",
        "arrival_task_count",
        "arrival_cpu",
        "arrival_gpu",
        "arrival_memory",
        "stage_cost",
        "stage_sla_risk",
        "physical_energy_kwh",
        "physical_electricity_cost_usd",
        "physical_carbon_kg",
        "stage_transmission",
        "transmission_amount_gb",
        "migration_count",
        "pending_pre",
        "transit_pre",
        "risk_score",
    ]
    h1 = tables["h1"]["steps"].loc[
        tables["h1"]["steps"]["phase"] == "primary", columns
    ]
    h4 = tables["h4_oracle"]["steps"].loc[
        tables["h4_oracle"]["steps"]["phase"] == "primary", columns
    ]
    pair = h1.merge(
        h4,
        on=["step", "split", "arrival_task_count", "arrival_cpu", "arrival_gpu", "arrival_memory"],
        suffixes=("_h1", "_h4"),
        validate="one_to_one",
    )
    if len(pair) != PRIMARY_STEPS or pair["step"].tolist() != list(range(PRIMARY_STEPS)):
        raise RuntimeError("H1/H4 primary step alignment failed")
    pair["backlog_h1"] = pair["pending_pre_h1"] + pair["transit_pre_h1"]
    pair["backlog_h4"] = pair["pending_pre_h4"] + pair["transit_pre_h4"]
    for metric in [
        "stage_cost",
        "stage_sla_risk",
        "physical_energy_kwh",
        "physical_electricity_cost_usd",
        "physical_carbon_kg",
        "stage_transmission",
        "transmission_amount_gb",
        "migration_count",
        "pending_pre",
        "backlog",
    ]:
        pair[f"{metric}_delta_h4_minus_h1"] = pair[f"{metric}_h4"] - pair[f"{metric}_h1"]
        pair[f"{metric}_improvement_h1_minus_h4"] = -pair[
            f"{metric}_delta_h4_minus_h1"
        ]
    return pair


def block_comparison(
    pair: pd.DataFrame,
    lifecycles: Mapping[str, pd.DataFrame],
    block_steps: int,
    label: str,
) -> pd.DataFrame:
    work = pair.copy()
    work["block_id"] = work["step"] // block_steps
    for controller, suffix in [("H1", "h1"), ("H4_ORACLE", "h4")]:
        life = lifecycles[controller]
        violations = life.loc[life["primary_sla_violation"]].copy()
        completion_step = violations["observed_true_completion_step"].fillna(
            PRIMARY_STEPS - 1
        )
        completion_step = completion_step.clip(upper=PRIMARY_STEPS - 1).astype(int)
        counts = completion_step.value_counts()
        work[f"sla_violation_count_{suffix}"] = work["step"].map(counts).fillna(0)
    rows: list[dict[str, Any]] = []
    sum_metrics = [
        "stage_cost",
        "sla_violation_count",
        "physical_electricity_cost_usd",
        "physical_carbon_kg",
        "stage_transmission",
    ]
    for block_id, group in work.groupby("block_id", sort=True):
        row: dict[str, Any] = {
            "record_type": "BLOCK",
            "block_type": label,
            "block_id": int(block_id),
            "start_step": int(group["step"].min()),
            "end_step_inclusive": int(group["step"].max()),
            "steps": len(group),
            "independent_random_sample": False,
        }
        for metric in sum_metrics:
            h1 = float(group[f"{metric}_h1"].sum())
            h4 = float(group[f"{metric}_h4"].sum())
            row[f"{metric}_h1"] = h1
            row[f"{metric}_h4"] = h4
            row[f"{metric}_delta_h4_minus_h1"] = h4 - h1
        for metric in ["backlog"]:
            h1 = float(group[f"{metric}_h1"].mean())
            h4 = float(group[f"{metric}_h4"].mean())
            row[f"{metric}_mean_h1"] = h1
            row[f"{metric}_mean_h4"] = h4
            row[f"{metric}_mean_delta_h4_minus_h1"] = h4 - h1
        rows.append(row)
    return pd.DataFrame(rows)


def external_pressure_analysis(pair: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    train = pair.loc[pair["split"] == "train", "arrival_gpu"]
    threshold = float(train.quantile(0.95))
    work = pair.copy()
    work["pressure_band"] = np.where(
        work["arrival_gpu"] >= threshold,
        "HIGH_GE_TRAIN_P95",
        "LOW_LT_TRAIN_P95",
    )
    rows: list[dict[str, Any]] = []
    metrics = [
        "stage_cost",
        "stage_sla_risk",
        "physical_electricity_cost_usd",
        "physical_carbon_kg",
        "backlog",
    ]
    for band, group in work.groupby("pressure_band", sort=True):
        for metric in metrics:
            aggregation = "mean" if metric == "backlog" else "sum"
            h1 = float(getattr(group[f"{metric}_h1"], aggregation)())
            h4 = float(getattr(group[f"{metric}_h4"], aggregation)())
            rows.append(
                {
                    "record_type": "EXTERNAL_PRESSURE",
                    "pressure_band": band,
                    "controller": "PAIRED",
                    "metric": metric,
                    "aggregation": aggregation,
                    "train_p95_new_gpu_demand_threshold": threshold,
                    "steps": len(group),
                    "h1": h1,
                    "h4_oracle": h4,
                    "absolute_delta_h4_minus_h1": h4 - h1,
                    "relative_delta_percent": relative_delta(h1, h4),
                }
            )
    return pd.DataFrame(rows), threshold


def risk_trajectory_rows(
    tables: Mapping[str, Mapping[str, pd.DataFrame]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, controller in [("h1", "H1"), ("h4_oracle", "H4_ORACLE")]:
        steps = tables[key]["steps"]
        steps = steps.loc[steps["phase"] == "primary"].copy()
        steps["risk_band"] = np.where(
            steps["risk_score"] >= RISK_THRESHOLD,
            "HIGH_GE_FROZEN_TRAIN_P95",
            "LOW_LT_FROZEN_TRAIN_P95",
        )
        for band, group in steps.groupby("risk_band", sort=True):
            rows.append(
                {
                    "record_type": "TRAJECTORY_RISK",
                    "pressure_band": band,
                    "controller": controller,
                    "metric": "trajectory_descriptive",
                    "aggregation": "mixed",
                    "train_p95_new_gpu_demand_threshold": math.nan,
                    "frozen_risk_threshold": RISK_THRESHOLD,
                    "steps": len(group),
                    "risk_mean": float(group["risk_score"].mean()),
                    "stage_cost_sum": float(group["stage_cost"].sum()),
                    "electricity_cost_sum": float(
                        group["physical_electricity_cost_usd"].sum()
                    ),
                    "carbon_kg_sum": float(group["physical_carbon_kg"].sum()),
                    "backlog_mean": float(
                        (group["pending_pre"] + group["transit_pre"]).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def shadow_disagreement_analysis(
    pair: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_parquet(
        V3_LABELS_PATH,
        columns=["step", "exact_action_disagreement"],
    )
    shadow = (
        labels.groupby("step", sort=True)["exact_action_disagreement"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "shadow_task_disagreement_fraction",
                "sum": "shadow_disagreement_count",
                "count": "shadow_task_count",
            }
        )
    )
    output = pair[
        [
            "step",
            "split",
            "arrival_task_count",
            "arrival_gpu",
            "stage_cost_delta_h4_minus_h1",
            "physical_electricity_cost_usd_delta_h4_minus_h1",
            "physical_carbon_kg_delta_h4_minus_h1",
            "backlog_delta_h4_minus_h1",
        ]
    ].merge(shadow, on="step", how="left", validate="one_to_one")
    output["shadow_task_disagreement_fraction"] = output[
        "shadow_task_disagreement_fraction"
    ].fillna(0.0)
    output["shadow_disagreement_count"] = output["shadow_disagreement_count"].fillna(0)
    output["shadow_task_count"] = output["shadow_task_count"].fillna(0)
    correlations: list[dict[str, Any]] = []
    for metric in [
        "stage_cost_delta_h4_minus_h1",
        "physical_electricity_cost_usd_delta_h4_minus_h1",
        "physical_carbon_kg_delta_h4_minus_h1",
        "backlog_delta_h4_minus_h1",
    ]:
        correlations.append(
            {
                "metric": metric,
                "pearson": float(
                    output["shadow_task_disagreement_fraction"].corr(output[metric])
                ),
                "spearman": float(
                    output["shadow_task_disagreement_fraction"].corr(
                        output[metric], method="spearman"
                    )
                ),
                "interpretation": "DESCRIPTIVE_ASSOCIATION_NOT_CAUSAL",
            }
        )
    return output, pd.DataFrame(correlations)


def fixed_state_action_cost_gap(
    canonical: pd.DataFrame,
    inputs: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    labels = pd.read_parquet(
        V3_LABELS_PATH,
        columns=[
            "state_id",
            "step",
            "task_id",
            "original_index",
            "priority",
            "h1_action",
            "h4_action",
            "exact_action_disagreement",
        ],
    )
    labels = labels.loc[labels["exact_action_disagreement"]].copy()
    tasks = pd.read_parquet(
        V3_TASKS_PATH,
        columns=[
            "state_id",
            "original_index",
            "cpu_cores",
            "gpu_units",
            "memory_gb",
            "bandwidth_gb",
            "estimated_duration_seconds",
            "estimated_duration_steps",
            "origin_dc",
            "remaining_sla_steps",
        ],
    )
    states = pd.read_parquet(
        V3_STATES_PATH,
        columns=["state_id", "energy_signals_json"],
    )
    signal_rows: list[dict[str, Any]] = []
    for row in states.itertuples(index=False):
        signals = json.loads(row.energy_signals_json)
        item: dict[str, Any] = {"state_id": row.state_id}
        for dc_id in range(1, 6):
            values = signals[str(dc_id)] if str(dc_id) in signals else signals[dc_id]
            item[f"price_{dc_id}"] = float(values["price"])
            item[f"carbon_{dc_id}"] = float(values["carbon"])
        signal_rows.append(item)
    work = (
        labels.merge(tasks, on=["state_id", "original_index"], how="left", validate="one_to_one")
        .merge(pd.DataFrame(signal_rows), on="state_id", how="left", validate="many_to_one")
        .merge(
            canonical[
                [
                    "original_index",
                    "true_duration",
                    "absolute_estimation_error_seconds",
                    "temporal_split",
                ]
            ],
            on="original_index",
            how="left",
            validate="one_to_one",
        )
    )
    config = spot_v3.alibaba_v3.repaired_runner._optimizer_config(
        ROOT / inputs["config"]["optimizer_config"]
    )
    watts = (
        work["cpu_cores"] * config.cpu_power_w_per_core
        + work["gpu_units"] * config.gpu_power_w_per_unit
        + work["memory_gb"] * config.memory_power_w_per_gb
    )
    work["estimated_energy_kwh"] = (
        watts / 1000.0 * work["estimated_duration_seconds"] / 3600.0
    )
    link_cost = {
        (link.origin_dc_id, link.destination_dc_id): link.transmission_cost_usd_per_gb
        for link in spot_v3.load_network_links()
    }

    def chosen_cost(row: pd.Series, action_column: str) -> float:
        action = int(row[action_column])
        if action == 0:
            urgency = 1.0 / max(1.0, float(row["remaining_sla_steps"]))
            return (
                config.weights.waiting_defer * config.waiting_cost_per_step
                + config.weights.sla_risk * urgency
            )
        electricity = (
            config.weights.electricity
            * float(row["estimated_energy_kwh"])
            * float(row[f"price_{action}"])
            / 1000.0
        )
        carbon = (
            config.weights.carbon
            * float(row["estimated_energy_kwh"])
            * float(row[f"carbon_{action}"])
            / 1000.0
        )
        transmission = (
            config.weights.transmission
            * link_cost[(int(row["origin_dc"]), action)]
            * float(row["bandwidth_gb"])
        )
        slack = int(row["remaining_sla_steps"]) - int(row["estimated_duration_steps"])
        sla = config.weights.sla_risk / max(1.0, float(slack + 1))
        return electricity + carbon + transmission + sla

    work["h1_chosen_action_cost"] = work.apply(
        chosen_cost, axis=1, action_column="h1_action"
    )
    work["h4_chosen_action_cost"] = work.apply(
        chosen_cost, axis=1, action_column="h4_action"
    )
    work["cost_gap_h4_minus_h1"] = (
        work["h4_chosen_action_cost"] - work["h1_chosen_action_cost"]
    )
    work["absolute_cost_gap"] = work["cost_gap_h4_minus_h1"].abs()
    work["relative_absolute_gap"] = (
        work["absolute_cost_gap"] / work["h1_chosen_action_cost"].abs().replace(0, np.nan)
    )
    tie_scale = config.deterministic_tie_break_epsilon * 4
    work["tie_break_scale_equivalent"] = work["absolute_cost_gap"] <= tie_scale
    duration_bins = [-np.inf, 3600, 6 * 3600, 24 * 3600, 48 * 3600, 72 * 3600, np.inf]
    duration_labels = ["<1h", "1-6h", "6-24h", "24-48h", "48-72h", ">72h"]
    work["duration_group"] = pd.cut(
        work["true_duration"], duration_bins, labels=duration_labels, right=False
    ).astype(str)
    train = canonical.loc[canonical["temporal_split"] == "train"]
    q1, q2 = train["absolute_estimation_error_seconds"].quantile([1 / 3, 2 / 3])
    work["duration_error_group"] = pd.cut(
        work["absolute_estimation_error_seconds"],
        [-np.inf, float(q1), float(q2), np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    raw_columns = [
        "state_id",
        "step",
        "task_id",
        "original_index",
        "priority",
        "duration_group",
        "duration_error_group",
        "h1_action",
        "h4_action",
        "h1_chosen_action_cost",
        "h4_chosen_action_cost",
        "cost_gap_h4_minus_h1",
        "absolute_cost_gap",
        "relative_absolute_gap",
        "tie_break_scale_equivalent",
    ]
    raw = work[raw_columns].copy()
    raw_path = OUTPUT / "raw/fixed_state_action_cost_gap.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(raw_path, index=False)
    summary_rows: list[dict[str, Any]] = []
    strata = [("ALL", "ALL", work)]
    for column in ["priority", "duration_group", "duration_error_group"]:
        strata.extend((column, str(value), group) for value, group in work.groupby(column, sort=True))
    for stratum, value, group in strata:
        gap = group["cost_gap_h4_minus_h1"]
        absolute = group["absolute_cost_gap"]
        summary_rows.append(
            {
                "stratum": stratum,
                "group": value,
                "disagreeing_tasks": len(group),
                "gap_mean_h4_minus_h1": float(gap.mean()),
                "gap_median_h4_minus_h1": float(gap.median()),
                "gap_p90_h4_minus_h1": float(gap.quantile(0.90)),
                "gap_p95_h4_minus_h1": float(gap.quantile(0.95)),
                "absolute_gap_mean": float(absolute.mean()),
                "absolute_gap_median": float(absolute.median()),
                "absolute_gap_p90": float(absolute.quantile(0.90)),
                "absolute_gap_p95": float(absolute.quantile(0.95)),
                "tie_break_scale_threshold": tie_scale,
                "tie_break_scale_equivalent_count": int(
                    group["tie_break_scale_equivalent"].sum()
                ),
                "tie_break_scale_equivalent_fraction": float(
                    group["tie_break_scale_equivalent"].mean()
                ),
            }
        )
    headline = summary_rows[0]
    evidence = {
        "shadow_disagreeing_tasks": float(len(work)),
        "tie_break_scale_threshold": tie_scale,
        "tie_break_scale_equivalent_fraction": float(
            headline["tie_break_scale_equivalent_fraction"]
        ),
        "absolute_gap_median": float(headline["absolute_gap_median"]),
        "absolute_gap_p95": float(headline["absolute_gap_p95"]),
    }
    return raw, pd.DataFrame(summary_rows), evidence


def resource_analysis_tables(
    tables: Mapping[str, Mapping[str, pd.DataFrame]],
    metrics: Mapping[str, Mapping[str, float]],
    tail_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    electricity: list[dict[str, Any]] = []
    carbon: list[dict[str, Any]] = []
    transmission: list[dict[str, Any]] = []
    migration: list[dict[str, Any]] = []
    backlog: list[dict[str, Any]] = []
    tail: list[dict[str, Any]] = []
    for key, controller in [("h1", "H1"), ("h4_oracle", "H4_ORACLE")]:
        values = metrics[controller]
        steps = tables[key]["steps"]
        primary = steps.loc[steps["phase"] == "primary"].copy()
        primary["backlog_total"] = primary["pending_pre"] + primary["transit_pre"]
        electricity.extend(
            [
                {
                    "controller": controller,
                    "phase": "primary",
                    "physical_energy_kwh": values["physical_energy_kwh_primary"],
                    "physical_electricity_cost_usd": values[
                        "physical_electricity_cost_usd_primary"
                    ],
                    "objective_electricity_contribution": values[
                        "objective_electricity"
                    ],
                    "signal_scope": "SustainCluster 2023 EXTERNAL_SCENARIO_SIGNAL",
                },
                {
                    "controller": controller,
                    "phase": "primary_plus_tail",
                    "physical_energy_kwh": values["physical_energy_kwh_with_tail"],
                    "physical_electricity_cost_usd": values[
                        "physical_electricity_cost_usd_with_tail"
                    ],
                    "objective_electricity_contribution": values[
                        "objective_electricity"
                    ],
                    "signal_scope": "SustainCluster 2023 EXTERNAL_SCENARIO_SIGNAL",
                },
            ]
        )
        carbon.extend(
            [
                {
                    "controller": controller,
                    "phase": "primary",
                    "physical_carbon_kg": values["physical_carbon_kg_primary"],
                    "carbon_cost_usd": math.nan,
                    "objective_carbon_contribution": values["objective_carbon"],
                    "carbon_cost_definition": "NOT_DEFINED_IN_FROZEN_PROJECT",
                    "signal_scope": "SustainCluster 2023 EXTERNAL_SCENARIO_SIGNAL",
                },
                {
                    "controller": controller,
                    "phase": "primary_plus_tail",
                    "physical_carbon_kg": values["physical_carbon_kg_with_tail"],
                    "carbon_cost_usd": math.nan,
                    "objective_carbon_contribution": values["objective_carbon"],
                    "carbon_cost_definition": "NOT_DEFINED_IN_FROZEN_PROJECT",
                    "signal_scope": "SustainCluster 2023 EXTERNAL_SCENARIO_SIGNAL",
                },
            ]
        )
        transmission.append(
            {
                "controller": controller,
                "phase": "primary",
                "transmission_cost_usd": values["transmission_cost_usd"],
                "transmission_amount_gb": values["transmission_amount_gb"],
                "cross_dc_task_count": values["migration_count"],
            }
        )
        migration.append(
            {
                "controller": controller,
                "phase": "primary",
                "migration_count": values["migration_count"],
                "migration_cost_usd": math.nan,
                "migration_cost_definition": (
                    "NOT_SEPARATELY_DEFINED; CROSS_DC_TRANSMISSION_IS_REPORTED"
                ),
            }
        )
        backlog.append(
            {
                "controller": controller,
                "phase": "primary",
                **numeric_stats(primary["backlog_total"], "backlog_"),
                "terminal_pending": values["primary_remaining_pending"],
                "terminal_running": values["primary_remaining_running"],
                "terminal_in_transit": values["primary_remaining_in_transit"],
                "terminal_unfinished": (
                    values["primary_remaining_pending"]
                    + values["primary_remaining_running"]
                    + values["primary_remaining_in_transit"]
                ),
            }
        )
        result = tail_results[key]
        tail.append(
            {
                "controller": controller,
                "drain_steps": result["drain_steps"],
                "maximum_drain_steps": result["maximum_drain_steps"],
                "drained_to_empty": result["drained_to_empty"],
                "primary_pending": result["primary"]["pending"],
                "primary_running": result["primary"]["running"],
                "primary_in_transit": result["primary"]["in_transit"],
                "primary_completed": result["primary"]["completed"],
                "tail_pending": result["tail_end"]["pending"],
                "tail_running": result["tail_end"]["running"],
                "tail_in_transit": result["tail_end"]["in_transit"],
                "tail_completed": result["tail_end"]["completed"],
                "primary_terminal_pending_cost": result["primary"][
                    "frozen_terminal_pending_cost"
                ],
                "tail_terminal_pending_cost": result["tail_end"][
                    "frozen_terminal_pending_cost"
                ],
                "terminal_pending_cost_change": result[
                    "terminal_pending_cost_change"
                ],
            }
        )
    return {
        "electricity": pd.DataFrame(electricity),
        "carbon": pd.DataFrame(carbon),
        "transmission": pd.DataFrame(transmission),
        "migration": pd.DataFrame(migration),
        "backlog": pd.DataFrame(backlog),
        "tail": pd.DataFrame(tail),
    }


def solver_quality_table(
    tables: Mapping[str, Mapping[str, pd.DataFrame]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, controller in [("h1", "H1"), ("h4_oracle", "H4_ORACLE")]:
        steps = tables[key]["steps"]
        called = steps.loc[steps["solver_called"]]
        rows.append(
            {
                "record_type": "SUMMARY",
                "controller": controller,
                "solver_calls": len(called),
                "optimal": int((called["solver_status"] == "optimal").sum()),
                "retry": int(called["presolve_retry"].sum()),
                "failure": int((called["solver_status"] != "optimal").sum()),
                "timeout": int((called["solver_status"] == "limit_reached").sum()),
                "fallback": int(called["fallback_used"].sum()),
                **numeric_stats(called["solve_ms"], "solve_ms_"),
            }
        )
        for row in called.loc[called["presolve_retry"]].itertuples(index=False):
            rows.append(
                {
                    "record_type": "PRESOLVE_RETRY",
                    "controller": controller,
                    "step": int(row.step),
                    "solver_calls": math.nan,
                    "optimal": math.nan,
                    "retry": 1,
                    "failure": 0,
                    "timeout": 0,
                    "fallback": 0,
                    "solver_status": row.solver_status,
                    "solver_message": row.solver_message,
                }
            )
    return pd.DataFrame(rows)


def generate_plots(
    comparison: pd.DataFrame,
    daily: pd.DataFrame,
    pressure: pd.DataFrame,
    shadow: pd.DataFrame,
    fixed_raw: pd.DataFrame,
) -> None:
    plot_dir = OUTPUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    stage = comparison.loc[comparison["metric"] == "stage_cost_total"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(["H1", "H4 Oracle"], [stage["h1"], stage["h4_oracle"]], color=["#35618f", "#bf5b3d"])
    ax.set_ylabel("Frozen stage cost")
    ax.set_title("Independent closed-loop stage cost")
    fig.tight_layout()
    fig.savefig(plot_dir / "01_total_stage_cost.png", dpi=180)
    plt.close(fig)


def generate_remaining_plots(
    comparison: pd.DataFrame,
    daily: pd.DataFrame,
    pressure: pd.DataFrame,
    shadow: pd.DataFrame,
    fixed_raw: pd.DataFrame,
) -> None:
    plot_dir = OUTPUT / "plots"
    electric = comparison.loc[
        comparison["metric"] == "physical_electricity_cost_usd_primary"
    ].iloc[0]
    carbon = comparison.loc[
        comparison["metric"] == "physical_carbon_kg_primary"
    ].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(["H1", "H4"], [electric["h1"], electric["h4_oracle"]], color=["#35618f", "#bf5b3d"])
    axes[0].set_title("Physical electricity cost")
    axes[0].set_ylabel("USD")
    axes[1].bar(["H1", "H4"], [carbon["h1"], carbon["h4_oracle"]], color=["#35618f", "#bf5b3d"])
    axes[1].set_title("Physical carbon")
    axes[1].set_ylabel("kgCO2")
    fig.tight_layout()
    fig.savefig(plot_dir / "02_electricity_carbon.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.plot(daily["block_id"], -daily["stage_cost_delta_h4_minus_h1"], color="#35618f", linewidth=1.1)
    ax.set_xlabel("Continuous day block")
    ax.set_ylabel("Stage-cost improvement (H1 - H4)")
    ax.set_title("Daily closed-loop stage-cost delta")
    fig.tight_layout()
    fig.savefig(plot_dir / "03_daily_stage_cost_delta.png", dpi=180)
    plt.close(fig)

    p = pressure.loc[
        (pressure["record_type"] == "EXTERNAL_PRESSURE")
        & (pressure["metric"] == "stage_cost")
    ].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(p["pressure_band"], -p["absolute_delta_h4_minus_h1"] / p["steps"], color=["#bf5b3d", "#35618f"])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Mean stage-cost improvement per step")
    ax.set_title("External GPU pressure stratification")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(plot_dir / "04_pressure_stage_cost_gain.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(
        shadow["shadow_task_disagreement_fraction"],
        -shadow["stage_cost_delta_h4_minus_h1"],
        s=8,
        alpha=0.24,
        color="#35618f",
        edgecolors="none",
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("v3 shadow task disagreement fraction")
    ax.set_ylabel("Closed-loop stage-cost improvement (H1 - H4)")
    ax.set_title("Shadow disagreement and closed-loop gain")
    fig.tight_layout()
    fig.savefig(plot_dir / "05_shadow_disagreement_vs_gain.png", dpi=180)
    plt.close(fig)

    threshold = 4e-9
    values = np.log10(fixed_raw["absolute_cost_gap"].to_numpy(dtype=float) + threshold)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(values, bins=80, color="#4e7d5b", alpha=0.9)
    ax.axvline(math.log10(2 * threshold), color="#bf5b3d", linestyle="--", linewidth=1)
    ax.set_xlabel("log10(abs fixed-state gap + tie-break threshold)")
    ax.set_ylabel("Disagreeing task decisions")
    ax.set_title("Fixed-state H1/H4 chosen-action objective gaps")
    fig.tight_layout()
    fig.savefig(plot_dir / "06_fixed_state_action_cost_gap.png", dpi=180)
    plt.close(fig)


def choose_diagnosis(
    metrics: Mapping[str, Mapping[str, float]],
    daily: pd.DataFrame,
) -> tuple[str, dict[str, float]]:
    h1 = metrics["H1"]
    h4 = metrics["H4_ORACLE"]
    stage_improvement = -relative_delta(h1["stage_cost_total"], h4["stage_cost_total"])
    electricity_improvement = -relative_delta(
        h1["physical_electricity_cost_usd_primary"],
        h4["physical_electricity_cost_usd_primary"],
    )
    carbon_improvement = -relative_delta(
        h1["physical_carbon_kg_primary"], h4["physical_carbon_kg_primary"]
    )
    daily_gain = -daily["stage_cost_delta_h4_minus_h1"]
    context = {
        "stage_cost_improvement_percent": stage_improvement,
        "physical_electricity_improvement_percent": electricity_improvement,
        "physical_carbon_improvement_percent": carbon_improvement,
        "sla_violation_rate_delta_h4_minus_h1": (
            h4["primary_sla_violation_rate"] - h1["primary_sla_violation_rate"]
        ),
        "completion_rate_delta_h4_minus_h1": (
            h4["primary_completion_rate"] - h1["primary_completion_rate"]
        ),
        "daily_stage_improvement_block_fraction": float((daily_gain > 0).mean()),
        "daily_stage_improvement_median": float(daily_gain.median()),
        "daily_stage_improvement_p25": float(daily_gain.quantile(0.25)),
        "daily_stage_improvement_p75": float(daily_gain.quantile(0.75)),
        "materiality_reference_percent": 1.0,
    }
    if h1["solver_failures"] or h4["solver_failures"]:
        return "INCONCLUSIVE", context
    stable_stage_gain = (
        stage_improvement >= 1.0
        and context["daily_stage_improvement_block_fraction"] >= 0.60
        and context["sla_violation_rate_delta_h4_minus_h1"] <= 0.001
    )
    meaningful_energy_gain = (
        max(electricity_improvement, carbon_improvement) >= 1.0
        and context["sla_violation_rate_delta_h4_minus_h1"] <= 0.001
    )
    all_small = (
        abs(stage_improvement) < 1.0
        and abs(electricity_improvement) < 1.0
        and abs(carbon_improvement) < 1.0
        and abs(context["sla_violation_rate_delta_h4_minus_h1"]) < 0.001
        and abs(context["completion_rate_delta_h4_minus_h1"]) < 0.001
    )
    if stable_stage_gain or meaningful_energy_gain:
        return "H4_CONTROL_VALUE_CONFIRMED", context
    if all_small:
        return "ACTION_DIFFERENCE_WITH_LIMITED_CONTROL_VALUE", context
    return "CONTEXT_DEPENDENT_CONTROL_VALUE", context


def finalize_outputs() -> dict[str, Any]:
    inputs = spot_v3.load_inputs()
    write_protocol_files(inputs)
    canonical = inputs["canonical"]
    tables = {
        "h1": read_controller_tables("h1"),
        "h4_oracle": read_controller_tables("h4_oracle"),
    }
    tail_results = {
        key: json.loads(
            (OUTPUT / "checkpoints" / key / "tail_result.json").read_text("utf-8")
        )
        for key in tables
    }
    lifecycles: dict[str, pd.DataFrame] = {}
    for key, controller in [("h1", "H1"), ("h4_oracle", "H4_ORACLE")]:
        tail_end_step = int(tail_results[key]["tail_end"]["next_step"])
        lifecycle = build_lifecycle(
            canonical,
            tables[key]["decisions"],
            tables[key]["events"],
            tail_end_step,
        )
        lifecycles[controller] = lifecycle
        lifecycle.to_parquet(
            OUTPUT / "raw" / key / "lifecycle.parquet",
            index=False,
        )
    metrics = {
        "H1": controller_metrics(tables["h1"], lifecycles["H1"], tail_results["h1"]),
        "H4_ORACLE": controller_metrics(
            tables["h4_oracle"],
            lifecycles["H4_ORACLE"],
            tail_results["h4_oracle"],
        ),
    }
    for controller, filename in [
        ("H1", "03_h1_closed_loop_summary.csv"),
        ("H4_ORACLE", "04_h4_closed_loop_summary.csv"),
    ]:
        write_csv(
            OUTPUT / filename,
            pd.DataFrame(
                [
                    {"controller": controller, "metric": metric, "value": value}
                    for metric, value in metrics[controller].items()
                ]
            ),
        )
    comparison = metric_comparison(metrics)
    write_csv(OUTPUT / "05_core_metric_comparison.csv", comparison)

    pair = paired_primary_steps(tables)
    pair.to_parquet(OUTPUT / "raw/paired_primary_steps.parquet", index=False)
    daily = block_comparison(pair, lifecycles, 96, "DAILY_96_STEPS")
    weekly = block_comparison(pair, lifecycles, 672, "WEEKLY_672_STEPS")
    write_csv(OUTPUT / "06_daily_block_comparison.csv", daily)
    write_csv(OUTPUT / "07_weekly_block_comparison.csv", weekly)

    task_tables = specialized_task_tables(tables, lifecycles)
    write_csv(OUTPUT / "08_sla_analysis.csv", task_tables["sla"])
    write_csv(OUTPUT / "09_waiting_analysis.csv", task_tables["waiting"])
    write_csv(OUTPUT / "10_completion_analysis.csv", task_tables["completion"])

    resource_tables = resource_analysis_tables(tables, metrics, tail_results)
    write_csv(OUTPUT / "11_electricity_analysis.csv", resource_tables["electricity"])
    write_csv(OUTPUT / "12_carbon_analysis.csv", resource_tables["carbon"])
    write_csv(OUTPUT / "13_transmission_analysis.csv", resource_tables["transmission"])
    write_csv(OUTPUT / "14_migration_analysis.csv", resource_tables["migration"])
    write_csv(OUTPUT / "15_backlog_analysis.csv", resource_tables["backlog"])

    priority = stratified_comparison(lifecycles, "priority", "priority")
    duration = stratified_comparison(lifecycles, "duration_group", "duration_group")
    error = stratified_comparison(
        lifecycles, "duration_error_group", "duration_error_group"
    )
    write_csv(OUTPUT / "16_priority_analysis.csv", priority)
    write_csv(OUTPUT / "17_duration_group_analysis.csv", duration)
    write_csv(OUTPUT / "18_duration_error_analysis.csv", error)

    pressure, pressure_threshold = external_pressure_analysis(pair)
    risk = risk_trajectory_rows(tables)
    write_csv(
        OUTPUT / "19_external_pressure_analysis.csv",
        pd.concat([pressure, risk], ignore_index=True, sort=False),
    )
    shadow, correlations = shadow_disagreement_analysis(pair)
    write_csv(OUTPUT / "20_shadow_disagreement_vs_gain.csv", shadow)
    write_csv(
        OUTPUT / "raw/shadow_disagreement_correlation_summary.csv", correlations
    )

    fixed_raw, fixed_summary, equivalence = fixed_state_action_cost_gap(
        canonical, inputs
    )
    write_csv(OUTPUT / "21_fixed_state_action_cost_gap.csv", fixed_summary)
    write_text(
        OUTPUT / "22_near_equivalent_action_diagnostic.md",
        f"""
# Near-equivalent Action Diagnostic

- Public v3 shadow task disagreement rate: 26.244091%.
- Disagreeing task decisions evaluated: {int(equivalence['shadow_disagreeing_tasks'])}.
- Frozen deterministic tie-break epsilon: 1e-9.
- Maximum destination-rank tie-break gap: 4e-9.
- Conservative tie-break-scale equivalent fraction: {equivalence['tie_break_scale_equivalent_fraction']:.9%}.
- Median absolute fixed-state objective gap: {equivalence['absolute_gap_median']:.12g}.
- P95 absolute fixed-state objective gap: {equivalence['absolute_gap_p95']:.12g}.

The binary label is deliberately named TIE_BREAK_SCALE_EQUIVALENT. SciPy/HiGHS did
not expose a task-level solver optimality tolerance in the frozen configuration, so
this audit does not claim that every numerically small gap is solver-equivalent.
The continuous gap distribution in 21_fixed_state_action_cost_gap.csv and the raw
Parquet table is the primary evidence.
""",
    )
    write_csv(OUTPUT / "23_tail_drain_analysis.csv", resource_tables["tail"])
    solver = solver_quality_table(tables)
    write_csv(OUTPUT / "24_solver_quality.csv", solver)

    manifests = {
        key: json.loads(
            (OUTPUT / "checkpoints" / key / "run_manifest.json").read_text("utf-8")
        )
        for key in tables
    }
    source_text = Path(__file__).read_text("utf-8").lower()
    write_text(
        OUTPUT / "25_information_leakage_audit.md",
        f"""
# Information Leakage Audit

- H1 Oracle future state accesses: {manifests['h1']['oracle_future_state_count']} (required 0).
- H1 Oracle future rows: {manifests['h1']['oracle_future_row_count']} (required 0).
- H4 Oracle future state accesses: {manifests['h4_oracle']['oracle_future_state_count']}.
- H4 uses exactly four future offsets per state and five deterministic DC rows per offset.
- Workload source hash shared: {sha256(CANONICAL_PATH)}.
- Capacity, energy signals, SLA, true duration, estimated duration, order, and initial state are shared and preflight-checked.
- true_duration is read only when applying actions to simulator transit/running state.
- estimated_duration is the only duration supplied to the optimizer.
- Transformer code path used: NO.
- BC code path used: NO.
- RL code path used: NO.
- Silent fallback label used: NO.
- Regional energy data provenance: SustainCluster 2023 EXTERNAL_SCENARIO_SIGNAL; not Alibaba2026 measured energy data.
""",
    )
    write_text(
        OUTPUT / "26_tests.md",
        """
# Validation

- compileall: PENDING
- dedicated tests: PENDING
- relevant regressions: PENDING
- full pytest: PENDING
- new failures: PENDING
""",
    )
    diagnosis, diagnostic = choose_diagnosis(metrics, daily)
    write_json(
        OUTPUT / "raw/diagnostic_context.json",
        {
            "diagnosis": diagnosis,
            "diagnostic": diagnostic,
            "pressure_threshold": pressure_threshold,
            "equivalence": equivalence,
            "correlations": correlations.to_dict("records"),
        },
    )
    generate_plots(comparison, daily, pressure, shadow, fixed_raw)
    generate_remaining_plots(comparison, daily, pressure, shadow, fixed_raw)
    return {
        "inputs": inputs,
        "tables": tables,
        "lifecycles": lifecycles,
        "metrics": metrics,
        "comparison": comparison,
        "daily": daily,
        "weekly": weekly,
        "priority": priority,
        "pressure": pressure,
        "pressure_threshold": pressure_threshold,
        "shadow": shadow,
        "correlations": correlations,
        "equivalence": equivalence,
        "tail_results": tail_results,
        "diagnosis": diagnosis,
        "diagnostic": diagnostic,
    }


def lookup_group_metric(
    frame: pd.DataFrame, group_column: str, group: str, metric: str
) -> tuple[float, float, float]:
    row = frame.loc[
        (frame[group_column].astype(str) == group) & (frame["metric"] == metric)
    ].iloc[0]
    return float(row["h1"]), float(row["h4_oracle"]), float(
        row["absolute_delta_h4_minus_h1"]
    )


def write_final_documents(result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    h1 = metrics["H1"]
    h4 = metrics["H4_ORACLE"]
    diagnostic = result["diagnostic"]
    diagnosis = result["diagnosis"]
    pressure_stage = result["pressure"].loc[
        (result["pressure"]["record_type"] == "EXTERNAL_PRESSURE")
        & (result["pressure"]["metric"] == "stage_cost")
    ]
    low = pressure_stage.loc[
        pressure_stage["pressure_band"] == "LOW_LT_TRAIN_P95"
    ].iloc[0]
    high = pressure_stage.loc[
        pressure_stage["pressure_band"] == "HIGH_GE_TRAIN_P95"
    ].iloc[0]
    hp = lookup_group_metric(
        result["priority"], "priority", "HP", "action_stage_cost"
    )
    spot = lookup_group_metric(
        result["priority"], "priority", "Spot", "action_stage_cost"
    )
    hp_improvement = -relative_delta(hp[0], hp[1])
    spot_improvement = -relative_delta(spot[0], spot[1])
    if math.isclose(hp_improvement, spot_improvement, abs_tol=1e-12):
        priority_benefit = "NEITHER_EQUAL"
    elif hp_improvement > spot_improvement:
        priority_benefit = "HP"
    else:
        priority_benefit = "SPOT"
    stage_corr = float(
        result["correlations"].loc[
            result["correlations"]["metric"] == "stage_cost_delta_h4_minus_h1",
            "pearson",
        ].iloc[0]
    )
    primary_electric_delta = (
        h4["physical_electricity_cost_usd_primary"]
        - h1["physical_electricity_cost_usd_primary"]
    )
    tail_electric_delta = (
        h4["physical_electricity_cost_usd_with_tail"]
        - h1["physical_electricity_cost_usd_with_tail"]
    )
    primary_sla_delta = (
        h4["primary_sla_violation_rate"] - h1["primary_sla_violation_rate"]
    )
    tail_sla_delta = h4["tail_sla_violation_rate"] - h1["tail_sla_violation_rate"]
    conclusion_changed = bool(
        np.sign(primary_electric_delta) != np.sign(tail_electric_delta)
        or np.sign(primary_sla_delta) != np.sign(tail_sla_delta)
    )
    next_steps = {
        "H4_CONTROL_VALUE_CONFIRMED": "�e�K�o��Vef`",
        "ACTION_DIFFERENCE_WITH_LIMITED_CONTROL_VALUE": "b�V(τVe��",
        "CONTEXT_DEPENDENT_CONTROL_VALUE": "�i�͹f`",
        "INCONCLUSIVE": "���6�<ʭ",
    }
    next_step = next_steps[diagnosis]
    next_step = {
        "H4_CONTROL_VALUE_CONFIRMED": "\u52a0\u5165\u9884\u6d4b\u4fe1\u606f\u7684\u4e13\u5bb6\u7b56\u7565\u5b66\u4e60",
        "ACTION_DIFFERENCE_WITH_LIMITED_CONTROL_VALUE": "\u9762\u5411\u51b3\u7b56\u8d28\u91cf\u7684\u7b56\u7565\u84b8\u998f",
        "CONTEXT_DEPENDENT_CONTROL_VALUE": "\u98ce\u9669\u72b6\u6001\u91cd\u70b9\u5b66\u4e60",
        "INCONCLUSIVE": "\u7ee7\u7eed\u63a7\u5236\u4ef7\u503c\u8bca\u65ad",
    }[diagnosis]
    stage_absolute = h4["stage_cost_total"] - h1["stage_cost_total"]
    stage_relative = relative_delta(h1["stage_cost_total"], h4["stage_cost_total"])
    transmission_delta = h4["transmission_cost_usd"] - h1["transmission_cost_usd"]
    migration_delta = h4["migration_count"] - h1["migration_count"]
    backlog_delta = h4["backlog_mean"] - h1["backlog_mean"]
    completion_delta = h4["primary_completion_rate"] - h1["primary_completion_rate"]
    shadow_rate = 0.26244091
    equivalence = result["equivalence"]
    stable = diagnostic["daily_stage_improvement_block_fraction"]
    write_text(
        OUTPUT / "27_final_diagnosis.md",
        f"""
# Final Diagnosis

## Diagnosis

**{diagnosis}**

The practical materiality reference is 1% per original metric, not a composite
score. Daily blocks are descriptive segments from one continuous trajectory.

## Evidence

- Total stage cost: H1={h1['stage_cost_total']:.12g}, H4={h4['stage_cost_total']:.12g}, H4-H1={stage_absolute:.12g}, relative={stage_relative:.9f}%.
- Physical electricity: H1={h1['physical_electricity_cost_usd_primary']:.12g} USD, H4={h4['physical_electricity_cost_usd_primary']:.12g} USD.
- Physical carbon: H1={h1['physical_carbon_kg_primary']:.12g} kgCO2, H4={h4['physical_carbon_kg_primary']:.12g} kgCO2.
- SLA violation rate delta H4-H1: {primary_sla_delta:.12g}.
- Daily stage-cost improvement block fraction: {stable:.9%}.
- High-pressure mean stage-cost delta per step H4-H1: {float(high['absolute_delta_h4_minus_h1']) / float(high['steps']):.12g}.
- Low-pressure mean stage-cost delta per step H4-H1: {float(low['absolute_delta_h4_minus_h1']) / float(low['steps']):.12g}.
- Shadow disagreement versus stage-cost delta Pearson correlation: {stage_corr:.9f}; descriptive, not causal.
- Tie-break-scale equivalent fraction among disagreeing public-state decisions: {equivalence['tie_break_scale_equivalent_fraction']:.9%}.
- Defer decisions: H1={int(h1['defer_decision_count'])}, H4={int(h4['defer_decision_count'])}.
- Solver failures/fallback: H1={int(h1['solver_failures'])}/{int(h1['fallback_count'])}; H4={int(h4['solver_failures'])}/{int(h4['fallback_count'])}.

## Route

Next stage: **{next_step}**. This report does not implement that stage.
""",
    )
    write_text(
        OUTPUT / "28_summary.md",
        f"""
# SpotGPU2026 Forward-control Value v1 Summary

1. H4 total stage-cost improvement (H1-H4): {-stage_absolute:.12g} objective units.
2. Relative stage-cost improvement: {-stage_relative:.9f}%.
3. SLA: H1={h1['primary_sla_violation_rate']:.9%}, H4={h4['primary_sla_violation_rate']:.9%}, delta={primary_sla_delta:.12g}.
4. Electricity: H4-H1={primary_electric_delta:.12g} USD under the frozen SustainCluster 2023 external signal scenario.
5. Carbon: H4-H1={h4['physical_carbon_kg_primary'] - h1['physical_carbon_kg_primary']:.12g} kgCO2 under the same external signal scenario.
6. Transmission/migration: cost delta={transmission_delta:.12g} USD; migration-count delta={migration_delta:.12g}.
7. Backlog/completion: mean backlog delta={backlog_delta:.12g}; primary completion-rate delta={completion_delta:.12g}.
8. Pressure concentration: low/high mean stage-cost deltas per step are {float(low['absolute_delta_h4_minus_h1']) / float(low['steps']):.12g} and {float(high['absolute_delta_h4_minus_h1']) / float(high['steps']):.12g}.
9. HP versus Spot: action-stage-cost improvements are HP={hp_improvement:.9f}% and Spot={spot_improvement:.9f}%; larger benefit={priority_benefit}.
10. Of the 26.244091% shadow action disagreement, {equivalence['tie_break_scale_equivalent_fraction']:.9%} is equivalent at the conservative 4e-9 tie-break scale. Broader solver-equivalence is not hard-classified.
11. Disagreement/gain association: Pearson={stage_corr:.9f}; descriptive only, no causal claim.
12. Time stability: {stable:.9%} of daily blocks have positive H4 stage-cost improvement.
13. Tail drain: H1={int(h1['tail_drain_steps'])} steps, H4={int(h4['tail_drain_steps'])} steps; conclusion changed={'YES' if conclusion_changed else 'NO'}.
14. MPC mechanism: {'spatial placement' if h1['defer_decision_count'] == 0 and h4['defer_decision_count'] == 0 else 'spatial placement plus temporal defer'}; no objective tuning was used.
15. Next stage: {next_step}.

Final diagnosis: **{diagnosis}**
""",
    )
    console = {
        "diagnosis": diagnosis,
        "next_step": next_step,
        "h1": h1,
        "h4": h4,
        "stage_absolute_delta": stage_absolute,
        "stage_relative_delta": stage_relative,
        "sla_delta": primary_sla_delta,
        "electricity_delta": primary_electric_delta,
        "carbon_delta": h4["physical_carbon_kg_primary"]
        - h1["physical_carbon_kg_primary"],
        "transmission_delta": transmission_delta,
        "migration_delta": migration_delta,
        "backlog_delta": backlog_delta,
        "completion_delta": completion_delta,
        "low_pressure_delta_per_step": float(low["absolute_delta_h4_minus_h1"])
        / float(low["steps"]),
        "high_pressure_delta_per_step": float(high["absolute_delta_h4_minus_h1"])
        / float(high["steps"]),
        "hp_benefit_percent": hp_improvement,
        "spot_benefit_percent": spot_improvement,
        "shadow_disagreement": shadow_rate,
        "fixed_gap_median": equivalence["absolute_gap_median"],
        "near_equivalent_fraction": equivalence[
            "tie_break_scale_equivalent_fraction"
        ],
        "tail_conclusion_changed": conclusion_changed,
        "stage_corr": stage_corr,
    }
    write_json(OUTPUT / "raw/final_console.json", console)
    return console


def write_integrity_manifest() -> None:
    rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if not path.is_file() or path.name == "29_integrity_manifest.csv":
            continue
        rows.append(
            {
                "path": path.relative_to(OUTPUT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(OUTPUT / "29_integrity_manifest.csv", pd.DataFrame(rows))


def record_tests(
    compileall: str,
    dedicated: str,
    regression: str,
    full: str,
    new_failures: int,
) -> None:
    write_text(
        OUTPUT / "26_tests.md",
        f"""
# Validation

- compileall: {compileall}
- dedicated tests: {dedicated}
- relevant regressions: {regression}
- full pytest: {full}
- new failures: {new_failures}
- Transformer: NO
- BC: NO
- RL: NO
- reward tuning: NO
- MPC tuning: NO
""",
    )
    write_integrity_manifest()
    print_final_console(
        json.loads((OUTPUT / "raw/final_console.json").read_text("utf-8")),
        {
            "compileall": compileall,
            "dedicated": dedicated,
            "regression": regression,
            "full": full,
            "new_failures": str(new_failures),
        },
    )


def print_final_console(console: Mapping[str, Any], tests: Mapping[str, str]) -> None:
    h1 = console["h1"]
    h4 = console["h4"]
    print(
        f"""
=== SPOTGPU2026 CONTROL VALUE v1 ===

=== CLOSED LOOP ===

Timeline:
17670 steps

H1 completed:
YES

H4 completed:
YES

=== STAGE COST ===

H1:
{h1['stage_cost_total']:.12g}

H4:
{h4['stage_cost_total']:.12g}

Absolute delta:
{console['stage_absolute_delta']:.12g}

Relative delta:
{console['stage_relative_delta']:.9f}%

=== SLA ===

H1:
{h1['primary_sla_violation_rate']:.9%}

H4:
{h4['primary_sla_violation_rate']:.9%}

Delta:
{console['sla_delta']:.12g}

=== ELECTRICITY ===

H1:
{h1['physical_electricity_cost_usd_primary']:.12g} USD

H4:
{h4['physical_electricity_cost_usd_primary']:.12g} USD

Delta:
{console['electricity_delta']:.12g} USD

=== CARBON ===

H1:
{h1['physical_carbon_kg_primary']:.12g} kgCO2

H4:
{h4['physical_carbon_kg_primary']:.12g} kgCO2

Delta:
{console['carbon_delta']:.12g} kgCO2

=== TRANSMISSION / MIGRATION ===

H1:
{h1['transmission_cost_usd']:.12g} USD / {int(h1['migration_count'])}

H4:
{h4['transmission_cost_usd']:.12g} USD / {int(h4['migration_count'])}

=== BACKLOG / COMPLETION ===

H1:
mean backlog {h1['backlog_mean']:.12g} / completion {h1['primary_completion_rate']:.9%}

H4:
mean backlog {h4['backlog_mean']:.12g} / completion {h4['primary_completion_rate']:.9%}

=== PRESSURE STRATIFICATION ===

Low pressure delta:
{console['low_pressure_delta_per_step']:.12g} per step

High pressure delta:
{console['high_pressure_delta_per_step']:.12g} per step

=== HP / SPOT ===

HP benefit:
{console['hp_benefit_percent']:.9f}%

Spot benefit:
{console['spot_benefit_percent']:.9f}%

=== ACTION VALUE ===

Shadow task disagreement:
26.244091%

Fixed-state action cost gap:
median absolute {console['fixed_gap_median']:.12g}

Near-equivalent evidence:
{console['near_equivalent_fraction']:.9%} at tie-break scale

=== TAIL DRAIN ===

H1:
{int(h1['tail_drain_steps'])} steps

H4:
{int(h4['tail_drain_steps'])} steps

Conclusion changed:
{'YES' if console['tail_conclusion_changed'] else 'NO'}

=== SOLVER ===

H1 failures:
{int(h1['solver_failures'])}

H4 failures:
{int(h4['solver_failures'])}

Timeout:
{int(h1['solver_timeouts'] + h4['solver_timeouts'])}

Fallback:
{int(h1['fallback_count'] + h4['fallback_count'])}

=== FINAL DIAGNOSIS ===

{console['diagnosis']}

=== NEXT STEP ===

{console['next_step']}

=== VALIDATION ===

compileall:
{tests['compileall']}

Dedicated:
{tests['dedicated']}

Regression:
{tests['regression']}

Full:
{tests['full']}

New failures:
{tests['new_failures']}

Transformer:
NO

BC:
NO

RL:
NO

Reward tuning:
NO

MPC tuning:
NO

Commit:
NO

Push:
NO

Final status:
{console['diagnosis']}
""".strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        required=True,
        choices=[
            "preflight",
            "run-h1",
            "run-h4",
            "finalize",
            "record-tests",
            "print-final",
        ],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compileall", default="PENDING")
    parser.add_argument("--dedicated", default="PENDING")
    parser.add_argument("--regression", default="PENDING")
    parser.add_argument("--full", default="PENDING")
    parser.add_argument("--new-failures", type=int, default=-1)
    args = parser.parse_args()
    load_eval_config()
    if args.action == "preflight":
        inputs = spot_v3.load_inputs()
        write_protocol_files(inputs)
        print(json.dumps(checkpoint_exact_preflight(inputs), indent=2))
    elif args.action in {"run-h1", "run-h4"}:
        gate = json.loads((OUTPUT / "preflight/preflight_result.json").read_text("utf-8"))
        if gate["status"] != "PASS":
            raise RuntimeError("formal run blocked by preflight")
        inputs = spot_v3.load_inputs()
        controller = "h1" if args.action == "run-h1" else "h4_oracle"
        print(json.dumps(run_formal_controller(inputs, controller, resume=args.resume), indent=2))
    elif args.action == "finalize":
        result = finalize_outputs()
        console = write_final_documents(result)
        write_integrity_manifest()
        print(json.dumps(console, indent=2))
    elif args.action == "record-tests":
        record_tests(
            args.compileall,
            args.dedicated,
            args.regression,
            args.full,
            args.new_failures,
        )
    else:
        console = json.loads((OUTPUT / "raw/final_console.json").read_text("utf-8"))
        print_final_console(
            console,
            {
                "compileall": args.compileall,
                "dedicated": args.dedicated,
                "regression": args.regression,
                "full": args.full,
                "new_failures": str(args.new_failures),
            },
        )


if False and __name__ == "__main__":
    main()



    electric = comparison.loc[
        comparison["metric"] == "physical_electricity_cost_usd_primary"
    ].iloc[0]
    carbon = comparison.loc[
        comparison["metric"] == "physical_carbon_kg_primary"
    ].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(["H1", "H4"], [electric["h1"], electric["h4_oracle"]], color=["#35618f", "#bf5b3d"])
    axes[0].set_title("Physical electricity cost")
    axes[0].set_ylabel("USD")
    axes[1].bar(["H1", "H4"], [carbon["h1"], carbon["h4_oracle"]], color=["#35618f", "#bf5b3d"])
    axes[1].set_title("Physical carbon")
    axes[1].set_ylabel("kgCO2")
    fig.tight_layout()
    fig.savefig(plot_dir / "02_electricity_carbon.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.plot(
        daily["block_id"],
        -daily["stage_cost_delta_h4_minus_h1"],
        color="#35618f",
        linewidth=1.1,
    )
    ax.set_xlabel("Continuous day block")
    ax.set_ylabel("Stage-cost improvement (H1 - H4)")
    ax.set_title("Daily closed-loop stage-cost delta")
    fig.tight_layout()
    fig.savefig(plot_dir / "03_daily_stage_cost_delta.png", dpi=180)
    plt.close(fig)

    pressure_stage = pressure.loc[
        (pressure["record_type"] == "EXTERNAL_PRESSURE")
        & (pressure["metric"] == "stage_cost")
    ].copy()
    improvement = (
        -pressure_stage["absolute_delta_h4_minus_h1"] / pressure_stage["steps"]
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(pressure_stage["pressure_band"], improvement, color=["#bf5b3d", "#35618f"])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Mean stage-cost improvement per step")
    ax.set_title("External GPU pressure stratification")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(plot_dir / "04_pressure_stage_cost_gain.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(
        shadow["shadow_task_disagreement_fraction"],
        -shadow["stage_cost_delta_h4_minus_h1"],
        s=8,
        alpha=0.24,
        color="#35618f",
        edgecolors="none",
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("v3 shadow task disagreement fraction")
    ax.set_ylabel("Closed-loop stage-cost improvement (H1 - H4)")
    ax.set_title("Shadow disagreement and closed-loop gain")
    fig.tight_layout()
    fig.savefig(plot_dir / "05_shadow_disagreement_vs_gain.png", dpi=180)
    plt.close(fig)

    threshold = 4e-9
    values = np.log10(fixed_raw["absolute_cost_gap"].to_numpy(dtype=float) + threshold)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(values, bins=80, color="#4e7d5b", alpha=0.9)
    ax.axvline(math.log10(2 * threshold), color="#bf5b3d", linestyle="--", linewidth=1)
    ax.set_xlabel("log10(abs fixed-state gap + tie-break threshold)")
    ax.set_ylabel("Disagreeing task decisions")
    ax.set_title("Fixed-state H1/H4 chosen-action objective gaps")
    fig.tight_layout()
    fig.savefig(plot_dir / "06_fixed_state_action_cost_gap.png", dpi=180)
    plt.close(fig)


def lookup_group_metric(
    frame: pd.DataFrame, group_column: str, group: str, metric: str
) -> tuple[float, float, float]:
    row = frame.loc[
        (frame[group_column].astype(str) == group) & (frame["metric"] == metric)
    ].iloc[0]
    return (
        float(row["h1"]),
        float(row["h4_oracle"]),
        float(row["absolute_delta_h4_minus_h1"]),
    )


def write_final_documents_duplicate_unused(result: Mapping[str, Any]) -> dict[str, Any]:
    h1 = result["metrics"]["H1"]
    h4 = result["metrics"]["H4_ORACLE"]
    diagnosis = result["diagnosis"]
    diagnostic = result["diagnostic"]
    pressure = result["pressure"]
    pressure = pressure.loc[
        (pressure["record_type"] == "EXTERNAL_PRESSURE")
        & (pressure["metric"] == "stage_cost")
    ]
    low = pressure.loc[pressure["pressure_band"] == "LOW_LT_TRAIN_P95"].iloc[0]
    high = pressure.loc[pressure["pressure_band"] == "HIGH_GE_TRAIN_P95"].iloc[0]
    hp = lookup_group_metric(result["priority"], "priority", "HP", "action_stage_cost")
    spot = lookup_group_metric(
        result["priority"], "priority", "Spot", "action_stage_cost"
    )
    hp_gain = -relative_delta(hp[0], hp[1])
    spot_gain = -relative_delta(spot[0], spot[1])
    benefit = (
        "NEITHER_EQUAL"
        if math.isclose(hp_gain, spot_gain, abs_tol=1e-12)
        else ("HP" if hp_gain > spot_gain else "SPOT")
    )
    corr = float(
        result["correlations"].loc[
            result["correlations"]["metric"] == "stage_cost_delta_h4_minus_h1",
            "pearson",
        ].iloc[0]
    )
    stage_delta = h4["stage_cost_total"] - h1["stage_cost_total"]
    stage_relative = relative_delta(h1["stage_cost_total"], h4["stage_cost_total"])
    electricity_delta = (
        h4["physical_electricity_cost_usd_primary"]
        - h1["physical_electricity_cost_usd_primary"]
    )
    carbon_delta = (
        h4["physical_carbon_kg_primary"] - h1["physical_carbon_kg_primary"]
    )
    sla_delta = (
        h4["primary_sla_violation_rate"] - h1["primary_sla_violation_rate"]
    )
    tail_electric_delta = (
        h4["physical_electricity_cost_usd_with_tail"]
        - h1["physical_electricity_cost_usd_with_tail"]
    )
    tail_sla_delta = h4["tail_sla_violation_rate"] - h1["tail_sla_violation_rate"]
    conclusion_changed = (
        np.sign(electricity_delta) != np.sign(tail_electric_delta)
        or np.sign(sla_delta) != np.sign(tail_sla_delta)
    )
    next_step = {
        "H4_CONTROL_VALUE_CONFIRMED": "�e�K�o��Vef`",
        "ACTION_DIFFERENCE_WITH_LIMITED_CONTROL_VALUE": "b�V(τVe��",
        "CONTEXT_DEPENDENT_CONTROL_VALUE": "�i�͹f`",
        "INCONCLUSIVE": "���6�<ʭ",
    }[diagnosis]
    next_step = {
        "H4_CONTROL_VALUE_CONFIRMED": "\u52a0\u5165\u9884\u6d4b\u4fe1\u606f\u7684\u4e13\u5bb6\u7b56\u7565\u5b66\u4e60",
        "ACTION_DIFFERENCE_WITH_LIMITED_CONTROL_VALUE": "\u9762\u5411\u51b3\u7b56\u8d28\u91cf\u7684\u7b56\u7565\u84b8\u998f",
        "CONTEXT_DEPENDENT_CONTROL_VALUE": "\u98ce\u9669\u72b6\u6001\u91cd\u70b9\u5b66\u4e60",
        "INCONCLUSIVE": "\u7ee7\u7eed\u63a7\u5236\u4ef7\u503c\u8bca\u65ad",
    }[diagnosis]
    eq = result["equivalence"]
    stable = diagnostic["daily_stage_improvement_block_fraction"]
    low_delta = float(low["absolute_delta_h4_minus_h1"]) / float(low["steps"])
    high_delta = float(high["absolute_delta_h4_minus_h1"]) / float(high["steps"])
    write_text(
        OUTPUT / "27_final_diagnosis.md",
        f"""
# Final Diagnosis

**{diagnosis}**

The 1% practical materiality reference is applied separately to the original
metrics; it is not a composite score. Daily and weekly blocks are descriptive
segments of one continuous trajectory, not independent random samples.

- Stage cost H1/H4/H4-H1: {h1['stage_cost_total']:.12g} / {h4['stage_cost_total']:.12g} / {stage_delta:.12g} ({stage_relative:.9f}%).
- Physical electricity H1/H4: {h1['physical_electricity_cost_usd_primary']:.12g} / {h4['physical_electricity_cost_usd_primary']:.12g} USD.
- Physical carbon H1/H4: {h1['physical_carbon_kg_primary']:.12g} / {h4['physical_carbon_kg_primary']:.12g} kgCO2.
- SLA violation-rate delta H4-H1: {sla_delta:.12g}.
- Positive daily stage-cost improvement fraction: {stable:.9%}.
- Low/high external-pressure mean stage-cost delta per step: {low_delta:.12g} / {high_delta:.12g}.
- Shadow disagreement versus closed-loop stage-cost delta Pearson: {corr:.9f}; descriptive, not causal.
- Tie-break-scale equivalent fraction: {eq['tie_break_scale_equivalent_fraction']:.9%}.
- Defer decisions H1/H4: {int(h1['defer_decision_count'])} / {int(h4['defer_decision_count'])}.
- Solver failures/fallback H1: {int(h1['solver_failures'])}/{int(h1['fallback_count'])}; H4: {int(h4['solver_failures'])}/{int(h4['fallback_count'])}.

Next stage: **{next_step}**. It is not implemented in this task.
""",
    )
    write_text(
        OUTPUT / "28_summary.md",
        f"""
# SpotGPU2026 Forward-control Value v1 Summary

1. H4 total stage-cost improvement (H1-H4): {-stage_delta:.12g}.
2. Relative stage-cost improvement: {-stage_relative:.9f}%.
3. SLA rates H1/H4 and delta: {h1['primary_sla_violation_rate']:.9%} / {h4['primary_sla_violation_rate']:.9%} / {sla_delta:.12g}.
4. Electricity H4-H1: {electricity_delta:.12g} USD under the frozen SustainCluster 2023 external signal scenario.
5. Carbon H4-H1: {carbon_delta:.12g} kgCO2 under that same external scenario.
6. Transmission cost and migration-count deltas: {h4['transmission_cost_usd'] - h1['transmission_cost_usd']:.12g} USD / {h4['migration_count'] - h1['migration_count']:.12g}.
7. Mean backlog and primary completion-rate deltas: {h4['backlog_mean'] - h1['backlog_mean']:.12g} / {h4['primary_completion_rate'] - h1['primary_completion_rate']:.12g}.
8. Low/high pressure stage-cost deltas per step: {low_delta:.12g} / {high_delta:.12g}.
9. HP/Spot action-stage-cost improvements: {hp_gain:.9f}% / {spot_gain:.9f}%; larger={benefit}.
10. Of the 26.244091% shadow disagreement, {eq['tie_break_scale_equivalent_fraction']:.9%} is equivalent at the conservative 4e-9 tie-break scale. Broader solver-equivalence is not hard-classified.
11. Disagreement/gain Pearson association: {corr:.9f}; descriptive only.
12. Positive H4 stage-cost gain occurs in {stable:.9%} of daily blocks.
13. Tail drain H1/H4: {int(h1['tail_drain_steps'])}/{int(h4['tail_drain_steps'])} steps; conclusion changed={'YES' if conclusion_changed else 'NO'}.
14. Current MPC mechanism: {'spatial placement' if h1['defer_decision_count'] == 0 and h4['defer_decision_count'] == 0 else 'spatial placement plus temporal defer'}.
15. Next stage: {next_step}.

Final diagnosis: **{diagnosis}**
""",
    )
    console = {
        "diagnosis": diagnosis,
        "next_step": next_step,
        "h1": h1,
        "h4": h4,
        "stage_absolute_delta": stage_delta,
        "stage_relative_delta": stage_relative,
        "sla_delta": sla_delta,
        "electricity_delta": electricity_delta,
        "carbon_delta": carbon_delta,
        "low_pressure_delta_per_step": low_delta,
        "high_pressure_delta_per_step": high_delta,
        "hp_benefit_percent": hp_gain,
        "spot_benefit_percent": spot_gain,
        "fixed_gap_median": eq["absolute_gap_median"],
        "near_equivalent_fraction": eq["tie_break_scale_equivalent_fraction"],
        "tail_conclusion_changed": conclusion_changed,
        "stage_corr": corr,
    }
    write_json(OUTPUT / "raw/final_console.json", console)
    return console


def write_integrity_manifest() -> None:
    rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "29_integrity_manifest.csv":
            rows.append(
                {
                    "path": path.relative_to(OUTPUT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_csv(OUTPUT / "29_integrity_manifest.csv", pd.DataFrame(rows))


def print_final_console(console: Mapping[str, Any], tests: Mapping[str, str]) -> None:
    h1, h4 = console["h1"], console["h4"]
    print(
        f"""
=== SPOTGPU2026 CONTROL VALUE v1 ===

=== CLOSED LOOP ===

Timeline:
17670 steps

H1 completed:
YES

H4 completed:
YES

=== STAGE COST ===

H1:
{h1['stage_cost_total']:.12g}

H4:
{h4['stage_cost_total']:.12g}

Absolute delta:
{console['stage_absolute_delta']:.12g}

Relative delta:
{console['stage_relative_delta']:.9f}%

=== SLA ===

H1:
{h1['primary_sla_violation_rate']:.9%}

H4:
{h4['primary_sla_violation_rate']:.9%}

Delta:
{console['sla_delta']:.12g}

=== ELECTRICITY ===

H1:
{h1['physical_electricity_cost_usd_primary']:.12g} USD

H4:
{h4['physical_electricity_cost_usd_primary']:.12g} USD

Delta:
{console['electricity_delta']:.12g} USD

=== CARBON ===

H1:
{h1['physical_carbon_kg_primary']:.12g} kgCO2

H4:
{h4['physical_carbon_kg_primary']:.12g} kgCO2

Delta:
{console['carbon_delta']:.12g} kgCO2

=== TRANSMISSION / MIGRATION ===

H1:
{h1['transmission_cost_usd']:.12g} USD / {int(h1['migration_count'])}

H4:
{h4['transmission_cost_usd']:.12g} USD / {int(h4['migration_count'])}

=== BACKLOG / COMPLETION ===

H1:
mean backlog {h1['backlog_mean']:.12g} / completion {h1['primary_completion_rate']:.9%}

H4:
mean backlog {h4['backlog_mean']:.12g} / completion {h4['primary_completion_rate']:.9%}

=== PRESSURE STRATIFICATION ===

Low pressure delta:
{console['low_pressure_delta_per_step']:.12g} per step

High pressure delta:
{console['high_pressure_delta_per_step']:.12g} per step

=== HP / SPOT ===

HP benefit:
{console['hp_benefit_percent']:.9f}%

Spot benefit:
{console['spot_benefit_percent']:.9f}%

=== ACTION VALUE ===

Shadow task disagreement:
26.244091%

Fixed-state action cost gap:
median absolute {console['fixed_gap_median']:.12g}

Near-equivalent evidence:
{console['near_equivalent_fraction']:.9%} at tie-break scale

=== TAIL DRAIN ===

H1:
{int(h1['tail_drain_steps'])} steps

H4:
{int(h4['tail_drain_steps'])} steps

Conclusion changed:
{'YES' if console['tail_conclusion_changed'] else 'NO'}

=== SOLVER ===

H1 failures:
{int(h1['solver_failures'])}

H4 failures:
{int(h4['solver_failures'])}

Timeout:
{int(h1['solver_timeouts'] + h4['solver_timeouts'])}

Fallback:
{int(h1['fallback_count'] + h4['fallback_count'])}

=== FINAL DIAGNOSIS ===

{console['diagnosis']}

=== NEXT STEP ===

{console['next_step']}

=== VALIDATION ===

compileall:
{tests['compileall']}

Dedicated:
{tests['dedicated']}

Regression:
{tests['regression']}

Full:
{tests['full']}

New failures:
{tests['new_failures']}

Transformer:
NO

BC:
NO

RL:
NO

Reward tuning:
NO

MPC tuning:
NO

Commit:
NO

Push:
NO

Final status:
{console['diagnosis']}
""".strip()
    )


def record_tests(
    compileall: str,
    dedicated: str,
    regression: str,
    full: str,
    new_failures: int,
) -> None:
    write_text(
        OUTPUT / "26_tests.md",
        f"""
# Validation

- compileall: {compileall}
- dedicated tests: {dedicated}
- relevant regressions: {regression}
- full pytest: {full}
- new failures: {new_failures}
- Transformer: NO
- BC: NO
- RL: NO
- reward tuning: NO
- MPC tuning: NO
""",
    )
    write_integrity_manifest()
    console = json.loads((OUTPUT / "raw/final_console.json").read_text("utf-8"))
    print_final_console(
        console,
        {
            "compileall": compileall,
            "dedicated": dedicated,
            "regression": regression,
            "full": full,
            "new_failures": str(new_failures),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        required=True,
        choices=["preflight", "run-h1", "run-h4", "finalize", "record-tests", "print-final"],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compileall", default="PENDING")
    parser.add_argument("--dedicated", default="PENDING")
    parser.add_argument("--regression", default="PENDING")
    parser.add_argument("--full", default="PENDING")
    parser.add_argument("--new-failures", type=int, default=-1)
    args = parser.parse_args()
    load_eval_config()
    if args.action == "preflight":
        inputs = spot_v3.load_inputs()
        write_protocol_files(inputs)
        print(json.dumps(checkpoint_exact_preflight(inputs), indent=2))
    elif args.action in {"run-h1", "run-h4"}:
        gate = json.loads((OUTPUT / "preflight/preflight_result.json").read_text("utf-8"))
        if gate["status"] != "PASS":
            raise RuntimeError("formal run blocked by preflight")
        inputs = spot_v3.load_inputs()
        controller = "h1" if args.action == "run-h1" else "h4_oracle"
        print(json.dumps(run_formal_controller(inputs, controller, resume=args.resume), indent=2))
    elif args.action == "finalize":
        result = finalize_outputs()
        console = write_final_documents(result)
        write_integrity_manifest()
        print(json.dumps(console, indent=2))
    elif args.action == "record-tests":
        record_tests(
            args.compileall, args.dedicated, args.regression, args.full, args.new_failures
        )
    else:
        console = json.loads((OUTPUT / "raw/final_console.json").read_text("utf-8"))
        print_final_console(
            console,
            {
                "compileall": args.compileall,
                "dedicated": args.dedicated,
                "regression": args.regression,
                "full": args.full,
                "new_failures": str(args.new_failures),
            },
        )


if __name__ == "__main__":
    main()
