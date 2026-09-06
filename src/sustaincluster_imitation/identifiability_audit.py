from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from forecasting.workload_forecast_provider import (
    ForecastTraceSource,
    OracleWorkloadForecastProvider,
)
from sustaincluster_imitation.bc_v2_offline import BCV2Arrays, sha256_file
from sustaincluster_mpc.forecast_pressure_adapter import distribute_global_forecast


AUDIT_NAME = "Privileged Distillation Identifiability Audit v1"
PARITY_STATUS = "CURRENT34_MISSING_CRITICAL_STATE"
HORIZONS_MINUTES = (15, 30, 45, 60)
RESOURCES = ("cpu", "gpu", "memory")
AUDIT_LABEL_COLUMNS = frozenset({"teacher_action_index", "h1_action_index"})


@dataclass(frozen=True)
class TrainOnlyStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    fitted_split: str = "train"

    @classmethod
    def fit(cls, values: np.ndarray, *, split: str) -> "TrainOnlyStandardizer":
        if split != "train":
            raise ValueError("Oracle future normalization must be fit on train only")
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != 60:
            raise ValueError("Oracle future matrix must have shape [N,60]")
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        scale = np.where(std > 1e-12, std, 1.0)
        return cls(mean.astype(np.float64), scale.astype(np.float64), split)

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.mean):
            raise ValueError("Oracle future matrix shape does not match normalizer")
        return ((matrix - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitted_split": self.fitted_split,
            "method": "per-dimension z-score",
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "zero_variance_scale_replacement": 1.0,
        }


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return dict(yaml.safe_load(stream)["privileged_distillation_identifiability_v1"])


def verify_frozen_inputs(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    root = Path(root)
    manifest = json.loads(
        (root / str(config["expert_dataset_manifest"])).read_text(encoding="utf-8")
    )
    full_dataset = root / str(config["expert_dataset_dir"]) / "expert_task_actions_full.parquet"
    digest = sha256_file(full_dataset)
    if digest != str(config["expected_dataset_sha256"]).upper():
        raise ValueError("Expert Dataset v2 SHA256 changed")
    if digest != manifest["primary_dataset_sha256"]:
        raise ValueError("Expert Dataset manifest SHA256 mismatch")
    if int(manifest["obs_dim"]) != int(config["current_obs_dim"]):
        raise ValueError("Current34 dimension changed")
    if int(manifest["action_dim"]) != int(config["action_dim"]):
        raise ValueError("action dimension changed")
    if (manifest["train_task_decisions"], manifest["val_task_decisions"], manifest["test_task_decisions"]) != (
        181944,
        38988,
        38988,
    ):
        raise ValueError("Expert Dataset v2 split sizes changed")
    return {
        "dataset_sha256": digest,
        "git_head": manifest["git_head"],
        "sustaincluster_commit": manifest["sustaincluster_commit"],
        "obs_dim": manifest["obs_dim"],
        "action_dim": manifest["action_dim"],
    }


def current_information_parity_rows() -> list[dict[str, Any]]:
    rows = [
        ("current_time", "Select current exogenous price/carbon and timestamp the state", "SchedulerState.exogenous + env managers", True, "0,1,2,3", "cyclic day/hour encoding", "aggregate", "Absolute timestamp and timestep are not explicit", "Optimizer consumes current signals directly rather than deriving them from time features"),
        ("timestep_minutes", "Convert duration, SLA, and transfer delay into optimization steps", "SchedulerState.exogenous.timestep_minutes", False, "", "fixed 15-minute constant", "implicit_constant", "No loss while the runtime resolution remains frozen", "Not a learned feature"),
        ("task_origin_dc", "Resolve origin and origin-to-destination network quantities", "TaskSnapshot.origin_dc_id", True, "4", "numeric DC id", "exact", "Nominal value is present", "DC identity is encoded as a scalar"),
        ("task_cpu_demand", "Energy and CPU capacity constraints", "TaskSnapshot.cpu_cores", True, "5", "raw cores", "exact", "No direct loss", "Per-task value"),
        ("task_gpu_demand", "Energy and GPU capacity constraints", "TaskSnapshot.gpu_units", True, "6", "raw GPU units", "exact", "No direct loss", "Per-task value"),
        ("task_memory_demand", "Energy and memory capacity constraints", "TaskSnapshot.memory_gb", False, "", "not represented", "missing", "Critical task demand is absent", "H1 writes task.memory_gb into the capacity matrix"),
        ("estimated_duration", "Energy, completion, and SLA feasibility", "TaskSnapshot.remaining_duration_minutes", True, "7", "estimated minutes", "exact_current_task", "No loss for a newly pending task; residual semantics are not separately marked", "Deployable duration contract"),
        ("remaining_sla", "Completion upper bounds and SLA-risk objective", "TaskSnapshot.remaining_sla_minutes", True, "8", "minutes to deadline", "exact", "No direct loss", "Student schema calls this SLA slack"),
        ("task_bandwidth", "Scale transmission cost and delay", "TaskSnapshot.bandwidth_gb", False, "", "not represented", "missing", "Critical task-specific network quantity is absent", "Cannot recover transfer cost/delay from origin alone"),
        ("transmission_cost", "Transmission objective for each destination", "TaskDestinationSnapshot.transmission_cost_usd", False, "", "not represented", "missing", "Destination cost is absent", "Fixed matrix is multiplied by missing task bandwidth"),
        ("transmission_delay", "Execution start, SLA feasibility, and feasible-action bounds", "TaskDestinationSnapshot.transmission_delay_seconds", False, "", "not represented", "missing", "Critical destination-specific delay is absent", "Affects transfer_steps and completion_step"),
        ("dc_cpu_available", "H1 CPU capacity upper bounds", "HorizonDataCenterSnapshot.cpu_available_cores[0]", True, "9,14,19,24,29", "available ratio", "aggregate", "Absolute schedulable capacity and reservations are collapsed", "DC totals are fixed but not explicit"),
        ("dc_gpu_available", "H1 GPU capacity upper bounds", "HorizonDataCenterSnapshot.gpu_available_units[0]", True, "10,15,20,25,30", "available ratio", "aggregate", "Absolute schedulable capacity and reservations are collapsed", "DC totals differ across sites"),
        ("dc_memory_available", "H1 memory capacity upper bounds", "HorizonDataCenterSnapshot.memory_available_gb[0]", True, "11,16,21,26,31", "available ratio", "aggregate", "Absolute schedulable capacity and reservations are collapsed", "Missing task memory compounds this loss"),
        ("current_carbon", "Current assignment carbon objective", "HorizonDataCenterSnapshot.carbon_intensity[0]", True, "12,17,22,27,32", "gCO2/kWh divided by 1000", "exact_scaled", "Invertible fixed scaling", "All five DC values present"),
        ("current_price", "Current assignment electricity objective", "HorizonDataCenterSnapshot.electricity_price[0]", True, "13,18,23,28,33", "price divided by 100", "exact_scaled", "Invertible fixed scaling", "All five DC values present"),
        ("queued_task_reservations", "Subtract queued CPU/GPU/memory from H1 capacity", "raw datacenter pending_tasks -> known reservations", False, "", "only reflected indirectly in raw availability if already allocated", "missing", "Critical schedulable-capacity state is absent", "Horizon adapter explicitly reserves queued tasks"),
        ("in_transit_reservations", "Reserve destination resources at arrival step", "env.in_transit_tasks", False, "", "not represented", "missing", "Critical current commitment state is absent", "Arrival, duration, and three resources are used"),
        ("running_task_release", "Release resources visible at the H1 node when release_step=0", "dc.running_tasks + controller finish time", False, "", "not represented", "missing", "Release timing cannot be reconstructed", "More influential for longer horizons but still part of H1 state"),
        ("joint_pending_task_set", "Joint MILP assignment and shared capacity competition", "SchedulerState.tasks", False, "", "one independent row contains only the focal task", "missing", "Critical multi-task coupling is absent", "Task-level ActorNet cannot see peer demands or queue composition"),
        ("task_position_tie_break", "Deterministic epsilon tie break across task/DC/step variables", "TaskSnapshot.original_index", False, "", "not represented in X", "missing", "Exact deterministic mapping can differ under ties", "sample_id is metadata, not a model feature"),
        ("allow_defer", "Enable backlog action and dispatch bounds", "SchedulerState.allow_defer + optimizer config", True, "mask", "feasible-action mask", "exact_separate_input", "No loss for action availability", "Defer has zero Teacher support in the primary traces"),
        ("per_task_feasible_actions", "Mask infeasible destination/defer logits", "deployable repaired-H4 feasibility adapter", True, "separate 6-d mask", "boolean mask", "aggregate", "Mask is per-task and does not encode joint MILP competition", "Teacher/H1 labels remain feasible"),
        ("objective_weights", "Weight energy, carbon, transmission, waiting, SLA, backlog", "frozen RollingHorizonConfig", False, "", "fixed experiment constants", "implicit_constant", "No loss within one frozen objective", "Not state information"),
        ("power_coefficients", "Convert CPU/GPU/memory and duration to task energy", "frozen RollingHorizonConfig", False, "", "fixed experiment constants", "implicit_constant", "No loss within one frozen power model", "Missing memory still prevents exact energy recovery"),
    ]
    columns = (
        "planner_input_name",
        "planner_usage",
        "source",
        "student_obs_available",
        "student_obs_indices",
        "representation",
        "exact_or_aggregate",
        "information_loss",
        "notes",
    )
    return [dict(zip(columns, row)) for row in rows]


def future_feature_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 34
    raw_names = {
        "cpu": "arriving_cpu_demand",
        "gpu": "arriving_gpu_demand",
        "memory": "arriving_memory_demand",
    }
    for dc_id in range(1, 6):
        for horizon_step, horizon_minutes in enumerate(HORIZONS_MINUTES, start=1):
            for resource in RESOURCES:
                rows.append(
                    {
                        "augmented_index": index,
                        "future_feature_index": index - 34,
                        "feature_name": f"oracle_dc{dc_id}_h{horizon_minutes}_{resource}_pressure",
                        "dc": dc_id,
                        "horizon_step": horizon_step,
                        "horizon_minutes": horizon_minutes,
                        "resource": resource,
                        "raw_source": f"Forecast Dataset v1 {raw_names[resource]}",
                        "aggregation": "global true future demand times deterministic origin probability",
                        "normalization": "train-split per-dimension z-score",
                        "timeline_semantics": f"future node {horizon_step} = +{horizon_minutes} minutes",
                        "used_by_repaired_h4": True,
                        "information_class": "NON_DEPLOYABLE_ORACLE_DIAGNOSTIC",
                    }
                )
                index += 1
    if index != 94:
        raise RuntimeError("Oracle future feature manifest is not 60-dimensional")
    return rows


class OracleFuturePressureBuilder:
    """Rebuild the exact workload-pressure quantities injected into repaired H4."""

    def __init__(self, forecast_dataset_root: Path, datacenter_config: Path) -> None:
        self.trace = ForecastTraceSource(forecast_dataset_root)
        self.oracle = OracleWorkloadForecastProvider()
        raw = yaml.safe_load(Path(datacenter_config).read_text(encoding="utf-8"))
        self.datacenters = tuple(
            sorted(raw["datacenters"], key=lambda item: int(item["dc_id"]))
        )
        if tuple(int(item["dc_id"]) for item in self.datacenters) != (1, 2, 3, 4, 5):
            raise ValueError("Oracle future builder requires canonical DC1-DC5 order")
        self._cache: dict[str, np.ndarray] = {}

    def vector(self, timestamp: str | pd.Timestamp) -> np.ndarray:
        key = pd.Timestamp(timestamp).isoformat()
        cached = self._cache.get(key)
        if cached is not None:
            return cached.copy()
        request = self.trace.request(timestamp, include_oracle_future=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bundle = self.oracle.forecast(request)
        pressures = distribute_global_forecast(bundle, self.datacenters)
        lookup = {
            (item.dc_id, item.horizon_minutes): item for item in pressures
        }
        values: list[float] = []
        for dc_id in range(1, 6):
            for horizon_minutes in HORIZONS_MINUTES:
                point = lookup[(dc_id, horizon_minutes)]
                values.extend((point.cpu_demand, point.gpu_demand, point.memory_demand))
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (60,) or not np.isfinite(vector).all() or np.any(vector < 0):
            raise ValueError("invalid Oracle future pressure vector")
        self._cache[key] = vector
        return vector.copy()

    def matrix(self, timestamps: Sequence[str | pd.Timestamp]) -> np.ndarray:
        return np.stack([self.vector(value) for value in timestamps]).astype(np.float64)


def read_audit_frame(path: Path) -> pd.DataFrame:
    columns = [
        "sample_id",
        "episode_id",
        "seed",
        "scenario",
        "step",
        "timestamp",
        "task_id",
        "student_observation",
        "feasible_action_mask",
        "teacher_action_index",
        "h1_action_index",
        "deployable_risk_score",
    ]
    return pq.read_table(path, columns=columns).to_pandas()


def build_audit_arrays(
    frame: pd.DataFrame,
    *,
    label_column: str,
    action_dim: int,
    future_features: np.ndarray | None = None,
) -> BCV2Arrays:
    if label_column not in AUDIT_LABEL_COLUMNS:
        raise ValueError("unsupported identifiability-audit label provenance")
    current = np.asarray(frame["student_observation"].tolist(), dtype=np.float32)
    if current.shape != (len(frame), 34):
        raise ValueError("Current34 matrix shape mismatch")
    if future_features is None:
        observations = current
    else:
        future = np.asarray(future_features, dtype=np.float32)
        if future.shape != (len(frame), 60):
            raise ValueError("Oracle future feature matrix shape mismatch")
        observations = np.concatenate((current, future), axis=1)
    masks = np.asarray(frame["feasible_action_mask"].tolist(), dtype=bool)
    labels = frame[label_column].to_numpy(dtype=np.int64, copy=True)
    if masks.shape != (len(frame), action_dim):
        raise ValueError("action mask shape mismatch")
    if not masks[np.arange(len(frame)), labels].all():
        raise ValueError(f"{label_column} contains an action outside the saved mask")
    if not np.isfinite(observations).all():
        raise ValueError("audit observation contains NaN or infinity")
    return BCV2Arrays(
        observations,
        labels,
        masks,
        tuple(str(value) for value in frame["sample_id"]),
    )


def deterministic_feature_hash(sample_ids: Sequence[str], values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for sample_id, row in zip(sample_ids, np.asarray(values, dtype="<f4")):
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(row.tobytes(order="C"))
    return digest.hexdigest().upper()


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, action_dim: int) -> list[list[int]]:
    matrix = np.zeros((action_dim, action_dim), dtype=np.int64)
    np.add.at(matrix, (labels.astype(int), predictions.astype(int)), 1)
    return matrix.tolist()


def per_action_accuracy(
    labels: np.ndarray, predictions: np.ndarray, action_dim: int
) -> list[dict[str, Any]]:
    rows = []
    for action in range(action_dim):
        selector = labels == action
        rows.append(
            {
                "action": action,
                "support": int(selector.sum()),
                "accuracy": float((predictions[selector] == action).mean())
                if selector.any()
                else float("nan"),
                "support_status": "OK" if selector.any() else "NO_LABEL_SUPPORT",
            }
        )
    return rows


def state_any_mismatch(
    frame: pd.DataFrame, labels: np.ndarray, predictions: np.ndarray
) -> pd.DataFrame:
    work = frame[["episode_id", "step"]].copy()
    work["mismatch"] = labels != predictions
    return (
        work.groupby(["episode_id", "step"], sort=False)["mismatch"]
        .agg(any_mismatch="any", mismatch_tasks="sum", task_count="size")
        .reset_index()
    )


def verify_same_split_identity(frames: Mapping[str, pd.DataFrame]) -> bool:
    seed_sets = {
        name: frozenset(int(value) for value in frame["seed"].unique())
        for name, frame in frames.items()
    }
    return (
        not seed_sets["train"] & seed_sets["validation"]
        and not seed_sets["train"] & seed_sets["test"]
        and not seed_sets["validation"] & seed_sets["test"]
        and sum(len(values) for values in seed_sets.values()) == 40
    )
