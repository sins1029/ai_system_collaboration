from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from sustaincluster_imitation.bc_policy import build_sustaincluster_actor
from sustaincluster_imitation.feature_encoder import (
    SemanticActionSpace,
    SustainClusterFeatureEncoder,
)
from sustaincluster_mpc.horizon_adapter import HorizonState


PRIMARY_SEEDS = tuple(range(3001, 3041))
CALIBRATION_SEEDS = tuple(range(1101, 1106))
EVALUATION_SEEDS = tuple(range(1201, 1206))
SPLIT_SHUFFLE_SEED = 2026
EXPECTED_STUDENT_OBS_DIM = 34
EXPECTED_ACTION_DIM = 6
FORBIDDEN_STUDENT_FEATURE_TOKENS = (
    "true_duration",
    "future_workload",
    "future_price",
    "future_carbon",
    "future_release",
    "teacher_objective",
    "teacher_future",
    "oracle",
    "future_task",
)


@dataclass(frozen=True)
class StudentTaskBatch:
    observations: np.ndarray
    feasible_action_mask: np.ndarray
    task_ids: tuple[str, ...]
    original_indices: tuple[int, ...]
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class BCDryRunBatch:
    observations: np.ndarray
    labels: np.ndarray
    masks: np.ndarray


def student_feature_names(dc_ids: Sequence[int]) -> tuple[str, ...]:
    names = [
        "current_time_day_sin",
        "current_time_day_cos",
        "current_time_hour_sin",
        "current_time_hour_cos",
        "current_task_origin_dc_id",
        "current_task_cpu_cores",
        "current_task_gpu_units",
        "current_task_estimated_duration_minutes",
        "current_task_sla_slack_minutes",
    ]
    for dc_id in dc_ids:
        names.extend(
            (
                f"current_dc{dc_id}_available_cpu_ratio",
                f"current_dc{dc_id}_available_gpu_ratio",
                f"current_dc{dc_id}_available_memory_ratio",
                f"current_dc{dc_id}_carbon_intensity_scaled",
                f"current_dc{dc_id}_electricity_price_scaled",
            )
        )
    return tuple(names)


def assert_deployable_feature_schema(feature_names: Sequence[str]) -> None:
    lowered = tuple(str(name).lower() for name in feature_names)
    leaks = sorted(
        {
            token
            for name in lowered
            for token in FORBIDDEN_STUDENT_FEATURE_TOKENS
            if token in name
        }
    )
    if leaks:
        raise ValueError(f"student feature schema contains forbidden tokens: {leaks}")
    if not any("estimated_duration" in name for name in lowered):
        raise ValueError("student feature schema must expose estimated_duration")


def build_student_task_batch(
    env: Any,
    deployable_mask_state: HorizonState,
    semantic_actions: SemanticActionSpace,
) -> StudentTaskBatch:
    if getattr(env, "information_mode", None) != "deployable":
        raise ValueError("student observations require a deployable environment")
    if deployable_mask_state.information_mode != "deployable":
        raise ValueError("student masks require a deployable planning state")
    observations = tuple(env._generate_per_task_obs_list())
    feature_names = student_feature_names(semantic_actions.dc_ids)
    assert_deployable_feature_schema(feature_names)
    if observations:
        matrix = np.asarray(observations, dtype=np.float32)
    else:
        matrix = np.empty((0, len(feature_names)), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"student observation shape mismatch: {matrix.shape}, "
            f"expected [N,{len(feature_names)}]"
        )
    encoded = SustainClusterFeatureEncoder(
        semantic_actions, horizon=deployable_mask_state.horizon
    ).encode(deployable_mask_state)
    expected_ids = tuple(
        task.task_id for task in deployable_mask_state.current.tasks
    )
    expected_indices = tuple(
        task.original_index for task in deployable_mask_state.current.tasks
    )
    if encoded.task_ids != expected_ids or encoded.original_indices != expected_indices:
        raise RuntimeError("feature-mask task identity alignment failed")
    raw_ids = tuple(str(task.job_name) for task in env.current_tasks)
    if raw_ids != expected_ids or len(matrix) != len(expected_ids):
        raise RuntimeError("student observation task identity alignment failed")
    if encoded.feasible_action_mask.shape != (
        len(expected_ids),
        semantic_actions.size,
    ):
        raise RuntimeError("student feasible mask shape mismatch")
    return StudentTaskBatch(
        matrix,
        encoded.feasible_action_mask.astype(bool, copy=False),
        expected_ids,
        expected_indices,
        feature_names,
    )


def semantic_action_name(
    action_index: int, semantic_actions: SemanticActionSpace
) -> str:
    dc_id = semantic_actions.dc_for_index(int(action_index))
    return "defer" if dc_id is None else f"assign_dc{dc_id}"


def stable_sample_id(episode_id: str, step: int, task_id: str) -> str:
    if not episode_id or not task_id or int(step) < 0:
        raise ValueError("sample identity fields must be non-empty and nonnegative")
    return f"{episode_id}::step_{int(step):03d}::task_{task_id}"


def build_split_manifest(
    seeds: Sequence[int] = PRIMARY_SEEDS,
    *,
    shuffle_seed: int = SPLIT_SHUFFLE_SEED,
    train_seed_count: int = 28,
    validation_seed_count: int = 6,
) -> dict[str, Any]:
    values = [int(seed) for seed in seeds]
    if len(values) != len(set(values)):
        raise ValueError("split seeds must be unique")
    if set(values) & (set(CALIBRATION_SEEDS) | set(EVALUATION_SEEDS)):
        raise ValueError("primary seeds overlap calibration/evaluation seeds")
    if train_seed_count <= 0 or validation_seed_count <= 0:
        raise ValueError("train and validation seed counts must be positive")
    if train_seed_count + validation_seed_count >= len(values):
        raise ValueError("test split must contain at least one seed")
    shuffled = list(values)
    random.Random(int(shuffle_seed)).shuffle(shuffled)
    train = tuple(sorted(shuffled[:train_seed_count]))
    validation = tuple(
        sorted(shuffled[train_seed_count : train_seed_count + validation_seed_count])
    )
    test = tuple(sorted(shuffled[train_seed_count + validation_seed_count :]))
    manifest = {
        "split_unit": "seed_and_complete_episode",
        "shuffle_seed": int(shuffle_seed),
        "train_seeds": list(train),
        "validation_seeds": list(validation),
        "test_seeds": list(test),
        "seed_to_split": {
            str(seed): split
            for split, split_seeds in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
            for seed in split_seeds
        },
    }
    validate_split_manifest(manifest, expected_seeds=values)
    return manifest


def validate_split_manifest(
    manifest: Mapping[str, Any], *, expected_seeds: Sequence[int]
) -> None:
    groups = {
        split: {int(seed) for seed in manifest[f"{split}_seeds"]}
        for split in ("train", "validation", "test")
    }
    if any(groups[left] & groups[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("split seed groups overlap")
    if set().union(*groups.values()) != {int(seed) for seed in expected_seeds}:
        raise ValueError("split seed groups do not cover expected seeds")


def validate_teacher_solution(result: Any) -> None:
    if getattr(result, "status", None) not in {"optimal", "feasible"}:
        raise RuntimeError(
            "teacher solver failed; fallback actions cannot become expert labels"
        )
    if not bool(getattr(result, "feasible", False)):
        raise RuntimeError(
            "teacher solver is not feasible; fallback actions cannot become labels"
        )


def environment_state_signature(env: Any) -> str:
    payload = {
        "current_time": str(env.current_time),
        "global_step": int(getattr(env, "global_step", 0)),
        "current_tasks": [
            (
                str(task.job_name),
                int(getattr(task, "wait_intervals", 0)),
                int(getattr(task, "scheduler_wait_intervals", 0)),
                bool(getattr(task, "temporarily_deferred", False)),
            )
            for task in env.current_tasks
        ],
        "transit": [
            (str(arrival), str(task.job_name), str(dc_name))
            for arrival, task, dc_name in env.in_transit_tasks
        ],
        "datacenters": [
            (
                str(name),
                float(dc.available_cores),
                float(dc.available_gpus),
                float(dc.available_mem),
                tuple(str(task.job_name) for task in dc.running_tasks),
                tuple(str(task.job_name) for task in dc.pending_tasks),
            )
            for name, dc in env.cluster_manager.datacenters.items()
        ],
        "python_rng": pickle.dumps(random.getstate()).hex(),
        "numpy_rng": pickle.dumps(np.random.get_state()).hex(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()


def make_disagreement_index(
    rows: pd.DataFrame, disagreement_column: str
) -> pd.DataFrame:
    if disagreement_column not in rows:
        raise ValueError(f"missing disagreement column {disagreement_column}")
    selected = rows.loc[
        rows[disagreement_column].astype(bool), ["row_id", "sample_id"]
    ].copy()
    selected.insert(2, "criterion", disagreement_column)
    return selected.reset_index(drop=True)


def build_state_disagreement_summary(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "episode_id",
        "seed",
        "scenario",
        "split",
        "step",
        "timestamp",
        "exact_action_disagreement",
        "semantic_disagreement",
        "target_dc_disagreement",
        "deployable_risk_score",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"state summary input missing columns: {sorted(missing)}")
    records = []
    group_columns = ["episode_id", "seed", "scenario", "split", "step", "timestamp"]
    for keys, group in rows.groupby(group_columns, sort=False, dropna=False):
        exact = int(group["exact_action_disagreement"].astype(bool).sum())
        target = int(group["target_dc_disagreement"].astype(bool).sum())
        semantic = int(group["semantic_disagreement"].astype(bool).sum())
        count = len(group)
        records.append(
            {
                **dict(zip(group_columns, keys)),
                "num_tasks": count,
                "num_exact_disagreements": exact,
                "num_target_dc_disagreements": target,
                "num_semantic_disagreements": semantic,
                "any_disagreement": exact > 0,
                "disagreement_rate": exact / max(1, count),
                "deployable_risk_score": float(group["deployable_risk_score"].iloc[0]),
            }
        )
    return pd.DataFrame(records)


def dataframe_content_sha256(
    rows: Iterable[Mapping[str, Any]],
) -> str:
    digest = hashlib.sha256()
    for row in rows:
        observation = np.asarray(row["student_observation"], dtype="<f4")
        mask = np.asarray(row["feasible_action_mask"], dtype=np.uint8)
        digest.update(str(row["sample_id"]).encode("utf-8"))
        digest.update(observation.tobytes())
        digest.update(mask.tobytes())
        digest.update(int(row["teacher_action_index"]).to_bytes(2, "little"))
    return digest.hexdigest().upper()


def stable_task_identity_pass(rows: pd.DataFrame) -> bool:
    static = (
        "origin_dc",
        "task_cpu_cores",
        "task_gpu_units",
        "task_memory_gb",
        "estimated_duration",
    )
    if rows.empty:
        return False
    grouped = rows.groupby(["episode_id", "task_id"], dropna=False)
    return all(
        int(group[list(static)].drop_duplicates().shape[0]) == 1
        for _, group in grouped
    )


def primary_quality_counts(
    rows: pd.DataFrame, *, obs_dim: int, action_dim: int
) -> dict[str, int]:
    observations = [np.asarray(value, dtype=np.float32) for value in rows["student_observation"]]
    masks = [np.asarray(value, dtype=bool) for value in rows["feasible_action_mask"]]
    labels = rows["teacher_action_index"].astype(int).to_numpy()
    wrong_observation_shape = sum(value.shape != (obs_dim,) for value in observations)
    wrong_mask_shape = sum(value.shape != (action_dim,) for value in masks)
    nan_rows = sum(not np.isfinite(value).all() for value in observations)
    invalid_labels = int(((labels < 0) | (labels >= action_dim)).sum())
    infeasible_labels = 0
    for label, mask in zip(labels, masks):
        if 0 <= label < len(mask) and not bool(mask[label]):
            infeasible_labels += 1
    return {
        "nan_observation_rows": int(nan_rows),
        "wrong_observation_shape_rows": int(wrong_observation_shape),
        "missing_or_wrong_mask_rows": int(wrong_mask_shape),
        "invalid_label_rows": invalid_labels,
        "infeasible_label_rows": int(infeasible_labels),
        "duplicate_sample_ids": int(rows["sample_id"].duplicated().sum()),
    }


def load_bc_dry_run_batch(
    rows: pd.DataFrame, *, batch_size: int = 64
) -> BCDryRunBatch:
    selected = rows.iloc[: min(int(batch_size), len(rows))]
    if selected.empty:
        raise ValueError("BC dry-run batch cannot be empty")
    observations = np.asarray(
        selected["student_observation"].tolist(), dtype=np.float32
    )
    labels = selected["teacher_action_index"].to_numpy(dtype=np.int64)
    masks = np.asarray(selected["feasible_action_mask"].tolist(), dtype=bool)
    if observations.ndim != 2 or labels.shape != (len(selected),):
        raise ValueError("BC dry-run observation/label shape mismatch")
    if masks.shape != (len(selected), masks.shape[-1]):
        raise ValueError("BC dry-run mask shape mismatch")
    if not np.isfinite(observations).all():
        raise ValueError("BC dry-run observations contain NaN/inf")
    if not masks[np.arange(len(labels)), labels].all():
        raise ValueError("BC dry-run contains infeasible labels")
    return BCDryRunBatch(observations, labels, masks)


def actor_forward_dry_run(batch: BCDryRunBatch) -> tuple[int, int]:
    import torch

    actor = build_sustaincluster_actor(
        batch.observations.shape[1], batch.masks.shape[1]
    )
    actor.eval()
    with torch.no_grad():
        logits = actor(torch.from_numpy(batch.observations))
    expected = (len(batch.labels), batch.masks.shape[1])
    if tuple(logits.shape) != expected:
        raise RuntimeError(
            f"ActorNet forward shape mismatch: {tuple(logits.shape)} != {expected}"
        )
    return expected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def finite_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise ValueError("ratio values must be finite")
    return float(numerator) / max(1.0, float(denominator))
