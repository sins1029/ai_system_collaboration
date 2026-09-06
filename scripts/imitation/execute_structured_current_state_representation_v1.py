from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.imitation import run_structured_current_state_representation_v1 as audit
from sustaincluster_imitation.bc_v2_offline import (
    BCV2Arrays,
    choose_recommended_run,
    classification_metrics,
    infer,
    load_checkpoint,
    per_action_metrics,
    sha256_file,
    train_seed,
)
from sustaincluster_imitation.current_collision import (
    current34_collision_audit,
    near_neighbor_audit,
    representation_collision_summary,
)
from sustaincluster_imitation.structured_current import (
    ACTION_DIM,
    DC_IDS,
    INFORMATION_CLASS,
    StructuredArrayStore,
    TrainOnlyStandardizer,
    dc_feature_names,
    dense_feature_names,
    infer_structured,
    load_structured_checkpoint,
    model_parameter_count,
    running_feature_names,
    task_feature_names,
    train_structured_seed,
)


FRAME_COLUMNS = (
    "row_id",
    "sample_id",
    "episode_id",
    "seed",
    "scenario",
    "split",
    "step",
    "timestamp",
    "task_id",
    "task_position",
    "student_observation",
    "feasible_action_mask",
    "teacher_action_index",
    "h1_action_index",
    "deployable_risk_score",
    "exact_action_disagreement",
)


def read_frame(config: Mapping[str, Any]) -> pd.DataFrame:
    path = ROOT / str(config["expert_dataset_dir"]) / "expert_task_actions_full.parquet"
    frame = pd.read_parquet(path, columns=list(FRAME_COLUMNS))
    frame = frame.sort_values("row_id").reset_index(drop=True)
    if len(frame) != 259920:
        raise RuntimeError("Expert Dataset v2 task count changed")
    if not np.array_equal(frame["row_id"].to_numpy(), np.arange(len(frame))):
        raise RuntimeError("Expert Dataset v2 row order changed")
    return frame


def task_matrix(frame: pd.DataFrame, column: str, dtype: Any) -> np.ndarray:
    return np.stack(frame[column].map(lambda value: np.asarray(value, dtype=dtype)))


def concatenate_state_ranges(
    values: np.ndarray, offsets: np.ndarray, selected_states: Sequence[int]
) -> np.ndarray:
    parts = [values[offsets[index] : offsets[index + 1]] for index in selected_states]
    if not parts:
        raise ValueError("selected state range is empty")
    nonempty = [part for part in parts if len(part)]
    if not nonempty:
        raise ValueError("selected state range contains no values")
    return np.concatenate(nonempty)


def schema_rows() -> pd.DataFrame:
    rows = []
    for index, name in enumerate(dense_feature_names()):
        if name.startswith("focal_"):
            group = "focal_task"
            source = "HorizonState.current.tasks[focal]"
            usage = "task objective, constraints, order, or current context"
        elif name.startswith("focal_to_dc"):
            group = "task_destination"
            source = "HorizonState.current.task_destinations"
            usage = "transmission objective and transfer/SLA semantics"
        elif name.startswith("dc"):
            group = "datacenter_current"
            source = "current DataCenterSnapshot + H1 horizon node 0"
            usage = "H1 current capacity constraint or current energy signal"
        elif name.startswith("pending_"):
            group = "pending_queue_aggregate"
            source = "HorizonState.current.tasks"
            usage = "joint current-batch competition summary"
        elif name.startswith("running_"):
            group = "running_task_aggregate"
            source = "HorizonState.running_tasks"
            usage = "controller-visible current release/capacity context"
        elif name.startswith("transit_"):
            group = "in_transit_aggregate"
            source = "HorizonState.transit_tasks"
            usage = "known current reservation context"
        else:
            raise RuntimeError(f"unclassified CurrentDense feature {name}")
        rows.append(
            {
                "feature_index": index,
                "feature_name": name,
                "group": group,
                "source": source,
                "deployable": True,
                "normalization": "train-split per-dimension z-score",
                "H1_usage": usage,
                "notes": "current information only; no action/objective/future input",
            }
        )
    return pd.DataFrame(rows)


def normalize_representations(
    replay: Mapping[str, Any],
    frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, TrainOnlyStandardizer]]:
    state_frame = replay["state_frame"]
    train_tasks = frame["split"].eq("train").to_numpy()
    train_states = state_frame.index[state_frame["split"].eq("train")].to_numpy(dtype=np.int64)
    dense_scaler = TrainOnlyStandardizer.fit(replay["dense_raw"][train_tasks], split="train")
    task_scaler = TrainOnlyStandardizer.fit(replay["pending_raw"][train_tasks], split="train")
    running_train = concatenate_state_ranges(
        replay["running_raw"], replay["running_offsets"], train_states
    )
    running_scaler = TrainOnlyStandardizer.fit(running_train, split="train")
    dc_scaler = TrainOnlyStandardizer.fit(
        replay["dc_raw"][train_states].reshape(-1, replay["dc_raw"].shape[-1]),
        split="train",
    )
    return (
        {
            "dense": dense_scaler.transform(replay["dense_raw"]),
            "pending": task_scaler.transform(replay["pending_raw"]),
            "running": running_scaler.transform(replay["running_raw"]),
            "dc": dc_scaler.transform(
                replay["dc_raw"].reshape(-1, replay["dc_raw"].shape[-1])
            ).reshape(replay["dc_raw"].shape),
        },
        {
            "current_dense": dense_scaler,
            "pending_and_focal": task_scaler,
            "running": running_scaler,
            "datacenter": dc_scaler,
        },
    )


def build_store(
    normalized: Mapping[str, np.ndarray],
    replay: Mapping[str, Any],
    frame: pd.DataFrame,
) -> StructuredArrayStore:
    masks = task_matrix(frame, "feasible_action_mask", bool)
    labels = frame["h1_action_index"].to_numpy(dtype=np.int64)
    state_frame = replay["state_frame"]
    return StructuredArrayStore(
        pending_values=normalized["pending"],
        state_offsets=replay["state_offsets"],
        running_values=normalized["running"],
        running_offsets=replay["running_offsets"],
        dc_values=normalized["dc"],
        labels=labels,
        masks=masks,
        state_ids=tuple(state_frame["state_id"].astype(str)),
        splits=tuple(state_frame["split"].astype(str)),
    )


def save_derived_arrays(
    output: Path,
    normalized: Mapping[str, np.ndarray],
    replay: Mapping[str, Any],
    frame: pd.DataFrame,
) -> Path:
    path = output / "dataset" / "structured_current_arrays_v1.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dense=normalized["dense"],
        pending=normalized["pending"],
        running=normalized["running"],
        datacenters=normalized["dc"],
        state_offsets=replay["state_offsets"],
        running_offsets=replay["running_offsets"],
        labels=frame["h1_action_index"].to_numpy(dtype=np.int8),
        masks=task_matrix(frame, "feasible_action_mask", bool),
        task_state_index=replay["task_state_index"],
    )
    replay["state_frame"].to_parquet(
        output / "dataset" / "state_metadata.parquet", index=False
    )
    return path


def bc_arrays(
    values: np.ndarray,
    frame: pd.DataFrame,
    split: str,
) -> BCV2Arrays:
    selector = frame["split"].eq(split).to_numpy()
    labels = frame.loc[selector, "h1_action_index"].to_numpy(dtype=np.int64)
    masks = task_matrix(frame.loc[selector], "feasible_action_mask", bool)
    if not masks[np.arange(len(labels)), labels].all():
        raise RuntimeError("H1 label is outside the frozen feasible mask")
    return BCV2Arrays(
        values[selector].astype(np.float32, copy=False),
        labels,
        masks,
        tuple(frame.loc[selector, "sample_id"].astype(str)),
    )


def train_dense(
    output: Path,
    config: Mapping[str, Any],
    normalized_dense: np.ndarray,
    frame: pd.DataFrame,
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    train = bc_arrays(normalized_dense, frame, "train")
    validation = bc_arrays(normalized_dense, frame, "validation")
    dense_config = dict(config)
    dense_config["obs_dim"] = normalized_dense.shape[1]
    dense_config["batch_size"] = int(config["task_batch_size"])
    runs = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        print(f"experiment=BC_DENSE seed={seed}", flush=True)
        runs.append(
            train_seed(
                seed=seed,
                config=dense_config,
                train=train,
                validation=validation,
                checkpoint_path=output / "checkpoints" / f"bc_dense_seed_{seed}_best.pt",
                history_path=output / "training_history" / f"bc_dense_seed_{seed}.csv",
                checkpoint_metadata={
                    **dict(provenance),
                    "experiment": "BC_DENSE_CURRENT_ONLY",
                    "label_provenance": "frozen h1_action_index",
                    "information_class": INFORMATION_CLASS,
                    "normalization": "train-only z-score",
                },
            )
        )
    return runs


def train_structured(
    output: Path,
    config: Mapping[str, Any],
    store: StructuredArrayStore,
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    runs = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        print(f"experiment=STRUCTURED_CURRENT seed={seed}", flush=True)
        runs.append(
            train_structured_seed(
                seed=seed,
                config=config,
                store=store,
                checkpoint_path=output / "checkpoints" / f"structured_current_seed_{seed}_best.pt",
                history_path=output / "training_history" / f"structured_current_seed_{seed}.csv",
                metadata={
                    **dict(provenance),
                    "experiment": "STRUCTURED_CURRENT_ONLY",
                    "label_provenance": "frozen h1_action_index",
                    "information_class": INFORMATION_CLASS,
                    "pending_context": "full current pending set excluding focal after phi",
                    "running_context": "full currently running set",
                    "dc_context": "five semantic DC rows with identity",
                    "joint_decoder": False,
                    "teacher_forcing": False,
                },
            )
        )
    return runs


def load_bc34_runs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = ROOT / str(config["bc34_artifacts"]) / "audit_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    runs = list(manifest["h1_runs"])
    if sorted(int(row["seed"]) for row in runs) != [11, 22, 33]:
        raise RuntimeError("BC34 frozen run set changed")
    return runs


def _prediction_record(
    model: str,
    run: Mapping[str, Any],
    test_indices: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    top2: np.ndarray,
    total_tasks: int,
) -> dict[str, Any]:
    full = np.full(total_tasks, -1, dtype=np.int64)
    full[test_indices] = predictions
    return {
        "model": model,
        "seed": int(run["seed"]),
        "run": dict(run),
        "test_indices": test_indices,
        "predictions": predictions,
        "probabilities": probabilities,
        "top2": top2,
        "full_predictions": full,
    }


def collect_predictions(
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    normalized_dense: np.ndarray,
    store: StructuredArrayStore,
    dense_runs: Sequence[Mapping[str, Any]],
    structured_runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    test_indices = np.flatnonzero(frame["split"].eq("test").to_numpy())
    current34 = task_matrix(frame, "student_observation", np.float32)
    current34_test = bc_arrays(current34, frame, "test")
    dense_test = bc_arrays(normalized_dense, frame, "test")
    records = []
    for run in load_bc34_runs(config):
        model, payload = load_checkpoint(Path(str(run["checkpoint"])))
        if payload.get("experiment") != "bc_h1_current34":
            raise RuntimeError("BC34 checkpoint provenance mismatch")
        pred, prob, top2 = infer(model, current34_test, int(config["inference_batch_size"]))
        records.append(
            _prediction_record("BC34", run, test_indices, pred, prob, top2, len(frame))
        )
    for run in dense_runs:
        model, payload = load_checkpoint(Path(str(run["checkpoint"])))
        if payload.get("experiment") != "BC_DENSE_CURRENT_ONLY":
            raise RuntimeError("CurrentDense checkpoint provenance mismatch")
        pred, prob, top2 = infer(model, dense_test, int(config["inference_batch_size"]))
        records.append(
            _prediction_record("CurrentDense", run, test_indices, pred, prob, top2, len(frame))
        )
    test_states = store.state_indices("test")
    for run in structured_runs:
        model, payload = load_structured_checkpoint(Path(str(run["checkpoint"])))
        if payload.get("experiment") != "STRUCTURED_CURRENT_ONLY":
            raise RuntimeError("StructuredCurrent checkpoint provenance mismatch")
        task_indices, pred, prob, top2 = infer_structured(model, store, test_states)
        if not np.array_equal(task_indices, test_indices):
            raise RuntimeError("StructuredCurrent test task order changed")
        records.append(
            _prediction_record(
                "StructuredCurrent", run, test_indices, pred, prob, top2, len(frame)
            )
        )
    return records


def quantile_boundaries(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=float), [1.0 / 3.0, 2.0 / 3.0])
    return float(low), float(high)


def bucket_names(values: np.ndarray, boundaries: tuple[float, float]) -> np.ndarray:
    low, high = boundaries
    return np.where(values <= low, "small", np.where(values <= high, "medium", "large"))


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    frame: pd.DataFrame,
    replay: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels_all = frame["h1_action_index"].to_numpy(dtype=np.int64)
    risk_all = frame["deployable_risk_score"].to_numpy(dtype=float)
    state_frame = replay["state_frame"]
    state_offsets = replay["state_offsets"]
    task_state = replay["task_state_index"]
    train_states = state_frame["split"].eq("train").to_numpy()
    queue_boundaries = quantile_boundaries(
        state_frame.loc[train_states, "queue_size"].to_numpy(dtype=float)
    )
    pressure_boundaries = quantile_boundaries(
        state_frame.loc[train_states, "resource_pressure"].to_numpy(dtype=float)
    )
    queue_bucket = bucket_names(
        state_frame["queue_size"].to_numpy(dtype=float), queue_boundaries
    )
    pressure_bucket = bucket_names(
        state_frame["resource_pressure"].to_numpy(dtype=float), pressure_boundaries
    )
    task_rows = []
    state_rows = []
    risk_rows = []
    queue_rows = []
    pressure_rows = []
    for record in records:
        indices = record["test_indices"]
        labels = labels_all[indices]
        predictions = record["predictions"]
        metrics = classification_metrics(
            labels,
            predictions,
            record["probabilities"],
            record["top2"],
            ACTION_DIM,
        )
        task_rows.append(
            {
                "model": record["model"],
                "seed": record["seed"],
                "samples": metrics["samples"],
                "overall_h1_accuracy": metrics["teacher_accuracy"],
                "top2_h1_accuracy": metrics["top2_accuracy"],
                "assignment_macro_f1": metrics["assignment_macro_f1"],
                "masked_cross_entropy": metrics["masked_cross_entropy"],
                "per_action_metrics_json": json.dumps(
                    per_action_metrics(labels, predictions, ACTION_DIM), sort_keys=True
                ),
                "best_epoch": record["run"]["best_epoch"],
                "best_val_loss": record["run"]["best_val_loss"],
                "checkpoint": record["run"]["checkpoint"],
                "selection_basis": "VALIDATION LOSS ONLY",
            }
        )
        full = record["full_predictions"]
        mismatches = []
        all_correct = []
        test_state_indices = state_frame.index[state_frame["split"].eq("test")]
        for state_index in test_state_indices:
            start, stop = state_offsets[state_index : state_index + 2]
            count = int((full[start:stop] != labels_all[start:stop]).sum())
            mismatches.append(count)
            all_correct.append(count == 0)
        state_rows.append(
            {
                "model": record["model"],
                "seed": record["seed"],
                "states": len(mismatches),
                "state_full_action_recovery": float(np.mean(all_correct)),
                "state_any_mismatch_rate": float(1.0 - np.mean(all_correct)),
                "mean_mismatched_tasks": float(np.mean(mismatches)),
                "median_mismatched_tasks": float(np.median(mismatches)),
                "max_mismatched_tasks": int(np.max(mismatches)),
            }
        )
        high = risk_all[indices] >= float(config["risk_threshold_p95"])
        for name, selector in (("LOW_RISK", ~high), ("HIGH_RISK", high)):
            risk_rows.append(
                {
                    "model": record["model"],
                    "seed": record["seed"],
                    "risk_group": name,
                    "risk_threshold": float(config["risk_threshold_p95"]),
                    "samples": int(selector.sum()),
                    "h1_accuracy": float((predictions[selector] == labels[selector]).mean()),
                    "mismatch_rate": float((predictions[selector] != labels[selector]).mean()),
                }
            )
        test_task_states = task_state[indices]
        for name in ("small", "medium", "large"):
            selector = queue_bucket[test_task_states] == name
            queue_rows.append(
                {
                    "model": record["model"],
                    "seed": record["seed"],
                    "queue_bucket": name,
                    "train_q33": queue_boundaries[0],
                    "train_q67": queue_boundaries[1],
                    "samples": int(selector.sum()),
                    "mean_queue_size": float(
                        state_frame["queue_size"].to_numpy()[test_task_states[selector]].mean()
                    ),
                    "h1_accuracy": float((predictions[selector] == labels[selector]).mean()),
                }
            )
            selector = pressure_bucket[test_task_states] == name
            pressure_rows.append(
                {
                    "model": record["model"],
                    "seed": record["seed"],
                    "pressure_bucket": name,
                    "train_q33": pressure_boundaries[0],
                    "train_q67": pressure_boundaries[1],
                    "samples": int(selector.sum()),
                    "mean_resource_pressure": float(
                        state_frame["resource_pressure"].to_numpy()[test_task_states[selector]].mean()
                    ),
                    "h1_accuracy": float((predictions[selector] == labels[selector]).mean()),
                }
            )
    return (
        pd.DataFrame(task_rows),
        pd.DataFrame(state_rows),
        pd.DataFrame(risk_rows),
        pd.DataFrame(queue_rows),
        pd.DataFrame(pressure_rows),
    )
