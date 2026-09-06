from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml

from sustaincluster_imitation.bc_policy import build_sustaincluster_actor


DATASET_NAME = "REPAIRED MPC EXPERT DATASET v2"
EXPECTED_DATASET_VERSION = "repaired_mpc_expert_dataset_v2"
TRAINING_COLUMNS = (
    "sample_id",
    "student_observation",
    "feasible_action_mask",
    "teacher_action_index",
)
PRIVILEGED_OR_DIAGNOSTIC_COLUMNS = frozenset(
    {
        "h1_action_index",
        "deployable_risk_score",
        "teacher_objective",
        "teacher_information_class",
        "scenario",
        "seed",
        "true_duration",
        "future_workload",
        "future_price",
        "future_carbon",
    }
)
PREDICTION_COLUMNS = (
    "sample_id",
    "episode_id",
    "seed",
    "scenario",
    "step",
    "task_id",
    "teacher_action",
    "h1_action",
    "bc_action",
    "teacher_h1_disagreement",
    "deployable_risk_score",
    "high_risk",
    "feasible_action_mask",
    "prob_defer",
    "prob_dc1",
    "prob_dc2",
    "prob_dc3",
    "prob_dc4",
    "prob_dc5",
    "predicted_probability",
    "teacher_action_probability",
    "h1_action_probability",
    "correct_teacher",
    "correct_h1",
)


@dataclass(frozen=True)
class BCV2Arrays:
    observations: np.ndarray
    labels: np.ndarray
    masks: np.ndarray
    sample_ids: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return dict(yaml.safe_load(stream)["bc_v2_offline"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_frozen_contract(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / str(config["dataset_manifest"])
    feature_path = root / str(config["feature_schema"])
    action_path = root / str(config["action_schema"])
    dataset_path = root / str(config["dataset_dir"]) / "expert_task_actions_full.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_schema = json.loads(feature_path.read_text(encoding="utf-8"))
    action_schema = json.loads(action_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(dataset_path)
    expected_hash = str(config["expected_dataset_sha256"]).upper()
    checks = {
        "dataset_name": DATASET_NAME,
        "dataset_version": manifest.get("dataset_version"),
        "dataset_sha256": actual_hash,
        "obs_dim": int(feature_schema["student_observation"]["shape"][0]),
        "action_dim": int(action_schema["action_dim"]),
        "true_duration_present": bool(
            feature_schema["student_observation"].get("true_duration_present", True)
        ),
        "student_information_mode": manifest.get("student_information_mode"),
        "teacher": manifest.get("teacher"),
    }
    if checks["dataset_version"] != EXPECTED_DATASET_VERSION:
        raise ValueError("BC v2 received the wrong dataset version")
    if actual_hash != expected_hash or actual_hash != manifest["primary_dataset_sha256"]:
        raise ValueError("Expert Dataset v2 SHA256 mismatch")
    if checks["obs_dim"] != int(config["obs_dim"]):
        raise ValueError("Student observation dimension mismatch")
    if checks["action_dim"] != int(config["action_dim"]):
        raise ValueError("Semantic action dimension mismatch")
    if checks["true_duration_present"]:
        raise ValueError("true_duration is present in Student features")
    if manifest.get("student_future_leakage_detected"):
        raise ValueError("Dataset manifest reports Student future leakage")
    return checks


def training_columns_are_deployable_only(
    columns: Sequence[str] = TRAINING_COLUMNS,
) -> bool:
    return not bool(set(columns) & PRIVILEGED_OR_DIAGNOSTIC_COLUMNS)


def load_training_arrays(path: Path, obs_dim: int, action_dim: int) -> BCV2Arrays:
    if not training_columns_are_deployable_only():
        raise RuntimeError("BC v2 training schema includes diagnostic metadata")
    table = pq.read_table(path, columns=list(TRAINING_COLUMNS))
    frame = table.to_pandas()
    observations = np.asarray(frame["student_observation"].tolist(), dtype=np.float32)
    labels = frame["teacher_action_index"].to_numpy(dtype=np.int64, copy=True)
    masks = np.asarray(frame["feasible_action_mask"].tolist(), dtype=bool)
    if observations.shape != (len(frame), obs_dim):
        raise ValueError("BC v2 observation matrix has the wrong shape")
    if masks.shape != (len(frame), action_dim):
        raise ValueError("BC v2 feasible mask matrix has the wrong shape")
    if not np.isfinite(observations).all():
        raise ValueError("BC v2 observations contain NaN or infinity")
    if np.any(labels < 0) or np.any(labels >= action_dim):
        raise ValueError("BC v2 labels are outside the action space")
    if not masks[np.arange(len(labels)), labels].all():
        raise ValueError("BC v2 contains an infeasible Teacher label")
    if not masks.any(axis=1).all():
        raise ValueError("BC v2 contains a row with no feasible action")
    return BCV2Arrays(
        observations,
        labels,
        masks,
        tuple(str(value) for value in frame["sample_id"]),
    )


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def masked_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if logits.shape != masks.shape:
        raise ValueError("feasible-action mask shape does not match logits")
    masks = masks.bool()
    if not bool(masks.any(dim=-1).all()):
        raise ValueError("every task must have at least one feasible action")
    return logits.masked_fill(~masks, torch.finfo(logits.dtype).min)


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    if labels.shape != logits.shape[:-1]:
        raise ValueError("label shape does not match logits")
    feasible = masks.bool().gather(1, labels.long().unsqueeze(1)).squeeze(1)
    if not bool(feasible.all()):
        raise ValueError("Teacher label is infeasible")
    return F.cross_entropy(
        masked_logits(logits, masks), labels.long(), reduction=reduction
    )


def build_actor(config: Mapping[str, Any]) -> torch.nn.Module:
    return build_sustaincluster_actor(
        int(config["obs_dim"]),
        int(config["action_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        use_layer_norm=bool(config["use_layer_norm"]),
    )


def actor_architecture(model: torch.nn.Module, config: Mapping[str, Any]) -> dict[str, Any]:
    layers = [
        {
            "index": index,
            "type": module.__class__.__name__,
            **(
                {"in_features": module.in_features, "out_features": module.out_features}
                if isinstance(module, torch.nn.Linear)
                else {}
            ),
            **(
                {"normalized_shape": list(module.normalized_shape)}
                if isinstance(module, torch.nn.LayerNorm)
                else {}
            ),
        }
        for index, module in enumerate(model.modules())
        if module is not model and not isinstance(module, torch.nn.Sequential)
    ]
    return {
        "class": model.__class__.__name__,
        "input_dim": int(config["obs_dim"]),
        "hidden_dims": [int(config["hidden_dim"]), int(config["hidden_dim"])],
        "normalization": "LayerNorm" if config["use_layer_norm"] else "NONE",
        "activation": "ReLU",
        "output_dim": int(config["action_dim"]),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "layers": layers,
    }


@torch.no_grad()
def evaluate_training_arrays(
    model: torch.nn.Module,
    arrays: BCV2Arrays,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    for start in range(0, len(arrays), batch_size):
        stop = min(len(arrays), start + batch_size)
        x = torch.from_numpy(arrays.observations[start:stop]).to(device)
        y = torch.from_numpy(arrays.labels[start:stop]).to(device)
        masks = torch.from_numpy(arrays.masks[start:stop]).to(device)
        logits = model(x)
        loss_sum += float(masked_cross_entropy(logits, y, masks, reduction="sum"))
        correct += int((masked_logits(logits, masks).argmax(dim=-1) == y).sum())
    return {"loss": loss_sum / len(arrays), "accuracy": correct / len(arrays)}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "obs_dim": int(config["obs_dim"]),
        "action_dim": int(config["action_dim"]),
        "actor_config": {
            "class": "ActorNet",
            "hidden_dim": int(config["hidden_dim"]),
            "use_layer_norm": bool(config["use_layer_norm"]),
            "activation": "ReLU",
        },
        **dict(metadata),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path, device: torch.device | str = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    actor_config = payload["actor_config"]
    config = {
        "obs_dim": int(payload["obs_dim"]),
        "action_dim": int(payload["action_dim"]),
        "hidden_dim": int(actor_config["hidden_dim"]),
        "use_layer_norm": bool(actor_config["use_layer_norm"]),
    }
    model = build_actor(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def train_seed(
    *,
    seed: int,
    config: Mapping[str, Any],
    train: BCV2Arrays,
    validation: BCV2Arrays,
    checkpoint_path: Path,
    history_path: Path,
    checkpoint_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    set_deterministic_seed(seed)
    device = torch.device(str(config["device"]))
    model = build_actor(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    x_train = torch.from_numpy(train.observations).to(device)
    y_train = torch.from_numpy(train.labels).to(device)
    m_train = torch.from_numpy(train.masks).to(device)
    batch_size = int(config["batch_size"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        train_loss_sum = 0.0
        train_correct = 0
        for start in range(0, len(train), batch_size):
            indices = order[start : start + batch_size].to(device)
            x = x_train[indices]
            y = y_train[indices]
            masks = m_train[indices]
            logits = model(x)
            loss = masked_cross_entropy(logits, y, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ):
                raise RuntimeError("non-finite gradient in BC v2 training")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip"])
            )
            optimizer.step()
            count = int(indices.numel())
            train_loss_sum += float(loss.detach()) * count
            train_correct += int(
                (masked_logits(logits.detach(), masks).argmax(dim=-1) == y).sum()
            )
        validation_metrics = evaluate_training_arrays(
            model, validation, int(config["inference_batch_size"]), device
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / len(train),
            "val_loss": validation_metrics["loss"],
            "train_accuracy": train_correct / len(train),
            "val_accuracy": validation_metrics["accuracy"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"seed={seed} epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_acc={row['val_accuracy']:.6f}",
            flush=True,
        )
        if row["val_loss"] < best_val_loss - float(config["early_stopping_min_delta"]):
            best_val_loss = float(row["val_loss"])
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= int(config["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("BC v2 did not produce a validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    metadata = {
        **dict(checkpoint_metadata),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_config": {
            key: config[key]
            for key in (
                "optimizer",
                "learning_rate",
                "weight_decay",
                "batch_size",
                "max_epochs",
                "early_stopping_patience",
                "early_stopping_min_delta",
                "gradient_clip",
                "deterministic",
                "class_weighting",
                "disagreement_weighting",
            )
        },
        "initialization": "RANDOM",
        "old_bc_weights_loaded": False,
        "selection_basis": "VALIDATION MASKED CROSS-ENTROPY LOSS ONLY",
    }
    save_checkpoint(checkpoint_path, model, config, metadata)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_completed": len(history),
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }


@torch.no_grad()
def infer(
    model: torch.nn.Module,
    arrays: BCV2Arrays,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    top2: list[np.ndarray] = []
    device = torch.device(device)
    for start in range(0, len(arrays), batch_size):
        stop = min(len(arrays), start + batch_size)
        x = torch.from_numpy(arrays.observations[start:stop]).to(device)
        masks = torch.from_numpy(arrays.masks[start:stop]).to(device)
        logits = masked_logits(model(x), masks)
        probs = torch.softmax(logits, dim=-1)
        predictions.append(logits.argmax(dim=-1).cpu().numpy())
        probabilities.append(probs.cpu().numpy())
        top2.append(logits.topk(2, dim=-1).indices.cpu().numpy())
    return (
        np.concatenate(predictions).astype(np.int64),
        np.concatenate(probabilities).astype(np.float32),
        np.concatenate(top2).astype(np.int64),
    )


def per_action_metrics(
    labels: np.ndarray, predictions: np.ndarray, action_dim: int
) -> list[dict[str, Any]]:
    names = ["defer"] + [f"assign_dc{index}" for index in range(1, action_dim)]
    rows: list[dict[str, Any]] = []
    for action, name in enumerate(names):
        expected = labels == action
        predicted = predictions == action
        tp = int((expected & predicted).sum())
        fp = int((~expected & predicted).sum())
        fn = int((expected & ~predicted).sum())
        support = int(expected.sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / support if support else float("nan")
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if support and np.isfinite(precision) and precision + recall
            else float("nan")
        )
        rows.append(
            {
                "action_index": action,
                "action": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "support_status": "OK" if support else "NO_TEACHER_SUPPORT",
            }
        )
    return rows


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    top2: np.ndarray,
    action_dim: int,
) -> dict[str, float | int]:
    teacher_probability = probabilities[np.arange(len(labels)), labels]
    action_rows = per_action_metrics(labels, predictions, action_dim)
    assignment_f1 = [row["f1"] for row in action_rows[1:] if np.isfinite(row["f1"])]
    return {
        "samples": len(labels),
        "masked_cross_entropy": float(-np.log(np.maximum(teacher_probability, 1e-30)).mean()),
        "teacher_accuracy": float((predictions == labels).mean()),
        "top2_accuracy": float((top2 == labels[:, None]).any(axis=1).mean()),
        "assignment_accuracy": float((predictions[labels != 0] == labels[labels != 0]).mean()),
        "assignment_macro_f1": float(np.mean(assignment_f1)),
    }


def recovery_metrics(
    teacher: np.ndarray,
    h1: np.ndarray,
    predictions: np.ndarray,
    selector: np.ndarray | None = None,
) -> dict[str, float | int]:
    if selector is None:
        selector = np.ones(len(teacher), dtype=bool)
    selector = np.asarray(selector, dtype=bool)
    disagreement = selector & (teacher != h1)
    count = int(disagreement.sum())
    recovered = int((disagreement & (predictions == teacher)).sum())
    fallback = int((disagreement & (predictions == h1)).sum())
    other = count - recovered - fallback
    return {
        "samples": int(selector.sum()),
        "teacher_accuracy": float((predictions[selector] == teacher[selector]).mean())
        if selector.any()
        else float("nan"),
        "disagreement_samples": count,
        "teacher_recovery_rate": recovered / count if count else float("nan"),
        "h1_fallback_rate": fallback / count if count else float("nan"),
        "other_rate": other / count if count else float("nan"),
        "recovered_count": recovered,
        "h1_fallback_count": fallback,
        "other_count": other,
    }


def three_way_categories(
    teacher: np.ndarray, h1: np.ndarray, predictions: np.ndarray
) -> np.ndarray:
    consensus = teacher == h1
    result = np.full(len(teacher), "E_TEACHER_NE_H1_BC_OTHER", dtype=object)
    result[consensus & (predictions == teacher)] = "A_TEACHER_EQ_H1_EQ_BC"
    result[consensus & (predictions != teacher)] = "B_TEACHER_EQ_H1_BC_DIFFERS"
    result[~consensus & (predictions == teacher)] = "C_TEACHER_NE_H1_BC_EQ_TEACHER"
    result[~consensus & (predictions == h1)] = "D_TEACHER_NE_H1_BC_EQ_H1"
    return result.astype(str)


def state_level_recovery(frame: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    working = frame[["episode_id", "step", "teacher_action_index", "h1_action_index"]].copy()
    working["bc_action"] = predictions
    working["disagreement"] = (
        working["teacher_action_index"] != working["h1_action_index"]
    )
    working["recovered"] = working["disagreement"] & (
        working["bc_action"] == working["teacher_action_index"]
    )
    rows = []
    for (episode_id, step), group in working.groupby(["episode_id", "step"], sort=False):
        disagreements = int(group["disagreement"].sum())
        if not disagreements:
            continue
        recovered = int(group["recovered"].sum())
        rows.append(
            {
                "episode_id": episode_id,
                "step": int(step),
                "num_disagreement_tasks": disagreements,
                "num_recovered_tasks": recovered,
                "any_recovered": recovered > 0,
                "all_recovered": recovered == disagreements,
            }
        )
    return pd.DataFrame(rows)


def build_prediction_frame(
    evaluation: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    risk_threshold: float,
) -> pd.DataFrame:
    teacher = evaluation["teacher_action_index"].to_numpy(dtype=np.int64)
    h1 = evaluation["h1_action_index"].to_numpy(dtype=np.int64)
    result = evaluation[
        ["sample_id", "episode_id", "seed", "scenario", "step", "task_id", "deployable_risk_score", "feasible_action_mask"]
    ].copy()
    result["teacher_action"] = teacher
    result["h1_action"] = h1
    result["bc_action"] = predictions
    result["teacher_h1_disagreement"] = teacher != h1
    result["high_risk"] = result["deployable_risk_score"] >= risk_threshold
    for index, name in enumerate(("defer", "dc1", "dc2", "dc3", "dc4", "dc5")):
        result[f"prob_{name}"] = probabilities[:, index]
    result["predicted_probability"] = probabilities[np.arange(len(result)), predictions]
    result["teacher_action_probability"] = probabilities[np.arange(len(result)), teacher]
    result["h1_action_probability"] = probabilities[np.arange(len(result)), h1]
    result["correct_teacher"] = predictions == teacher
    result["correct_h1"] = predictions == h1
    return result[list(PREDICTION_COLUMNS)]


def choose_recommended_run(runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not runs:
        raise ValueError("no BC v2 runs available for selection")
    return min(runs, key=lambda run: (float(run["best_val_loss"]), int(run["seed"])))


def aggregate(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}
