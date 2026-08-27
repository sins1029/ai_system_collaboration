from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from sustaincluster_imitation.bc_policy import (
    BCPolicy,
    BCPolicyConfig,
    weighted_masked_cross_entropy,
)
from sustaincluster_imitation.dataset_reader import ExpertDatasetReader


@dataclass(frozen=True)
class ClassificationMetrics:
    sample_count: int
    loss: float
    overall_accuracy: float
    top2_accuracy: float
    defer_precision: float
    defer_recall: float
    defer_f1: float
    assign_accuracy: float
    migration_precision: float
    migration_recall: float
    migration_f1: float
    expert_action_agreement: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]


class BehaviorCloningTrainer:
    def __init__(
        self,
        dataset_dir: Path,
        split_manifest_path: Path,
        config_path: Path,
        artifact_dir: Path,
    ) -> None:
        self.reader = ExpertDatasetReader(dataset_dir)
        self.split_manifest = json.loads(
            Path(split_manifest_path).read_text(encoding="utf-8")
        )
        with Path(config_path).open(encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)["behavior_cloning"]
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.rows = self.reader.read_tasks()
        self.action_dc_ids = tuple(int(x) for x in self.config["canonical_dc_ids"])
        self.action_dim = len(self.action_dc_ids) + 1
        self.feature_dim = len(self.rows[0]["feature_vector"])
        self._validate_rows()

    def train_all(self) -> dict[str, Any]:
        runs = [self._train_seed(int(seed)) for seed in self.config["seeds"]]
        best = min(runs, key=lambda item: item["validation"]["loss"])
        result = {
            "config": self.config,
            "dataset_manifest": self.reader.manifest,
            "feature_dim": self.feature_dim,
            "action_dim": self.action_dim,
            "model_parameter_count": best["parameter_count"],
            "best_checkpoint": best["checkpoint"],
            "runs": runs,
            "aggregate": self._aggregate_runs(runs),
        }
        (self.artifact_dir / "bc_training_results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result

    def _train_seed(self, seed: int) -> dict[str, Any]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device(self.config["device"])
        policy = BCPolicy(
            BCPolicyConfig(
                self.feature_dim,
                self.action_dc_ids,
                int(self.config["hidden_dim"]),
                bool(self.config["use_layer_norm"]),
            )
        ).to(device)
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        train = self._arrays("train")
        validation = self._arrays("validation")
        test = self._arrays("test")
        class_weights = self._class_weights(train[1], device)
        batch_size = int(self.config["batch_size"])
        history = []
        best_validation = float("inf")
        best_state = None
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        for epoch in range(int(self.config["epochs"])):
            order = rng.permutation(len(train[1]))
            policy.train()
            losses = []
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                features, labels, masks, _ = self._tensor_batch(
                    train, indices, device
                )
                logits = policy(features)
                task_mask = torch.ones_like(labels, dtype=torch.bool)
                loss = weighted_masked_cross_entropy(
                    logits,
                    labels,
                    task_mask,
                    masks,
                    class_weights,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    float(self.config["gradient_clip_norm"]),
                )
                optimizer.step()
                losses.append(float(loss.detach()))
            validation_metrics = self._evaluate(
                policy, validation, class_weights, device
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(losses)),
                    "validation_loss": validation_metrics.loss,
                    "validation_accuracy": validation_metrics.overall_accuracy,
                }
            )
            if validation_metrics.loss < best_validation:
                best_validation = validation_metrics.loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in policy.state_dict().items()
                }
        if best_state is None:
            raise RuntimeError("行为克隆未生成检查点")
        policy.load_state_dict(best_state)
        train_metrics = self._evaluate(policy, train, class_weights, device)
        validation_metrics = self._evaluate(
            policy, validation, class_weights, device
        )
        test_metrics = self._evaluate(policy, test, class_weights, device)
        checkpoint = self.artifact_dir / f"bc_actor_seed_{seed}.pt"
        policy.save_checkpoint(
            checkpoint,
            {
                "seed": seed,
                "validation_loss": validation_metrics.loss,
                "dataset_variant": "deployable_baseline_forecast",
            },
        )
        return {
            "seed": seed,
            "checkpoint": str(checkpoint.resolve()),
            "training_seconds": time.perf_counter() - started,
            "parameter_count": sum(p.numel() for p in policy.parameters()),
            "class_weights": class_weights.cpu().tolist(),
            "history": history,
            "train": asdict(train_metrics),
            "validation": asdict(validation_metrics),
            "test": asdict(test_metrics),
        }

    def _arrays(
        self, split: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        assignments = self.split_manifest["episode_assignments"]
        rows = [
            row for row in self.rows if assignments[row["episode_id"]] == split
        ]
        if not rows:
            raise ValueError(f"数据集划分 {split!r} 为空")
        return (
            np.asarray([row["feature_vector"] for row in rows], dtype=np.float32),
            np.asarray([row["semantic_label_index"] for row in rows], dtype=np.int64),
            np.asarray([row["feasible_action_mask"] for row in rows], dtype=bool),
            rows,
        )

    @staticmethod
    def _tensor_batch(
        arrays: tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]],
        indices: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        features, labels, masks, rows = arrays
        return (
            torch.as_tensor(features[indices], device=device),
            torch.as_tensor(labels[indices], device=device),
            torch.as_tensor(masks[indices], device=device),
            [rows[int(index)] for index in indices],
        )

    def _class_weights(
        self, labels: np.ndarray, device: torch.device
    ) -> torch.Tensor:
        counts = np.bincount(labels, minlength=self.action_dim).astype(float)
        inverse = len(labels) / np.maximum(counts, 1.0)
        inverse /= inverse.mean()
        inverse = np.clip(inverse, 0.1, float(self.config["class_weight_max"]))
        return torch.as_tensor(inverse, dtype=torch.float32, device=device)

    def _evaluate(
        self,
        policy: BCPolicy,
        arrays: tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]],
        class_weights: torch.Tensor,
        device: torch.device,
    ) -> ClassificationMetrics:
        features, labels, masks, rows = arrays
        policy.eval()
        predictions = []
        top2_correct = 0
        losses = []
        batch_size = int(self.config["batch_size"])
        with torch.no_grad():
            for start in range(0, len(labels), batch_size):
                stop = min(len(labels), start + batch_size)
                x = torch.as_tensor(features[start:stop], device=device)
                y = torch.as_tensor(labels[start:stop], device=device)
                feasible = torch.as_tensor(masks[start:stop], device=device)
                logits = policy(x)
                task_mask = torch.ones_like(y, dtype=torch.bool)
                loss = weighted_masked_cross_entropy(
                    logits, y, task_mask, feasible, class_weights
                )
                losses.append(float(loss))
                masked = logits.masked_fill(
                    ~feasible, torch.finfo(logits.dtype).min
                )
                predictions.extend(masked.argmax(dim=-1).cpu().tolist())
                top2 = masked.topk(min(2, self.action_dim), dim=-1).indices
                top2_correct += int((top2 == y.unsqueeze(1)).any(dim=1).sum())
        return self._metrics(
            labels,
            np.asarray(predictions, dtype=int),
            rows,
            float(np.mean(losses)),
            top2_correct / len(labels),
        )

    def _metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        rows: list[dict[str, Any]],
        loss: float,
        top2_accuracy: float,
    ) -> ClassificationMetrics:
        confusion = np.zeros((self.action_dim, self.action_dim), dtype=int)
        for expected, predicted in zip(labels, predictions):
            confusion[int(expected), int(predicted)] += 1
        per_class = {}
        names = ["defer"] + [f"dc_{dc_id}" for dc_id in self.action_dc_ids]
        for index, name in enumerate(names):
            precision, recall, f1 = self._prf(confusion, index)
            per_class[name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(confusion[index].sum()),
            }
        assign = labels != 0
        assign_accuracy = float(
            (predictions[assign] == labels[assign]).mean()
        ) if assign.any() else 0.0
        origin_labels = np.asarray(
            [
                self.action_dc_ids.index(int(row["origin_dc_id"])) + 1
                if int(row["origin_dc_id"]) in self.action_dc_ids
                else -1
                for row in rows
            ]
        )
        true_migration = (labels != 0) & (labels != origin_labels)
        predicted_migration = (predictions != 0) & (
            predictions != origin_labels
        )
        migration_precision, migration_recall, migration_f1 = self._binary_prf(
            true_migration, predicted_migration
        )
        return ClassificationMetrics(
            len(labels),
            loss,
            float((labels == predictions).mean()),
            top2_accuracy,
            per_class["defer"]["precision"],
            per_class["defer"]["recall"],
            per_class["defer"]["f1"],
            assign_accuracy,
            migration_precision,
            migration_recall,
            migration_f1,
            float((labels == predictions).mean()),
            per_class,
            confusion.tolist(),
        )

    @staticmethod
    def _prf(confusion: np.ndarray, index: int) -> tuple[float, float, float]:
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        return BehaviorCloningTrainer._prf_counts(
            true_positive, false_positive, false_negative
        )

    @staticmethod
    def _binary_prf(
        expected: np.ndarray, predicted: np.ndarray
    ) -> tuple[float, float, float]:
        true_positive = int((expected & predicted).sum())
        false_positive = int((~expected & predicted).sum())
        false_negative = int((expected & ~predicted).sum())
        return BehaviorCloningTrainer._prf_counts(
            true_positive, false_positive, false_negative
        )

    @staticmethod
    def _prf_counts(
        true_positive: int, false_positive: int, false_negative: int
    ) -> tuple[float, float, float]:
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        return precision, recall, f1

    @staticmethod
    def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
        fields = (
            "overall_accuracy",
            "top2_accuracy",
            "defer_f1",
            "assign_accuracy",
            "migration_f1",
            "loss",
        )
        return {
            split: {
                field: {
                    "mean": float(np.mean([run[split][field] for run in runs])),
                    "std": float(np.std([run[split][field] for run in runs])),
                }
                for field in fields
            }
            for split in ("train", "validation", "test")
        }

    def _validate_rows(self) -> None:
        if not self.rows:
            raise ValueError("expert task dataset 为空")
        for row in self.rows:
            if len(row["feature_vector"]) != self.feature_dim:
                raise ValueError("特征维度不一致")
            if len(row["feasible_action_mask"]) != self.action_dim:
                raise ValueError("动作掩码维度不一致")
            label = int(row["semantic_label_index"])
            if label < 0 or label >= self.action_dim:
                raise ValueError("语义标签超出范围")
            if not row["feasible_action_mask"][label]:
                raise ValueError("专家标签在数据集中不可行")
