from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from sustaincluster_imitation.bc_v2_offline import (
    masked_cross_entropy,
    masked_logits,
    set_deterministic_seed,
)


DC_IDS = (1, 2, 3, 4, 5)
ACTION_DIM = 6
INFORMATION_CLASS = "DEPLOYABLE_CURRENT_ONLY"


@dataclass(frozen=True)
class TrainOnlyStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    fitted_split: str = "train"

    @classmethod
    def fit(cls, values: np.ndarray, *, split: str) -> "TrainOnlyStandardizer":
        values = np.asarray(values, dtype=np.float64)
        if split != "train":
            raise ValueError("normalization may only be fit on the train split")
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("standardizer requires a finite non-empty matrix")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-12] = 1.0
        return cls(mean.astype(np.float64), scale.astype(np.float64))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError("standardizer input dimension mismatch")
        transformed = (values - self.mean) / self.scale
        if not np.isfinite(transformed).all():
            raise ValueError("standardization produced non-finite values")
        return transformed.astype(np.float32)

    def to_dict(self, feature_names: Sequence[str]) -> dict[str, Any]:
        if len(feature_names) != len(self.mean):
            raise ValueError("feature names do not match standardizer dimension")
        return {
            "method": "per-dimension z-score",
            "fitted_split": self.fitted_split,
            "feature_names": list(feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }


def task_feature_names() -> tuple[str, ...]:
    return (
        "cpu_cores",
        "gpu_units",
        "memory_gb",
        "estimated_duration_minutes",
        "remaining_duration_minutes",
        "remaining_sla_minutes",
        "bandwidth_gb",
        "wait_intervals",
        "was_deferred",
        "scheduler_wait_intervals",
        "decision_position",
        *(f"origin_dc_{dc_id}" for dc_id in DC_IDS),
    )


def running_feature_names() -> tuple[str, ...]:
    return (
        "cpu_cores",
        "gpu_units",
        "memory_gb",
        "estimated_release_step",
        *(f"dc_{dc_id}" for dc_id in DC_IDS),
    )


def dc_feature_names() -> tuple[str, ...]:
    return (
        *(f"identity_dc_{dc_id}" for dc_id in DC_IDS),
        "optimizer_order_position",
        "cpu_total_cores",
        "gpu_total_units",
        "memory_total_gb",
        "cpu_available_cores_raw",
        "gpu_available_units_raw",
        "memory_available_gb_raw",
        "cpu_reserved_cores",
        "gpu_reserved_units",
        "memory_reserved_gb",
        "h1_cpu_available_cores",
        "h1_gpu_available_units",
        "h1_memory_available_gb",
        "current_electricity_price_usd_per_mwh",
        "current_carbon_intensity_gco2_per_kwh",
        "running_task_count",
        "queued_task_count",
        "in_transit_task_count",
    )


def dense_feature_names() -> tuple[str, ...]:
    names = [f"focal_{name}" for name in task_feature_names()]
    for dc_id in DC_IDS:
        names.extend(
            (
                f"focal_to_dc{dc_id}_transmission_cost_usd",
                f"focal_to_dc{dc_id}_transmission_delay_seconds",
            )
        )
    dc_names = dc_feature_names()[5:]
    for dc_id in DC_IDS:
        names.extend(f"dc{dc_id}_{name}" for name in dc_names)
    names.append("pending_count")
    queue_variables = (
        "cpu_cores",
        "gpu_units",
        "memory_gb",
        "estimated_duration_minutes",
        "remaining_sla_minutes",
        "bandwidth_gb",
        "wait_intervals",
    )
    for variable in queue_variables:
        for statistic in ("sum", "mean", "max", "p50", "p90"):
            names.append(f"pending_{variable}_{statistic}")
    names.extend(f"pending_origin_dc{dc_id}_count" for dc_id in DC_IDS)
    names.extend(("pending_gpu_positive_count", "pending_deferred_count"))
    for dc_id in DC_IDS:
        names.extend(
            f"running_dc{dc_id}_{name}"
            for name in (
                "count",
                "cpu_sum",
                "gpu_sum",
                "memory_sum",
                "release_step_mean",
                "release_step_min",
                "release_step_max",
            )
        )
    for dc_id in DC_IDS:
        names.extend(
            f"transit_dc{dc_id}_{name}"
            for name in (
                "count",
                "cpu_sum",
                "gpu_sum",
                "memory_sum",
                "arrival_step_mean",
                "arrival_step_min",
                "arrival_step_max",
                "duration_step_mean",
            )
        )
    return tuple(names)


def task_feature_row(task: Any) -> np.ndarray:
    values = [
        float(task.cpu_cores),
        float(task.gpu_units),
        float(task.memory_gb),
        float(task.duration_minutes),
        float(task.remaining_duration_minutes),
        float(task.remaining_sla_minutes),
        float(task.bandwidth_gb),
        float(task.wait_intervals),
        float(task.was_deferred),
        float(task.scheduler_wait_intervals),
        float(task.original_index),
    ]
    values.extend(float(task.origin_dc_id == dc_id) for dc_id in DC_IDS)
    return np.asarray(values, dtype=np.float32)


def running_feature_matrix(state: Any) -> np.ndarray:
    rows = []
    for task in state.running_tasks:
        row = [
            float(task.cpu_cores),
            float(task.gpu_units),
            float(task.memory_gb),
            float(task.release_step),
        ]
        row.extend(float(task.dc_id == dc_id) for dc_id in DC_IDS)
        rows.append(row)
    if not rows:
        return np.empty((0, len(running_feature_names())), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def dc_feature_matrix(state: Any) -> np.ndarray:
    current = {int(dc.dc_id): dc for dc in state.current.datacenters}
    horizon = {int(dc.dc_id): dc for dc in state.datacenters}
    order = {int(dc.dc_id): index for index, dc in enumerate(state.datacenters)}
    if set(current) != set(DC_IDS) or set(horizon) != set(DC_IDS):
        raise ValueError("structured representation requires semantic DC1-DC5")
    rows = []
    for dc_id in DC_IDS:
        dc = current[dc_id]
        hdc = horizon[dc_id]
        row = [float(dc_id == candidate) for candidate in DC_IDS]
        row.extend(
            (
                float(order[dc_id]),
                float(dc.cpu_total_cores),
                float(dc.gpu_total_units),
                float(dc.memory_total_gb),
                float(dc.cpu_available_cores),
                float(dc.gpu_available_units),
                float(dc.memory_available_gb),
                float(dc.cpu_reserved_cores),
                float(dc.gpu_reserved_units),
                float(dc.memory_reserved_gb),
                float(hdc.cpu_available_cores[0]),
                float(hdc.gpu_available_units[0]),
                float(hdc.memory_available_gb[0]),
                float(hdc.electricity_price_usd_per_mwh[0]),
                float(hdc.carbon_intensity_gco2_per_kwh[0]),
                float(dc.running_task_count),
                float(dc.queued_task_count),
                float(dc.in_transit_task_count),
            )
        )
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def _summary(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [0.0] * 5
    return [
        float(array.sum()),
        float(array.mean()),
        float(array.max()),
        float(np.percentile(array, 50)),
        float(np.percentile(array, 90)),
    ]


def _queue_summary(tasks: Sequence[Any]) -> list[float]:
    values = [float(len(tasks))]
    accessors = (
        "cpu_cores",
        "gpu_units",
        "memory_gb",
        "duration_minutes",
        "remaining_sla_minutes",
        "bandwidth_gb",
        "wait_intervals",
    )
    for name in accessors:
        values.extend(_summary([float(getattr(task, name)) for task in tasks]))
    values.extend(float(sum(task.origin_dc_id == dc_id for task in tasks)) for dc_id in DC_IDS)
    values.append(float(sum(task.gpu_units > 0 for task in tasks)))
    values.append(float(sum(bool(task.was_deferred) for task in tasks)))
    return values


def _running_summary(state: Any) -> list[float]:
    values = []
    for dc_id in DC_IDS:
        tasks = [task for task in state.running_tasks if int(task.dc_id) == dc_id]
        releases = [float(task.release_step) for task in tasks]
        values.extend(
            (
                float(len(tasks)),
                float(sum(task.cpu_cores for task in tasks)),
                float(sum(task.gpu_units for task in tasks)),
                float(sum(task.memory_gb for task in tasks)),
                float(np.mean(releases)) if releases else 0.0,
                float(np.min(releases)) if releases else 0.0,
                float(np.max(releases)) if releases else 0.0,
            )
        )
    return values


def _transit_summary(state: Any) -> list[float]:
    values = []
    for dc_id in DC_IDS:
        tasks = [
            task
            for task in state.transit_tasks
            if int(task.destination_dc_id) == dc_id
        ]
        arrivals = [float(task.arrival_step) for task in tasks]
        durations = [float(task.duration_steps) for task in tasks]
        values.extend(
            (
                float(len(tasks)),
                float(sum(task.cpu_cores for task in tasks)),
                float(sum(task.gpu_units for task in tasks)),
                float(sum(task.memory_gb for task in tasks)),
                float(np.mean(arrivals)) if arrivals else 0.0,
                float(np.min(arrivals)) if arrivals else 0.0,
                float(np.max(arrivals)) if arrivals else 0.0,
                float(np.mean(durations)) if durations else 0.0,
            )
        )
    return values


@dataclass(frozen=True)
class RawCurrentStateRepresentation:
    pending: np.ndarray
    running: np.ndarray
    datacenters: np.ndarray
    dense: np.ndarray
    resource_pressure: float


def extract_current_state(state: Any) -> RawCurrentStateRepresentation:
    if int(state.horizon) != 1 or state.forecast_mode != "no_future_arrivals":
        raise ValueError("Current representations require deployable H1 state")
    if state.information_mode != "deployable" or state.future_arrivals:
        raise ValueError("Current representations must not contain Oracle/future arrivals")
    pending = np.stack([task_feature_row(task) for task in state.current.tasks]).astype(
        np.float32
    )
    running = running_feature_matrix(state)
    dcs = dc_feature_matrix(state)
    destinations = {
        (item.original_index, int(item.destination_dc_id)): item
        for item in state.current.task_destinations
    }
    global_values = []
    global_values.extend(dcs[:, 5:].reshape(-1).tolist())
    global_values.extend(_queue_summary(state.current.tasks))
    global_values.extend(_running_summary(state))
    global_values.extend(_transit_summary(state))
    dense_rows = []
    for task, task_values in zip(state.current.tasks, pending):
        destination_values = []
        for dc_id in DC_IDS:
            item = destinations[(task.original_index, dc_id)]
            destination_values.extend(
                (float(item.transmission_cost_usd), float(item.transmission_delay_seconds))
            )
        dense_rows.append(
            np.concatenate(
                (
                    task_values,
                    np.asarray(destination_values, dtype=np.float32),
                    np.asarray(global_values, dtype=np.float32),
                )
            )
        )
    dense = np.stack(dense_rows).astype(np.float32)
    if dense.shape[1] != len(dense_feature_names()):
        raise RuntimeError("CurrentDense feature declaration mismatch")
    pressure_values = []
    for row in dcs:
        totals = row[6:9]
        available = row[15:18]
        pressure_values.extend((1.0 - available / np.maximum(totals, 1e-12)).tolist())
    return RawCurrentStateRepresentation(
        pending=pending,
        running=running,
        datacenters=dcs,
        dense=dense,
        resource_pressure=float(np.max(pressure_values)),
    )


def canonical_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest().upper()


def label_entropy(labels: Sequence[int], action_dim: int = ACTION_DIM) -> float:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=action_dim)
    probabilities = counts[counts > 0] / max(1, counts.sum())
    return float(-(probabilities * np.log2(probabilities)).sum())


class StructuredCurrentPolicy(nn.Module):
    def __init__(
        self,
        task_dim: int,
        running_dim: int,
        dc_dim: int,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.task_dim = int(task_dim)
        self.running_dim = int(running_dim)
        self.dc_dim = int(dc_dim)
        self.embedding_dim = int(embedding_dim)
        self.focal_encoder = self._encoder(task_dim, embedding_dim)
        self.pending_phi = self._encoder(task_dim, embedding_dim)
        self.pending_rho = self._encoder(embedding_dim, embedding_dim)
        self.running_phi = self._encoder(running_dim, embedding_dim)
        self.running_rho = self._encoder(embedding_dim, embedding_dim)
        self.dc_encoder = self._encoder(dc_dim, embedding_dim)
        self.dc_rho = self._encoder(embedding_dim, embedding_dim)
        base_dim = embedding_dim * 4
        self.defer_head = nn.Sequential(
            nn.Linear(base_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, 1)
        )
        self.assignment_head = nn.Sequential(
            nn.Linear(base_dim + embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    @staticmethod
    def _encoder(input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(int(input_dim), int(output_dim)),
            nn.LayerNorm(int(output_dim)),
            nn.ReLU(),
        )

    def forward_state(
        self,
        pending: torch.Tensor,
        running: torch.Tensor,
        datacenters: torch.Tensor,
        dc_ids: torch.Tensor,
    ) -> torch.Tensor:
        if pending.ndim != 2 or pending.shape[1] != self.task_dim or not len(pending):
            raise ValueError("pending set has the wrong shape")
        if running.ndim != 2 or running.shape[1] != self.running_dim:
            raise ValueError("running set has the wrong shape")
        if datacenters.shape != (len(DC_IDS), self.dc_dim):
            raise ValueError("DC set must contain exactly five feature rows")
        if dc_ids.shape != (len(DC_IDS),) or set(dc_ids.detach().cpu().tolist()) != set(DC_IDS):
            raise ValueError("DC identity must contain semantic DC1-DC5")
        dc_order = torch.argsort(dc_ids)
        datacenters = datacenters[dc_order]
        focal = self.focal_encoder(pending)
        pending_phi = self.pending_phi(pending)
        pending_excluding_focal = pending_phi.sum(dim=0, keepdim=True) - pending_phi
        pending_context = self.pending_rho(pending_excluding_focal)
        if len(running):
            running_sum = self.running_phi(running).sum(dim=0, keepdim=True)
        else:
            running_sum = torch.zeros(
                (1, self.embedding_dim), device=pending.device, dtype=pending.dtype
            )
        running_context = self.running_rho(running_sum).expand(len(pending), -1)
        dc_embeddings = self.dc_encoder(datacenters)
        dc_global = self.dc_rho(dc_embeddings.sum(dim=0, keepdim=True)).expand(
            len(pending), -1
        )
        base = torch.cat((focal, pending_context, running_context, dc_global), dim=-1)
        defer = self.defer_head(base)
        task_dc = torch.cat(
            (
                base[:, None, :].expand(-1, len(DC_IDS), -1),
                dc_embeddings[None, :, :].expand(len(pending), -1, -1),
            ),
            dim=-1,
        )
        assignments = self.assignment_head(task_dc).squeeze(-1)
        return torch.cat((defer, assignments), dim=-1)


@dataclass(frozen=True)
class StructuredArrayStore:
    pending_values: np.ndarray
    state_offsets: np.ndarray
    running_values: np.ndarray
    running_offsets: np.ndarray
    dc_values: np.ndarray
    labels: np.ndarray
    masks: np.ndarray
    state_ids: tuple[str, ...]
    splits: tuple[str, ...]

    def __post_init__(self) -> None:
        state_count = len(self.state_ids)
        if len(self.state_offsets) != state_count + 1:
            raise ValueError("pending state offsets are invalid")
        if len(self.running_offsets) != state_count + 1:
            raise ValueError("running state offsets are invalid")
        if self.dc_values.shape[:2] != (state_count, len(DC_IDS)):
            raise ValueError("DC state array is invalid")
        if int(self.state_offsets[-1]) != len(self.labels):
            raise ValueError("task labels do not align with state offsets")
        if self.masks.shape != (len(self.labels), ACTION_DIM):
            raise ValueError("feasible masks do not align with labels")
        if not self.masks[np.arange(len(self.labels)), self.labels].all():
            raise ValueError("H1 labels must be feasible")

    def state_indices(self, split: str) -> np.ndarray:
        return np.asarray(
            [index for index, value in enumerate(self.splits) if value == split],
            dtype=np.int64,
        )

    def state_arrays(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        task_start, task_stop = self.state_offsets[index : index + 2]
        run_start, run_stop = self.running_offsets[index : index + 2]
        return (
            self.pending_values[task_start:task_stop],
            self.running_values[run_start:run_stop],
            self.dc_values[index],
            self.labels[task_start:task_stop],
            self.masks[task_start:task_stop],
        )


def structured_checkpoint_payload(
    model: StructuredCurrentPolicy,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_type": "StructuredCurrentPolicy_DeepSets",
        "model_state_dict": model.state_dict(),
        "task_dim": model.task_dim,
        "running_dim": model.running_dim,
        "dc_dim": model.dc_dim,
        "embedding_dim": model.embedding_dim,
        "action_dim": ACTION_DIM,
        **dict(metadata),
    }


def load_structured_checkpoint(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[StructuredCurrentPolicy, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = StructuredCurrentPolicy(
        payload["task_dim"],
        payload["running_dim"],
        payload["dc_dim"],
        payload["embedding_dim"],
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


@torch.no_grad()
def evaluate_structured_loss(
    model: StructuredCurrentPolicy,
    store: StructuredArrayStore,
    state_indices: Sequence[int],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    task_count = 0
    dc_ids = torch.tensor(DC_IDS, dtype=torch.long, device=device)
    for index in state_indices:
        pending, running, dcs, labels, masks = store.state_arrays(int(index))
        logits = model.forward_state(
            torch.from_numpy(pending).to(device),
            torch.from_numpy(running).to(device),
            torch.from_numpy(dcs).to(device),
            dc_ids,
        )
        y = torch.from_numpy(labels).to(device)
        mask = torch.from_numpy(masks).to(device)
        loss_sum += float(masked_cross_entropy(logits, y, mask, reduction="sum"))
        correct += int((masked_logits(logits, mask).argmax(dim=-1) == y).sum())
        task_count += len(labels)
    return {"loss": loss_sum / task_count, "accuracy": correct / task_count}


def train_structured_seed(
    *,
    seed: int,
    config: Mapping[str, Any],
    store: StructuredArrayStore,
    checkpoint_path: Path,
    history_path: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    set_deterministic_seed(seed)
    device = torch.device(str(config["device"]))
    model = StructuredCurrentPolicy(
        store.pending_values.shape[1],
        store.running_values.shape[1],
        store.dc_values.shape[2],
        int(config["structured_embedding_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_states = store.state_indices("train")
    validation_states = store.state_indices("validation")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improvement = 0
    history = []
    batch_size = int(config["state_batch_size"])
    dc_ids = torch.tensor(DC_IDS, dtype=torch.long, device=device)
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_states), generator=generator).numpy()
        train_loss_sum = 0.0
        train_correct = 0
        train_tasks = 0
        for start in range(0, len(order), batch_size):
            logits_parts = []
            label_parts = []
            mask_parts = []
            for local_index in order[start : start + batch_size]:
                state_index = int(train_states[int(local_index)])
                pending, running, dcs, labels, masks = store.state_arrays(state_index)
                logits_parts.append(
                    model.forward_state(
                        torch.from_numpy(pending).to(device),
                        torch.from_numpy(running).to(device),
                        torch.from_numpy(dcs).to(device),
                        dc_ids,
                    )
                )
                label_parts.append(torch.from_numpy(labels).to(device))
                mask_parts.append(torch.from_numpy(masks).to(device))
            logits = torch.cat(logits_parts)
            labels = torch.cat(label_parts)
            masks = torch.cat(mask_parts)
            loss = masked_cross_entropy(logits, labels, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ):
                raise RuntimeError("non-finite gradient in structured training")
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            optimizer.step()
            count = len(labels)
            train_loss_sum += float(loss.detach()) * count
            train_correct += int(
                (masked_logits(logits.detach(), masks).argmax(dim=-1) == labels).sum()
            )
            train_tasks += count
        validation = evaluate_structured_loss(model, store, validation_states, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_tasks,
            "val_loss": validation["loss"],
            "train_accuracy": train_correct / train_tasks,
            "val_accuracy": validation["accuracy"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"structured seed={seed} epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_acc={row['val_accuracy']:.6f}",
            flush=True,
        )
        if row["val_loss"] < best_loss - float(config["early_stopping_min_delta"]):
            best_loss = float(row["val_loss"])
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
        raise RuntimeError("structured training produced no validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = structured_checkpoint_payload(
        model,
        {
            **dict(metadata),
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "selection_basis": "VALIDATION MASKED CROSS-ENTROPY LOSS ONLY",
            "training_config": {
                "optimizer": config["optimizer"],
                "learning_rate": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                "state_batch_size": config["state_batch_size"],
                "max_epochs": config["max_epochs"],
                "early_stopping_patience": config["early_stopping_patience"],
                "gradient_clip": config["gradient_clip"],
                "class_weighting": False,
                "risk_weighting": False,
                "disagreement_weighting": False,
                "focal_loss": False,
            },
        },
    )
    torch.save(payload, checkpoint_path)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "epochs_completed": len(history),
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }


@torch.no_grad()
def infer_structured(
    model: StructuredCurrentPolicy,
    store: StructuredArrayStore,
    state_indices: Sequence[int],
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device(device)
    model.eval()
    predictions = []
    probabilities = []
    top2 = []
    task_indices = []
    dc_ids = torch.tensor(DC_IDS, dtype=torch.long, device=device)
    for state_index in state_indices:
        index = int(state_index)
        pending, running, dcs, _, masks = store.state_arrays(index)
        logits = model.forward_state(
            torch.from_numpy(pending).to(device),
            torch.from_numpy(running).to(device),
            torch.from_numpy(dcs).to(device),
            dc_ids,
        )
        mask = torch.from_numpy(masks).to(device)
        logits = masked_logits(logits, mask)
        probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
        predictions.append(logits.argmax(dim=-1).cpu().numpy())
        top2.append(logits.topk(2, dim=-1).indices.cpu().numpy())
        start, stop = store.state_offsets[index : index + 2]
        task_indices.append(np.arange(start, stop, dtype=np.int64))
    return (
        np.concatenate(task_indices),
        np.concatenate(predictions).astype(np.int64),
        np.concatenate(probabilities).astype(np.float32),
        np.concatenate(top2).astype(np.int64),
    )


def model_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def dataframe_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def json_histogram(labels: Sequence[int]) -> str:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=ACTION_DIM)
    return json.dumps({str(index): int(value) for index, value in enumerate(counts)})


def finite_or_nan(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")
