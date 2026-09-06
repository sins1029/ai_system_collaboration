from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from forecasting.loader import LoadedForecastDataset, load_forecast_dataset
from forecasting.transformer_forecaster import (
    TransformerForecastConfig,
    TransformerForecaster,
)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_epochs", "patience"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "gradient_clip_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.weight_decay)) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


@dataclass(frozen=True)
class TrainingResult:
    seed: int
    best_epoch: int
    best_val_loss: float
    stopped_epoch: int
    checkpoint: str
    history: pd.DataFrame
    stable: bool


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def load_training_splits(
    dataset_root: str | Path,
    *,
    history_length: int = 96,
    horizon: int = 4,
    loader: Callable[..., LoadedForecastDataset] = load_forecast_dataset,
) -> tuple[LoadedForecastDataset, LoadedForecastDataset]:
    """Training boundary: this function never requests the test split."""
    train = loader(
        "train",
        history_length,
        horizon,
        True,
        dataset_root=dataset_root,
    )
    validation = loader(
        "val",
        history_length,
        horizon,
        True,
        dataset_root=dataset_root,
    )
    return train, validation


def _data_loader(
    data: LoadedForecastDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(data.X, dtype=np.float32)),
        torch.from_numpy(np.asarray(data.Y, dtype=np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _mean_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    samples = 0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            loss = criterion(model(features), targets)
            batch = int(features.shape[0])
            total += float(loss.item()) * batch
            samples += batch
    return total / samples


def save_checkpoint(
    path: str | Path,
    model: TransformerForecaster,
    *,
    metadata: dict[str, Any],
) -> None:
    payload = {
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "model_config": model.config.to_dict(),
        **metadata,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[TransformerForecaster, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = TransformerForecaster(
        TransformerForecastConfig(**payload["model_config"])
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def train_one_seed(
    *,
    seed: int,
    model_config: TransformerForecastConfig,
    training_config: TrainingConfig,
    train_data: LoadedForecastDataset,
    validation_data: LoadedForecastDataset,
    checkpoint_path: str | Path,
    device: torch.device,
    checkpoint_metadata: dict[str, Any],
) -> TrainingResult:
    set_global_seed(seed)
    model = TransformerForecaster(model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    train_loader = _data_loader(
        train_data,
        batch_size=training_config.batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _data_loader(
        validation_data,
        batch_size=training_config.batch_size,
        shuffle=False,
        seed=seed,
    )

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_epoch = training_config.max_epochs
    for epoch in range(1, training_config.max_epochs + 1):
        model.train()
        total = 0.0
        samples = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at seed={seed}, epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config.gradient_clip_norm
            )
            optimizer.step()
            batch = int(features.shape[0])
            total += float(loss.item()) * batch
            samples += batch
        train_loss = total / samples
        val_loss = _mean_loss(model, validation_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                metadata={
                    **checkpoint_metadata,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "seed": seed,
                    "training_config": asdict(training_config),
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.patience:
                stopped_epoch = epoch
                break

    stable = math.isfinite(best_val_loss) and best_epoch > 0
    if not stable:
        raise RuntimeError(f"seed {seed} did not produce a finite validation checkpoint")
    return TrainingResult(
        seed=seed,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        stopped_epoch=stopped_epoch,
        checkpoint=str(Path(checkpoint_path)),
        history=pd.DataFrame(history),
        stable=stable,
    )


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
