from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from forecasting.h60_dataset import H60_TARGET_NAMES, H60WindowDataset
from forecasting.transformer_forecaster import (
    TransformerForecastConfig,
    TransformerForecaster,
)


@dataclass(frozen=True)
class H60TrainingConfig:
    batch_size: int = 256
    max_epochs: int = 80
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    min_delta: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_epochs", "patience"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "gradient_clip_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "min_delta"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class H60TrainingResult:
    model_type: str
    seed: int
    best_epoch: int
    stopped_epoch: int
    best_validation_loss: float
    training_seconds: float
    checkpoint_path: str
    history: tuple[dict[str, float | int], ...]


class H60MLP(nn.Module):
    """The same compact MLP architecture used by the Night 1 preflight."""

    def __init__(self, input_dim: int, history_length: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_length = int(history_length)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim * self.history_length, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, len(H60_TARGET_NAMES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).unsqueeze(1)

    def config_dict(self) -> dict[str, int]:
        return {
            "input_dim": self.input_dim,
            "history_length": self.history_length,
            "hidden_dim": self.hidden_dim,
        }


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _loader(
    window: H60WindowDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(window.X, dtype=np.float32)),
            torch.from_numpy(np.asarray(window.Y, dtype=np.float32)),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def predict_h60(
    model: nn.Module,
    features: np.ndarray,
    *,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or torch.device("cpu")
    values = torch.from_numpy(np.asarray(features, dtype=np.float32))
    loader = DataLoader(TensorDataset(values), batch_size=batch_size, shuffle=False)
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            predictions.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def denormalize_h60(
    values: np.ndarray,
    scaler: Mapping[str, Any],
    *,
    clip_nonnegative: bool = True,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for index, name in enumerate(H60_TARGET_NAMES):
        parameter = scaler["parameters"][name]
        result[:, :, index] = (
            result[:, :, index] * float(parameter["scale"])
            + float(parameter["mean"])
        )
    return np.maximum(result, 0.0) if clip_nonnegative else result


def h60_metric_summary(
    truth_normalized: np.ndarray,
    prediction_normalized: np.ndarray,
    *,
    scaler: Mapping[str, Any],
) -> dict[str, float]:
    truth = np.asarray(truth_normalized, dtype=np.float64)
    prediction = np.asarray(prediction_normalized, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 3 or truth.shape[1:] != (1, 4):
        raise ValueError("H60 metric arrays must share [samples,1,4] shape")
    error = prediction - truth
    result: dict[str, float] = {
        "normalized_macro_mae": float(np.abs(error).mean()),
        "normalized_rmse": float(np.sqrt(np.square(error).mean())),
    }
    raw_truth = denormalize_h60(truth, scaler, clip_nonnegative=False)
    raw_prediction = denormalize_h60(prediction, scaler, clip_nonnegative=True)
    raw_error = raw_prediction - raw_truth
    result["raw_macro_mae"] = float(np.abs(raw_error).mean())
    result["raw_rmse"] = float(np.sqrt(np.square(raw_error).mean()))
    for index, name in enumerate(H60_TARGET_NAMES):
        result[f"normalized_mae_{name}"] = float(
            np.abs(error[:, 0, index]).mean()
        )
        result[f"normalized_rmse_{name}"] = float(
            np.sqrt(np.square(error[:, 0, index]).mean())
        )
        result[f"raw_mae_{name}"] = float(
            np.abs(raw_error[:, 0, index]).mean()
        )
        result[f"raw_rmse_{name}"] = float(
            np.sqrt(np.square(raw_error[:, 0, index]).mean())
        )
    return result


def _mean_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    criterion = nn.MSELoss()
    total = 0.0
    samples = 0
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            loss = criterion(model(features), targets)
            batch = int(len(features))
            total += float(loss.item()) * batch
            samples += batch
    return total / samples


def train_h60_model(
    *,
    model_type: str,
    model_factory: Callable[[], nn.Module],
    model_config: Mapping[str, Any],
    seed: int,
    train_window: H60WindowDataset,
    validation_window: H60WindowDataset,
    training_config: H60TrainingConfig,
    checkpoint_path: str | Path,
    checkpoint_metadata: Mapping[str, Any],
    device: torch.device | None = None,
) -> H60TrainingResult:
    """Train with early stopping driven exclusively by validation MSE."""
    device = device or torch.device("cpu")
    set_deterministic_seed(seed)
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    criterion = nn.MSELoss()
    train_loader = _loader(
        train_window,
        batch_size=training_config.batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _loader(
        validation_window,
        batch_size=training_config.batch_size,
        shuffle=False,
        seed=seed,
    )
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    without_improvement = 0
    stopped_epoch = training_config.max_epochs
    started = time.perf_counter()
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
                raise RuntimeError(
                    f"non-finite {model_type} loss at seed={seed}, epoch={epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config.gradient_clip_norm
            )
            optimizer.step()
            batch = int(len(features))
            total += float(loss.item()) * batch
            samples += batch
        train_loss = total / samples
        validation_loss = _mean_mse(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - training_config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            without_improvement = 0
            payload = {
                "model_type": model_type,
                "model_config": dict(model_config),
                "model_state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "seed": int(seed),
                "best_epoch": int(best_epoch),
                "best_validation_loss": float(best_loss),
                "training_config": asdict(training_config),
                **dict(checkpoint_metadata),
            }
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(checkpoint)
        else:
            without_improvement += 1
            if without_improvement >= training_config.patience:
                stopped_epoch = epoch
                break
    elapsed = time.perf_counter() - started
    if best_epoch == 0 or not math.isfinite(best_loss):
        raise RuntimeError(f"{model_type} seed {seed} produced no valid checkpoint")
    return H60TrainingResult(
        model_type=model_type,
        seed=int(seed),
        best_epoch=int(best_epoch),
        stopped_epoch=int(stopped_epoch),
        best_validation_loss=float(best_loss),
        training_seconds=float(elapsed),
        checkpoint_path=str(checkpoint),
        history=tuple(history),
    )


def load_h60_checkpoint(
    path: str | Path,
    *,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=True)
    model_type = str(payload["model_type"])
    config = dict(payload["model_config"])
    if model_type == "MLP":
        model: nn.Module = H60MLP(**config)
    elif model_type == "Transformer":
        model = TransformerForecaster(TransformerForecastConfig(**config))
    else:
        raise ValueError(f"unsupported H60 model type {model_type!r}")
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload
