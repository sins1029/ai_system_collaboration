from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from forecasting.forecast_evaluation import (
    forecast_timestamps,
    inverse_transform_targets,
    normalize_history,
    schema_names,
)
from forecasting.training import load_checkpoint


@dataclass
class TransformerForecastService:
    model: torch.nn.Module
    checkpoint_metadata: dict[str, Any]
    scaler: dict[str, Any]
    feature_schema: dict[str, Any]
    device: torch.device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        dataset_root: str | Path,
        device: str | torch.device | None = None,
    ) -> "TransformerForecastService":
        selected_device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        root = Path(dataset_root)
        scaler = json.loads((root / "09_scaler_stats.json").read_text("utf-8"))
        schema = json.loads((root / "10_feature_schema.json").read_text("utf-8"))
        model, metadata = load_checkpoint(checkpoint, device=selected_device)
        return cls(model, metadata, scaler, schema, selected_device)

    @property
    def target_names(self) -> list[str]:
        return schema_names(self.feature_schema, "Y")

    def forecast(self, history: np.ndarray) -> np.ndarray:
        config = self.model.config
        raw = np.asarray(history, dtype=np.float32)
        expected = (config.history_length, config.input_dim)
        if raw.shape != expected:
            raise ValueError(f"history must have shape {expected}, got {raw.shape}")
        normalized = normalize_history(raw, self.scaler, self.feature_schema["X"])
        features = torch.from_numpy(normalized[None, ...]).to(self.device)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(features).cpu().numpy()[0]
        return inverse_transform_targets(
            prediction,
            self.scaler,
            self.target_names,
        )

    def forecast_with_timestamps(
        self,
        history: np.ndarray,
        history_end_timestamp: str | np.datetime64 | pd.Timestamp,
    ) -> list[dict[str, Any]]:
        prediction = self.forecast(history)
        timestamps = forecast_timestamps(
            np.asarray([np.datetime64(pd.Timestamp(history_end_timestamp).to_datetime64())]),
            horizon=prediction.shape[0],
            resolution_minutes=15,
        )[0]
        rows: list[dict[str, Any]] = []
        for index, timestamp in enumerate(timestamps):
            row: dict[str, Any] = {
                "forecast_timestamp": pd.Timestamp(timestamp).isoformat(),
                "horizon_minutes": (index + 1) * 15,
            }
            row.update(
                {
                    name: float(prediction[index, target_index])
                    for target_index, name in enumerate(self.target_names)
                }
            )
            rows.append(row)
        return rows
