from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class TransformerForecastConfig:
    input_dim: int
    target_dim: int = 4
    history_length: int = 96
    forecast_horizon: int = 4
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    head_hidden_dim: int = 128

    def __post_init__(self) -> None:
        for name in (
            "input_dim",
            "target_dim",
            "history_length",
            "forecast_horizon",
            "d_model",
            "nhead",
            "num_layers",
            "dim_feedforward",
            "head_hidden_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int, dropout: float) -> None:
        super().__init__()
        if d_model <= 0 or max_length <= 0:
            raise ValueError("d_model and max_length must be positive")
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            encoding[:, 1::2] = torch.cos(
                position * div_term[: encoding[:, 1::2].shape[1]]
            )
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"positional input must be [B,L,D], got {tuple(x.shape)}")
        if x.shape[1] > self.encoding.shape[1] or x.shape[2] != self.encoding.shape[2]:
            raise ValueError("positional encoding shape mismatch")
        return self.dropout(x + self.encoding[:, : x.shape[1]].to(x.dtype))


class TransformerForecaster(nn.Module):
    def __init__(self, config: TransformerForecastConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_projection = nn.Linear(config.input_dim, config.d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(
            config.d_model,
            config.history_length,
            config.dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )
        self.forecast_head = nn.Sequential(
            nn.Linear(config.d_model, config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.head_hidden_dim,
                config.forecast_horizon * config.target_dim,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (
            self.config.history_length,
            self.config.input_dim,
        )
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                "Transformer input must be "
                f"[B,{expected[0]},{expected[1]}], got {tuple(x.shape)}"
            )
        hidden = self.feature_projection(x)
        hidden = self.positional_encoding(hidden)
        encoded = self.encoder(hidden)
        forecast = self.forecast_head(encoded[:, -1, :])
        result = forecast.reshape(
            x.shape[0],
            self.config.forecast_horizon,
            self.config.target_dim,
        )
        expected_output = (
            x.shape[0],
            self.config.forecast_horizon,
            self.config.target_dim,
        )
        if tuple(result.shape) != expected_output:
            raise RuntimeError("Transformer output shape contract failed")
        return result


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
