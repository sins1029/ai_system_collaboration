from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from forecasting.forecast_evaluation import (
    compute_metrics,
    forecast_timestamps,
    inverse_transform_targets,
    normalize_history,
    persistence_forecast,
)
from forecasting.loader import LoadedForecastDataset, load_forecast_dataset
from forecasting.training import load_checkpoint, load_training_splits, save_checkpoint
from forecasting.transformer_forecaster import (
    SinusoidalPositionalEncoding,
    TransformerForecastConfig,
    TransformerForecaster,
)


def _config() -> TransformerForecastConfig:
    return TransformerForecastConfig(
        input_dim=8,
        target_dim=4,
        history_length=96,
        forecast_horizon=4,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.1,
        head_hidden_dim=24,
    )


def _scaler() -> dict[str, object]:
    return {
        "fitted_on": "TRAIN_ONLY",
        "parameters": {
            "a": {"mean": 10.0, "scale": 2.0},
            "b": {"mean": 20.0, "scale": 4.0},
            "c": {"mean": 30.0, "scale": 5.0},
            "d": {"mean": 40.0, "scale": 8.0},
        },
    }


def test_transformer_output_shape() -> None:
    model = TransformerForecaster(_config()).eval()
    result = model(torch.zeros(3, 96, 8))

    assert result.shape == (3, 4, 4)


def test_positional_encoding_preserves_shape() -> None:
    encoding = SinusoidalPositionalEncoding(16, 96, 0.0)
    value = torch.randn(2, 96, 16)

    assert encoding(value).shape == value.shape


def test_forward_is_deterministic_in_evaluation_mode() -> None:
    torch.manual_seed(9)
    model = TransformerForecaster(_config()).eval()
    value = torch.randn(2, 96, 8)

    first = model(value)
    second = model(value)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_frozen_dataset_loader_is_compatible() -> None:
    root = Path(__file__).resolve().parents[1] / "artifacts/forecast_dataset_v1"
    data = load_forecast_dataset("train", 96, 4, True, dataset_root=root)

    assert data.X.shape == (3261, 96, 8)
    assert data.Y.shape == (3261, 4, 4)
    assert data.normalized is True


def test_checkpoint_reload_prediction_equivalence(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model = TransformerForecaster(_config()).eval()
    value = torch.randn(2, 96, 8)
    expected = model(value)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model,
        metadata={
            "input_dim": 8,
            "target_dim": 4,
            "history_length": 96,
            "forecast_horizon": 4,
            "feature_schema": [],
            "target_schema": [],
            "best_epoch": 1,
            "best_val_loss": 0.5,
            "seed": 5,
            "scaler_reference": "scaler.json",
            "dataset_manifest_reference": "manifest.json",
            "git_head": "test",
        },
    )

    reloaded, metadata = load_checkpoint(checkpoint)
    actual = reloaded(value)

    torch.testing.assert_close(actual, expected)
    assert metadata["best_val_loss"] == 0.5


def test_inverse_scaling_correctness() -> None:
    normalized = np.ones((2, 4, 4), dtype=np.float32)

    actual = inverse_transform_targets(normalized, _scaler(), ["a", "b", "c", "d"])

    np.testing.assert_allclose(actual[0, 0], [12.0, 24.0, 35.0, 48.0])


def test_persistence_repeats_last_workload_observation() -> None:
    history = np.arange(2 * 3 * 6, dtype=np.float64).reshape(2, 3, 6)

    actual = persistence_forecast(history, target_indices=[0, 2, 3, 5], horizon=4)

    expected = np.repeat(history[:, -1, [0, 2, 3, 5]][:, None, :], 4, axis=1)
    np.testing.assert_array_equal(actual, expected)


def test_horizon_timestamp_alignment() -> None:
    history_end = np.asarray([np.datetime64("2026-01-01T00:00")])

    actual = forecast_timestamps(history_end, horizon=4, resolution_minutes=15)

    np.testing.assert_array_equal(
        actual[0],
        np.asarray(
            [
                np.datetime64("2026-01-01T00:15"),
                np.datetime64("2026-01-01T00:30"),
                np.datetime64("2026-01-01T00:45"),
                np.datetime64("2026-01-01T01:00"),
            ],
            dtype="datetime64[ns]",
        ),
    )


def test_metrics_computation_by_target_and_horizon() -> None:
    truth = np.zeros((2, 4, 4), dtype=np.float64)
    prediction = np.ones_like(truth)

    metrics = compute_metrics(
        truth,
        prediction,
        target_names=["a", "b", "c", "d"],
        model="Transformer",
        seed=11,
    )

    assert metrics.shape[0] == 16
    assert metrics["horizon_minutes"].tolist()[:4] == [15, 15, 15, 15]
    np.testing.assert_allclose(metrics["mae"], 1.0)
    np.testing.assert_allclose(metrics["rmse"], 1.0)


def test_training_split_loader_never_requests_test() -> None:
    requested: list[str] = []

    def spy_loader(
        split: str,
        history_length: int,
        horizon: int,
        normalized: bool,
        *,
        dataset_root: str | Path,
    ) -> LoadedForecastDataset:
        requested.append(split)
        return LoadedForecastDataset(
            X=np.zeros((2, history_length, 8), dtype=np.float32),
            Y=np.zeros((2, horizon, 4), dtype=np.float32),
            history_end_timestamp=np.zeros(2, dtype="datetime64[ns]"),
            target_start_timestamp=np.zeros(2, dtype="datetime64[ns]"),
            target_end_timestamp=np.zeros(2, dtype="datetime64[ns]"),
            feature_schema={},
            split=split,
            normalized=normalized,
        )

    load_training_splits("unused", loader=spy_loader)

    assert requested == ["train", "val"]
    assert "test" not in requested


def test_raw_history_normalization_respects_schema_roles() -> None:
    history = np.asarray([[12.0, 24.0, 35.0, 48.0, 0.5]], dtype=np.float32)
    schema = [
        {"index": index, "name": name, "normalization": "TRAIN_STANDARD_SCORE"}
        for index, name in enumerate(["a", "b", "c", "d"])
    ] + [{"index": 4, "name": "calendar", "normalization": "NONE"}]

    actual = normalize_history(history, _scaler(), schema)

    np.testing.assert_allclose(actual[0, :4], 1.0)
    assert actual[0, 4] == 0.5


def test_model_config_round_trip_is_json_serializable() -> None:
    restored = TransformerForecastConfig(**json.loads(json.dumps(_config().to_dict())))

    assert restored == _config()
