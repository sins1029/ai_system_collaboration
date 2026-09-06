from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml


WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting.forecast_evaluation import schema_names, write_json
from forecasting.training import TrainingConfig, load_training_splits, train_one_seed
from forecasting.transformer_forecaster import (
    TransformerForecastConfig,
    TransformerForecaster,
    count_parameters,
)


def _git(*args: str, cwd: Path = WORKSPACE) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8", errors="replace"
    ).strip()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE).as_posix()


def _dataset_gate(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((root / "13_dataset_manifest.json").read_text("utf-8"))
    schema = json.loads((root / "10_feature_schema.json").read_text("utf-8"))
    leakage = pd.read_csv(root / "11_leakage_audit.csv")
    critical = leakage[leakage["severity"] == "CRITICAL"]
    failures = critical[critical["status"] != "PASS"]
    if failures.shape[0]:
        raise RuntimeError("BLOCKING DATASET ISSUE: critical leakage audit failed")
    required = {
        "source_type": "FIRST_VERIFIED_UNIQUE_CYCLE",
        "full_year_repetition_used": False,
        "random_split_used": False,
        "future_leakage_detected": False,
        "scaler_fit_split": "TRAIN_ONLY",
        "history_length": 96,
        "forecast_horizon": 4,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"BLOCKING DATASET ISSUE: {key}={manifest.get(key)!r}, expected {expected!r}"
            )
    return manifest, schema


def train(config_path: Path) -> Path:
    raw_config = yaml.safe_load(config_path.read_text("utf-8"))[
        "transformer_forecast_v1"
    ]
    dataset_root = (WORKSPACE / raw_config["dataset_root"]).resolve()
    output = (WORKSPACE / raw_config["output_dir"]).resolve()
    training_dir = output / "training"
    checkpoint_dir = output / "checkpoints"
    training_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest, schema = _dataset_gate(dataset_root)
    sustain_repo = WORKSPACE / "references/external_repos/sustain-cluster"
    sustain_head = _git("rev-parse", "HEAD", cwd=sustain_repo)
    sustain_status = _git("status", "--short", cwd=sustain_repo)
    if sustain_head != "3f6ea95cb835b89ba50b0ef76d66d14b8037643e":
        raise RuntimeError(f"unexpected SustainCluster commit: {sustain_head}")
    if sustain_status:
        raise RuntimeError("SustainCluster worktree must remain clean")

    history_length = int(raw_config["history_length"])
    horizon = int(raw_config["forecast_horizon"])
    train_data, validation_data = load_training_splits(
        dataset_root,
        history_length=history_length,
        horizon=horizon,
    )
    feature_names = schema_names(schema, "X")
    target_names = schema_names(schema, "Y")
    expected_train = (int(manifest["train_samples"]), history_length, len(feature_names))
    expected_val = (int(manifest["val_samples"]), history_length, len(feature_names))
    if train_data.X.shape != expected_train or validation_data.X.shape != expected_val:
        raise RuntimeError("Dataset v1 input shape contract changed")
    if train_data.Y.shape != (expected_train[0], horizon, len(target_names)):
        raise RuntimeError("Dataset v1 target shape contract changed")

    model_values = raw_config["model"]
    model_config = TransformerForecastConfig(
        input_dim=len(feature_names),
        target_dim=len(target_names),
        history_length=history_length,
        forecast_horizon=horizon,
        d_model=int(model_values["d_model"]),
        nhead=int(model_values["nhead"]),
        num_layers=int(model_values["num_layers"]),
        dim_feedforward=int(model_values["dim_feedforward"]),
        dropout=float(model_values["dropout"]),
        head_hidden_dim=int(model_values["head_hidden_dim"]),
    )
    training_values = raw_config["training"]
    if training_values["optimizer"] != "AdamW":
        raise ValueError("Transformer Forecast v1 requires AdamW")
    training_config = TrainingConfig(
        batch_size=int(training_values["batch_size"]),
        max_epochs=int(training_values["max_epochs"]),
        patience=int(training_values["early_stopping_patience"]),
        learning_rate=float(training_values["learning_rate"]),
        weight_decay=float(training_values["weight_decay"]),
        gradient_clip_norm=float(training_values["gradient_clip_norm"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameter_count = count_parameters(TransformerForecaster(model_config))
    git_head = _git("rev-parse", "HEAD")
    git_branch = _git("branch", "--show-current")

    model_record = {
        "architecture": "standard Transformer encoder with last-token head",
        "input_shape": [None, history_length, len(feature_names)],
        "output_shape": [None, horizon, len(target_names)],
        "model": model_config.to_dict(),
        "training": {"optimizer": "AdamW", **asdict(training_config)},
        "seeds": [int(seed) for seed in raw_config["seeds"]],
        "parameter_count": parameter_count,
        "feature_names": feature_names,
        "target_names": target_names,
    }
    write_json(output / "02_model_config.json", model_record)
    environment = {
        "device": str(device),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms": True,
        "git_branch": git_branch,
        "git_head": git_head,
        "sustaincluster_commit": sustain_head,
    }
    write_json(output / "03_training_environment.json", environment)

    metadata = {
        "input_dim": len(feature_names),
        "target_dim": len(target_names),
        "history_length": history_length,
        "forecast_horizon": horizon,
        "feature_schema": schema["X"],
        "target_schema": schema["Y"],
        "scaler_reference": _relative(dataset_root / "09_scaler_stats.json"),
        "dataset_manifest_reference": _relative(dataset_root / "13_dataset_manifest.json"),
        "git_head": git_head,
        "sustaincluster_commit": sustain_head,
    }
    results: list[dict[str, Any]] = []
    for seed_value in raw_config["seeds"]:
        seed = int(seed_value)
        checkpoint = checkpoint_dir / f"transformer_seed_{seed}_best.pt"
        print(f"TRAIN seed={seed} device={device}", flush=True)
        result = train_one_seed(
            seed=seed,
            model_config=model_config,
            training_config=training_config,
            train_data=train_data,
            validation_data=validation_data,
            checkpoint_path=checkpoint,
            device=device,
            checkpoint_metadata=metadata,
        )
        history_path = training_dir / f"training_history_seed_{seed}.csv"
        result.history.to_csv(history_path, index=False)
        record = {
            "seed": seed,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "stopped_epoch": result.stopped_epoch,
            "checkpoint": _relative(Path(result.checkpoint)),
            "history": _relative(history_path),
            "stable": result.stable,
        }
        results.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    summary = {
        "status": "PASS",
        "selection_split": "validation",
        "test_split_accessed": False,
        "dataset": "Forecast Dataset v1",
        "train_samples": int(train_data.X.shape[0]),
        "validation_samples": int(validation_data.X.shape[0]),
        "parameter_count": parameter_count,
        "device": str(device),
        "results": results,
    }
    write_json(training_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE / "configs/forecasting/transformer_forecast_v1.yaml",
    )
    args = parser.parse_args()
    train(args.config.resolve())


if __name__ == "__main__":
    main()
