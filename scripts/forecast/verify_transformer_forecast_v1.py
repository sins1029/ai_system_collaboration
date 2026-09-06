from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting.inference import TransformerForecastService
from forecasting.loader import load_forecast_dataset
from forecasting.training import load_checkpoint


REQUIRED_FILES = (
    "01_summary.md",
    "02_model_config.json",
    "03_training_environment.json",
    "04_metrics_by_target_horizon.csv",
    "05_metrics_summary.csv",
    "06_gpu_peak_diagnostics.csv",
    "07_horizon_degradation.csv",
    "08_persistence_comparison.md",
    "09_recommended_checkpoint.json",
    "10_inference_benchmark.json",
    "11_leakage_check.md",
    "12_evidence_index.md",
    "13_change_manifest.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8", errors="replace"
    ).strip()


def verify(dataset_root: Path, output: Path) -> dict[str, object]:
    missing = [name for name in REQUIRED_FILES if not (output / name).is_file()]
    if missing:
        raise AssertionError(f"missing required artifacts: {missing}")

    dataset_manifest = json.loads(
        (dataset_root / "13_dataset_manifest.json").read_text("utf-8")
    )
    for name, metadata in dataset_manifest["dataset_files"].items():
        path = dataset_root / "dataset" / name
        if path.stat().st_size != int(metadata["bytes"]):
            raise AssertionError(f"Dataset v1 size changed: {name}")
        if _sha256(path) != metadata["sha256"]:
            raise AssertionError(f"Dataset v1 hash changed: {name}")

    metrics = pd.read_csv(output / "04_metrics_by_target_horizon.csv")
    if len(metrics) != 64 or metrics[["mae", "rmse"]].isna().any().any():
        raise AssertionError("metrics must contain 16 Persistence and 48 Transformer rows")
    if set(metrics["horizon_minutes"]) != {15, 30, 45, 60}:
        raise AssertionError("metric horizons changed")

    prediction_rows: dict[int, int] = {}
    for seed in (11, 22, 33):
        history = pd.read_csv(
            output / "training" / f"training_history_seed_{seed}.csv"
        )
        if history.empty or list(history.columns) != [
            "epoch",
            "train_loss",
            "val_loss",
            "learning_rate",
        ]:
            raise AssertionError(f"invalid training history for seed {seed}")
        if not np.isfinite(history[["train_loss", "val_loss"]]).all().all():
            raise AssertionError(f"non-finite history for seed {seed}")
        model, metadata = load_checkpoint(
            output / "checkpoints" / f"transformer_seed_{seed}_best.pt"
        )
        if metadata["seed"] != seed or model.config.forecast_horizon != 4:
            raise AssertionError(f"invalid checkpoint metadata for seed {seed}")
        prediction = pd.read_parquet(
            output / "predictions" / f"test_predictions_seed_{seed}.parquet"
        )
        required_columns = {
            "history_end_timestamp",
            "forecast_timestamp",
            "horizon_step",
            "horizon_minutes",
            "target",
            "y_true",
            "y_pred",
            "persistence_pred",
        }
        if set(prediction.columns) != required_columns or len(prediction) != 573 * 4 * 4:
            raise AssertionError(f"invalid prediction contract for seed {seed}")
        if prediction[["y_true", "y_pred", "persistence_pred"]].isna().any().any():
            raise AssertionError(f"NaN prediction for seed {seed}")
        prediction_rows[seed] = len(prediction)

    recommended = json.loads(
        (output / "09_recommended_checkpoint.json").read_text("utf-8")
    )
    if recommended["selection_basis"] != "lowest validation normalized MSE":
        raise AssertionError("recommended checkpoint is not validation-only")
    if recommended["test_metrics_used_for_selection"] is not False:
        raise AssertionError("test metrics entered checkpoint selection")

    # Pure API smoke test deliberately uses train raw history, not test data.
    train = load_forecast_dataset(
        "train", 96, 4, False, dataset_root=dataset_root
    )
    service = TransformerForecastService.from_checkpoint(
        WORKSPACE / recommended["checkpoint"],
        dataset_root=dataset_root,
        device="cpu",
    )
    forecast = service.forecast(train.X[0])
    timed = service.forecast_with_timestamps(
        train.X[0], train.history_end_timestamp[0]
    )
    if forecast.shape != (4, 4) or len(timed) != 4:
        raise AssertionError("pure inference API contract failed")
    if [row["horizon_minutes"] for row in timed] != [15, 30, 45, 60]:
        raise AssertionError("pure inference timestamp contract failed")

    sustain_repo = WORKSPACE / "references/external_repos/sustain-cluster"
    sustain_head = _git("rev-parse", "HEAD", cwd=sustain_repo)
    sustain_status = _git("status", "--short", cwd=sustain_repo)
    if sustain_head != "3f6ea95cb835b89ba50b0ef76d66d14b8037643e":
        raise AssertionError("SustainCluster commit changed")
    if sustain_status:
        raise AssertionError("SustainCluster worktree is not clean")

    return {
        "status": "PASS",
        "required_artifacts": len(REQUIRED_FILES),
        "metric_rows": len(metrics),
        "prediction_rows_by_seed": prediction_rows,
        "dataset_hashes_unchanged": True,
        "pure_inference_shape": list(forecast.shape),
        "sustaincluster_commit": sustain_head,
        "sustaincluster_clean": True,
        "test_npz_loaded_by_verifier": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=WORKSPACE / "artifacts/forecast_dataset_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "artifacts/transformer_forecast_v1",
    )
    args = parser.parse_args()
    result = verify(args.dataset_root.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
