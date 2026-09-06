from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml


matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting.forecast_evaluation import (
    compute_metrics,
    inverse_transform_targets,
    persistence_forecast,
    prediction_long_frame,
    relative_improvement,
    schema_names,
    write_json,
)
from forecasting.inference import TransformerForecastService
from forecasting.loader import load_forecast_dataset
from forecasting.training import load_checkpoint
from forecasting.transformer_forecaster import count_parameters


def _git(*args: str, cwd: Path = WORKSPACE) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8", errors="replace"
    ).strip()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE).as_posix()


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _predict(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(
                np.asarray(features[start : start + batch_size], dtype=np.float32)
            ).to(device)
            output.append(model(batch).cpu().numpy())
    return np.concatenate(output, axis=0)


def _raw_history(
    normalized: np.ndarray,
    *,
    scaler: dict[str, Any],
    feature_names: list[str],
    target_names: list[str],
) -> tuple[np.ndarray, list[int]]:
    result = np.asarray(normalized, dtype=np.float64).copy()
    indices = [feature_names.index(name) for name in target_names]
    result[..., indices] = inverse_transform_targets(
        result[..., indices], scaler, target_names
    )
    return result, indices


def _metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    transformer = metrics[metrics["model"] == "Transformer"]
    persistence = metrics[metrics["model"] == "Persistence"]
    for scope, group_column in (("target", "target"), ("horizon", "horizon_minutes")):
        for value, group in transformer.groupby(group_column, sort=False):
            per_seed = group.groupby("seed", as_index=False)[["mae", "rmse"]].mean()
            rows.append(
                {
                    "model": "Transformer",
                    "summary_scope": scope,
                    "target": value if scope == "target" else "ALL",
                    "horizon_minutes": value if scope == "horizon" else "ALL",
                    "mae_mean": float(per_seed["mae"].mean()),
                    "mae_std": float(per_seed["mae"].std(ddof=0)),
                    "rmse_mean": float(per_seed["rmse"].mean()),
                    "rmse_std": float(per_seed["rmse"].std(ddof=0)),
                }
            )
        for value, group in persistence.groupby(group_column, sort=False):
            rows.append(
                {
                    "model": "Persistence",
                    "summary_scope": scope,
                    "target": value if scope == "target" else "ALL",
                    "horizon_minutes": value if scope == "horizon" else "ALL",
                    "mae_mean": float(group["mae"].mean()),
                    "mae_std": 0.0,
                    "rmse_mean": float(group["rmse"].mean()),
                    "rmse_std": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _gpu_peak_rows(
    y_true: np.ndarray,
    predictions: dict[int, np.ndarray],
    persistence: np.ndarray,
    *,
    gpu_index: int,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[str, int | str, np.ndarray]] = [
        ("Persistence", "NA", persistence)
    ] + [("Transformer", seed, value) for seed, value in predictions.items()]
    for model, seed, candidate in candidates:
        for horizon_index in [None, 0, 1, 2, 3]:
            truth = y_true[:, :, gpu_index]
            pred = candidate[:, :, gpu_index]
            if horizon_index is not None:
                truth = truth[:, horizon_index]
                pred = pred[:, horizon_index]
            mask = truth >= threshold
            error = pred[mask] - truth[mask]
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "segment": "GPU_TRUE_GE_TEST_P90",
                    "horizon_minutes": "ALL" if horizon_index is None else (horizon_index + 1) * 15,
                    "threshold": threshold,
                    "point_count": int(mask.sum()),
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "selection_basis": "evaluation diagnostic only; not model selection",
                }
            )
    return pd.DataFrame(rows)


def _horizon_degradation(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, seed, target), group in metrics.groupby(
        ["model", "seed", "target"], sort=False
    ):
        ordered = group.sort_values("horizon_step")
        mae = ordered["mae"].to_numpy()
        rmse = ordered["rmse"].to_numpy()
        monotonic_mae = bool(np.all(np.diff(mae) >= -1e-12))
        monotonic_rmse = bool(np.all(np.diff(rmse) >= -1e-12))
        for row in ordered.itertuples(index=False):
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "target": target,
                    "horizon_minutes": row.horizon_minutes,
                    "mae": row.mae,
                    "rmse": row.rmse,
                    "mae_delta_vs_15m": row.mae - mae[0],
                    "rmse_delta_vs_15m": row.rmse - rmse[0],
                    "mae_monotonic_non_decreasing": monotonic_mae,
                    "rmse_monotonic_non_decreasing": monotonic_rmse,
                }
            )
    return pd.DataFrame(rows)


def _diagnostic_figures(
    *,
    timestamps: np.ndarray,
    y_true: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    target_names: list[str],
    seed: int,
    output: Path,
) -> list[Path]:
    gpu_index = target_names.index("arriving_gpu_demand")
    task_index = target_names.index("new_task_count")
    gpu = y_true[:, 0, gpu_index]
    normal = int(np.argmin(np.abs(gpu - np.median(gpu))))
    peak = int(np.argmax(gpu))
    drop = int(np.argmin(np.diff(gpu))) + 1
    centers = {"ordinary": normal, "gpu_peak": peak, "workload_drop": drop}
    paths: list[Path] = []
    for name, center in centers.items():
        start = max(0, center - 24)
        end = min(len(gpu), start + 48)
        start = max(0, end - 48)
        x = pd.to_datetime(timestamps[start:end]) + pd.Timedelta(minutes=15)
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
        for axis, target_index, label in (
            (axes[0], gpu_index, "arriving_gpu_demand"),
            (axes[1], task_index, "new_task_count"),
        ):
            axis.plot(x, y_true[start:end, 0, target_index], color="#111827", label="True", linewidth=1.8)
            axis.plot(x, prediction[start:end, 0, target_index], color="#0072B2", label="Transformer", linewidth=1.4)
            axis.plot(x, persistence[start:end, 0, target_index], color="#D55E00", label="Persistence", linewidth=1.2, alpha=0.85)
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
        axes[0].legend(ncol=3, frameon=False)
        axes[1].set_xlabel("Forecast timestamp (UTC), +15 min horizon")
        fig.suptitle(
            f"Test set | seed {seed} | {name} | {x[0]} to {x[-1]}"
        )
        fig.tight_layout()
        path = output / f"diagnostic_{name}_seed_{seed}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _benchmark(
    service: TransformerForecastService,
    history: np.ndarray,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        service.forecast(history)
    if service.device.type == "cuda":
        torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        service.forecast(history)
        if service.device.type == "cuda":
            torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
    return {
        "device": str(service.device),
        "batch_size": 1,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "timestep_minutes": 15,
    }


def evaluate(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text("utf-8"))["transformer_forecast_v1"]
    dataset_root = (WORKSPACE / config["dataset_root"]).resolve()
    output = (WORKSPACE / config["output_dir"]).resolve()
    predictions_dir = output / "predictions"
    figures_dir = output / "figures"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((dataset_root / "13_dataset_manifest.json").read_text("utf-8"))
    scaler = json.loads((dataset_root / "09_scaler_stats.json").read_text("utf-8"))
    schema = json.loads((dataset_root / "10_feature_schema.json").read_text("utf-8"))
    training_summary = json.loads(
        (output / "training/training_summary.json").read_text("utf-8")
    )
    if training_summary.get("test_split_accessed") is not False:
        raise RuntimeError("training boundary evidence is missing")
    selected = min(training_summary["results"], key=lambda item: item["best_val_loss"])
    recommended_path = WORKSPACE / selected["checkpoint"]
    recommended = {
        "seed": int(selected["seed"]),
        "checkpoint": selected["checkpoint"],
        "best_epoch": int(selected["best_epoch"]),
        "best_validation_normalized_mse": float(selected["best_val_loss"]),
        "selection_basis": "lowest validation normalized MSE",
        "test_metrics_used_for_selection": False,
    }
    write_json(output / "09_recommended_checkpoint.json", recommended)

    # This is the only test-split load, and it happens after checkpoint selection.
    test = load_forecast_dataset(
        "test",
        int(config["history_length"]),
        int(config["forecast_horizon"]),
        True,
        dataset_root=dataset_root,
    )
    feature_names = schema_names(schema, "X")
    target_names = schema_names(schema, "Y")
    if test.X.shape != (int(manifest["test_samples"]), 96, len(feature_names)):
        raise RuntimeError("test input shape contract changed")
    raw_x, target_indices = _raw_history(
        test.X,
        scaler=scaler,
        feature_names=feature_names,
        target_names=target_names,
    )
    y_true = inverse_transform_targets(test.Y, scaler, target_names)
    persistence = persistence_forecast(
        raw_x,
        target_indices=target_indices,
        horizon=int(config["forecast_horizon"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation_config = config["evaluation"]
    predictions: dict[int, np.ndarray] = {}
    metric_frames: list[pd.DataFrame] = [
        compute_metrics(
            y_true,
            persistence,
            target_names=target_names,
            model="Persistence",
            seed="NA",
        )
    ]
    for result in training_summary["results"]:
        seed = int(result["seed"])
        model, _ = load_checkpoint(WORKSPACE / result["checkpoint"], device=device)
        normalized_prediction = _predict(
            model,
            test.X,
            device=device,
            batch_size=int(evaluation_config["inference_batch_size"]),
        )
        prediction = inverse_transform_targets(
            normalized_prediction, scaler, target_names
        )
        predictions[seed] = prediction
        metric_frames.append(
            compute_metrics(
                y_true,
                prediction,
                target_names=target_names,
                model="Transformer",
                seed=seed,
            )
        )
        long_frame = prediction_long_frame(
            history_end_timestamp=test.history_end_timestamp,
            y_true=y_true,
            y_pred=prediction,
            persistence_pred=persistence,
            target_names=target_names,
        )
        long_frame.to_parquet(
            predictions_dir / f"test_predictions_seed_{seed}.parquet",
            index=False,
            engine="pyarrow",
            compression="zstd",
        )

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(output / "04_metrics_by_target_horizon.csv", index=False)
    _metric_summary(metrics).to_csv(output / "05_metrics_summary.csv", index=False)

    gpu_index = target_names.index("arriving_gpu_demand")
    gpu_threshold = float(
        np.quantile(y_true[:, :, gpu_index], float(evaluation_config["gpu_peak_quantile"]))
    )
    peak = _gpu_peak_rows(
        y_true,
        predictions,
        persistence,
        gpu_index=gpu_index,
        threshold=gpu_threshold,
    )
    peak.to_csv(output / "06_gpu_peak_diagnostics.csv", index=False)
    degradation = _horizon_degradation(metrics)
    degradation.to_csv(output / "07_horizon_degradation.csv", index=False)

    transformer_mean = (
        metrics[metrics["model"] == "Transformer"]
        .groupby(["target", "horizon_minutes"], as_index=False)[["mae", "rmse"]]
        .mean()
    )
    persistence_metrics = metrics[metrics["model"] == "Persistence"].copy()
    comparison = transformer_mean.merge(
        persistence_metrics[["target", "horizon_minutes", "mae", "rmse"]],
        on=["target", "horizon_minutes"],
        suffixes=("_transformer", "_persistence"),
    )
    comparison["mae_relative_improvement"] = comparison.apply(
        lambda row: relative_improvement(row.mae_persistence, row.mae_transformer),
        axis=1,
    )
    comparison["rmse_relative_improvement"] = comparison.apply(
        lambda row: relative_improvement(row.rmse_persistence, row.rmse_transformer),
        axis=1,
    )
    wins = int((comparison["mae_transformer"] < comparison["mae_persistence"]).sum())
    gpu_comparison = comparison[comparison["target"] == "arriving_gpu_demand"]
    gpu_wins = int((gpu_comparison["mae_transformer"] < gpu_comparison["mae_persistence"]).sum())
    peak_all = peak[peak["horizon_minutes"] == "ALL"].copy()
    peak_persistence_mae = float(peak_all[peak_all["model"] == "Persistence"]["mae"].iloc[0])
    peak_transformer_mae = float(peak_all[peak_all["model"] == "Transformer"]["mae"].mean())
    peak_better = peak_transformer_mae < peak_persistence_mae
    if wins >= 12 and gpu_wins == 4 and peak_better:
        judgement = "CLEARLY_BETTER"
    elif wins > 0 or gpu_wins > 0 or peak_better:
        judgement = "MIXED"
    else:
        judgement = "NO_BETTER"
    h60 = comparison[comparison["horizon_minutes"] == 60]
    h60_wins = int((h60["mae_transformer"] < h60["mae_persistence"]).sum())
    gpu_h60_better = bool(
        gpu_comparison.loc[gpu_comparison["horizon_minutes"] == 60, "mae_relative_improvement"].iloc[0] > 0
    )
    if h60_wins >= 3 and gpu_h60_better:
        h60_judgement = "USEFUL"
    elif h60_wins > 0:
        h60_judgement = "WEAK"
    else:
        h60_judgement = "NO_CLEAR_SIGNAL"

    comparison_lines = [
        "# Transformer vs Persistence",
        "",
        f"Sanity judgement: **{judgement}**.",
        f"Transformer mean MAE wins {wins}/16 target-horizon cells; GPU wins {gpu_wins}/4 horizons; GPU P90-segment mean MAE better: {peak_better}.",
        "",
        "| target | horizon_min | Transformer MAE | Persistence MAE | relative improvement |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        comparison_lines.append(
            f"| {row.target} | {row.horizon_minutes} | {row.mae_transformer:.6f} | {row.mae_persistence:.6f} | {row.mae_relative_improvement:.2%} |"
        )
    _write_text(output / "08_persistence_comparison.md", "\n".join(comparison_lines))

    recommended_prediction = predictions[int(recommended["seed"])]
    figure_paths = _diagnostic_figures(
        timestamps=test.history_end_timestamp,
        y_true=y_true,
        prediction=recommended_prediction,
        persistence=persistence,
        target_names=target_names,
        seed=int(recommended["seed"]),
        output=figures_dir,
    )
    service = TransformerForecastService.from_checkpoint(
        recommended_path,
        dataset_root=dataset_root,
        device=device,
    )
    benchmark = _benchmark(
        service,
        raw_x[0],
        warmup=int(evaluation_config["latency_warmup"]),
        iterations=int(evaluation_config["latency_iterations"]),
    )
    benchmark["parameter_count"] = count_parameters(service.model)
    write_json(output / "10_inference_benchmark.json", benchmark)

    leakage = pd.read_csv(dataset_root / "11_leakage_audit.csv")
    critical_pass = bool(
        leakage.loc[leakage["severity"] == "CRITICAL", "status"].eq("PASS").all()
    )
    leakage_text = f"""# Leakage Check

Status: **{'PASS' if critical_pass else 'FAIL'}**

- Training loader requests only `train` and `val`; it has no test-split branch.
- Training loss uses normalized train X/Y only.
- Validation normalized MSE is used only for early stopping and checkpoint selection.
- The recommended seed is selected before the single final test load, by lowest validation normalized MSE only.
- Test labels are used only for final metrics, figures, and the P90 diagnostic segmentation.
- Scaler metadata says `TRAIN_ONLY`; no scaler fitting occurs in model training, evaluation, or inference.
- Inputs follow the frozen schema: observed history plus legal known-in-advance calendar features.
- No future truth, test metric, test reward, or MPC outcome participates in model/architecture selection.
"""
    _write_text(output / "11_leakage_check.md", leakage_text)

    best_lines = "\n".join(
        f"- seed {item['seed']}: epoch {item['best_epoch']}, val MSE {item['best_val_loss']:.8f}, stopped epoch {item['stopped_epoch']}"
        for item in training_summary["results"]
    )
    gpu_lines = "\n".join(
        f"- +{int(row.horizon_minutes)}m: Transformer MAE {row.mae_transformer:.6f}, RMSE {row.rmse_transformer:.6f}; Persistence MAE {row.mae_persistence:.6f}, RMSE {row.rmse_persistence:.6f}; MAE improvement {row.mae_relative_improvement:.2%}."
        for row in gpu_comparison.itertuples(index=False)
    )
    target_improvement = (
        comparison.groupby("target")["mae_relative_improvement"].mean().sort_values(ascending=False)
    )
    strongest = str(target_improvement.index[0])
    gpu_degradation = degradation[
        (degradation["model"] == "Transformer")
        & (degradation["target"] == "arriving_gpu_demand")
    ].groupby("horizon_minutes")[["mae", "rmse"]].mean()
    stable = all(bool(item["stable"]) for item in training_summary["results"])
    ready = stable and critical_pass and judgement != "NO_BETTER"
    summary = f"""# Transformer Forecast v1 Summary

1. **Q1. 实际输入 shape？** `[N, 96, {len(feature_names)}]`；feature 顺序来自冻结 schema。
2. **Q2. 输出 shape？** `[N, 4, {len(target_names)}]`，对应 +15/+30/+45/+60 min。
3. **Q3. 参数量？** `{benchmark['parameter_count']}`。
4. **Q4. 三个 seed 是否稳定？** `{'YES' if stable else 'NO'}`，均以有限 validation loss 保存 checkpoint。
5. **Q5. 各 seed 最佳 validation epoch/loss？**\n{best_lines}
6. **Q6. 推荐 checkpoint？** `{recommended['checkpoint']}`，seed `{recommended['seed']}`。
7. **Q7. 是否优于 Persistence？** `{judgement}`；Transformer mean MAE 赢 `{wins}/16` 个 target-horizon cells。
8. **Q8. 改善最明显的 target？** `{strongest}`，按四个 horizon 的 mean relative MAE improvement 判断。
9. **Q9. GPU demand 是否优于 Persistence？** GPU MAE 赢 `{gpu_wins}/4` horizons。\n{gpu_lines}
10. **Q10. GPU peak 是否优于 Persistence？** `{'YES' if peak_better else 'NO'}`；test true P90=`{gpu_threshold:.6f}`，Transformer 三 seed mean MAE=`{peak_transformer_mae:.6f}`，Persistence=`{peak_persistence_mae:.6f}`。
11. **Q11. Horizon 误差如何变化？** GPU Transformer mean MAE 为 `{', '.join(f'+{int(i)}m={row.mae:.6f}' for i, row in gpu_degradation.iterrows())}`；完整 target/seed 明细见 `07_horizon_degradation.csv`。
12. **Q12. +60 min 是否仍可预测？** `{h60_judgement}`；四 target 中 `{h60_wins}/4` 的 mean MAE 优于 Persistence，GPU +60m 更优=`{gpu_h60_better}`。
13. **Q13. 是否发现 leakage？** `{'NO; PASS' if critical_pass else 'YES; FAIL'}`。
14. **Q14. Inference latency？** batch=1, `{benchmark['device']}`, mean `{benchmark['mean_ms']:.3f}` ms, P95 `{benchmark['p95_ms']:.3f}` ms。
15. **Q15. READY FOR FORECAST-AWARE MPC V1？** `{'YES' if ready else 'NO'}`。本轮未连接或修改 MPC。

Dataset v1 remained frozen. Test evaluation was run once after validation-only checkpoint selection.
"""
    _write_text(output / "01_summary.md", summary)

    evidence = f"""# Evidence Index

| Evidence | Location |
|---|---|
| Frozen dataset manifest | `artifacts/forecast_dataset_v1/13_dataset_manifest.json` |
| Frozen feature/target order | `artifacts/forecast_dataset_v1/10_feature_schema.json` |
| Train-only scaler | `artifacts/forecast_dataset_v1/09_scaler_stats.json` |
| Training boundary | `src/forecasting/training.py` |
| Transformer architecture | `src/forecasting/transformer_forecaster.py` |
| Pure inference API | `src/forecasting/inference.py` |
| Validation-only selection | `artifacts/transformer_forecast_v1/09_recommended_checkpoint.json` |
| Per-cell test metrics | `artifacts/transformer_forecast_v1/04_metrics_by_target_horizon.csv` |
| GPU P90 diagnostics | `artifacts/transformer_forecast_v1/06_gpu_peak_diagnostics.csv` |
| Test predictions | `artifacts/transformer_forecast_v1/predictions/` |
| Diagnostic figures | `artifacts/transformer_forecast_v1/figures/` |
| Git HEAD | `{_git('rev-parse', 'HEAD')}` |
| SustainCluster HEAD | `{_git('rev-parse', 'HEAD', cwd=WORKSPACE / 'references/external_repos/sustain-cluster')}` |
"""
    _write_text(output / "12_evidence_index.md", evidence)
    change_manifest = """# Change Manifest

## Added for Transformer Forecast v1

- Standard Transformer encoder, training/checkpoint utilities, evaluation helpers and pure inference service under `src/forecasting/`.
- Fixed v1 configuration and isolated train/evaluation scripts.
- Forecasting model tests covering shape, determinism, loader, checkpoint, scaling, baseline, timestamps, metrics and test-split isolation.
- Three seed histories/checkpoints, strict final test predictions, metrics, GPU diagnostics, figures and reports under this artifact directory.

## Explicit non-changes

- Forecast Dataset v1 files, split, schema, targets and scaler were not modified.
- SustainCluster third-party source was not modified.
- MPC, reward, BC and SAC code was not modified for this round.
- No MPC experiment, commit or push was performed.
"""
    _write_text(output / "13_change_manifest.md", change_manifest)

    final_record = {
        "status": "READY FOR FORECAST-AWARE MPC V1" if ready else "BLOCKED",
        "transformer_vs_persistence": judgement,
        "forecast_60_min": h60_judgement,
        "leakage": "PASS" if critical_pass else "FAIL",
        "recommended_checkpoint": recommended["checkpoint"],
        "figures": [_relative(path) for path in figure_paths],
        "output": _relative(output),
    }
    write_json(output / "evaluation_summary.json", final_record)
    print(json.dumps(final_record, ensure_ascii=False), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE / "configs/forecasting/transformer_forecast_v1.yaml",
    )
    args = parser.parse_args()
    evaluate(args.config.resolve())


if __name__ == "__main__":
    main()
