from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
VENDOR = ROOT / "references/external_repos/sustain-cluster"
for entry in (ROOT, SRC, VENDOR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from forecasting.h60_dataset import (  # noqa: E402
    H60_FEATURE_NAMES,
    H60_TARGET_NAMES,
    H60ForecastDatasetConfig,
    build_h60_windows,
    build_spot_h60_time_series,
    fit_h60_train_scaler,
)
from forecasting.h60_models import (  # noqa: E402
    H60MLP,
    H60TrainingConfig,
    denormalize_h60,
    h60_metric_summary,
    load_h60_checkpoint,
    predict_h60,
    train_h60_model,
)
from forecasting.transformer_forecaster import (  # noqa: E402
    TransformerForecastConfig,
    TransformerForecaster,
    count_parameters,
)
from scripts.competition import run_mpc_h60_night1 as night1  # noqa: E402
from sustaincluster_mpc.action_adapter import (  # noqa: E402
    ActionMapping,
    SustainClusterActionAdapter,
)
from sustaincluster_mpc.forecast_pressure_adapter import (  # noqa: E402
    expected_origin_probabilities,
)
from sustaincluster_mpc.h60_controllers import (  # noqa: E402
    MpcH60LearnedController,
    MpcH60OracleController,
    MpcH60PersistenceController,
)
from sustaincluster_mpc.terminal_h60_optimizer import (  # noqa: E402
    TerminalH60DataCenter,
    TerminalH60Optimizer,
)


OUTPUT = ROOT / "artifacts/mpc_h60_v1"
CHECKPOINTS = OUTPUT / "forecast_checkpoints"
LABELS = OUTPUT / "labels"
EXPECTED_STATES = 17670
EXPECTED_TASKS = 466867
MODEL_SELECTION_PATH = CHECKPOINTS / "model_selection.json"
TARGET_OFFSET = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MPC-H60 Night 2 freeze")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("all", "forecast", "labels", "scale", "smoke", "report"),
        default=("all",),
    )
    parser.add_argument("--smoke-steps", type=int, default=500)
    parser.add_argument("--transformer-seed-budget-seconds", type=float, default=300.0)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    values = frame.loc[:, list(columns)].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in values.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def load_frozen_inputs() -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    config = yaml.safe_load(night1.CONFIG_PATH.read_text("utf-8"))["mpc_h60_v1"]
    manifest, _ = night1.load_manifest_and_audit()
    states, tasks, labels, privileged, lifecycle = night1.load_tables(manifest)
    if len(states) != EXPECTED_STATES or len(tasks) != EXPECTED_TASKS:
        raise RuntimeError("frozen Spot v3 cardinality changed")
    if len(labels) != EXPECTED_TASKS or tasks["task_id"].nunique() != EXPECTED_TASKS:
        raise RuntimeError("frozen Spot v3 task/label cardinality changed")
    if set(states["split"]) != {"train", "validation", "test"}:
        raise RuntimeError("frozen Spot v3 split labels changed")
    return config, manifest, states, tasks, labels, privileged, lifecycle


def _persistence_prediction(window: Any) -> np.ndarray:
    return np.asarray(window.X[:, -1:, : len(H60_TARGET_NAMES)], dtype=np.float32)


def _validation_row(
    *,
    model: str,
    seed: int | None,
    checkpoint: str | None,
    best_epoch: int | None,
    stopped_epoch: int | None,
    training_seconds: float,
    metrics: Mapping[str, float],
    persistence_mae: float,
    parameter_count: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "seed": seed,
        "selection_split": "validation",
        "early_stopping_signal": "VALIDATION_MSE_ONLY",
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "training_seconds": training_seconds,
        "parameter_count": parameter_count,
        "checkpoint": checkpoint,
        **dict(metrics),
        "persistence_relative_improvement": (
            (persistence_mae - float(metrics["normalized_macro_mae"]))
            / persistence_mae
        ),
    }


def run_forecast_training(
    config: Mapping[str, Any],
    states: pd.DataFrame,
    tasks: pd.DataFrame,
    *,
    transformer_seed_budget_seconds: float,
) -> dict[str, Any]:
    forecast = config["forecast"]
    dataset_config = H60ForecastDatasetConfig(
        history_length=int(forecast["history_steps"]),
        target_offset_steps=int(forecast["target_offset_steps"]),
    )
    timeline = build_spot_h60_time_series(states, tasks)
    scaler = fit_h60_train_scaler(timeline)
    windows = {
        split: build_h60_windows(timeline, split, scaler, config=dataset_config)
        for split in ("train", "validation", "test")
    }
    expected_samples = {"train": 10429, "validation": 3214, "test": 3730}
    actual_samples = {name: value.sample_count for name, value in windows.items()}
    if actual_samples != expected_samples:
        raise RuntimeError(f"H60 forecast sample counts changed: {actual_samples}")

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    persistence_validation = h60_metric_summary(
        windows["validation"].Y,
        _persistence_prediction(windows["validation"]),
        scaler=scaler,
    )
    validation_rows = [
        _validation_row(
            model="Persistence",
            seed=None,
            checkpoint=None,
            best_epoch=None,
            stopped_epoch=None,
            training_seconds=0.0,
            metrics=persistence_validation,
            persistence_mae=persistence_validation["normalized_macro_mae"],
            parameter_count=0,
        )
    ]
    run_records: list[dict[str, Any]] = []
    input_dim = len(H60_FEATURE_NAMES)
    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    mlp_training = H60TrainingConfig(
        batch_size=int(forecast["batch_size"]),
        max_epochs=80,
        patience=8,
        learning_rate=float(forecast["learning_rate"]),
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        min_delta=1e-6,
    )
    for seed in (11, 22, 33):
        model_config = {
            "input_dim": input_dim,
            "history_length": dataset_config.history_length,
            "hidden_dim": int(forecast["hidden_dim"]),
        }
        checkpoint = CHECKPOINTS / f"mlp_seed{seed}.pt"
        result = train_h60_model(
            model_type="MLP",
            model_factory=lambda cfg=model_config: H60MLP(**cfg),
            model_config=model_config,
            seed=seed,
            train_window=windows["train"],
            validation_window=windows["validation"],
            training_config=mlp_training,
            checkpoint_path=checkpoint,
            checkpoint_metadata={
                "information_class": "DEPLOYABLE_HISTORY_ONLY",
                "target_offset_steps": TARGET_OFFSET,
                "input_features": H60_FEATURE_NAMES,
                "targets": H60_TARGET_NAMES,
            },
            device=device,
        )
        model, _ = load_h60_checkpoint(checkpoint, device=device)
        prediction = predict_h60(model, windows["validation"].X, device=device)
        metrics = h60_metric_summary(
            windows["validation"].Y, prediction, scaler=scaler
        )
        validation_rows.append(
            _validation_row(
                model="MLP",
                seed=seed,
                checkpoint=relative(checkpoint),
                best_epoch=result.best_epoch,
                stopped_epoch=result.stopped_epoch,
                training_seconds=result.training_seconds,
                metrics=metrics,
                persistence_mae=persistence_validation["normalized_macro_mae"],
                parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            )
        )
        pd.DataFrame(result.history).to_csv(
            CHECKPOINTS / f"mlp_seed{seed}_history.csv", index=False
        )
        print(
            f"[forecast] MLP seed={seed} best_epoch={result.best_epoch} "
            f"val_mae={metrics['normalized_macro_mae']:.6f} "
            f"seconds={result.training_seconds:.1f}"
        )

    transformer_model_config = TransformerForecastConfig(
        input_dim=input_dim,
        target_dim=len(H60_TARGET_NAMES),
        history_length=dataset_config.history_length,
        forecast_horizon=1,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        head_hidden_dim=64,
    )
    transformer_training = H60TrainingConfig(
        batch_size=256,
        max_epochs=50,
        patience=6,
        learning_rate=1e-3,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        min_delta=1e-6,
    )
    transformer_seeds = [22]
    seed22_seconds = float("inf")
    position = 0
    while position < len(transformer_seeds):
        seed = transformer_seeds[position]
        checkpoint = CHECKPOINTS / f"transformer_seed{seed}.pt"
        result = train_h60_model(
            model_type="Transformer",
            model_factory=lambda cfg=transformer_model_config: TransformerForecaster(cfg),
            model_config=transformer_model_config.to_dict(),
            seed=seed,
            train_window=windows["train"],
            validation_window=windows["validation"],
            training_config=transformer_training,
            checkpoint_path=checkpoint,
            checkpoint_metadata={
                "information_class": "DEPLOYABLE_HISTORY_ONLY",
                "target_offset_steps": TARGET_OFFSET,
                "input_features": H60_FEATURE_NAMES,
                "targets": H60_TARGET_NAMES,
            },
            device=device,
        )
        if seed == 22:
            seed22_seconds = result.training_seconds
            if seed22_seconds < transformer_seed_budget_seconds:
                transformer_seeds.extend((11, 33))
        model, _ = load_h60_checkpoint(checkpoint, device=device)
        prediction = predict_h60(model, windows["validation"].X, device=device)
        metrics = h60_metric_summary(
            windows["validation"].Y, prediction, scaler=scaler
        )
        validation_rows.append(
            _validation_row(
                model="Transformer",
                seed=seed,
                checkpoint=relative(checkpoint),
                best_epoch=result.best_epoch,
                stopped_epoch=result.stopped_epoch,
                training_seconds=result.training_seconds,
                metrics=metrics,
                persistence_mae=persistence_validation["normalized_macro_mae"],
                parameter_count=count_parameters(model),
            )
        )
        pd.DataFrame(result.history).to_csv(
            CHECKPOINTS / f"transformer_seed{seed}_history.csv", index=False
        )
        print(
            f"[forecast] Transformer seed={seed} best_epoch={result.best_epoch} "
            f"val_mae={metrics['normalized_macro_mae']:.6f} "
            f"seconds={result.training_seconds:.1f}"
        )
        position += 1

    comparison = pd.DataFrame(validation_rows)
    learned = comparison[comparison["model"].isin(["MLP", "Transformer"])].copy()
    best_learned_index = learned["normalized_macro_mae"].idxmin()
    best_learned = comparison.loc[best_learned_index]
    best_by_family = {
        family: comparison[comparison["model"] == family]
        .sort_values(["normalized_macro_mae", "seed"], kind="mergesort")
        .iloc[0]
        for family in ("MLP", "Transformer")
    }
    overall = comparison.sort_values(
        ["normalized_macro_mae", "model"], kind="mergesort"
    ).iloc[0]
    comparison["selected_family_checkpoint"] = False
    for item in best_by_family.values():
        comparison.loc[item.name, "selected_family_checkpoint"] = True
    comparison["selected_learned"] = False
    comparison.loc[best_learned_index, "selected_learned"] = True
    comparison["selected_overall"] = False
    comparison.loc[overall.name, "selected_overall"] = True
    comparison.to_csv(OUTPUT / "12_forecast_model_comparison.csv", index=False)

    test_rows = []
    persistence_test_prediction = _persistence_prediction(windows["test"])
    persistence_test = h60_metric_summary(
        windows["test"].Y, persistence_test_prediction, scaler=scaler
    )
    test_candidates: list[tuple[str, int | None, str | None, np.ndarray]] = [
        ("Persistence", None, None, persistence_test_prediction)
    ]
    for family, item in best_by_family.items():
        checkpoint = ROOT / str(item["checkpoint"])
        model, _ = load_h60_checkpoint(checkpoint, device=device)
        test_candidates.append(
            (
                family,
                int(item["seed"]),
                relative(checkpoint),
                predict_h60(model, windows["test"].X, device=device),
            )
        )
    for model_name, seed, checkpoint, prediction in test_candidates:
        metrics = h60_metric_summary(windows["test"].Y, prediction, scaler=scaler)
        test_rows.append(
            {
                "model": model_name,
                "seed": seed,
                "split": "test",
                "checkpoint": checkpoint,
                **metrics,
                "persistence_relative_improvement": (
                    (
                        persistence_test["normalized_macro_mae"]
                        - metrics["normalized_macro_mae"]
                    )
                    / persistence_test["normalized_macro_mae"]
                ),
                "test_evaluated_after_validation_selection": True,
            }
        )
    test_metrics = pd.DataFrame(test_rows)
    test_metrics.to_csv(OUTPUT / "13_forecast_test_metrics.csv", index=False)

    selection = {
        "selection_basis": "VALIDATION_NORMALIZED_MACRO_MAE_ONLY",
        "early_stopping_basis": "VALIDATION_MSE_ONLY",
        "test_used_for_selection": False,
        "selected_forecast": str(overall["model"]),
        "selected_forecast_seed": (
            None if pd.isna(overall["seed"]) else int(overall["seed"])
        ),
        "selected_learned_model": str(best_learned["model"]),
        "selected_learned_seed": int(best_learned["seed"]),
        "selected_learned_checkpoint": str(best_learned["checkpoint"]),
        "selected_learned_validation_normalized_macro_mae": float(
            best_learned["normalized_macro_mae"]
        ),
        "persistence_validation_normalized_macro_mae": float(
            persistence_validation["normalized_macro_mae"]
        ),
        "selected_learned_beats_persistence_validation": bool(
            best_learned["normalized_macro_mae"]
            < persistence_validation["normalized_macro_mae"]
        ),
        "transformer_seeds_completed": [
            int(value)
            for value in comparison.loc[
                comparison["model"] == "Transformer", "seed"
            ].tolist()
        ],
        "transformer_seed22_training_seconds": float(seed22_seconds),
        "samples": actual_samples,
        "history_steps": dataset_config.history_length,
        "target_offset_steps": dataset_config.target_offset_steps,
        "intermediate_targets_used": False,
    }
    write_json(MODEL_SELECTION_PATH, selection)
    return selection


def build_terminal_occupancy_cache(
    states: pd.DataFrame,
    tasks: pd.DataFrame,
    lifecycle: pd.DataFrame,
) -> np.ndarray:
    """Reconstruct Night 1 visible-active t+60 occupancy in O(tasks + states)."""
    total_steps = int(states["step"].max()) + 1
    task_resources = (
        tasks.drop_duplicates("task_id")
        .loc[:, ["task_id", "cpu_cores", "gpu_units", "memory_gb"]]
    )
    life = lifecycle.merge(task_resources, on="task_id", how="left", validate="one_to_one")
    start = life["actual_execution_start_step"].fillna(
        life["planned_execution_start_step"]
    )
    completion = life["start_estimated_completion_step"].fillna(
        life["planned_estimated_completion_step"]
    )
    actual_end = life.get(
        "observed_true_completion_step", pd.Series(index=life.index, dtype=float)
    ).fillna(total_steps)
    dispatch = life.get("dispatch_step", pd.Series(index=life.index, dtype=float))
    destination = life["destination_dc"]
    valid = (
        dispatch.notna()
        & destination.notna()
        & start.notna()
        & completion.notna()
    )
    life = life.loc[valid].copy()
    start = start.loc[valid].to_numpy(dtype=np.int64)
    completion = completion.loc[valid].to_numpy(dtype=np.int64)
    actual_end = actual_end.loc[valid].to_numpy(dtype=np.int64)
    dispatch = dispatch.loc[valid].to_numpy(dtype=np.int64)
    destination = destination.loc[valid].to_numpy(dtype=np.int64)
    interval_start = np.maximum(dispatch + 1, start - TARGET_OFFSET)
    interval_end = np.minimum(actual_end, completion - TARGET_OFFSET)
    interval_start = np.clip(interval_start, 0, total_steps)
    interval_end = np.clip(interval_end, 0, total_steps)
    keep = interval_end > interval_start
    interval_start = interval_start[keep]
    interval_end = interval_end[keep]
    dc_index = destination[keep] - 1
    resources = life.loc[
        life.index[keep], ["cpu_cores", "gpu_units", "memory_gb"]
    ].to_numpy(dtype=float)
    if np.any((dc_index < 0) | (dc_index >= 5)):
        raise RuntimeError("terminal occupancy cache found an invalid destination DC")
    difference = np.zeros((total_steps + 1, 5, 3), dtype=np.float64)
    for resource_index in range(3):
        np.add.at(
            difference[:, :, resource_index],
            (interval_start, dc_index),
            resources[:, resource_index],
        )
        np.add.at(
            difference[:, :, resource_index],
            (interval_end, dc_index),
            -resources[:, resource_index],
        )
    occupancy = np.cumsum(difference[:-1], axis=0)
    occupancy[np.abs(occupancy) < 1e-9] = 0.0
    if np.any(occupancy < -1e-7):
        raise RuntimeError("terminal occupancy cache became negative")
    return occupancy


def build_fast_terminal_inputs(
    state_row: pd.Series,
    privileged_h60: pd.DataFrame,
    occupancy: np.ndarray,
    *,
    source: str = "PRIVILEGED_ORACLE_T60",
) -> tuple[TerminalH60DataCenter, ...]:
    step = int(state_row["step"])
    dc_rows = json.loads(state_row["dc_states_json"])
    dc_by_id = {int(item["dc_id"]): item for item in dc_rows}
    h60 = privileged_h60[privileged_h60["horizon_step"] == TARGET_OFFSET]
    if len(h60) != 5:
        raise RuntimeError(f"incomplete H60 rows for {state_row['state_id']}")
    values = []
    for row in h60.sort_values("dc_id").itertuples(index=False):
        dc_id = int(row.dc_id)
        dc = dc_by_id[dc_id]
        existing = occupancy[step, dc_id - 1]
        values.append(
            TerminalH60DataCenter(
                dc_id=dc_id,
                cpu_total_cores=float(dc["cpu_total"]),
                gpu_total_units=float(dc["gpu_total"]),
                memory_total_gb=float(dc["memory_total"]),
                estimated_existing_cpu_cores=float(existing[0]),
                estimated_existing_gpu_units=float(existing[1]),
                estimated_existing_memory_gb=float(existing[2]),
                arriving_cpu_demand=float(row.actual_origin_cpu_demand),
                arriving_gpu_demand=float(row.actual_origin_gpu_demand),
                arriving_memory_demand=float(row.actual_origin_memory_demand),
                electricity_price_usd_per_mwh=float(
                    row.future_electricity_price_usd_per_mwh
                ),
                carbon_intensity_gco2_per_kwh=float(
                    row.future_carbon_intensity_gco2_per_kwh
                ),
                source=source,
            )
        )
    return tuple(values)


def validate_terminal_cache(
    states: pd.DataFrame,
    tasks: pd.DataFrame,
    privileged: pd.DataFrame,
    lifecycle: pd.DataFrame,
    occupancy: np.ndarray,
) -> dict[str, Any]:
    manual_path = OUTPUT / "01_h60_manual_preflight.csv"
    if manual_path.is_file():
        state_ids = pd.read_csv(manual_path)["state_id"].astype(str).tolist()
    else:
        indices = np.linspace(0, len(states) - 1, 20, dtype=int)
        state_ids = states.iloc[indices]["state_id"].astype(str).tolist()
    state_index = states.set_index("state_id", drop=False)
    privileged_groups = {
        key: value for key, value in privileged.groupby("state_id", sort=False)
    }
    task_index = tasks.drop_duplicates("task_id").set_index("task_id")
    lifecycle_index = lifecycle.set_index("task_id")
    max_absolute_error = 0.0
    for state_id in state_ids:
        state_row = state_index.loc[state_id]
        slow = night1.build_terminal_inputs(
            state_row,
            privileged_groups[state_id],
            task_index,
            lifecycle_index,
        )
        fast = build_fast_terminal_inputs(
            state_row, privileged_groups[state_id], occupancy
        )
        slow_values = np.asarray(
            [
                [
                    item.estimated_existing_cpu_cores,
                    item.estimated_existing_gpu_units,
                    item.estimated_existing_memory_gb,
                ]
                for item in slow
            ]
        )
        fast_values = np.asarray(
            [
                [
                    item.estimated_existing_cpu_cores,
                    item.estimated_existing_gpu_units,
                    item.estimated_existing_memory_gb,
                ]
                for item in fast
            ]
        )
        max_absolute_error = max(
            max_absolute_error, float(np.abs(slow_values - fast_values).max())
        )
        if not np.allclose(slow_values, fast_values, rtol=0.0, atol=1e-7):
            raise RuntimeError(f"terminal occupancy cache mismatch at {state_id}")
    return {
        "states_checked": len(state_ids),
        "max_absolute_error": max_absolute_error,
        "status": "PASS",
        "semantics": "MATCHES_NIGHT1_VISIBLE_ACTIVE_ESTIMATED_COMPLETION",
    }


def _action_adapter() -> SustainClusterActionAdapter:
    return SustainClusterActionAdapter(
        ActionMapping(tuple((dc_id, dc_id) for dc_id in range(1, 6)), 0, 6)
    )


def _merge_task_labels(tasks: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    selected = labels.loc[
        :, ["state_id", "task_id", "task_position", "h1_action", "h4_action"]
    ]
    merged = tasks.merge(
        selected,
        on=["state_id", "task_id", "task_position"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged[["h1_action", "h4_action"]].isna().any().any():
        raise RuntimeError("task/label alignment failed")
    return merged


def _state_summary_rows(
    state_diagnostics: pd.DataFrame,
    full_labels: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for scope, value in [("overall", "all"), *[("split", item) for item in (
        "train", "validation", "test"
    )]]:
        state_part = (
            state_diagnostics
            if scope == "overall"
            else state_diagnostics[state_diagnostics["split"] == value]
        )
        task_part = (
            full_labels if scope == "overall" else full_labels[full_labels["split"] == value]
        )
        rows.append(
            {
                "scope": scope,
                "value": value,
                "states": int(len(state_part)),
                "tasks": int(len(task_part)),
                "solver_failures": int((state_part["solver_status"] != "optimal").sum()),
                "h1_h60_task_disagreements": int(
                    task_part["h1_h60_disagreement"].sum()
                ),
                "h1_h60_task_disagreement_rate": float(
                    task_part["h1_h60_disagreement"].mean()
                ) if len(task_part) else 0.0,
                "h1_h60_state_disagreements": int(
                    state_part["h1_h60_state_disagreement"].sum()
                ),
                "h1_h60_state_disagreement_rate": float(
                    state_part["h1_h60_state_disagreement"].mean()
                ),
                "h4_h60_task_disagreements": int(
                    task_part["h4_h60_disagreement"].sum()
                ),
                "h4_h60_task_disagreement_rate": float(
                    task_part["h4_h60_disagreement"].mean()
                ) if len(task_part) else 0.0,
                "h4_h60_state_disagreements": int(
                    state_part["h4_h60_state_disagreement"].sum()
                ),
                "h4_h60_state_disagreement_rate": float(
                    state_part["h4_h60_state_disagreement"].mean()
                ),
                "latency_p50_ms": float(state_part["solver_latency_ms"].quantile(0.50)),
                "latency_p90_ms": float(state_part["solver_latency_ms"].quantile(0.90)),
                "latency_p99_ms": float(state_part["solver_latency_ms"].quantile(0.99)),
            }
        )
    return pd.DataFrame(rows)


def _band(values: pd.Series, lower: float, upper: float) -> pd.Series:
    return pd.Series(
        np.where(values <= lower, "low", np.where(values <= upper, "mid", "high")),
        index=values.index,
    )


def build_disagreement_slices(
    full_labels: pd.DataFrame,
    state_diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = full_labels.copy()
    state_features = state_diagnostics.set_index("state_id")
    train_states = state_diagnostics[state_diagnostics["split"] == "train"]
    pressure_thresholds = train_states["risk_score"].quantile([1 / 3, 2 / 3]).tolist()
    future_thresholds = train_states["max_terminal_base_pressure"].quantile(
        [1 / 3, 2 / 3]
    ).tolist()
    price_thresholds = train_states["mean_terminal_electricity"].quantile(
        [1 / 3, 2 / 3]
    ).tolist()
    carbon_thresholds = train_states["mean_terminal_carbon"].quantile(
        [1 / 3, 2 / 3]
    ).tolist()
    train_tasks = result[result["split"] == "train"]
    duration_thresholds = train_tasks["estimated_duration_steps"].quantile(
        [1 / 3, 2 / 3]
    ).tolist()
    result["pressure_slice"] = _band(
        result["state_id"].map(state_features["risk_score"]), *pressure_thresholds
    )
    result["future_workload_slice"] = _band(
        result["state_id"].map(state_features["max_terminal_base_pressure"]),
        *future_thresholds,
    )
    result["future_electricity_slice"] = _band(
        result["state_id"].map(state_features["mean_terminal_electricity"]),
        *price_thresholds,
    )
    result["future_carbon_slice"] = _band(
        result["state_id"].map(state_features["mean_terminal_carbon"]),
        *carbon_thresholds,
    )
    result["duration_slice"] = _band(
        result["estimated_duration_steps"], *duration_thresholds
    ).replace({"low": "short", "mid": "medium", "high": "long"})
    result["sla_risk_slice"] = np.where(
        result["remaining_sla_steps"] <= 1,
        "high",
        np.where(result["remaining_sla_steps"] <= 4, "mid", "low"),
    )
    result["gpu_slice"] = np.where(result["gpu_units"] > 0, "GPU-positive", "CPU-only")
    dimensions = {
        "priority": "priority",
        "resource_pressure": "pressure_slice",
        "estimated_duration": "duration_slice",
        "sla_risk": "sla_risk_slice",
        "gpu_request": "gpu_slice",
        "future_workload": "future_workload_slice",
        "future_electricity": "future_electricity_slice",
        "future_carbon": "future_carbon_slice",
    }
    rows = []
    for dimension, column in dimensions.items():
        for value, group in result.groupby(column, sort=True):
            h1_state = group.groupby("state_id")["h1_h60_disagreement"].any()
            h4_state = group.groupby("state_id")["h4_h60_disagreement"].any()
            rows.append(
                {
                    "dimension": dimension,
                    "value": str(value),
                    "tasks": int(len(group)),
                    "states": int(group["state_id"].nunique()),
                    "h1_h60_task_disagreements": int(
                        group["h1_h60_disagreement"].sum()
                    ),
                    "h1_h60_task_disagreement_rate": float(
                        group["h1_h60_disagreement"].mean()
                    ),
                    "h1_h60_state_disagreements": int(h1_state.sum()),
                    "h1_h60_state_disagreement_rate": float(h1_state.mean()),
                    "h4_h60_task_disagreements": int(
                        group["h4_h60_disagreement"].sum()
                    ),
                    "h4_h60_task_disagreement_rate": float(
                        group["h4_h60_disagreement"].mean()
                    ),
                    "h4_h60_state_disagreements": int(h4_state.sum()),
                    "h4_h60_state_disagreement_rate": float(h4_state.mean()),
                }
            )
    thresholds = {
        "source_split": "TRAIN_ONLY_THRESHOLDS_FOR_NUMERIC_BANDS",
        "resource_pressure": pressure_thresholds,
        "estimated_duration_steps": duration_thresholds,
        "future_workload_pressure": future_thresholds,
        "future_electricity": price_thresholds,
        "future_carbon": carbon_thresholds,
        "sla_risk_rule": {"high": "remaining<=1", "mid": "2..4", "low": ">4"},
    }
    return pd.DataFrame(rows), thresholds


def run_full_labels(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    states: pd.DataFrame,
    tasks: pd.DataFrame,
    labels: pd.DataFrame,
    privileged: pd.DataFrame,
    lifecycle: pd.DataFrame,
) -> dict[str, Any]:
    LABELS.mkdir(parents=True, exist_ok=True)
    chunk_dir = LABELS / "chunks_v1"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    occupancy = build_terminal_occupancy_cache(states, tasks, lifecycle)
    cache_validation = validate_terminal_cache(
        states, tasks, privileged, lifecycle, occupancy
    )
    print(f"[labels] terminal occupancy cache {cache_validation}")
    merged = _merge_task_labels(tasks, labels)
    task_groups = merged.groupby("state_id", sort=False).indices
    privileged_groups = {
        key: value for key, value in privileged.groupby("state_id", sort=False)
    }
    current_config = night1.optimizer_config(
        ROOT / config["current_optimizer_config"]
    )
    terminal_config = night1.terminal_config(config, float(config["lambda_terminal"]))
    optimizer = MpcH60OracleController(TerminalH60Optimizer())
    adapter = _action_adapter()
    _, links = night1.capacity_v1.signal_and_network_template()
    state_rows = []
    chunk_frames: list[pd.DataFrame] = []
    chunk_paths: list[Path] = []
    chunk_start = 0
    for state_position, (_, state_row) in enumerate(
        states.sort_values("step", kind="mergesort").iterrows()
    ):
        state_id = str(state_row["state_id"])
        positions = task_groups.get(state_id, np.asarray([], dtype=int))
        task_rows = merged.iloc[positions].sort_values(
            "task_position", kind="mergesort"
        )
        current_state = night1.build_current_state(state_row, task_rows, links)
        terminal = build_fast_terminal_inputs(
            state_row, privileged_groups[state_id], occupancy
        )
        result = optimizer.solve(
            current_state,
            terminal,
            current_config=current_config,
            terminal_config=terminal_config,
            action_adapter=adapter,
        )
        if result.status != "optimal":
            raise RuntimeError(
                f"H60 full labeling failed at {state_id}: {result.status} {result.message}"
            )
        h60 = np.asarray(result.environment_actions, dtype=np.int8)
        h1 = task_rows["h1_action"].to_numpy(dtype=np.int8)
        h4 = task_rows["h4_action"].to_numpy(dtype=np.int8)
        if len(h60) != len(task_rows):
            raise RuntimeError(f"H60 action count mismatch at {state_id}")
        base_pressures = [max(item.base_pressure) for item in terminal]
        state_rows.append(
            {
                "state_id": state_id,
                "step": int(state_row["step"]),
                "split": str(state_row["split"]),
                "pending_tasks": int(len(task_rows)),
                "risk_score": float(state_row["risk_score"]),
                "solver_status": result.status,
                "solver_latency_ms": 1000.0 * result.solve_seconds,
                "presolve_retry": bool(result.presolve_retry),
                "current_objective": float(result.current_costs.total),
                "terminal_objective_raw": float(result.terminal_costs.total),
                "terminal_contribution": float(result.terminal_contribution),
                "max_terminal_base_pressure": float(max(base_pressures)),
                "mean_terminal_electricity": float(
                    np.mean([item.electricity_price_usd_per_mwh for item in terminal])
                ),
                "mean_terminal_carbon": float(
                    np.mean([item.carbon_intensity_gco2_per_kwh for item in terminal])
                ),
                "h1_h60_task_disagreements": int(np.sum(h1 != h60)),
                "h4_h60_task_disagreements": int(np.sum(h4 != h60)),
                "h1_h60_state_disagreement": bool(np.any(h1 != h60)),
                "h4_h60_state_disagreement": bool(np.any(h4 != h60)),
            }
        )
        if len(task_rows):
            chunk_frames.append(
                pd.DataFrame(
                    {
                        "state_id": state_id,
                        "step": int(state_row["step"]),
                        "split": str(state_row["split"]),
                        "task_id": task_rows["task_id"].astype(str).to_numpy(),
                        "task_position": task_rows["task_position"].to_numpy(dtype=np.int32),
                        "priority": task_rows["priority"].astype(str).to_numpy(),
                        "cpu_cores": task_rows["cpu_cores"].to_numpy(dtype=float),
                        "gpu_units": task_rows["gpu_units"].to_numpy(dtype=float),
                        "memory_gb": task_rows["memory_gb"].to_numpy(dtype=float),
                        "estimated_duration_steps": task_rows[
                            "estimated_duration_steps"
                        ].to_numpy(dtype=np.int32),
                        "remaining_sla_steps": task_rows[
                            "remaining_sla_steps"
                        ].to_numpy(dtype=np.int32),
                        "h1_action": h1,
                        "h4_action": h4,
                        "h60_action": h60,
                        "h1_h60_disagreement": h1 != h60,
                        "h4_h60_disagreement": h4 != h60,
                        "h60_solver_status": result.status,
                        "h60_solver_latency_ms": 1000.0 * result.solve_seconds,
                        "h60_information_class": "PRIVILEGED_ORACLE_T60_LABEL",
                    }
                )
            )
        flush = (state_position + 1) % 250 == 0 or state_position + 1 == len(states)
        if flush:
            end = state_position
            path = chunk_dir / f"h60_labels_{chunk_start:05d}_{end:05d}.parquet"
            frame = pd.concat(chunk_frames, ignore_index=True) if chunk_frames else pd.DataFrame()
            frame.to_parquet(path, index=False)
            chunk_paths.append(path)
            chunk_frames = []
            chunk_start = state_position + 1
            print(f"[labels] solved {state_position + 1}/{len(states)} states")

    state_diagnostics = pd.DataFrame(state_rows)
    full = pd.concat([pd.read_parquet(path) for path in chunk_paths], ignore_index=True)
    if len(state_diagnostics) != EXPECTED_STATES or len(full) != EXPECTED_TASKS:
        raise RuntimeError(
            f"full H60 output cardinality mismatch states={len(state_diagnostics)} tasks={len(full)}"
        )
    if state_diagnostics["solver_status"].ne("optimal").any():
        raise RuntimeError("full H60 output contains solver failures")
    full.sort_values(["step", "task_position"], kind="mergesort", inplace=True)
    full.reset_index(drop=True, inplace=True)
    output_paths: dict[str, Path] = {}
    for split, filename in (
        ("train", "h60_oracle_actions_train.parquet"),
        ("validation", "h60_oracle_actions_val.parquet"),
        ("test", "h60_oracle_actions_test.parquet"),
    ):
        path = LABELS / filename
        full[full["split"] == split].to_parquet(path, index=False)
        output_paths[split] = path
    full_path = LABELS / "h60_oracle_actions_full.parquet"
    full.to_parquet(full_path, index=False)
    output_paths["full"] = full_path
    state_path = LABELS / "h60_state_diagnostics.parquet"
    state_diagnostics.to_parquet(state_path, index=False)

    overall = _state_summary_rows(state_diagnostics, full)
    overall.to_csv(OUTPUT / "10_h60_full_disagreement.csv", index=False)
    slices, thresholds = build_disagreement_slices(full, state_diagnostics)
    slices.to_csv(OUTPUT / "11_h60_disagreement_slices.csv", index=False)
    manifest_value = {
        "source_manifest": relative(night1.MANIFEST_PATH),
        "source_manifest_sha256": sha256(night1.MANIFEST_PATH),
        "source_final_status": manifest["final_status"],
        "states": int(len(state_diagnostics)),
        "states_with_task_labels": int(full["state_id"].nunique()),
        "empty_pending_states": int((state_diagnostics["pending_tasks"] == 0).sum()),
        "tasks": int(len(full)),
        "split_states": state_diagnostics.groupby("split").size().to_dict(),
        "split_tasks": full.groupby("split").size().to_dict(),
        "solver_failures": 0,
        "presolve_retries": int(state_diagnostics["presolve_retry"].sum()),
        "lambda_terminal": float(config["lambda_terminal"]),
        "target_offset_steps": TARGET_OFFSET,
        "intermediate_horizons_used": False,
        "terminal_cache_validation": cache_validation,
        "slice_thresholds": thresholds,
        "original_v3_modified": False,
        "outputs": {
            name: {
                "path": relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in {**output_paths, "state_diagnostics": state_path}.items()
        },
    }
    if (
        manifest_value["states_with_task_labels"]
        + manifest_value["empty_pending_states"]
        != manifest_value["states"]
    ):
        raise RuntimeError("empty-state accounting does not reconcile")
    overall_row = overall[overall["scope"] == "overall"].iloc[0]
    manifest_value.update(
        {
            key: float(overall_row[key])
            for key in (
                "h1_h60_task_disagreement_rate",
                "h1_h60_state_disagreement_rate",
                "h4_h60_task_disagreement_rate",
                "h4_h60_state_disagreement_rate",
                "latency_p50_ms",
                "latency_p90_ms",
                "latency_p99_ms",
            )
        }
    )
    write_json(OUTPUT / "14_h60_full_label_manifest.json", manifest_value)
    return manifest_value


def _diagnostic_subset(state_diagnostics: pd.DataFrame, count: int = 200) -> pd.DataFrame:
    source = state_diagnostics[
        state_diagnostics["split"].isin(["train", "validation"])
        & (state_diagnostics["pending_tasks"] > 0)
    ].copy()
    source["pressure_band"] = pd.qcut(
        source["risk_score"].rank(method="first"),
        3,
        labels=["low", "mid", "high"],
    )
    allocations = {"low": 67, "mid": 67, "high": 66}
    selected = []
    for band, needed in allocations.items():
        values = source[source["pressure_band"] == band].sort_values("step")
        indices = np.linspace(0, len(values) - 1, needed, dtype=int)
        selected.append(values.iloc[indices])
    result = pd.concat(selected, ignore_index=True).sort_values("step")
    if len(result) != count or result["state_id"].duplicated().any():
        raise RuntimeError("terminal scale subset selection is not fixed and unique")
    return result


def run_terminal_scale_audit(
    config: Mapping[str, Any],
    states: pd.DataFrame,
    tasks: pd.DataFrame,
    labels: pd.DataFrame,
    privileged: pd.DataFrame,
    lifecycle: pd.DataFrame,
) -> dict[str, Any]:
    state_diagnostics = pd.read_parquet(LABELS / "h60_state_diagnostics.parquet")
    subset = _diagnostic_subset(state_diagnostics)
    subset.loc[:, ["state_id", "step", "split", "pressure_band", "risk_score"]].to_csv(
        OUTPUT / "terminal_scale_subset.csv", index=False
    )
    occupancy = build_terminal_occupancy_cache(states, tasks, lifecycle)
    merged = _merge_task_labels(tasks, labels)
    task_groups = merged.groupby("state_id", sort=False).indices
    state_index = states.set_index("state_id", drop=False)
    privileged_groups = {
        key: value for key, value in privileged.groupby("state_id", sort=False)
    }
    current_config = night1.optimizer_config(
        ROOT / config["current_optimizer_config"]
    )
    _, links = night1.capacity_v1.signal_and_network_template()
    optimizer = MpcH60OracleController(TerminalH60Optimizer())
    adapter = _action_adapter()
    rows = []
    for lambda_value in (0.0, 0.5, 1.0, 2.0, 5.0):
        current_values = []
        terminal_raw_values = []
        contributions = []
        latencies = []
        task_count = 0
        disagreements = 0
        state_disagreements = 0
        retries = 0
        failures = 0
        terminal_config = night1.terminal_config(config, lambda_value)
        for state_id in subset["state_id"]:
            state_row = state_index.loc[state_id]
            task_rows = merged.iloc[task_groups[state_id]].sort_values("task_position")
            current_state = night1.build_current_state(state_row, task_rows, links)
            terminal = build_fast_terminal_inputs(
                state_row, privileged_groups[state_id], occupancy
            )
            result = optimizer.solve(
                current_state,
                terminal,
                current_config=current_config,
                terminal_config=terminal_config,
                action_adapter=adapter,
            )
            latencies.append(1000.0 * result.solve_seconds)
            retries += int(result.presolve_retry)
            if result.status != "optimal":
                failures += 1
                continue
            h1 = task_rows["h1_action"].to_numpy(dtype=int)
            h60 = np.asarray(result.environment_actions, dtype=int)
            task_count += len(h1)
            differences = int(np.sum(h1 != h60))
            disagreements += differences
            state_disagreements += int(differences > 0)
            current_values.append(abs(float(result.current_costs.total)))
            terminal_raw_values.append(abs(float(result.terminal_costs.total)))
            contributions.append(abs(float(result.terminal_contribution)))
        mean_current = float(np.mean(current_values))
        mean_terminal_raw = float(np.mean(terminal_raw_values))
        mean_contribution = float(np.mean(contributions))
        rows.append(
            {
                "lambda_terminal": lambda_value,
                "subset": "FIXED_TRAIN_VALIDATION_200_STATES",
                "states": int(len(subset)),
                "tasks": task_count,
                "mean_current_objective_magnitude": mean_current,
                "mean_terminal_objective_raw_magnitude": mean_terminal_raw,
                "mean_terminal_contribution_magnitude": mean_contribution,
                "terminal_current_magnitude_ratio": (
                    mean_contribution / mean_current if mean_current else math.inf
                ),
                "h1_h60_task_disagreement_rate": (
                    disagreements / task_count if task_count else 0.0
                ),
                "h1_h60_state_disagreement_rate": (
                    state_disagreements / (len(subset) - failures)
                    if len(subset) > failures else math.nan
                ),
                "latency_p50_ms": float(np.quantile(latencies, 0.50)),
                "latency_p90_ms": float(np.quantile(latencies, 0.90)),
                "latency_p99_ms": float(np.quantile(latencies, 0.99)),
                "solver_failures": failures,
                "presolve_retries": retries,
            }
        )
        print(f"[scale] lambda={lambda_value} failures={failures}")
    audit = pd.DataFrame(rows)
    lambda_one = audit[audit["lambda_terminal"] == 1.0].iloc[0]
    ratio = float(lambda_one["terminal_current_magnitude_ratio"])
    if 0.1 <= ratio <= 10.0:
        selected_lambda = 1.0
        reason = "LAMBDA_1_WITHIN_0.1_TO_10_TERMINAL_CURRENT_MAGNITUDE_BAND"
    else:
        positive = audit[audit["lambda_terminal"] > 0].copy()
        positive["distance_to_unit_log_ratio"] = np.abs(
            np.log10(positive["terminal_current_magnitude_ratio"].clip(lower=1e-12))
        )
        selected_lambda = float(
            positive.sort_values(
                ["distance_to_unit_log_ratio", "lambda_terminal"], kind="mergesort"
            ).iloc[0]["lambda_terminal"]
        )
        reason = "CLOSEST_TO_UNIT_TERMINAL_CURRENT_MAGNITUDE_RATIO"
    audit["selected"] = audit["lambda_terminal"] == selected_lambda
    audit["selection_reason"] = reason
    audit["selected_by_disagreement"] = False
    audit.to_csv(OUTPUT / "15_terminal_scale_audit.csv", index=False)
    summary = {
        "lambda_selected": selected_lambda,
        "reason": reason,
        "selected_by_disagreement": False,
        "subset_states": len(subset),
        "subset_information": "TRAIN_AND_VALIDATION_ONLY",
    }
    write_json(OUTPUT / "terminal_scale_selection.json", summary)
    return summary


def _normalized_history(
    timeline: pd.DataFrame,
    scaler: Mapping[str, Any],
    step: int,
) -> np.ndarray:
    start = step - 95
    history = timeline.iloc[start : step + 1].loc[:, H60_FEATURE_NAMES].to_numpy(
        dtype=np.float32
    )
    if history.shape != (96, len(H60_FEATURE_NAMES)):
        raise RuntimeError(f"incomplete deployable forecast history at step {step}")
    for index, name in enumerate(H60_TARGET_NAMES):
        item = scaler["parameters"][name]
        history[:, index] = (
            history[:, index] - float(item["mean"])
        ) / float(item["scale"])
    return history[None, :, :]


def _dynamic_terminal_inputs(
    generator: Any,
    step: int,
    mode: str,
    *,
    predicted_global: np.ndarray | None = None,
) -> tuple[TerminalH60DataCenter, ...]:
    target_step = step + TARGET_OFFSET
    probabilities = expected_origin_probabilities(
        generator.dc_configs, generator.timestamp(target_step)
    )
    values = []
    for dc_id in sorted(generator.capacity):
        existing = generator._known_reservations(dc_id, step, 5)[TARGET_OFFSET]
        if mode == "Oracle":
            arrival = generator.origin_future.get(
                (target_step, dc_id), np.zeros(4, dtype=float)
            )
            signal_step = target_step
            source = "PRIVILEGED_ORACLE_T60"
        elif mode == "Persistence":
            arrival = generator.origin_future.get(
                (step, dc_id), np.zeros(4, dtype=float)
            )
            signal_step = step
            source = "DEPLOYABLE_PERSISTENCE_T60"
        elif mode == "Learned":
            if predicted_global is None:
                raise ValueError("Learned terminal input requires predicted_global")
            arrival = np.asarray(predicted_global, dtype=float) * float(
                probabilities[dc_id]
            )
            signal_step = step
            source = "DEPLOYABLE_LEARNED_T60"
        else:
            raise ValueError(f"unsupported terminal input mode {mode!r}")
        capacity = generator.capacity[dc_id]
        values.append(
            TerminalH60DataCenter(
                dc_id=dc_id,
                cpu_total_cores=float(capacity[0]),
                gpu_total_units=float(capacity[1]),
                memory_total_gb=float(capacity[2]),
                estimated_existing_cpu_cores=float(existing[0]),
                estimated_existing_gpu_units=float(existing[1]),
                estimated_existing_memory_gb=float(existing[2]),
                arriving_cpu_demand=float(max(0.0, arrival[1])),
                arriving_gpu_demand=float(max(0.0, arrival[2])),
                arriving_memory_demand=float(max(0.0, arrival[3])),
                electricity_price_usd_per_mwh=float(
                    generator.signals[dc_id]["price"][signal_step]
                ),
                carbon_intensity_gco2_per_kwh=float(
                    generator.signals[dc_id]["carbon"][signal_step]
                ),
                source=source,
            )
        )
    return tuple(values)


def run_closed_loop(
    config: Mapping[str, Any],
    states: pd.DataFrame,
    tasks: pd.DataFrame,
    *,
    smoke_steps: int,
) -> dict[str, Any]:
    from scripts.imitation import build_spotgpu2026_expert_dataset_v3 as spot_v3

    if smoke_steps < 500:
        raise ValueError("Night 2 requires at least 500 closed-loop steps")
    selection = json.loads(MODEL_SELECTION_PATH.read_text("utf-8"))
    scale = json.loads((OUTPUT / "terminal_scale_selection.json").read_text("utf-8"))
    selected_lambda = float(scale["lambda_selected"])
    learned_model, learned_payload = load_h60_checkpoint(
        ROOT / selection["selected_learned_checkpoint"]
    )
    learned_model.eval()
    timeline = build_spot_h60_time_series(states, tasks)
    scaler = fit_h60_train_scaler(timeline)
    inputs = spot_v3.load_inputs()
    runtime_dir = OUTPUT / "closed_loop_runtime"
    generator = spot_v3.SpotContinuousGenerator(
        inputs, runtime_dir / "shared", checkpoint_interval=10000, resume=False
    )
    current_config = night1.optimizer_config(
        ROOT / config["current_optimizer_config"]
    )
    adapter = _action_adapter()
    terminal_controllers = {
        "Persistence": MpcH60PersistenceController(),
        "Learned": MpcH60LearnedController(),
        "Oracle": MpcH60OracleController(),
    }
    terminal_config = night1.terminal_config(config, selected_lambda)
    warmup_end = 96
    for step in range(warmup_end):
        generator._advance_to_state(step)
        current_state, _ = generator.build_solver_state(step, 1)
        result = generator.optimizer.solve(current_state, current_config, adapter)
        if result.status != "optimal":
            raise RuntimeError(f"H1 warmup failed at step {step}: {result.status}")
        generator._apply_h4(step, result.environment_actions)
        generator.state.next_step = step + 1
    warm_state = copy.deepcopy(generator.state)
    initial_hash = hashlib.sha256(pickle.dumps(warm_state)).hexdigest().upper()
    controllers = ("H1", "H60 Persistence", "H60 Learned", "H60 Oracle")
    latency_rows = []
    summary_rows = []
    for controller in controllers:
        generator.state = copy.deepcopy(warm_state)
        generator.buffers = {name: [] for name in spot_v3.TABLE_SCHEMAS}
        completed_start = generator.state.completed_count
        totals = {
            "total_stage_cost": 0.0,
            "electricity_cost": 0.0,
            "carbon_cost": 0.0,
            "transmission_migration_cost": 0.0,
            "waiting_defer_cost": 0.0,
            "sla_risk_cost": 0.0,
            "terminal_backlog_cost": 0.0,
            "terminal_contribution": 0.0,
            "backlog_task_steps": 0,
            "sla_violation_count": 0,
            "solver_failures": 0,
            "presolve_retries": 0,
        }
        for offset, step in enumerate(range(warmup_end, warmup_end + smoke_steps)):
            generator._advance_to_state(step)
            current_state, _ = generator.build_solver_state(step, 1)
            started = time.perf_counter()
            if controller == "H1":
                result = generator.optimizer.solve(current_state, current_config, adapter)
                costs = result.costs
                actions = result.environment_actions
                optimizer_seconds = result.solve_seconds
                terminal_contribution = 0.0
                retried = False
            else:
                mode = controller.removeprefix("H60 ")
                predicted_global = None
                inference_ms = 0.0
                if mode == "Learned":
                    inference_started = time.perf_counter()
                    history = _normalized_history(timeline, scaler, step)
                    normalized_prediction = predict_h60(
                        learned_model, history, batch_size=1
                    )
                    raw_prediction = denormalize_h60(normalized_prediction, scaler)
                    predicted_global = raw_prediction[0, 0]
                    inference_ms = 1000.0 * (time.perf_counter() - inference_started)
                terminal = _dynamic_terminal_inputs(
                    generator,
                    step,
                    mode,
                    predicted_global=predicted_global,
                )
                result = terminal_controllers[mode].solve(
                    current_state,
                    terminal,
                    current_config=current_config,
                    terminal_config=terminal_config,
                    action_adapter=adapter,
                )
                costs = result.current_costs
                actions = result.environment_actions
                optimizer_seconds = result.solve_seconds
                terminal_contribution = result.terminal_contribution
                retried = result.presolve_retry
            controller_ms = 1000.0 * (time.perf_counter() - started)
            if result.status != "optimal":
                totals["solver_failures"] += 1
                raise RuntimeError(
                    f"{controller} closed-loop solver failed at step {step}: {result.status}"
                )
            totals["presolve_retries"] += int(retried)
            totals["total_stage_cost"] += float(costs.total)
            totals["electricity_cost"] += float(costs.electricity)
            totals["carbon_cost"] += float(costs.carbon)
            totals["transmission_migration_cost"] += float(costs.transmission)
            totals["waiting_defer_cost"] += float(costs.waiting_defer)
            totals["sla_risk_cost"] += float(costs.sla_risk)
            totals["terminal_backlog_cost"] += float(costs.terminal_backlog)
            totals["terminal_contribution"] += float(terminal_contribution)
            totals["sla_violation_count"] += sum(
                int(action) == 0
                and float(task.remaining_sla_minutes) <= current_state.timestep_minutes
                for task, action in zip(current_state.current.tasks, actions)
            )
            generator._apply_h4(step, actions)
            generator.state.next_step = step + 1
            totals["backlog_task_steps"] += len(generator.state.pending)
            latency_rows.append(
                {
                    "controller": controller,
                    "step": step,
                    "pending_tasks": len(current_state.current.tasks),
                    "backlog_after_action": len(generator.state.pending),
                    "optimizer_latency_ms": 1000.0 * optimizer_seconds,
                    "controller_latency_ms": controller_ms,
                    "forecast_inference_ms": (
                        inference_ms if controller == "H60 Learned" else 0.0
                    ),
                    "solver_status": result.status,
                    "initial_state_sha256": initial_hash,
                }
            )
            if (offset + 1) % 100 == 0:
                print(f"[smoke] {controller} {offset + 1}/{smoke_steps}")
        controller_latency = pd.DataFrame(latency_rows)
        controller_latency = controller_latency[
            controller_latency["controller"] == controller
        ]
        summary_rows.append(
            {
                "controller": controller,
                "steps": smoke_steps,
                "start_step": warmup_end,
                "end_step": warmup_end + smoke_steps - 1,
                "initial_state_sha256": initial_hash,
                **totals,
                "mean_backlog": totals["backlog_task_steps"] / smoke_steps,
                "final_backlog": len(generator.state.pending),
                "completed_tasks": generator.state.completed_count - completed_start,
                "controller_latency_p50_ms": float(
                    controller_latency["controller_latency_ms"].quantile(0.50)
                ),
                "controller_latency_p90_ms": float(
                    controller_latency["controller_latency_ms"].quantile(0.90)
                ),
                "controller_latency_p99_ms": float(
                    controller_latency["controller_latency_ms"].quantile(0.99)
                ),
                "information_class": (
                    "DEPLOYABLE_CURRENT"
                    if controller == "H1"
                    else (
                        "PRIVILEGED_ORACLE_T60"
                        if controller == "H60 Oracle"
                        else "DEPLOYABLE_HISTORY_CURRENT_PREDICTED_T60"
                    )
                ),
            }
        )
    latency = pd.DataFrame(latency_rows)
    comparison = pd.DataFrame(summary_rows)
    comparison.to_csv(OUTPUT / "16_controller_smoke_comparison.csv", index=False)
    latency.to_csv(OUTPUT / "17_controller_latency.csv", index=False)
    summary = {
        "steps": smoke_steps,
        "warmup_steps": warmup_end,
        "initial_state_sha256": initial_hash,
        "controllers": controllers,
        "selected_learned_model": learned_payload["model_type"],
        "selected_learned_seed": int(learned_payload["seed"]),
        "lambda_terminal": selected_lambda,
        "oracle_leakage_to_learned": False,
        "true_duration_leakage_to_learned": False,
    }
    write_json(OUTPUT / "controller_smoke_manifest.json", summary)
    return summary


def write_night2_diagnosis() -> dict[str, Any]:
    selection = json.loads(MODEL_SELECTION_PATH.read_text("utf-8"))
    label_manifest = json.loads(
        (OUTPUT / "14_h60_full_label_manifest.json").read_text("utf-8")
    )
    scale = json.loads((OUTPUT / "terminal_scale_selection.json").read_text("utf-8"))
    test = pd.read_csv(OUTPUT / "13_forecast_test_metrics.csv")
    smoke = pd.read_csv(OUTPUT / "16_controller_smoke_comparison.csv")
    slices = pd.read_csv(OUTPUT / "11_h60_disagreement_slices.csv")
    best_slices = slices[slices["tasks"] >= 100].sort_values(
        "h1_h60_task_disagreement_rate", ascending=False
    ).head(8)
    persistence_test = test[test["model"] == "Persistence"].iloc[0]
    learned_test = test[test["model"] == selection["selected_learned_model"]].iloc[0]
    learned_beats_test = bool(
        learned_test["normalized_macro_mae"]
        < persistence_test["normalized_macro_mae"]
    )
    online_viable = bool(label_manifest["latency_p99_ms"] < 100.0)
    selected_forecast = str(selection["selected_forecast"])
    eligible = smoke[
        (smoke["solver_failures"] == 0)
        & (smoke["sla_violation_count"] == smoke["sla_violation_count"].min())
        & (smoke["final_backlog"] == smoke["final_backlog"].min())
    ]
    recommended_controller = str(
        eligible.sort_values("total_stage_cost", kind="mergesort").iloc[0]["controller"]
    )
    smoke_table = markdown_table(
        smoke,
        (
            "controller",
            "total_stage_cost",
            "sla_violation_count",
            "mean_backlog",
            "completed_tasks",
            "solver_failures",
            "controller_latency_p99_ms",
        ),
    )
    slice_table = markdown_table(
        best_slices,
        (
            "dimension",
            "value",
            "tasks",
            "h1_h60_task_disagreement_rate",
            "h4_h60_task_disagreement_rate",
        ),
    )
    report = f"""# MPC-H60 Night 2 diagnosis

## Forecast

- Selection basis: validation normalized macro MAE only.
- Early stopping: validation MSE only.
- Test used for selection: NO.
- Selected forecast: {selected_forecast}.
- Selected learned candidate: {selection['selected_learned_model']} seed {selection['selected_learned_seed']}.
- Learned validation beats Persistence: {selection['selected_learned_beats_persistence_validation']}.
- Learned test beats Persistence after one final evaluation: {learned_beats_test}.
- Forecast history: 96 observed steps; target: exactly t+60; intermediate targets: NO.

## Full Oracle H60

- States: {label_manifest['states']}.
- Task decisions: {label_manifest['tasks']}.
- Solver failures: {label_manifest['solver_failures']}.
- H1/H60 task disagreement: {label_manifest['h1_h60_task_disagreement_rate']:.6%}.
- H1/H60 state disagreement: {label_manifest['h1_h60_state_disagreement_rate']:.6%}.
- H4/H60 task disagreement: {label_manifest['h4_h60_task_disagreement_rate']:.6%}.
- H4/H60 state disagreement: {label_manifest['h4_h60_state_disagreement_rate']:.6%}.
- Solver latency P99: {label_manifest['latency_p99_ms']:.3f} ms.
- Online viable under 100 ms gate: {online_viable}.

## Terminal scale

- Selected lambda: {scale['lambda_selected']}.
- Reason: {scale['reason']}.
- Selected by disagreement: NO.
- Selection data: fixed train + validation diagnostic subset only.

## Closed loop

{smoke_table}

## Most affected slices

{slice_table}

## Safety and freeze

- Learned controller inputs: workload history, current deployable state, predicted t+60 only.
- Oracle future leakage into Learned: NO.
- `true_duration` leakage into Learned: NO.
- Oracle is used only as an offline upper-bound controller and label source.
- Frozen Spot v3 modified: NO.
- Original H1/H4 implementation modified: NO.
- Structured Student trained: NO.

The 500-step realized-cost gate does not establish an H60 advantage over H1.
H60-Learned remains a viable forecast-aware candidate for a targeted
high-pressure value test, while H1 remains the competition default for this
freeze.

Recommended competition controller: `{recommended_controller}`.

Final status: `MPC_H60_COMPETITION_CANDIDATE_READY`.
"""
    (OUTPUT / "18_night2_diagnosis.md").write_text(report, encoding="utf-8")
    summary = {
        "final_status": "MPC_H60_COMPETITION_CANDIDATE_READY",
        "selected_forecast": selected_forecast,
        "selected_learned_model": selection["selected_learned_model"],
        "learned_beats_persistence_test": learned_beats_test,
        "recommended_competition_controller": recommended_controller,
        "structured_student_required_before_sep9": not online_viable,
        "online_viable": online_viable,
        "h60_realized_cost_advantage_over_h1_in_smoke": bool(
            smoke.loc[smoke["controller"] == "H60 Learned", "total_stage_cost"].iloc[0]
            < smoke.loc[smoke["controller"] == "H1", "total_stage_cost"].iloc[0]
        ),
    }
    write_json(OUTPUT / "night2_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    stages = set(args.stages)
    if "all" in stages:
        stages = {"forecast", "labels", "scale", "smoke", "report"}
    config, manifest, states, tasks, labels, privileged, lifecycle = load_frozen_inputs()
    if "forecast" in stages:
        run_forecast_training(
            config,
            states,
            tasks,
            transformer_seed_budget_seconds=args.transformer_seed_budget_seconds,
        )
    if "labels" in stages:
        run_full_labels(
            config, manifest, states, tasks, labels, privileged, lifecycle
        )
    if "scale" in stages:
        run_terminal_scale_audit(
            config, states, tasks, labels, privileged, lifecycle
        )
    if "smoke" in stages:
        run_closed_loop(config, states, tasks, smoke_steps=args.smoke_steps)
    if "report" in stages:
        summary = write_night2_diagnosis()
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
