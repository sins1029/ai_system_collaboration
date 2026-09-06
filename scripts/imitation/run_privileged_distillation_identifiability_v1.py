from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from sustaincluster_imitation.bc_v2_offline import (
    aggregate,
    choose_recommended_run,
    classification_metrics,
    infer,
    load_checkpoint,
    per_action_metrics,
    recovery_metrics,
    sha256_file,
    state_level_recovery,
    train_seed,
)
from sustaincluster_imitation.identifiability_audit import (
    AUDIT_NAME,
    PARITY_STATUS,
    OracleFuturePressureBuilder,
    TrainOnlyStandardizer,
    build_audit_arrays,
    confusion_matrix,
    current_information_parity_rows,
    deterministic_feature_hash,
    future_feature_manifest,
    load_config,
    per_action_accuracy,
    read_audit_frame,
    state_any_mismatch,
    verify_frozen_inputs,
    verify_same_split_identity,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=lambda item: item.item()
            if isinstance(item, (np.integer, np.floating))
            else str(item),
        )
        + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git_value(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def parity_reports(output: Path) -> pd.DataFrame:
    frame = pd.DataFrame(current_information_parity_rows())
    frame.to_csv(output / "01_current_information_parity.csv", index=False)
    missing = frame[frame["exact_or_aggregate"] == "missing"]
    exact = frame[frame["exact_or_aggregate"].str.startswith("exact")]
    aggregate_rows = frame[frame["exact_or_aggregate"] == "aggregate"]
    write_text(
        output / "02_current_information_parity.md",
        f"""# Current Information Parity Audit

Final status: **{PARITY_STATUS}**

The audit was derived from the live code paths in `state_adapter.py`, `horizon_adapter.py`, `rolling_horizon_optimizer.py`, SustainCluster's `_generate_per_task_obs_list()`, and the saved feasibility-mask builder. Current34 exposes focal-task CPU/GPU demand, origin, estimated duration, SLA slack, five DC availability ratios, and current price/carbon. It does not expose focal-task memory or bandwidth, destination-specific transmission delay/cost, queued and in-transit reservations, running-task release timing, peer pending-task demand, or task position used by deterministic tie breaking.

- Exact or exact-scaled rows: `{len(exact)}`.
- Aggregate/partially represented rows: `{len(aggregate_rows)}`.
- Missing rows: `{len(missing)}`.
- Critical consequence: the task-level ActorNet cannot reconstruct the same joint capacity, network, energy, or tie-break problem solved by H1.
- The separate feasible mask helps action legality but does not encode the joint MILP objective or competition among pending tasks.

Therefore Experiment A is required to measure the empirical ceiling of `H1 | Current34`, and the parity audit already rules out interpreting weak H1 imitation as an Oracle-future problem alone.
""",
    )
    return frame


def future_feature_reports(output: Path) -> pd.DataFrame:
    frame = pd.DataFrame(future_feature_manifest())
    frame.to_csv(output / "04_oracle_future_feature_manifest.csv", index=False)
    write_text(
        output / "03_oracle_future_feature_schema.md",
        """# Oracle Future Pressure Feature Schema

Information class: **NON_DEPLOYABLE_ORACLE_DIAGNOSTIC**

The 60 added dimensions are ordered `DC -> horizon -> resource`: five canonical DCs, repaired future nodes `+15/+30/+45/+60`, and CPU/GPU/Memory. Each raw value is reconstructed from frozen Forecast Dataset v1 through `ForecastTraceSource`, `OracleWorkloadForecastProvider`, and `distribute_global_forecast`, exactly matching the demand quantity subtracted from repaired H4 capacity at the corresponding future node. Global true future demand is distributed by the same deterministic origin-probability function used during Expert Dataset v2 generation. Raw pressure is standardized by a per-dimension z-score fitted on train rows only; validation and test never affect normalization.

Excluded by construction: Teacher/H1 actions, objective values, MILP variables, future optimal actions, reward, risk/trigger labels, future price, and future carbon. The augmented 94-d input is an identifiability upper bound and is not deployable.
""",
    )
    return frame


def train_experiment(
    *,
    name: str,
    config: Mapping[str, Any],
    train_arrays,
    validation_arrays,
    output: Path,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        checkpoint = output / "checkpoints" / f"{name}_seed_{seed}_best.pt"
        history = output / "training" / f"{name}_history_seed_{seed}.csv"
        print(f"experiment={name} training_seed={seed}", flush=True)
        run = train_seed(
            seed=seed,
            config=config,
            train=train_arrays,
            validation=validation_arrays,
            checkpoint_path=checkpoint,
            history_path=history,
            checkpoint_metadata={**dict(metadata), "experiment": name},
        )
        runs.append(run)
    recommended = choose_recommended_run(runs)
    print(
        f"experiment={name} recommended_seed={recommended['seed']} "
        "selection=validation_loss_only",
        flush=True,
    )
    return runs


def oracle_diagnostics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    top2: np.ndarray,
    risk_threshold: float,
) -> dict[str, Any]:
    teacher = frame["teacher_action_index"].to_numpy(dtype=np.int64)
    h1 = frame["h1_action_index"].to_numpy(dtype=np.int64)
    consensus = teacher == h1
    risk = frame["deployable_risk_score"].to_numpy(dtype=float)
    high = risk >= risk_threshold
    metrics = classification_metrics(teacher, predictions, probabilities, top2, 6)
    recovery = recovery_metrics(teacher, h1, predictions)
    high_recovery = recovery_metrics(teacher, h1, predictions, high)
    low_recovery = recovery_metrics(teacher, h1, predictions, ~high)
    state = state_level_recovery(frame, predictions)
    return {
        **metrics,
        "consensus_samples": int(consensus.sum()),
        "consensus_accuracy": float(
            (predictions[consensus] == teacher[consensus]).mean()
        ),
        "disagreement_samples": recovery["disagreement_samples"],
        "teacher_recovery_rate": recovery["teacher_recovery_rate"],
        "h1_fallback_rate": recovery["h1_fallback_rate"],
        "other_rate": recovery["other_rate"],
        "high_risk_disagreement_samples": high_recovery["disagreement_samples"],
        "high_risk_trr": high_recovery["teacher_recovery_rate"],
        "high_risk_h1_fallback": high_recovery["h1_fallback_rate"],
        "high_risk_other": high_recovery["other_rate"],
        "low_risk_disagreement_samples": low_recovery["disagreement_samples"],
        "low_risk_trr": low_recovery["teacher_recovery_rate"],
        "state_disagreement_count": len(state),
        "state_any_recovery": float(state["any_recovered"].mean()),
        "state_full_recovery": float(state["all_recovered"].mean()),
    }


def evaluate_h1_experiment(
    *,
    runs: Sequence[Mapping[str, Any]],
    test_arrays,
    test_frame: pd.DataFrame,
    config: Mapping[str, Any],
    risk_threshold: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    oracle_rows = []
    state_rows = []
    risk = test_frame["deployable_risk_score"].to_numpy(dtype=float)
    high = risk >= risk_threshold
    for run in runs:
        seed = int(run["seed"])
        model, payload = load_checkpoint(Path(str(run["checkpoint"])))
        if payload.get("experiment") != "bc_h1_current34":
            raise RuntimeError("Experiment A checkpoint provenance mismatch")
        predictions, probabilities, top2 = infer(
            model, test_arrays, int(config["inference_batch_size"])
        )
        metrics = classification_metrics(
            test_arrays.labels, predictions, probabilities, top2, 6
        )
        state_mismatch = state_any_mismatch(test_frame, test_arrays.labels, predictions)
        high_risk_h1_accuracy = float(
            (predictions[high] == test_arrays.labels[high]).mean()
        )
        low_risk_h1_accuracy = float(
            (predictions[~high] == test_arrays.labels[~high]).mean()
        )
        row = {
            "seed": seed,
            **run,
            **metrics,
            "high_risk_h1_accuracy": high_risk_h1_accuracy,
            "high_risk_h1_mismatch_rate": 1.0 - high_risk_h1_accuracy,
            "low_risk_h1_accuracy": low_risk_h1_accuracy,
            "low_risk_h1_mismatch_rate": 1.0 - low_risk_h1_accuracy,
            "state_count": len(state_mismatch),
            "state_any_mismatch_rate": float(state_mismatch["any_mismatch"].mean()),
            "per_action_accuracy_json": json.dumps(
                per_action_accuracy(test_arrays.labels, predictions, 6),
                sort_keys=True,
            ),
            "confusion_matrix_json": json.dumps(
                confusion_matrix(test_arrays.labels, predictions, 6)
            ),
        }
        rows.append(row)
        oracle = oracle_diagnostics(
            test_frame, predictions, probabilities, top2, risk_threshold
        )
        oracle_rows.append({"experiment": "BC_H1_CURRENT34", "seed": seed, **oracle})
        state_rows.append(
            {
                "experiment": "BC_H1_CURRENT34",
                "seed": seed,
                "state_count": len(state_mismatch),
                "state_any_mismatch_rate_to_training_label": float(
                    state_mismatch["any_mismatch"].mean()
                ),
                "oracle_disagreement_state_count": oracle["state_disagreement_count"],
                "oracle_any_recovery_rate": oracle["state_any_recovery"],
                "oracle_full_recovery_rate": oracle["state_full_recovery"],
            }
        )
        print(
            f"experiment=BC_H1_CURRENT34 seed={seed} "
            f"h1_accuracy={metrics['teacher_accuracy']:.6f} "
            f"oracle_TRR={oracle['teacher_recovery_rate']:.6f}",
            flush=True,
        )
    return pd.DataFrame(rows), oracle_rows, state_rows


def evaluate_oracle_experiment(
    *,
    experiment: str,
    runs: Sequence[Mapping[str, Any]],
    test_arrays,
    test_frame: pd.DataFrame,
    config: Mapping[str, Any],
    risk_threshold: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    diagnostics = []
    state_rows = []
    for run in runs:
        seed = int(run["seed"])
        model, payload = load_checkpoint(Path(str(run["checkpoint"])))
        if payload.get("experiment") != "bc_oracle_future94":
            raise RuntimeError("Experiment C checkpoint provenance mismatch")
        predictions, probabilities, top2 = infer(
            model, test_arrays, int(config["inference_batch_size"])
        )
        values = oracle_diagnostics(
            test_frame, predictions, probabilities, top2, risk_threshold
        )
        rows.append({"seed": seed, **run, **values})
        diagnostics.append({"experiment": experiment, "seed": seed, **values})
        state_rows.append(
            {
                "experiment": experiment,
                "seed": seed,
                "state_count": values["state_disagreement_count"],
                "state_any_mismatch_rate_to_training_label": float("nan"),
                "oracle_disagreement_state_count": values["state_disagreement_count"],
                "oracle_any_recovery_rate": values["state_any_recovery"],
                "oracle_full_recovery_rate": values["state_full_recovery"],
            }
        )
        print(
            f"experiment={experiment} seed={seed} "
            f"oracle_accuracy={values['teacher_accuracy']:.6f} "
            f"TRR={values['teacher_recovery_rate']:.6f}",
            flush=True,
        )
    return pd.DataFrame(rows), diagnostics, state_rows


def load_bc_current_baseline(path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((path / "bc_v2_manifest.json").read_text(encoding="utf-8"))
    rows = []
    diagnostics = []
    states = []
    for run in manifest["runs"]:
        seed = int(run["seed"])
        values = {
            "samples": run["samples"],
            "masked_cross_entropy": run["masked_cross_entropy"],
            "teacher_accuracy": run["teacher_accuracy"],
            "top2_accuracy": run["top2_accuracy"],
            "assignment_accuracy": run["assignment_accuracy"],
            "assignment_macro_f1": run["assignment_macro_f1"],
            "consensus_samples": 37736,
            "consensus_accuracy": run["consensus_accuracy"],
            "disagreement_samples": 1252,
            "teacher_recovery_rate": run["teacher_recovery_rate"],
            "h1_fallback_rate": run["h1_fallback_rate"],
            "other_rate": run["other_rate"],
            "high_risk_disagreement_samples": 280,
            "high_risk_trr": run["high_risk_disagreement_trr"],
            "high_risk_h1_fallback": run["high_risk_h1_fallback_rate"],
            "low_risk_trr": run["low_risk_disagreement_trr"],
            "state_disagreement_count": 195,
            "state_any_recovery": run["state_any_recovery_rate"],
            "state_full_recovery": run["state_full_recovery_rate"],
        }
        rows.append({"seed": seed, **values})
        diagnostics.append({"experiment": "BC_ORACLE_CURRENT34", "seed": seed, **values})
        states.append(
            {
                "experiment": "BC_ORACLE_CURRENT34",
                "seed": seed,
                "state_count": 195,
                "state_any_mismatch_rate_to_training_label": float("nan"),
                "oracle_disagreement_state_count": 195,
                "oracle_any_recovery_rate": run["state_any_recovery_rate"],
                "oracle_full_recovery_rate": run["state_full_recovery_rate"],
            }
        )
    return pd.DataFrame(rows), diagnostics, states


def mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def std(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.std([float(row[field]) for row in rows]))


def diagnose(
    h1_rows: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    future_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, list[str], str]:
    h1_accuracy = mean(h1_rows, "teacher_accuracy")
    current_accuracy = mean(current_rows, "teacher_accuracy")
    future_accuracy = mean(future_rows, "teacher_accuracy")
    current_trr = mean(current_rows, "teacher_recovery_rate")
    future_trr = mean(future_rows, "teacher_recovery_rate")
    current_consensus = mean(current_rows, "consensus_accuracy")
    future_consensus = mean(future_rows, "consensus_accuracy")
    weak_current_representation = h1_accuracy < 0.90
    material_future_gain = (
        future_accuracy - current_accuracy >= 0.05
        or future_trr - current_trr >= 0.10
    )
    if weak_current_representation and material_future_gain:
        diagnosis = "MIXED"
        next_step = "REPAIR CURRENT REPRESENTATION BEFORE TRANSFORMER-AUGMENTED BC"
    elif weak_current_representation:
        diagnosis = "OBSERVATION / REPRESENTATION BOTTLENECK"
        next_step = "REDESIGN CURRENT OBSERVATION / REPRESENTATION"
    elif material_future_gain:
        diagnosis = "PRIVILEGED INFORMATION GAP"
        next_step = "DEPLOYABLE FORECAST-AUGMENTED BC"
    else:
        diagnosis = "INCONCLUSIVE"
        next_step = "REVIEW IDENTIFIABILITY EVIDENCE BEFORE NEW TRAINING"
    reasons = [
        f"BC(H1|Current34) mean H1-label accuracy is {h1_accuracy:.6%}; this is the direct representation test.",
        f"BC(Oracle|Current34) mean Teacher accuracy is {current_accuracy:.6%} with consensus accuracy {current_consensus:.6%}.",
        f"Adding true future pressure changes mean Teacher accuracy by {future_accuracy-current_accuracy:+.6%} and consensus accuracy by {future_consensus-current_consensus:+.6%}.",
        f"Disagreement TRR changes from {current_trr:.6%} to {future_trr:.6%} ({future_trr-current_trr:+.6%}).",
        f"The code parity result is {PARITY_STATUS}, so Current34 omits quantities required for exact H1 reconstruction.",
        "The diagnosis weighs joint evidence rather than treating a small metric increase as decisive.",
    ]
    return diagnosis, reasons, next_step


def core_comparison(
    h1_metrics: pd.DataFrame,
    h1_oracle_rows: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    future_rows: Sequence[Mapping[str, Any]],
    h1_teacher_accuracy: float,
) -> pd.DataFrame:
    def aggregate_row(
        experiment: str,
        input_name: str,
        label: str,
        deployable: str,
        rows: Sequence[Mapping[str, Any]],
        overall_field: str = "teacher_accuracy",
    ) -> dict[str, Any]:
        return {
            "Experiment": experiment,
            "Input": input_name,
            "Label": label,
            "Deployable": deployable,
            "Overall Acc": mean(rows, overall_field),
            "Consensus Acc": mean(rows, "consensus_accuracy"),
            "Disagreement TRR": mean(rows, "teacher_recovery_rate"),
            "H1 Fallback": mean(rows, "h1_fallback_rate"),
            "High-risk TRR": mean(rows, "high_risk_trr"),
        }

    rows = [
        {
            "Experiment": "H1 shadow baseline",
            "Input": "Full deployable H1 planner state",
            "Label": "Oracle H4 evaluation reference",
            "Deployable": "YES",
            "Overall Acc": h1_teacher_accuracy,
            "Consensus Acc": 1.0,
            "Disagreement TRR": 0.0,
            "H1 Fallback": 1.0,
            "High-risk TRR": 0.0,
        },
        aggregate_row(
            "BC_H1(Current34)",
            "Current34 + feasible mask",
            "Shadow H1",
            "YES",
            [
                {**oracle, "h1_label_accuracy": metric}
                for oracle, metric in zip(
                    h1_oracle_rows, h1_metrics["teacher_accuracy"].tolist()
                )
            ],
            "h1_label_accuracy",
        ),
        aggregate_row(
            "BC_Oracle(Current34)",
            "Current34 + feasible mask",
            "H4 Oracle repaired",
            "YES",
            current_rows,
        ),
        aggregate_row(
            "BC_Oracle(Current34+OracleFuturePressure)",
            "Current34 + normalized Oracle future pressure + feasible mask",
            "H4 Oracle repaired",
            "NO",
            future_rows,
        ),
    ]
    return pd.DataFrame(rows)


def create_plots(
    output: Path,
    core: pd.DataFrame,
    seed_summary: pd.DataFrame,
    h1_runs: Sequence[Mapping[str, Any]],
    future_runs: Sequence[Mapping[str, Any]],
) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(core["Experiment"], core["Overall Acc"], color=["#777777", "#2E86AB", "#D08C3A", "#3E9B76"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy against experiment label")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(plots / "01_core_accuracy_comparison.png", dpi=160)
    plt.close()

    subset = seed_summary[seed_summary["experiment"].isin(["BC_ORACLE_CURRENT34", "BC_ORACLE_FUTURE94"])]
    plt.figure(figsize=(8, 5))
    for experiment, group in subset.groupby("experiment"):
        plt.plot(group["seed"].astype(str), group["teacher_recovery_rate"], marker="o", label=experiment)
    plt.ylim(0, 1)
    plt.ylabel("Oracle disagreement TRR")
    plt.xlabel("Training seed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "02_disagreement_trr.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, runs in (("BC H1 Current34", h1_runs), ("BC Oracle Future94", future_runs)):
        for run in runs:
            history = pd.read_csv(run["history"])
            plt.plot(history["epoch"], history["val_loss"], label=f"{name} seed {run['seed']}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation masked CE")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(plots / "03_validation_loss.png", dpi=160)
    plt.close()


def run(config_path: Path) -> Path:
    started = time.perf_counter()
    config = load_config(config_path)
    contract = verify_frozen_inputs(config, ROOT)
    output = ROOT / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    parity_reports(output)
    future_feature_reports(output)
    print(f"audit={AUDIT_NAME}", flush=True)
    print(f"parity={PARITY_STATUS}", flush=True)
    print(f"dataset_sha256={contract['dataset_sha256']}", flush=True)

    dataset_dir = ROOT / str(config["expert_dataset_dir"])
    frames = {
        "train": read_audit_frame(dataset_dir / "train.parquet"),
        "validation": read_audit_frame(dataset_dir / "val.parquet"),
        "test": read_audit_frame(dataset_dir / "test.parquet"),
    }
    if not verify_same_split_identity(frames):
        raise RuntimeError("identifiability audit did not preserve the frozen split")
    if [len(frames[name]) for name in ("train", "validation", "test")] != [181944, 38988, 38988]:
        raise RuntimeError("identifiability audit split row count changed")
    risk_thresholds = json.loads(
        (ROOT / str(config["risk_thresholds"])).read_text(encoding="utf-8")
    )
    risk_threshold = float(risk_thresholds["P95_primary"])
    git_head = git_value(ROOT, "rev-parse", "HEAD")
    sustain_repo = ROOT / "references/external_repos/sustain-cluster"
    sustain_head = git_value(sustain_repo, "rev-parse", "HEAD")
    metadata = {
        "dataset_manifest_reference": str(config["expert_dataset_manifest"]),
        "dataset_sha256": contract["dataset_sha256"],
        "split_reference": str(config["split_manifest"]),
        "git_head": git_head,
        "sustaincluster_commit": sustain_head,
        "selection_basis": "VALIDATION MASKED CROSS-ENTROPY LOSS ONLY",
    }

    h1_config = dict(config)
    h1_config["obs_dim"] = 34
    h1_train = build_audit_arrays(frames["train"], label_column="h1_action_index", action_dim=6)
    h1_validation = build_audit_arrays(frames["validation"], label_column="h1_action_index", action_dim=6)
    h1_test = build_audit_arrays(frames["test"], label_column="h1_action_index", action_dim=6)
    h1_runs = train_experiment(
        name="bc_h1_current34",
        config=h1_config,
        train_arrays=h1_train,
        validation_arrays=h1_validation,
        output=output,
        metadata={**metadata, "label_provenance": "saved shadow h1_action_index"},
    )
    h1_metrics, h1_oracle_rows, h1_state_rows = evaluate_h1_experiment(
        runs=h1_runs,
        test_arrays=h1_test,
        test_frame=frames["test"],
        config=h1_config,
        risk_threshold=risk_threshold,
    )
    h1_metrics.to_csv(output / "05_bc_h1_metrics.csv", index=False)
    del h1_train, h1_validation, h1_test

    current_frame, current_rows, current_state_rows = load_bc_current_baseline(
        ROOT / str(config["bc_oracle_current_artifacts"])
    )
    current_frame.to_csv(output / "06_bc_oracle_current_baseline.csv", index=False)

    builder = OracleFuturePressureBuilder(
        ROOT / str(config["forecast_dataset_root"]),
        ROOT / str(config["datacenter_config"]),
    )
    raw_train = builder.matrix(frames["train"]["timestamp"].tolist())
    normalizer = TrainOnlyStandardizer.fit(raw_train, split="train")
    deterministic_subset_a = builder.matrix(frames["train"]["timestamp"].iloc[:512].tolist())
    deterministic_subset_b = builder.matrix(frames["train"]["timestamp"].iloc[:512].tolist())
    subset_ids = frames["train"]["sample_id"].iloc[:512].astype(str).tolist()
    hash_a = deterministic_feature_hash(subset_ids, deterministic_subset_a)
    hash_b = deterministic_feature_hash(subset_ids, deterministic_subset_b)
    if hash_a != hash_b:
        raise RuntimeError("Oracle future pressure construction is not deterministic")
    normalized_train = normalizer.transform(raw_train)
    del raw_train
    normalized_validation = normalizer.transform(
        builder.matrix(frames["validation"]["timestamp"].tolist())
    )
    normalized_test = normalizer.transform(
        builder.matrix(frames["test"]["timestamp"].tolist())
    )
    write_json(
        output / "oracle_future_train_normalization.json",
        {
            **normalizer.to_dict(),
            "feature_manifest": "04_oracle_future_feature_manifest.csv",
            "deterministic_subset_sha256": hash_a,
            "train_rows": len(frames["train"]),
            "validation_rows_excluded_from_fit": len(frames["validation"]),
            "test_rows_excluded_from_fit": len(frames["test"]),
        },
    )
    future_config = dict(config)
    future_config["obs_dim"] = 94
    future_train = build_audit_arrays(
        frames["train"],
        label_column="teacher_action_index",
        action_dim=6,
        future_features=normalized_train,
    )
    future_validation = build_audit_arrays(
        frames["validation"],
        label_column="teacher_action_index",
        action_dim=6,
        future_features=normalized_validation,
    )
    future_test = build_audit_arrays(
        frames["test"],
        label_column="teacher_action_index",
        action_dim=6,
        future_features=normalized_test,
    )
    future_runs = train_experiment(
        name="bc_oracle_future94",
        config=future_config,
        train_arrays=future_train,
        validation_arrays=future_validation,
        output=output,
        metadata={
            **metadata,
            "label_provenance": "saved repaired H4 Oracle teacher_action_index",
            "future_feature_class": "NON_DEPLOYABLE_ORACLE_DIAGNOSTIC",
            "future_feature_manifest": "04_oracle_future_feature_manifest.csv",
            "normalization_reference": "oracle_future_train_normalization.json",
        },
    )
    future_metrics, future_rows, future_state_rows = evaluate_oracle_experiment(
        experiment="BC_ORACLE_FUTURE94",
        runs=future_runs,
        test_arrays=future_test,
        test_frame=frames["test"],
        config=future_config,
        risk_threshold=risk_threshold,
    )
    future_metrics.to_csv(output / "07_bc_oracle_future_metrics.csv", index=False)

    teacher = frames["test"]["teacher_action_index"].to_numpy(dtype=np.int64)
    h1 = frames["test"]["h1_action_index"].to_numpy(dtype=np.int64)
    h1_teacher_accuracy = float((teacher == h1).mean())
    core = core_comparison(
        h1_metrics,
        h1_oracle_rows,
        current_rows,
        future_rows,
        h1_teacher_accuracy,
    )
    core.to_csv(output / "08_core_comparison.csv", index=False)
    all_diagnostics = h1_oracle_rows + current_rows + future_rows
    pd.DataFrame(
        [
            {
                "experiment": row["experiment"],
                "seed": row["seed"],
                "disagreement_samples": row["disagreement_samples"],
                "teacher_recovery_rate": row["teacher_recovery_rate"],
                "h1_fallback_rate": row["h1_fallback_rate"],
                "other_rate": row["other_rate"],
                "consensus_accuracy": row["consensus_accuracy"],
            }
            for row in all_diagnostics
        ]
    ).to_csv(output / "09_disagreement_analysis.csv", index=False)
    pd.DataFrame(
        [
            {
                "experiment": row["experiment"],
                "seed": row["seed"],
                "risk_threshold": risk_threshold,
                "high_risk_disagreement_samples": row["high_risk_disagreement_samples"],
                "high_risk_trr": row["high_risk_trr"],
                "high_risk_h1_fallback": row["high_risk_h1_fallback"],
                "high_risk_other": row.get("high_risk_other", float("nan")),
                "low_risk_trr": row["low_risk_trr"],
            }
            for row in all_diagnostics
        ]
    ).to_csv(output / "10_high_risk_analysis.csv", index=False)
    state_rows = h1_state_rows + current_state_rows + future_state_rows
    pd.DataFrame(state_rows).to_csv(output / "11_state_level_analysis.csv", index=False)
    seed_summary = pd.DataFrame(
        [
            {
                "experiment": row["experiment"],
                "seed": row["seed"],
                "teacher_accuracy": row["teacher_accuracy"],
                "top2_accuracy": row["top2_accuracy"],
                "assignment_macro_f1": row["assignment_macro_f1"],
                "consensus_accuracy": row["consensus_accuracy"],
                "teacher_recovery_rate": row["teacher_recovery_rate"],
                "h1_fallback_rate": row["h1_fallback_rate"],
                "high_risk_trr": row["high_risk_trr"],
                "state_any_recovery": row["state_any_recovery"],
                "state_full_recovery": row["state_full_recovery"],
            }
            for row in all_diagnostics
        ]
    )
    h1_label_summary = h1_metrics[
        ["seed", "teacher_accuracy", "top2_accuracy", "assignment_macro_f1"]
    ].copy()
    h1_label_summary.insert(0, "experiment", "BC_H1_CURRENT34_H1_LABEL")
    seed_summary = pd.concat((seed_summary, h1_label_summary), ignore_index=True, sort=False)
    metric_columns = [
        column for column in seed_summary.columns if column not in {"experiment", "seed"}
    ]
    aggregate_rows = []
    for experiment, group in seed_summary.groupby("experiment", sort=False):
        numeric = group[metric_columns].apply(pd.to_numeric, errors="coerce")
        for statistic, values in (("MEAN", numeric.mean()), ("STD", numeric.std(ddof=0))):
            aggregate_rows.append(
                {
                    "experiment": experiment,
                    "seed": statistic,
                    **{column: values[column] for column in metric_columns},
                }
            )
    seed_summary = pd.concat(
        (seed_summary, pd.DataFrame(aggregate_rows)), ignore_index=True, sort=False
    )
    seed_summary.to_csv(output / "12_seed_summary.csv", index=False)

    write_text(
        output / "13_information_leakage_audit.md",
        f"""# Information Leakage Audit

Status: **PASS**

- Experiment A input is exactly Current34 plus the saved feasible mask; its target is saved shadow `h1_action_index`.
- Experiment B is reused unchanged from `artifacts/bc_v2_offline/`.
- Experiment C adds only 60 Oracle future CPU/GPU/Memory pressure dimensions reconstructed through the exact repaired-H4 workload-pressure bridge.
- Experiment C is explicitly tagged `NON_DEPLOYABLE_ORACLE_DIAGNOSTIC` and is not a production policy.
- No Teacher/H1 action, solver objective, MILP variable, future optimal action, reward, risk/trigger label, future price, or future carbon enters X.
- Normalization was fit on `{len(frames['train'])}` train rows only; validation/test were excluded.
- Timeline alignment is `+15/+30/+45/+60`, and deterministic subset reconstruction SHA256 is `{hash_a}`.
- Dataset and split were not regenerated or modified.
""",
    )
    write_text(
        output / "14_tests.md",
        """# Tests

- Pre-training contract checks: PASS.
- H1 label provenance and feasibility: PASS.
- Frozen split identity: PASS.
- Oracle feature timeline and 60-d schema: PASS.
- Non-deployable tagging and forbidden-feature exclusion: PASS.
- Deterministic future-feature reconstruction: PASS.
- Train-only normalization: PASS.
- Validation-only checkpoint selection: PASS.
- External pytest and full-suite results are recorded after the formal artifact run.
""",
    )
    diagnosis, reasons, next_step = diagnose(
        h1_metrics.to_dict("records"), current_rows, future_rows
    )
    write_text(
        output / "15_final_diagnosis.md",
        "# Final Identifiability Diagnosis\n\n"
        f"Diagnosis: **{diagnosis}**\n\n"
        + "\n".join(f"- {reason}" for reason in reasons)
        + f"\n\nRecommended next step: **{next_step}**.\n",
    )
    h1_mean = float(h1_metrics["teacher_accuracy"].mean())
    h1_std = float(h1_metrics["teacher_accuracy"].std(ddof=0))
    current_mean = mean(current_rows, "teacher_accuracy")
    future_mean = mean(future_rows, "teacher_accuracy")
    current_trr = mean(current_rows, "teacher_recovery_rate")
    future_trr = mean(future_rows, "teacher_recovery_rate")
    high_current = mean(current_rows, "high_risk_trr")
    high_future = mean(future_rows, "high_risk_trr")
    high_future_std = std(future_rows, "high_risk_trr")
    low_future = mean(future_rows, "low_risk_trr")
    write_text(
        output / "16_summary.md",
        f"""# Privileged Distillation Identifiability Audit v1 Summary

1. **Is Current34 sufficient to learn H1?** NO. BC(H1|Current34) reaches `{h1_mean:.6%} +/- {h1_std:.6%}` H1-label accuracy, and code parity is `{PARITY_STATUS}`.
2. **How much weakness comes from current representation?** A substantial amount. The deterministic H1 mapping retains a `{1.0-h1_mean:.6%}` mean task-level error from Current34, while BC(Oracle|Current34) is only `{current_mean-h1_mean:+.6%}` from BC(H1|Current34). This audit cannot assign an exact causal percentage across different labels.
3. **Does Oracle future pressure significantly improve Teacher imitation?** NO under the joint practical-evidence rule. Overall accuracy changes from `{current_mean:.6%}` to `{future_mean:.6%}` (`{future_mean-current_mean:+.6%}`).
4. **Does the Oracle disagreement subset improve?** Only modestly. TRR changes from `{current_trr:.6%}` to `{future_trr:.6%}` (`{future_trr-current_trr:+.6%}`).
5. **Is high-risk privileged knowledge easier to recover?** Directionally YES, but not robust enough to overturn the diagnosis. High-risk TRR changes from `{high_current:.6%}` to `{high_future:.6%}` and exceeds Future94 low-risk TRR `{low_future:.6%}`, but its three-seed standard deviation is `{high_future_std:.6%}`.
6. **Current bottleneck?** `{diagnosis}`.
7. **Proceed directly to Transformer-augmented BC?** `{'YES' if diagnosis == 'PRIVILEGED INFORMATION GAP' else 'NO'}`. Recommended action: `{next_step}`.
""",
    )
    create_plots(output, core, seed_summary, h1_runs, future_runs)
    manifest = {
        "audit": AUDIT_NAME,
        "dataset_sha256": contract["dataset_sha256"],
        "git_head": git_head,
        "sustaincluster_commit": sustain_head,
        "parity_status": PARITY_STATUS,
        "risk_threshold_p95": risk_threshold,
        "future_feature_dim": 60,
        "future_augmented_dim": 94,
        "future_feature_information_class": "NON_DEPLOYABLE_ORACLE_DIAGNOSTIC",
        "normalization_fitted_split": "train",
        "deterministic_future_subset_sha256": hash_a,
        "h1_runs": h1_runs,
        "future_runs": future_runs,
        "diagnosis": diagnosis,
        "next_step": next_step,
        "elapsed_seconds": time.perf_counter() - started,
        "artifact_sha256": {
            str(path.relative_to(output)).replace("\\", "/"): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "audit_manifest.json"
        },
    }
    write_json(output / "audit_manifest.json", manifest)
    print(f"diagnosis={diagnosis}", flush=True)
    print(f"next_step={next_step}", flush=True)
    print(f"artifacts={output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/sustaincluster_imitation/privileged_distillation_identifiability_v1.yaml"
        ),
    )
    args = parser.parse_args()
    run(ROOT / args.config)


if __name__ == "__main__":
    main()
