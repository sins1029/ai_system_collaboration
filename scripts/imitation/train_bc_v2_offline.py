from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sustaincluster_imitation.bc_v2_offline import (
    DATASET_NAME,
    PREDICTION_COLUMNS,
    actor_architecture,
    aggregate,
    build_actor,
    build_prediction_frame,
    choose_recommended_run,
    classification_metrics,
    infer,
    load_checkpoint,
    load_config,
    load_training_arrays,
    masked_cross_entropy,
    masked_logits,
    per_action_metrics,
    recovery_metrics,
    set_deterministic_seed,
    sha256_file,
    state_level_recovery,
    three_way_categories,
    train_seed,
    training_columns_are_deployable_only,
    validate_frozen_contract,
)


EVALUATION_COLUMNS = [
    "sample_id",
    "episode_id",
    "seed",
    "scenario",
    "step",
    "task_id",
    "origin_dc",
    "student_observation",
    "feasible_action_mask",
    "teacher_action_index",
    "h1_action_index",
    "deployable_risk_score",
    "teacher_is_local",
    "teacher_is_migration",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def split_metrics(
    teacher: np.ndarray,
    h1: np.ndarray,
    predictions: np.ndarray,
    top2: np.ndarray,
    selector: np.ndarray,
) -> dict[str, Any]:
    selector = np.asarray(selector, dtype=bool)
    selected_predictions = predictions[selector]
    distribution = np.bincount(selected_predictions, minlength=6)
    return {
        "samples": int(selector.sum()),
        "teacher_accuracy": float((predictions[selector] == teacher[selector]).mean()),
        "top2_teacher_accuracy": float(
            (top2[selector] == teacher[selector, None]).any(axis=1).mean()
        ),
        **{f"predicted_action_{index}_count": int(distribution[index]) for index in range(6)},
    }


def migration_metrics(
    evaluation: pd.DataFrame, predictions: np.ndarray
) -> dict[str, Any]:
    teacher = evaluation["teacher_action_index"].to_numpy(dtype=np.int64)
    h1 = evaluation["h1_action_index"].to_numpy(dtype=np.int64)
    origin = evaluation["origin_dc"].to_numpy(dtype=np.int64)
    teacher_migration = evaluation["teacher_is_migration"].to_numpy(dtype=bool)
    teacher_local = evaluation["teacher_is_local"].to_numpy(dtype=bool)
    bc_migration = (predictions != 0) & (predictions != origin)
    h1_migration = (h1 != 0) & (h1 != origin)
    return {
        "migration_samples": int(teacher_migration.sum()),
        "local_samples": int(teacher_local.sum()),
        "migration_accuracy": float(
            (predictions[teacher_migration] == teacher[teacher_migration]).mean()
        ),
        "local_accuracy": float((predictions[teacher_local] == teacher[teacher_local]).mean()),
        "bc_migration_fraction": float(bc_migration.mean()),
        "teacher_migration_fraction": float(teacher_migration.mean()),
        "h1_migration_fraction": float(h1_migration.mean()),
    }


def determine_outcome(runs: list[dict[str, Any]]) -> str:
    consensus = np.asarray([run["consensus_accuracy"] for run in runs])
    trr = np.asarray([run["teacher_recovery_rate"] for run in runs])
    fallback = np.asarray([run["h1_fallback_rate"] for run in runs])
    high = np.asarray([run["high_risk_disagreement_trr"] for run in runs])
    overall = np.asarray([run["teacher_accuracy"] for run in runs])
    if overall.mean() < 0.85 or consensus.mean() < 0.90:
        return "GENERAL IMITATION WEAK"
    if trr.std() > 0.10 or high.std() > 0.10:
        return "MIXED"
    if trr.mean() >= 0.35 and high.mean() >= 0.35 and fallback.mean() < 0.60:
        return "PRIVILEGED KNOWLEDGE DISTILLED"
    if trr.mean() < 0.20 and fallback.mean() > 0.50:
        return "MOSTLY H1-LIKE IMITATION"
    return "MIXED"


def run_smoke(root: Path, config: Mapping[str, Any]) -> None:
    contract = validate_frozen_contract(config, root)
    train_path = root / str(config["dataset_dir"]) / "train.parquet"
    arrays = load_training_arrays(train_path, int(config["obs_dim"]), int(config["action_dim"]))
    subset = slice(0, min(512, len(arrays)))
    set_deterministic_seed(11)
    model = build_actor(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    x = torch.from_numpy(arrays.observations[subset])
    labels = torch.from_numpy(arrays.labels[subset])
    masks = torch.from_numpy(arrays.masks[subset])
    logits = model(x)
    loss = masked_cross_entropy(logits, labels, masks)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    optimizer.step()
    if not bool(torch.isfinite(loss)) or not finite_gradients or logits.shape != (len(x), 6):
        raise RuntimeError("BC v2 smoke training failed")
    print(f"Dataset: {DATASET_NAME}")
    print(f"obs_dim: {contract['obs_dim']}")
    print(f"action_dim: {contract['action_dim']}")
    print(f"Dataset SHA256: {contract['dataset_sha256']}")
    print(f"smoke_loss: {float(loss):.8f}")
    print("smoke_status: PASS")


def create_figures(
    output: Path,
    runs: list[dict[str, Any]],
    histories: dict[int, pd.DataFrame],
    h1_accuracy: float,
    risk_deciles: pd.DataFrame,
) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for seed, history in histories.items():
        plt.plot(history["epoch"], history["train_loss"], label=f"train {seed}")
        plt.plot(history["epoch"], history["val_loss"], linestyle="--", label=f"val {seed}")
    plt.xlabel("Epoch")
    plt.ylabel("Masked cross-entropy")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(figure_dir / "01_train_val_loss.png", dpi=160)
    plt.close()

    labels = ["H1"] + [f"BC {run['seed']}" for run in runs]
    values = [h1_accuracy] + [run["teacher_accuracy"] for run in runs]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values, color=["#777777", "#2E86AB", "#3E9B76", "#D08C3A"])
    plt.ylim(0, 1)
    plt.ylabel("Teacher accuracy")
    plt.tight_layout()
    plt.savefig(figure_dir / "02_h1_vs_bc_accuracy.png", dpi=160)
    plt.close()

    x = np.arange(len(runs))
    recovered = [run["teacher_recovery_rate"] for run in runs]
    fallback = [run["h1_fallback_rate"] for run in runs]
    other = [run["other_rate"] for run in runs]
    plt.figure(figsize=(7, 4))
    plt.bar(x, recovered, label="Teacher recovered", color="#2E86AB")
    plt.bar(x, fallback, bottom=recovered, label="H1 fallback", color="#D08C3A")
    plt.bar(x, other, bottom=np.asarray(recovered) + np.asarray(fallback), label="Other", color="#888888")
    plt.xticks(x, [str(run["seed"]) for run in runs])
    plt.ylim(0, 1)
    plt.xlabel("Training seed")
    plt.ylabel("Disagreement outcome fraction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "03_disagreement_outcomes.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for seed, group in risk_deciles.groupby("model_seed"):
        plt.plot(group["risk_decile"], group["teacher_recovery_rate"], marker="o", label=str(seed))
    plt.xlabel("Test risk decile")
    plt.ylabel("Teacher recovery rate")
    plt.legend(title="Seed")
    plt.tight_layout()
    plt.savefig(figure_dir / "04_trr_by_risk_decile.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    values = [run["high_risk_disagreement_trr"] for run in runs]
    plt.bar([str(run["seed"]) for run in runs], values, color="#B24745")
    plt.ylim(0, 1)
    plt.xlabel("Training seed")
    plt.ylabel("High-risk disagreement TRR")
    plt.tight_layout()
    plt.savefig(figure_dir / "05_high_risk_disagreement_trr.png", dpi=160)
    plt.close()


def generate_reports(
    *,
    root: Path,
    config: Mapping[str, Any],
    output: Path,
    contract: Mapping[str, Any],
    architecture: Mapping[str, Any],
    training_runs: list[dict[str, Any]],
    evaluated_runs: list[dict[str, Any]],
    recommended: Mapping[str, Any],
    h1_accuracy: float,
    risk_threshold: float,
    overall_frame: pd.DataFrame,
    per_action_frame: pd.DataFrame,
    consensus_frame: pd.DataFrame,
    recovery_frame: pd.DataFrame,
    high_risk_frame: pd.DataFrame,
    scenario_frame: pd.DataFrame,
    risk_decile_frame: pd.DataFrame,
    three_way_frame: pd.DataFrame,
    state_frame: pd.DataFrame,
    migration_frame: pd.DataFrame,
    defer_frame: pd.DataFrame,
    git_head: str,
    sustain_head: str,
    elapsed_seconds: float,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    overall_frame.to_csv(output / "05_overall_metrics.csv", index=False)
    per_action_frame.to_csv(output / "06_per_action_metrics.csv", index=False)
    consensus_frame.to_csv(output / "07_consensus_disagreement_metrics.csv", index=False)
    recovery_frame.to_csv(output / "08_teacher_recovery_summary.csv", index=False)
    high_risk_frame.to_csv(output / "09_high_risk_recovery.csv", index=False)
    scenario_frame.to_csv(output / "10_scenario_breakdown.csv", index=False)
    risk_decile_frame.to_csv(output / "11_risk_decile_diagnostics.csv", index=False)
    three_way_frame.to_csv(output / "12_three_way_action_analysis.csv", index=False)
    state_frame.to_csv(output / "13_state_level_recovery.csv", index=False)
    migration_frame.to_csv(output / "14_migration_local_metrics.csv", index=False)
    defer_frame.to_csv(output / "15_predicted_defer_diagnostics.csv", index=False)

    write_json(output / "02_training_config.json", dict(config))
    write_json(output / "03_actor_architecture.json", dict(architecture))
    write_json(
        output / "04_training_environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": config["device"],
            "torch_threads": torch.get_num_threads(),
            "elapsed_seconds": elapsed_seconds,
            "git_head": git_head,
            "sustaincluster_commit": sustain_head,
        },
    )
    selection = {
        "selection_metric": "validation masked cross-entropy loss only",
        "test_metrics_used_for_selection": False,
        "recommended_seed": int(recommended["seed"]),
        "recommended_checkpoint": recommended["checkpoint"],
        "runs": training_runs,
    }
    write_json(output / "16_checkpoint_selection.json", selection)
    write_json(output / "recommended_checkpoint.json", selection)

    write_text(
        output / "17_information_leakage_audit.md",
        f"""# BC v2 Information Leakage Audit

Status: **PASS**

- Training columns: `student_observation`, `feasible_action_mask`, `teacher_action_index`, and `sample_id` only.
- Student observation dimension is `{contract['obs_dim']}` and is loaded from the frozen feature schema.
- H1 action, deployable risk, scenario, seed, Teacher objective, and privileged future metadata are not read by the training loader.
- Risk and H1 are joined only after inference for diagnostic slicing.
- `true_duration` is absent; `estimated_duration` remains feature index 7.
- Changing diagnostic metadata cannot change training tensor X because those Parquet columns are not selected.
- Future information in Student: **NO**.
""",
    )
    write_text(
        output / "18_bc_v1_vs_v2_notes.md",
        """# BC v1 vs BC v2 Notes

BC v1 used a 233-dimensional observation and pre-audit/pre-repair Teacher data. BC v2 uses the frozen 34-dimensional deployable current observation and repaired H4 Oracle privileged labels. The shared ActorNet architecture is reused, but BC v2 starts from random weights and does not load the old 233-dimensional checkpoint. Accuracy values are not directly comparable because the Teacher, observation, dataset, and information contract all changed.
""",
    )
    evidence = [
        ("Frozen data contract", "artifacts/repaired_mpc_expert_dataset_v2/20_dataset_manifest.json"),
        ("Training config", "02_training_config.json"),
        ("Actor architecture", "03_actor_architecture.json"),
        ("Overall metrics", "05_overall_metrics.csv"),
        ("Teacher recovery", "08_teacher_recovery_summary.csv"),
        ("High-risk recovery", "09_high_risk_recovery.csv"),
        ("State recovery", "13_state_level_recovery.csv"),
        ("Checkpoint selection", "16_checkpoint_selection.json"),
        ("Test predictions", "predictions/test_predictions_seed_*.parquet"),
    ]
    write_text(
        output / "19_evidence_index.md",
        "# BC v2 Evidence Index\n\n| Evidence | Path |\n|---|---|\n"
        + "\n".join(f"| {name} | `{path}` |" for name, path in evidence),
    )
    write_text(
        output / "20_change_manifest.md",
        """# BC v2 Change Manifest

## Added

- A deployable-only Dataset v2 loader, standard masked-CE training utilities, checkpoint contract, and offline recovery metrics.
- A three-seed BC v2 training/evaluation runner, dedicated configuration, tests, predictions, checkpoints, diagnostics, and reports.

## Frozen / unchanged

- Expert Dataset v2 Parquet files, labels, and split.
- Repaired MPC Teacher, timeline, objective, reward, Transformer, Forecast Dataset, and SustainCluster source.
- No closed-loop evaluation, SAC training, commit, or push.
""",
    )

    def metric_mean(name: str) -> float:
        return float(np.mean([run[name] for run in evaluated_runs]))

    def metric_std(name: str) -> float:
        return float(np.std([run[name] for run in evaluated_runs]))

    outcome = determine_outcome(evaluated_runs)
    recovered = [run["recovered_count"] for run in evaluated_runs]
    damaged = [run["consensus_damaged_count"] for run in evaluated_runs]
    seed_stability = "YES" if metric_std("teacher_accuracy") < 0.01 else "MIXED"
    write_text(
        output / "01_summary.md",
        f"""# BC v2 Offline Privileged-Knowledge Distillation

1. **Q1. Student input?** Frozen 34-d `student_observation` plus 6-d `feasible_action_mask`.
2. **Q2. Future information used?** NO.
3. **Q3. H1 action used as input?** NO.
4. **Q4. obs_dim?** `34`.
5. **Q5. action_dim?** `6`.
6. **Q6. Teacher defer support?** `0`; `DEFER_BEHAVIOR_NOT_COVERED_BY_PRIMARY_TEACHER_DATA`.
7. **Q7. Three seeds stable?** `{seed_stability}`; overall accuracy std `{metric_std('teacher_accuracy'):.6f}`.
8. **Q8. Recommended checkpoint?** `{recommended['checkpoint']}` selected only by validation loss.
9. **Q9. Test BC Teacher accuracy?** `{metric_mean('teacher_accuracy'):.6%} +/- {metric_std('teacher_accuracy'):.6%}`.
10. **Q10. H1 Teacher accuracy?** `{h1_accuracy:.6%}`.
11. **Q11. Consensus accuracy?** `{metric_mean('consensus_accuracy'):.6%} +/- {metric_std('consensus_accuracy'):.6%}`.
12. **Q12. Disagreement TRR?** `{metric_mean('teacher_recovery_rate'):.6%} +/- {metric_std('teacher_recovery_rate'):.6%}`.
13. **Q13. H1 fallback?** `{metric_mean('h1_fallback_rate'):.6%} +/- {metric_std('h1_fallback_rate'):.6%}`.
14. **Q14. High-risk disagreement TRR?** `{metric_mean('high_risk_disagreement_trr'):.6%} +/- {metric_std('high_risk_disagreement_trr'):.6%}`.
15. **Q15. Low-risk disagreement TRR?** `{metric_mean('low_risk_disagreement_trr'):.6%} +/- {metric_std('low_risk_disagreement_trr'):.6%}`.
16. **Q16. Disagreement preference?** Mean TRR `{metric_mean('teacher_recovery_rate'):.6%}` versus H1 fallback `{metric_mean('h1_fallback_rate'):.6%}`.
17. **Q17. Recovered privileged decisions?** Per seed `{recovered}`.
18. **Q18. Consensus damaged decisions?** Per seed `{damaged}`.
19. **Q19. Abnormal defer prediction?** Mean predicted defer rate `{metric_mean('predicted_defer_rate'):.6%}`; no post-hoc action suppression was used.
20. **Q20. Offline outcome?** `{outcome}`.
21. **Q21. Ready for closed loop?** `{'YES' if outcome != 'GENERAL IMITATION WEAK' else 'NO'}`.
""",
    )

    manifest = {
        "experiment": "bc_v2_offline_privileged_distillation",
        "dataset": DATASET_NAME,
        "dataset_sha256": contract["dataset_sha256"],
        "git_head": git_head,
        "sustaincluster_commit": sustain_head,
        "obs_dim": contract["obs_dim"],
        "action_dim": contract["action_dim"],
        "training_rows": 181944,
        "validation_rows": 38988,
        "test_rows": 38988,
        "risk_threshold_p95": risk_threshold,
        "recommended_checkpoint": recommended["checkpoint"],
        "recommended_seed": recommended["seed"],
        "outcome": outcome,
        "runs": evaluated_runs,
        "artifact_sha256": {
            str(path.relative_to(output)).replace("\\", "/"): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "bc_v2_manifest.json"
        },
    }
    write_json(output / "bc_v2_manifest.json", manifest)


def run_formal(root: Path, config: Mapping[str, Any]) -> None:
    started = time.perf_counter()
    contract = validate_frozen_contract(config, root)
    dataset_dir = root / str(config["dataset_dir"])
    output = root / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {DATASET_NAME}", flush=True)
    print(f"obs_dim: {contract['obs_dim']}", flush=True)
    print(f"action_dim: {contract['action_dim']}", flush=True)
    print(f"Dataset SHA256: {contract['dataset_sha256']}", flush=True)
    if not training_columns_are_deployable_only():
        raise RuntimeError("BC v2 training loader leaks diagnostic information")
    train = load_training_arrays(dataset_dir / "train.parquet", 34, 6)
    validation = load_training_arrays(dataset_dir / "val.parquet", 34, 6)
    if (len(train), len(validation)) != (181944, 38988):
        raise RuntimeError("frozen train/validation row count mismatch")
    git_head = git_value(root, "rev-parse", "HEAD")
    sustain_head = git_value(
        root / "references/external_repos/sustain-cluster", "rev-parse", "HEAD"
    )
    metadata = {
        "feature_schema_reference": str(config["feature_schema"]),
        "action_schema_reference": str(config["action_schema"]),
        "dataset_manifest_reference": str(config["dataset_manifest"]),
        "dataset_sha256": contract["dataset_sha256"],
        "train_val_test_split_reference": str(config["split_manifest"]),
        "git_head": git_head,
        "sustaincluster_commit": sustain_head,
    }
    architecture = actor_architecture(build_actor(config), config)
    training_runs: list[dict[str, Any]] = []
    histories: dict[int, pd.DataFrame] = {}
    for seed in config["seeds"]:
        seed = int(seed)
        checkpoint = output / "checkpoints" / f"bc_v2_seed_{seed}_best.pt"
        history = output / "training" / f"training_history_seed_{seed}.csv"
        print(f"training_seed={seed}", flush=True)
        run = train_seed(
            seed=seed,
            config=config,
            train=train,
            validation=validation,
            checkpoint_path=checkpoint,
            history_path=history,
            checkpoint_metadata=metadata,
        )
        training_runs.append(run)
        histories[seed] = pd.read_csv(history)
    recommended = dict(choose_recommended_run(training_runs))
    print(f"recommended_seed={recommended['seed']} by validation loss only", flush=True)

    test = load_training_arrays(dataset_dir / "test.parquet", 34, 6)
    if len(test) != 38988:
        raise RuntimeError("frozen test row count mismatch")
    evaluation = pd.read_parquet(dataset_dir / "test.parquet", columns=EVALUATION_COLUMNS)
    if tuple(evaluation["sample_id"].astype(str)) != test.sample_ids:
        raise RuntimeError("evaluation metadata is misaligned with training tensors")
    risk_thresholds = json.loads(
        (root / str(config["risk_thresholds"])).read_text(encoding="utf-8")
    )
    risk_threshold = float(risk_thresholds["P95_primary"])
    teacher = test.labels
    h1 = evaluation["h1_action_index"].to_numpy(dtype=np.int64)
    h1_accuracy = float((h1 == teacher).mean())
    risk = evaluation["deployable_risk_score"].to_numpy(dtype=float)
    high_risk = risk >= risk_threshold
    consensus = teacher == h1
    decile_labels = [f"D{index}" for index in range(1, 11)]
    evaluation["risk_decile"] = pd.qcut(
        pd.Series(risk).rank(method="first"), 10, labels=decile_labels
    ).astype(str)

    overall_rows: list[dict[str, Any]] = [
        {
            "model": "H1",
            "seed": "NA",
            "samples": len(test),
            "teacher_accuracy": h1_accuracy,
            "masked_cross_entropy": float("nan"),
            "top2_accuracy": float("nan"),
            "assignment_accuracy": h1_accuracy,
            "assignment_macro_f1": float("nan"),
        }
    ]
    per_action_rows: list[dict[str, Any]] = []
    consensus_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    high_risk_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    three_way_rows: list[dict[str, Any]] = []
    state_frames: list[pd.DataFrame] = []
    migration_rows: list[dict[str, Any]] = []
    defer_rows: list[dict[str, Any]] = []
    evaluated_runs: list[dict[str, Any]] = []
    for training_run in training_runs:
        seed = int(training_run["seed"])
        model, payload = load_checkpoint(root / training_run["checkpoint"])
        if payload["old_bc_weights_loaded"]:
            raise RuntimeError("old BC weights were loaded into BC v2")
        predictions, probabilities, top2 = infer(
            model, test, int(config["inference_batch_size"])
        )
        prediction_frame = build_prediction_frame(
            evaluation, predictions, probabilities, risk_threshold
        )
        if list(prediction_frame.columns) != list(PREDICTION_COLUMNS):
            raise RuntimeError("test prediction schema mismatch")
        prediction_path = output / "predictions" / f"test_predictions_seed_{seed}.parquet"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_frame.to_parquet(prediction_path, index=False)

        overall = classification_metrics(teacher, predictions, probabilities, top2, 6)
        overall_rows.append({"model": "BC", "seed": seed, **overall})
        per_action_rows.extend(
            {"model_seed": seed, **row}
            for row in per_action_metrics(teacher, predictions, 6)
        )
        consensus_metric = split_metrics(teacher, h1, predictions, top2, consensus)
        disagreement_metric = split_metrics(teacher, h1, predictions, top2, ~consensus)
        consensus_rows.extend(
            [
                {"model_seed": seed, "subset": "CONSENSUS", **consensus_metric},
                {"model_seed": seed, "subset": "DISAGREEMENT", **disagreement_metric},
            ]
        )
        recovery = recovery_metrics(teacher, h1, predictions)
        disagreement_selector = ~consensus
        confidence = {
            "mean_teacher_probability": float(
                probabilities[np.arange(len(test)), teacher][disagreement_selector].mean()
            ),
            "mean_h1_probability": float(
                probabilities[np.arange(len(test)), h1][disagreement_selector].mean()
            ),
        }
        confidence["teacher_minus_h1_probability_margin"] = (
            confidence["mean_teacher_probability"] - confidence["mean_h1_probability"]
        )
        recovery_rows.append({"model_seed": seed, **recovery, **confidence})
        for subset_name, selector in (
            ("HIGH_RISK_ALL", high_risk),
            ("LOW_RISK_ALL", ~high_risk),
            ("HIGH_RISK_DISAGREEMENT", high_risk & ~consensus),
            ("LOW_RISK_DISAGREEMENT", ~high_risk & ~consensus),
        ):
            high_risk_rows.append(
                {
                    "model_seed": seed,
                    "subset": subset_name,
                    "risk_threshold": risk_threshold,
                    **recovery_metrics(teacher, h1, predictions, selector),
                }
            )
        for scenario in ("normal_trace", "high_load_trace"):
            selector = evaluation["scenario"].to_numpy() == scenario
            scenario_recovery = recovery_metrics(teacher, h1, predictions, selector)
            high_scenario = selector & high_risk
            high_scenario_recovery = recovery_metrics(
                teacher, h1, predictions, high_scenario
            )
            scenario_rows.append(
                {
                    "model_seed": seed,
                    "scenario": scenario,
                    "samples": int(selector.sum()),
                    "overall_accuracy": float(
                        (predictions[selector] == teacher[selector]).mean()
                    ),
                    "disagreement_rate": float((teacher[selector] != h1[selector]).mean()),
                    "disagreement_trr": scenario_recovery["teacher_recovery_rate"],
                    "high_risk_disagreement_samples": high_scenario_recovery[
                        "disagreement_samples"
                    ],
                    "high_risk_disagreement_trr": high_scenario_recovery[
                        "teacher_recovery_rate"
                    ],
                }
            )
        for decile in decile_labels:
            selector = evaluation["risk_decile"].to_numpy() == decile
            values = recovery_metrics(teacher, h1, predictions, selector)
            decile_rows.append(
                {
                    "model_seed": seed,
                    "risk_decile": decile,
                    "risk_min": float(risk[selector].min()),
                    "risk_max": float(risk[selector].max()),
                    "samples": int(selector.sum()),
                    "teacher_h1_disagreement_rate": float(
                        (teacher[selector] != h1[selector]).mean()
                    ),
                    "bc_teacher_accuracy": values["teacher_accuracy"],
                    "teacher_recovery_rate": values["teacher_recovery_rate"],
                }
            )
        categories = three_way_categories(teacher, h1, predictions)
        for category in (
            "A_TEACHER_EQ_H1_EQ_BC",
            "B_TEACHER_EQ_H1_BC_DIFFERS",
            "C_TEACHER_NE_H1_BC_EQ_TEACHER",
            "D_TEACHER_NE_H1_BC_EQ_H1",
            "E_TEACHER_NE_H1_BC_OTHER",
        ):
            count = int((categories == category).sum())
            three_way_rows.append(
                {
                    "model_seed": seed,
                    "category": category,
                    "count": count,
                    "fraction": count / len(test),
                }
            )
        state = state_level_recovery(evaluation, predictions)
        state.insert(0, "model_seed", seed)
        state_frames.append(state)
        state_any = float(state["any_recovered"].mean())
        state_full = float(state["all_recovered"].mean())
        migration = migration_metrics(evaluation, predictions)
        migration_rows.append({"model_seed": seed, **migration})
        defer_count = int((predictions == 0).sum())
        defer_rows.append(
            {
                "model_seed": seed,
                "teacher_defer_labels": int((teacher == 0).sum()),
                "bc_predicted_defer_count": defer_count,
                "bc_predicted_defer_rate": defer_count / len(test),
                "claim_bc_learned_defer": False,
                "coverage_status": "DEFER_BEHAVIOR_NOT_COVERED_BY_PRIMARY_TEACHER_DATA",
            }
        )
        high_values = recovery_metrics(teacher, h1, predictions, high_risk)
        low_values = recovery_metrics(teacher, h1, predictions, ~high_risk)
        evaluated_runs.append(
            {
                **training_run,
                **overall,
                "consensus_accuracy": consensus_metric["teacher_accuracy"],
                "teacher_recovery_rate": recovery["teacher_recovery_rate"],
                "h1_fallback_rate": recovery["h1_fallback_rate"],
                "other_rate": recovery["other_rate"],
                "recovered_count": recovery["recovered_count"],
                "consensus_damaged_count": int(
                    (consensus & (predictions != teacher)).sum()
                ),
                "high_risk_disagreement_trr": high_values["teacher_recovery_rate"],
                "high_risk_h1_fallback_rate": high_values["h1_fallback_rate"],
                "low_risk_disagreement_trr": low_values["teacher_recovery_rate"],
                "state_any_recovery_rate": state_any,
                "state_full_recovery_rate": state_full,
                "predicted_defer_count": defer_count,
                "predicted_defer_rate": defer_count / len(test),
                "prediction_file": str(prediction_path),
            }
        )
        print(
            f"seed={seed} test_acc={overall['teacher_accuracy']:.6f} "
            f"TRR={recovery['teacher_recovery_rate']:.6f} "
            f"H1_fallback={recovery['h1_fallback_rate']:.6f}",
            flush=True,
        )

    overall_frame = pd.DataFrame(overall_rows)
    per_action_frame = pd.DataFrame(per_action_rows)
    consensus_frame = pd.DataFrame(consensus_rows)
    recovery_frame = pd.DataFrame(recovery_rows)
    high_risk_frame = pd.DataFrame(high_risk_rows)
    scenario_frame = pd.DataFrame(scenario_rows)
    risk_decile_frame = pd.DataFrame(decile_rows)
    three_way_frame = pd.DataFrame(three_way_rows)
    state_frame = pd.concat(state_frames, ignore_index=True)
    migration_frame = pd.DataFrame(migration_rows)
    defer_frame = pd.DataFrame(defer_rows)
    create_figures(output, evaluated_runs, histories, h1_accuracy, risk_decile_frame)
    generate_reports(
        root=root,
        config=config,
        output=output,
        contract=contract,
        architecture=architecture,
        training_runs=training_runs,
        evaluated_runs=evaluated_runs,
        recommended=recommended,
        h1_accuracy=h1_accuracy,
        risk_threshold=risk_threshold,
        overall_frame=overall_frame,
        per_action_frame=per_action_frame,
        consensus_frame=consensus_frame,
        recovery_frame=recovery_frame,
        high_risk_frame=high_risk_frame,
        scenario_frame=scenario_frame,
        risk_decile_frame=risk_decile_frame,
        three_way_frame=three_way_frame,
        state_frame=state_frame,
        migration_frame=migration_frame,
        defer_frame=defer_frame,
        git_head=git_head,
        sustain_head=sustain_head,
        elapsed_seconds=time.perf_counter() - started,
    )
    print(f"outcome={determine_outcome(evaluated_runs)}", flush=True)
    print(f"artifacts={output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sustaincluster_imitation/bc_v2_offline.yaml"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--train", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / args.config)
    if args.smoke:
        run_smoke(root, config)
    else:
        run_formal(root, config)


if __name__ == "__main__":
    main()
