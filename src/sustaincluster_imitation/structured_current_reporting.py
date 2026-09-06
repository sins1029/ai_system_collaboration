from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sustaincluster_imitation.bc_v2_offline import choose_recommended_run
from sustaincluster_imitation.structured_current import dense_feature_names


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def seed_summary(task: pd.DataFrame, state: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    high = risk.loc[risk["risk_group"] == "HIGH_RISK", ["model", "seed", "h1_accuracy"]]
    high = high.rename(columns={"h1_accuracy": "high_risk_accuracy"})
    low = risk.loc[risk["risk_group"] == "LOW_RISK", ["model", "seed", "h1_accuracy"]]
    low = low.rename(columns={"h1_accuracy": "low_risk_accuracy"})
    base = task.merge(state, on=["model", "seed"], how="inner").merge(
        high, on=["model", "seed"], how="inner"
    ).merge(low, on=["model", "seed"], how="inner")
    selected = [
        "model",
        "seed",
        "overall_h1_accuracy",
        "top2_h1_accuracy",
        "assignment_macro_f1",
        "high_risk_accuracy",
        "low_risk_accuracy",
        "state_full_action_recovery",
        "state_any_mismatch_rate",
        "mean_mismatched_tasks",
        "best_epoch",
        "best_val_loss",
    ]
    base = base[selected]
    numeric = [column for column in selected if column not in {"model", "seed"}]
    mean = base.groupby("model", sort=False)[numeric].mean().reset_index()
    mean.insert(1, "seed", "MEAN")
    std = base.groupby("model", sort=False)[numeric].std(ddof=0).reset_index()
    std.insert(1, "seed", "STD")
    return pd.concat((base, mean, std), ignore_index=True, sort=False)


def core_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    means = summary.loc[summary["seed"].astype(str) == "MEAN"].set_index("model")
    rows = []
    for model, observation, structure in (
        ("BC34", "Current34", "Flat"),
        ("CurrentDense", "CurrentDense", "Flat"),
        ("StructuredCurrent", "Full Current Sets", "Set"),
    ):
        row = means.loc[model]
        rows.append(
            {
                "Model": model,
                "Observation": observation,
                "Structure": structure,
                "Label": "H1",
                "Deployable Current Info Only": "YES",
                "Overall H1 Accuracy": row["overall_h1_accuracy"],
                "Assignment Macro F1": row["assignment_macro_f1"],
                "High-risk Accuracy": row["high_risk_accuracy"],
                "Low-risk Accuracy": row["low_risk_accuracy"],
                "State Full Recovery": row["state_full_action_recovery"],
                "Mean Mismatched Tasks": row["mean_mismatched_tasks"],
                "Three-seed Accuracy Std": summary.loc[
                    (summary["model"] == model)
                    & (summary["seed"].astype(str) == "STD"),
                    "overall_h1_accuracy",
                ].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def recommended_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    selected = {}
    for model in ("BC34", "CurrentDense", "StructuredCurrent"):
        candidates = [record for record in records if record["model"] == model]
        best = choose_recommended_run([record["run"] for record in candidates])
        selected[model] = next(
            record for record in candidates if int(record["seed"]) == int(best["seed"])
        )
    return selected


def structured_fingerprint(
    task_index: int,
    normalized: Mapping[str, np.ndarray],
    replay: Mapping[str, Any],
) -> np.ndarray:
    state = int(replay["task_state_index"][task_index])
    start, stop = replay["state_offsets"][state : state + 2]
    run_start, run_stop = replay["running_offsets"][state : state + 2]
    running = normalized["running"][run_start:run_stop]
    return np.concatenate(
        (
            normalized["pending"][task_index],
            normalized["pending"][start:stop].sum(axis=0),
            running.sum(axis=0)
            if len(running)
            else np.zeros(normalized["running"].shape[1]),
            normalized["dc"][state].reshape(-1),
        )
    )


def write_case_studies(
    output: Path,
    pairs: Sequence[Mapping[str, Any]],
    frame: pd.DataFrame,
    normalized: Mapping[str, np.ndarray],
    replay: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    test_pairs = [row for row in pairs if row["split"] == "test"][:15]
    if len(test_pairs) < 10:
        test_pairs = list(pairs[:15])
    if len(test_pairs) < 10:
        raise RuntimeError("near-neighbor audit found fewer than 10 case-study pairs")
    selected = recommended_records(records)
    feature_names = dense_feature_names()
    state_frame = replay["state_frame"]
    cases = []
    sections = [
        "# Current34 Collision Case Studies",
        "",
        "Pairs are nearest opposite-H1-label neighbors under train-only standardized Current34 and the same feasible mask. CurrentDense distance and a fixed structured fingerprint are diagnostic only; neither was used for model selection.",
        "",
    ]
    task_indices = set()
    for number, pair in enumerate(test_pairs, start=1):
        left = int(pair["left_index"])
        right = int(pair["right_index"])
        task_indices.update((left, right))
        difference = np.abs(normalized["dense"][left] - normalized["dense"][right])
        top = np.argsort(difference)[-8:][::-1]
        left_state = state_frame.iloc[int(replay["task_state_index"][left])]
        right_state = state_frame.iloc[int(replay["task_state_index"][right])]
        structured_distance = float(
            np.linalg.norm(
                structured_fingerprint(left, normalized, replay)
                - structured_fingerprint(right, normalized, replay)
            )
        )
        case = {
            "case": number,
            "left_index": left,
            "right_index": right,
            "left_sample_id": str(frame.iloc[left]["sample_id"]),
            "right_sample_id": str(frame.iloc[right]["sample_id"]),
            "left_h1_action": int(frame.iloc[left]["h1_action_index"]),
            "right_h1_action": int(frame.iloc[right]["h1_action_index"]),
            "current34_distance": float(pair["current34_distance"]),
            "current_dense_distance": float(
                np.linalg.norm(normalized["dense"][left] - normalized["dense"][right])
            ),
            "structured_fingerprint_distance": structured_distance,
            "left_queue_size": int(left_state["queue_size"]),
            "right_queue_size": int(right_state["queue_size"]),
            "left_running_size": int(left_state["running_size"]),
            "right_running_size": int(right_state["running_size"]),
            "top_distinguishing_dense_features": " | ".join(
                f"{feature_names[index]}:{difference[index]:.3f}" for index in top
            ),
        }
        cases.append(case)
        sections.extend(
            (
                f"## Case {number}",
                "",
                f"- Current34 distance: `{case['current34_distance']:.6g}`; same feasible mask, but H1 actions are `{case['left_h1_action']}` vs `{case['right_h1_action']}`.",
                f"- CurrentDense distance: `{case['current_dense_distance']:.6g}`; structured fingerprint distance: `{structured_distance:.6g}`.",
                f"- Pending batch size: `{case['left_queue_size']}` vs `{case['right_queue_size']}`; running set size: `{case['left_running_size']}` vs `{case['right_running_size']}`.",
                f"- Main newly visible differences: `{case['top_distinguishing_dense_features']}`.",
                f"- Samples: `{case['left_sample_id']}` and `{case['right_sample_id']}`.",
                "",
            )
        )
    write_text(output / "08_collision_case_studies.md", "\n".join(sections))
    selected_indices = np.asarray(sorted(task_indices), dtype=np.int64)
    labels = frame["h1_action_index"].to_numpy(dtype=np.int64)
    accuracy = {}
    for model, record in selected.items():
        full = record["full_predictions"]
        valid = full[selected_indices] >= 0
        accuracy[model] = float(
            (full[selected_indices][valid] == labels[selected_indices][valid]).mean()
        )
    return cases, accuracy


def diagnose(
    core: pd.DataFrame,
    determinism: pd.DataFrame,
    collision: pd.DataFrame,
) -> tuple[str, list[str], str, bool]:
    values = core.set_index("Model")
    bc = float(values.loc["BC34", "Overall H1 Accuracy"])
    dense = float(values.loc["CurrentDense", "Overall H1 Accuracy"])
    structured = float(values.loc["StructuredCurrent", "Overall H1 Accuracy"])
    bc_high = float(values.loc["BC34", "High-risk Accuracy"])
    dense_high = float(values.loc["CurrentDense", "High-risk Accuracy"])
    structured_high = float(values.loc["StructuredCurrent", "High-risk Accuracy"])
    bc_state = float(values.loc["BC34", "State Full Recovery"])
    dense_state = float(values.loc["CurrentDense", "State Full Recovery"])
    structured_state = float(values.loc["StructuredCurrent", "State Full Recovery"])
    dense_gain = dense - bc
    structural_gain = structured - dense
    dense_std = float(values.loc["CurrentDense", "Three-seed Accuracy Std"])
    structured_std = float(values.loc["StructuredCurrent", "Three-seed Accuracy Std"])
    exact_deterministic = bool(determinism["exact_repeat_deterministic"].all())
    order_sensitive = int((~determinism["reversed_order_same_by_task"]).sum())
    flat_material = dense_gain > max(0.05, 3.0 * dense_std)
    structural_material = structural_gain > max(0.03, 3.0 * structured_std)
    ambiguity_evidence = order_sensitive > 0 and structured < 0.90
    if flat_material and structural_material:
        diagnosis = "MIXED"
    elif structural_material:
        diagnosis = "STRUCTURAL CONTEXT BOTTLENECK"
    elif flat_material:
        diagnosis = "FLAT FEATURE INSUFFICIENCY"
    elif ambiguity_evidence:
        diagnosis = "H1 LABEL / DECISION-CONTEXT AMBIGUITY"
    else:
        diagnosis = "INCONCLUSIVE"
    current34_collision = collision.loc[
        (collision["representation"] == "Current34")
        & collision["method"].eq("raw_exact")
    ].iloc[0]
    structured_collision = collision.loc[
        (collision["representation"] == "StructuredCurrent")
        & collision["method"].eq("raw_exact")
    ].iloc[0]
    collision_reduced = int(structured_collision["multi_label_collision_groups"]) < int(
        current34_collision["multi_label_collision_groups"]
    )
    best_gain = max(dense, structured) - bc
    best_high_gain = max(dense_high, structured_high) - bc_high
    best_state_gain = max(dense_state, structured_state) - bc_state
    stable = min(dense_std, structured_std) < 0.02
    gate = bool(
        exact_deterministic
        and best_gain >= 0.10
        and best_high_gain >= 0.05
        and best_state_gain >= 0.05
        and stable
        and collision_reduced
    )
    if gate:
        next_step = "FORECAST-AUGMENTED STRUCTURED DISTILLATION"
    elif ambiguity_evidence:
        next_step = "JOINT / AUTOREGRESSIVE DECODER AUDIT"
    else:
        next_step = "CURRENT REPRESENTATION REPAIR CONTINUES"
    reasons = [
        f"BC34 mean H1 accuracy is {bc:.6%}; CurrentDense is {dense:.6%} ({dense_gain:+.6%}).",
        f"StructuredCurrent is {structured:.6%} ({structural_gain:+.6%} versus CurrentDense).",
        f"High-risk accuracy is BC34 {bc_high:.6%}, CurrentDense {dense_high:.6%}, StructuredCurrent {structured_high:.6%}.",
        f"State full recovery is BC34 {bc_state:.6%}, CurrentDense {dense_state:.6%}, StructuredCurrent {structured_state:.6%}.",
        f"Exact repeated H1 solves deterministic={exact_deterministic}; reversed-order-sensitive representatives={order_sensitive}/{len(determinism)}.",
        f"Raw exact multi-label collision groups change from {int(current34_collision['multi_label_collision_groups'])} to {int(structured_collision['multi_label_collision_groups'])}.",
        "The classification uses joint evidence from task/state/risk/collision/seed behavior, not a single 95% accuracy threshold.",
    ]
    return diagnosis, reasons, next_step, gate


def create_plots(
    output: Path,
    core: pd.DataFrame,
    risk: pd.DataFrame,
    queue: pd.DataFrame,
    exact: pd.DataFrame,
    state: pd.DataFrame,
) -> None:
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(core["Model"], core["Overall H1 Accuracy"], color=["#777777", "#277DA1", "#43AA8B"])
    plt.ylim(0, 1)
    plt.ylabel("H1 task accuracy")
    plt.tight_layout()
    plt.savefig(plots / "01_h1_task_accuracy.png", dpi=160)
    plt.close()

    risk_mean = risk.groupby(["model", "risk_group"], sort=False)["h1_accuracy"].mean().unstack()
    risk_mean[["LOW_RISK", "HIGH_RISK"]].plot(
        kind="bar", color=["#4D908E", "#F94144"], figsize=(8, 5)
    )
    plt.ylim(0, 1)
    plt.ylabel("H1 accuracy")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plots / "02_risk_stratified_accuracy.png", dpi=160)
    plt.close()

    queue_mean = queue.groupby(["model", "queue_bucket"], sort=False)["h1_accuracy"].mean().unstack()
    queue_mean[["small", "medium", "large"]].plot(
        kind="bar", color=["#90BE6D", "#F9C74F", "#F9844A"], figsize=(8, 5)
    )
    plt.ylim(0, 1)
    plt.ylabel("H1 accuracy")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plots / "03_queue_size_accuracy.png", dpi=160)
    plt.close()

    collision_entropy = exact.loc[
        exact["method"].str.contains("round") & exact["label_collision"]
    ]
    plt.figure(figsize=(8, 5))
    plt.hist(collision_entropy["label_entropy_bits"], bins=20, color="#577590")
    plt.xlabel("Current34 collision label entropy (bits)")
    plt.ylabel("Groups")
    plt.tight_layout()
    plt.savefig(plots / "04_current34_collision_entropy.png", dpi=160)
    plt.close()

    state_mean = state.groupby("model", sort=False)["state_full_action_recovery"].mean()
    plt.figure(figsize=(8, 5))
    plt.bar(state_mean.index, state_mean.values, color=["#777777", "#277DA1", "#43AA8B"])
    plt.ylim(0, 1)
    plt.ylabel("State full action recovery")
    plt.tight_layout()
    plt.savefig(plots / "05_state_full_recovery.png", dpi=160)
    plt.close()
    write_text(
        plots / "README.md",
        """# Plot Source Data

- Figure 1: `../21_core_comparison.csv`
- Figure 2: `../16_risk_stratified_metrics.csv`
- Figure 3: `../17_queue_size_stratified_metrics.csv`
- Figure 4: `../05_current34_collision_exact.csv`
- Figure 5: `../15_state_level_metrics.csv`
""",
    )
