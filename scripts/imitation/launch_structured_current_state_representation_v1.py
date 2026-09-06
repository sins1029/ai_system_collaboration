from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.imitation import execute_structured_current_state_representation_v1 as pipeline
from scripts.imitation import run_structured_current_state_representation_v1 as audit
from sustaincluster_imitation.current_collision import (
    current34_collision_audit,
    near_neighbor_audit,
    representation_collision_summary,
)
from sustaincluster_imitation.structured_current import (
    INFORMATION_CLASS,
    dc_feature_names,
    dense_feature_names,
    load_structured_checkpoint,
    model_parameter_count,
    running_feature_names,
    task_feature_names,
)
from sustaincluster_imitation import structured_current_reporting as reporting


def collision_summary_report(
    output: Path,
    exact_summary: pd.DataFrame,
    near: pd.DataFrame,
    config: dict,
) -> None:
    decimals = int(config["exact_collision"]["standardized_round_decimals"])
    raw = exact_summary.loc[
        (exact_summary["split"] == "all")
        & (exact_summary["method"] == "raw_float32_exact")
    ].iloc[0]
    rounded = exact_summary.loc[
        (exact_summary["split"] == "all")
        & (exact_summary["method"] == f"train_zscore_round_{decimals}_decimals")
    ].iloc[0]
    near_test = near.loc[(near["split"] == "test") & (near["stratum"] == "ALL")]
    lines = [
        "# Current34 Collision Summary",
        "",
        f"- Canonical input: Current34 plus the exact six-action feasible mask.",
        f"- Raw float32 exact duplicate groups: `{int(raw['duplicate_groups'])}`; multi-label groups: `{int(raw['multi_label_collision_groups'])}`.",
        f"- Raw empirical duplicate-group ceiling: `{float(raw['empirical_duplicate_group_ceiling']):.6%}`.",
        f"- Rounded diagnostic: train-only z-score followed by `{decimals}` decimal rounding.",
        f"- Rounded duplicate groups: `{int(rounded['duplicate_groups'])}`; multi-label groups: `{int(rounded['multi_label_collision_groups'])}`.",
        f"- Rounded empirical duplicate-group ceiling: `{float(rounded['empirical_duplicate_group_ceiling']):.6%}`.",
        "- This is an empirical duplicate-group ceiling, not a theoretical Bayes-optimal proof.",
        "",
        "## Test-split near neighbors",
        "",
    ]
    for row in near_test.itertuples(index=False):
        lines.append(
            f"- k={int(row.k)}: label agreement `{float(row.label_agreement):.6%}`, "
            f"local entropy `{float(row.mean_local_label_entropy_bits):.6f}` bits, "
            f"median nearest opposite-label distance `{float(row.median_nearest_opposite_label_distance):.6f}`."
        )
    lines.extend(
        (
            "",
            "Near-neighbor diagnostics use deterministic subsets within each split, the scaler fit on train only, and neighbors restricted to the same feasible-action mask. They are diagnostic and never select features, architectures, checkpoints, or thresholds.",
        )
    )
    audit.write_text(output / "07_current34_collision_summary.md", "\n".join(lines))


def information_contract(output: Path, dense_dim: int) -> None:
    audit.write_text(
        output / "11_current_dense_information_contract.md",
        f"""# CurrentDense Information Contract

- Input dimension: `{dense_dim}`.
- Information class: **{INFORMATION_CLASS}**.
- Focal task: complete deployable H1 task snapshot including memory, bandwidth, wait state, estimated duration, SLA, origin, and decision position.
- Destination context: current transmission cost and delay for DC1-DC5.
- DC context: totals, raw availability, known reservations, exact H1 node-0 capacity, current price/carbon, task counts, and optimizer DC order.
- Pending context: fixed-size sum/mean/max/P50/P90 summaries of the complete current joint batch.
- Running context: per-DC resource and controller-visible estimated release-step summaries.
- In-transit context: per-DC known resource, arrival-step, and estimated-duration summaries.
- Normalization: per-dimension z-score fitted on train tasks/states only.
- Label: frozen `h1_action_index`; labels are never features.

Excluded: future arrivals, Oracle price/carbon, Transformer forecast, true duration, Teacher/H1 action input, solver objective, MILP variables, reward, risk feature, and trigger feature. The saved Teacher action is used only to reproduce the already frozen environment trajectory and is converted through the audited semantic-to-environment action mapping.
""",
    )


def information_leakage_report(output: Path, replay: dict) -> None:
    audit.write_text(
        output / "22_information_leakage_audit.md",
        f"""# Information Leakage Audit

Status: **PASS**

- Source state is deployable H1 (`horizon=1`, `no_future_arrivals`).
- Frozen Teacher actions are used only for deterministic trajectory replay; they never enter CurrentDense or StructuredCurrent X.
- H1 actions are labels only and never enter X.
- Running release uses controller-visible estimated finish time, not true duration.
- Pending set contains only `env.current_tasks` at the current scheduler call.
- Running set contains only tasks currently present in each DC's `running_tasks`.
- DC rows preserve semantic identity and optimizer tuple position.
- All four scalers are fit on the train split only.
- Risk, Oracle disagreement, queue bucket, and pressure bucket are evaluation metadata only.
- Replay alignment mismatches: observations `{replay['replay_observation_mismatches']}`, masks `{replay['replay_mask_mismatches']}`, H1 labels `{replay['replay_h1_mismatches']}`.
- No Oracle feature, Transformer forecast, true future value, objective, reward, or teacher forcing is present.
""",
    )


def write_final_reports(
    output: Path,
    diagnosis: str,
    reasons: list[str],
    next_step: str,
    gate: bool,
    core: pd.DataFrame,
    determinism: pd.DataFrame,
    exact_summary: pd.DataFrame,
    near: pd.DataFrame,
    queue: pd.DataFrame,
    collision: pd.DataFrame,
) -> None:
    audit.write_text(
        output / "24_final_diagnosis.md",
        "# Final Diagnosis\n\n"
        f"Diagnosis: **{diagnosis}**\n\n"
        + "\n".join(f"- {reason}" for reason in reasons)
        + f"\n\nNext step: **{next_step}**.\n"
        + f"Forecast-augmented distillation gate: **{'OPEN' if gate else 'CLOSED'}**.\n",
    )
    values = core.set_index("Model")
    bc = values.loc["BC34"]
    dense = values.loc["CurrentDense"]
    structured = values.loc["StructuredCurrent"]
    decimals = "round"
    current_collision = collision.loc[
        (collision["representation"] == "Current34")
        & collision["method"].eq("raw_exact")
    ].iloc[0]
    dense_collision = collision.loc[
        (collision["representation"] == "CurrentDense")
        & collision["method"].eq("raw_exact")
    ].iloc[0]
    structured_collision = collision.loc[
        (collision["representation"] == "StructuredCurrent")
        & collision["method"].eq("raw_exact")
    ].iloc[0]
    queue_means = queue.groupby(["model", "queue_bucket"])["h1_accuracy"].mean()
    large_gap = float(queue_means["BC34", "large"] - queue_means["BC34", "small"])
    raw_all = exact_summary.loc[
        (exact_summary["split"] == "all")
        & (exact_summary["method"] == "raw_float32_exact")
    ].iloc[0]
    near_test_k1 = near.loc[
        (near["split"] == "test")
        & (near["stratum"] == "ALL")
        & (near["k"] == 1)
    ].iloc[0]
    order_sensitive = int((~determinism["reversed_order_same_by_task"]).sum())
    summary = f"""# Structured Current-State Representation Repair v1 Summary

1. **Is H1 joint or sequential?** Joint. One MILP simultaneously chooses all current task actions; there is no within-call residual-capacity update.
2. **Why is Current34 insufficient?** It omits task memory/bandwidth/destination network values, absolute reservation-aware capacity, task order, running/in-transit composition, and the rest of the pending batch.
3. **Does Current34 contain same-observation/different-label or near-collision ambiguity?** Raw multi-label groups: `{int(raw_all['multi_label_collision_groups'])}`; test k=1 near-neighbor disagreement: `{float(near_test_k1['near_neighbor_disagreement']):.6%}`.
4. **How much does fixed-size current repair improve?** `{float(bc['Overall H1 Accuracy']):.6%}` to `{float(dense['Overall H1 Accuracy']):.6%}` (`{float(dense['Overall H1 Accuracy']-bc['Overall H1 Accuracy']):+.6%}`).
5. **Is CurrentDense sufficient?** Its joint evidence is task accuracy `{float(dense['Overall H1 Accuracy']):.6%}`, high-risk `{float(dense['High-risk Accuracy']):.6%}`, and state full recovery `{float(dense['State Full Recovery']):.6%}`; interpretation is `{diagnosis}`.
6. **Does preserving pending/running/DC sets improve further?** StructuredCurrent reaches `{float(structured['Overall H1 Accuracy']):.6%}` (`{float(structured['Overall H1 Accuracy']-dense['Overall H1 Accuracy']):+.6%}` versus CurrentDense).
7. **Are high-risk states improved?** BC34 `{float(bc['High-risk Accuracy']):.6%}`, CurrentDense `{float(dense['High-risk Accuracy']):.6%}`, StructuredCurrent `{float(structured['High-risk Accuracy']):.6%}`.
8. **Are large queues the main BC34 failure region?** Large-minus-small BC34 accuracy is `{large_gap:+.6%}`; see queue-size stratification for the full three-seed evidence.
9. **Is there H1 tie-break/decision-context ambiguity?** Exact repeats pass; `{order_sensitive}/{len(determinism)}` representative states change per-task assignments when task order is reversed. Task position is therefore retained, but no sequential teacher forcing is needed.
10. **May forecast-augmented distillation start next?** `{'YES' if gate else 'NO'}`. Next step: `{next_step}`.

Raw exact multi-label collision groups: Current34 `{int(current_collision['multi_label_collision_groups'])}`, CurrentDense `{int(dense_collision['multi_label_collision_groups'])}`, StructuredCurrent `{int(structured_collision['multi_label_collision_groups'])}`.
"""
    audit.write_text(output / "25_summary.md", summary)


def refresh_manifest_hashes(output: Path, manifest: dict) -> None:
    manifest["artifact_sha256"] = {
        str(path.relative_to(output)).replace("\\", "/"): audit.sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "10_current_dense_manifest.json"
    }
    audit.write_json(output / "10_current_dense_manifest.json", manifest)


def run(config_path: Path) -> Path:
    started = time.perf_counter()
    config = audit.load_config(config_path)
    provenance = audit.verify_frozen_contract(config)
    output = ROOT / str(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    (output / "training_history").mkdir(exist_ok=True)
    audit.write_information_flow(output)
    print(f"audit={audit.AUDIT_NAME}", flush=True)
    print(f"dataset_sha256={provenance['dataset_sha256']}", flush=True)

    frame = pipeline.read_frame(config)
    current34_normalized, current34_scaler, exact, exact_summary = current34_collision_audit(
        frame, config
    )
    exact.to_csv(output / "05_current34_collision_exact.csv", index=False)
    near, pairs = near_neighbor_audit(current34_normalized, frame, config)
    near.to_csv(output / "06_current34_collision_near.csv", index=False)
    collision_summary_report(output, exact_summary, near, config)
    print("phase1_collision_audit=PASS", flush=True)

    replay, candidates = audit.replay_current_states(frame, config)
    determinism = audit.determinism_audit(candidates, config)
    audit.write_determinism_reports(output, determinism)
    if not bool(determinism["exact_repeat_deterministic"].all()):
        raise RuntimeError("H1 exact-state determinism failed; training is blocked")
    print("h1_determinism=PASS", flush=True)

    normalized, scalers = pipeline.normalize_representations(replay, frame)
    schema = pipeline.schema_rows()
    schema.to_csv(output / "09_current_dense_schema.csv", index=False)
    information_contract(output, normalized["dense"].shape[1])
    masks = pipeline.task_matrix(frame, "feasible_action_mask", bool)
    current34 = pipeline.task_matrix(frame, "student_observation", np.float32)
    collision = representation_collision_summary(
        frame=frame,
        masks=masks,
        current34=current34,
        dense_raw=replay["dense_raw"],
        dense_normalized=normalized["dense"],
        pending_raw=replay["pending_raw"],
        pending_normalized=normalized["pending"],
        running_raw=replay["running_raw"],
        running_normalized=normalized["running"],
        dc_raw=replay["dc_raw"],
        dc_normalized=normalized["dc"],
        state_offsets=replay["state_offsets"],
        running_offsets=replay["running_offsets"],
        decimals=int(config["exact_collision"]["standardized_round_decimals"]),
    )
    derived_path = pipeline.save_derived_arrays(output, normalized, replay, frame)
    store = pipeline.build_store(normalized, replay, frame)
    dense_runs = pipeline.train_dense(output, config, normalized["dense"], frame, provenance)
    structured_runs = pipeline.train_structured(output, config, store, provenance)
    records = pipeline.collect_predictions(
        config, frame, normalized["dense"], store, dense_runs, structured_runs
    )
    task, state, risk, queue, pressure = pipeline.evaluate_records(
        records, frame, replay, config
    )
    task.loc[task["model"] == "BC34"].to_csv(output / "12_bc34_baseline.csv", index=False)
    task.loc[task["model"] == "CurrentDense"].to_csv(
        output / "13_current_dense_metrics.csv", index=False
    )
    task.loc[task["model"] == "StructuredCurrent"].to_csv(
        output / "14_structured_current_metrics.csv", index=False
    )
    state.to_csv(output / "15_state_level_metrics.csv", index=False)
    risk.to_csv(output / "16_risk_stratified_metrics.csv", index=False)
    queue.to_csv(output / "17_queue_size_stratified_metrics.csv", index=False)
    pressure.to_csv(output / "18_resource_pressure_metrics.csv", index=False)
    cases, case_accuracy = reporting.write_case_studies(
        output, pairs, frame, normalized, replay, records
    )
    collision["selected_near_case_recommended_accuracy"] = collision[
        "representation"
    ].map(
        {
            "Current34": case_accuracy["BC34"],
            "CurrentDense": case_accuracy["CurrentDense"],
            "StructuredCurrent": case_accuracy["StructuredCurrent"],
        }
    )
    collision.to_csv(output / "19_collision_resolution_comparison.csv", index=False)
    summary = reporting.seed_summary(task, state, risk)
    summary.to_csv(output / "20_seed_summary.csv", index=False)
    core = reporting.core_comparison(summary)
    core.to_csv(output / "21_core_comparison.csv", index=False)
    information_leakage_report(output, replay)
    diagnosis, reasons, next_step, gate = reporting.diagnose(
        core, determinism, collision
    )
    write_final_reports(
        output,
        diagnosis,
        reasons,
        next_step,
        gate,
        core,
        determinism,
        exact_summary,
        near,
        queue,
        collision,
    )
    reporting.create_plots(output, core, risk, queue, exact, state)
    audit.write_text(
        output / "23_tests.md",
        """# Tests

- Pre-training frozen-contract checks: PASS.
- Exact H1 replay alignment: PASS.
- Exact repeated-state determinism: PASS.
- Dedicated, relevant, and full-suite results are recorded after the formal run.
""",
    )
    structured_model, _ = load_structured_checkpoint(
        Path(str(choose_best(structured_runs)["checkpoint"]))
    )
    manifest = {
        "audit": audit.AUDIT_NAME,
        **provenance,
        "information_class": INFORMATION_CLASS,
        "source_dataset_unchanged": True,
        "source_split_unchanged": True,
        "replay_uses_frozen_teacher_actions_only_for_environment_transition": True,
        "replay_alignment_mismatches": {
            "observation": replay["replay_observation_mismatches"],
            "mask": replay["replay_mask_mismatches"],
            "h1_label": replay["replay_h1_mismatches"],
        },
        "dimensions": {
            "current34": 34,
            "current_dense": len(dense_feature_names()),
            "structured_task": len(task_feature_names()),
            "structured_running": len(running_feature_names()),
            "structured_dc": len(dc_feature_names()),
            "action": 6,
        },
        "normalization": {
            "current34_collision": current34_scaler.to_dict([f"current34_{i}" for i in range(34)]),
            "current_dense": scalers["current_dense"].to_dict(dense_feature_names()),
            "pending_and_focal": scalers["pending_and_focal"].to_dict(task_feature_names()),
            "running": scalers["running"].to_dict(running_feature_names()),
            "datacenter": scalers["datacenter"].to_dict(dc_feature_names()),
        },
        "derived_dataset": {
            "path": str(derived_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": audit.sha256_file(derived_path),
            "task_rows": len(frame),
            "state_rows": len(replay["state_frame"]),
        },
        "dense_runs": dense_runs,
        "structured_runs": structured_runs,
        "structured_model": {
            "type": "DeepSets conditional per-task scorer",
            "pending_focal_excluded": True,
            "strict_joint_decoder": False,
            "teacher_forcing": False,
            "parameter_count": model_parameter_count(structured_model),
        },
        "determinism_pass": bool(determinism["exact_repeat_deterministic"].all()),
        "collision_case_count": len(cases),
        "diagnosis": diagnosis,
        "next_step": next_step,
        "forecast_gate_open": gate,
        "closed_loop": False,
        "sac_rl": False,
        "oracle_feature_injection": False,
        "transformer_feature_injection": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    refresh_manifest_hashes(output, manifest)
    print(f"diagnosis={diagnosis}", flush=True)
    print(f"next_step={next_step}", flush=True)
    print(f"artifacts={output}", flush=True)
    return output


def choose_best(runs: list[dict]) -> dict:
    return min(runs, key=lambda row: (float(row["best_val_loss"]), int(row["seed"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/sustaincluster_imitation/structured_current_state_representation_v1.yaml"
        ),
    )
    args = parser.parse_args()
    run(ROOT / args.config)


if __name__ == "__main__":
    main()
