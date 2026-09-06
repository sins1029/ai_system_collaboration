from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.imitation import execute_structured_current_state_representation_v1 as pipeline
from scripts.imitation import launch_structured_current_state_representation_v1 as launcher
from scripts.imitation import run_structured_current_state_representation_v1 as audit
from sustaincluster_imitation.current_collision import (
    current34_collision_audit,
    near_neighbor_audit,
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


def completed_runs(output: Path, prefix: str) -> list[dict]:
    runs = []
    for checkpoint in sorted((output / "checkpoints").glob(f"{prefix}_seed_*_best.pt")):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        seed = int(payload["seed"])
        history_name = (
            f"bc_dense_seed_{seed}.csv"
            if prefix == "bc_dense"
            else f"structured_current_seed_{seed}.csv"
        )
        history = output / "training_history" / history_name
        if not history.exists():
            raise RuntimeError(f"missing completed training history: {history}")
        runs.append(
            {
                "seed": seed,
                "best_epoch": int(payload["best_epoch"]),
                "best_val_loss": float(payload["best_val_loss"]),
                "epochs_completed": len(pd.read_csv(history)),
                "checkpoint": str(checkpoint),
                "history": str(history),
            }
        )
    if [row["seed"] for row in runs] != [11, 22, 33]:
        raise RuntimeError(f"incomplete formal checkpoints for {prefix}")
    return runs


def finalize(config_path: Path) -> Path:
    started = time.perf_counter()
    config = audit.load_config(config_path)
    provenance = audit.verify_frozen_contract(config)
    output = ROOT / str(config["output_dir"])
    dense_runs = completed_runs(output, "bc_dense")
    structured_runs = completed_runs(output, "structured_current")

    frame = pipeline.read_frame(config)
    _, current34_scaler, exact, exact_summary = current34_collision_audit(frame, config)
    near, pairs = near_neighbor_audit(_, frame, config)
    replay, candidates = audit.replay_current_states(frame, config)
    determinism = audit.determinism_audit(candidates, config)
    audit.write_determinism_reports(output, determinism)
    if not bool(determinism["exact_repeat_deterministic"].all()):
        raise RuntimeError("H1 exact-state determinism failed during finalization")
    normalized, scalers = pipeline.normalize_representations(replay, frame)

    collision = pd.read_csv(output / "19_collision_resolution_comparison.csv")
    task = pd.concat(
        [
            pd.read_csv(output / "12_bc34_baseline.csv"),
            pd.read_csv(output / "13_current_dense_metrics.csv"),
            pd.read_csv(output / "14_structured_current_metrics.csv"),
        ],
        ignore_index=True,
    )
    state = pd.read_csv(output / "15_state_level_metrics.csv")
    risk = pd.read_csv(output / "16_risk_stratified_metrics.csv")
    queue = pd.read_csv(output / "17_queue_size_stratified_metrics.csv")
    pressure = pd.read_csv(output / "18_resource_pressure_metrics.csv")
    summary = reporting.seed_summary(task, state, risk)
    summary.to_csv(output / "20_seed_summary.csv", index=False)
    core = reporting.core_comparison(summary)
    core.to_csv(output / "21_core_comparison.csv", index=False)
    launcher.information_leakage_report(output, replay)
    diagnosis, reasons, next_step, gate = reporting.diagnose(
        core, determinism, collision
    )
    launcher.write_final_reports(
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
- Dedicated, relevant, and full-suite results are recorded after finalization.
""",
    )

    best_structured = launcher.choose_best(structured_runs)
    structured_model, _payload = load_structured_checkpoint(
        Path(best_structured["checkpoint"])
    )
    derived_path = output / "dataset" / "structured_current_arrays_v1.npz"
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
            "current34_collision": current34_scaler.to_dict(
                [f"current34_{index}" for index in range(34)]
            ),
            "current_dense": scalers["current_dense"].to_dict(dense_feature_names()),
            "pending_and_focal": scalers["pending_and_focal"].to_dict(
                task_feature_names()
            ),
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
        "determinism_pass": True,
        "collision_case_count": min(15, len(pairs)),
        "diagnosis": diagnosis,
        "next_step": next_step,
        "forecast_gate_open": gate,
        "closed_loop": False,
        "sac_rl": False,
        "oracle_feature_injection": False,
        "transformer_feature_injection": False,
        "finalized_from_completed_formal_checkpoints": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    launcher.refresh_manifest_hashes(output, manifest)
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
            "configs/sustaincluster_imitation/structured_current_state_representation_v1.yaml"
        ),
    )
    args = parser.parse_args()
    finalize(ROOT / args.config)


if __name__ == "__main__":
    main()
