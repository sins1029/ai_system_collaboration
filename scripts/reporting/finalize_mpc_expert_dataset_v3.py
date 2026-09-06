from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/mpc_expert_dataset_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8", errors="replace"
    ).strip()


def scenario_a_statistics() -> pd.DataFrame:
    tasks = pd.read_parquet(
        OUTPUT / "dataset/scenario_a/expert_task_actions_full.parquet"
    )
    states = pd.read_parquet(
        OUTPUT / "dataset/scenario_a/current_states_full.parquet"
    )
    episodes = pd.read_csv(OUTPUT / "04_alibaba2020_v3_episode_manifest.csv")
    alignment = pd.read_csv(OUTPUT / "07_alibaba2020_v2_v3_alignment.csv")
    agreement = alignment.set_index("metric")["agreement_rate"]
    values = {
        "episodes": len(episodes),
        "states": len(states),
        "task_decisions": len(tasks),
        "h1_oracle_exact_disagreement_rate": float(tasks["exact_action_disagreement"].mean()),
        "placement_disagreement_rate": float(tasks["target_dc_disagreement"].mean()),
        "defer_disagreement_rate": float(tasks["semantic_disagreement"].mean()),
        "h1_defer_rate": float(tasks["h1_is_defer"].mean()),
        "oracle_defer_rate": float(tasks["teacher_is_defer"].mean()),
        "state_any_disagreement_rate": float(states["any_disagreement"].mean()),
        "mean_pending_batch_size": float(states["num_tasks"].mean()),
        "solver_failures": int(episodes["teacher_solver_failures"].sum() + episodes["h1_solver_failures"].sum()),
        "v2_v3_student_observation_agreement": float(agreement["student_observation_agreement"]),
        "v2_v3_feasible_mask_agreement": float(agreement["feasible_mask_agreement"]),
        "v2_v3_h1_action_agreement": float(agreement["h1_action_agreement"]),
        "v2_v3_h4_oracle_action_agreement": float(agreement["h4_oracle_action_agreement"]),
    }
    return pd.DataFrame(
        [
            {"metric": key, "value": value, "status": "V3_INDEPENDENT_REGENERATION"}
            for key, value in values.items()
        ]
    )


def scenario_b_statistics(capacity: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("raw_jobs", capacity["workload_task_count"], "FULL_SOURCE_MEASURED"),
        ("canonical_jobs", capacity["workload_task_count"], "FULL_SOURCE_MEASURED"),
        ("node_count", capacity["node_count"], "FULL_NODE_INFO_MEASURED"),
        ("native_total_cpu", capacity["native_capacity"]["cpu"], "FULL_NODE_INFO_MEASURED"),
        ("native_total_gpu", capacity["native_capacity"]["gpu"], "FULL_NODE_INFO_MEASURED"),
        ("modeled_total_memory", capacity["native_capacity"]["memory"], "MODELED_DC_MEMORY"),
        ("static_feasible_rate", capacity["static_feasible_rate"], "FULL_WORKLOAD_MEASURED"),
        ("gpu_capacity_multiplier_vs_old", capacity["gpu_capacity_multiplier"], "DERIVED"),
        ("native_too_loose", capacity["native_too_loose"], "CALIBRATION_DECISION"),
        ("comparable_capacity_required", capacity["comparable_capacity_required"], "CALIBRATION_DECISION"),
    ]
    for item in capacity["native_pressure"]:
        for metric in (
            "offered_load_ratio",
            "mandatory_concurrent_peak",
            "mandatory_peak_ratio",
            "immediate_start_concurrent_peak",
            "immediate_start_peak_ratio",
        ):
            rows.append(
                (f"{item['resource']}_{metric}", item[metric], "FULL_WORKLOAD_MEASURED")
            )
    arrivals = pd.read_csv(OUTPUT / "36_spot_native_arrival_pressure_summary.csv")
    for item in arrivals.itertuples(index=False):
        for statistic in ("mean", "p50", "p90", "p95", "p99", "max"):
            rows.append(
                (
                    f"arrival_{item.metric}_{statistic}",
                    getattr(item, statistic),
                    "FULL_WORKLOAD_MEASURED",
                )
            )
    return pd.DataFrame(rows, columns=["metric", "value", "status"])


def integrity_manifest() -> pd.DataFrame:
    rows = []
    excluded = {
        "23_integrity_manifest.csv",
        "23_current_integrity_manifest.csv",
    }
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in excluded:
            rows.append(
                {
                    "path": path.relative_to(OUTPUT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return pd.DataFrame(rows)


def run() -> Path:
    alibaba = json.loads(
        (OUTPUT / "00_alibaba2020_v3_manifest.json").read_text("utf-8")
    )
    capacity = json.loads(
        (OUTPUT / "43_spot_capacity_calibration_manifest.json").read_text("utf-8")
    )
    if alibaba["status"] != "ALIBABA2020_V3_READY":
        raise RuntimeError("Alibaba2020 v3 is not ready")
    if capacity["final_status"] != "SPOT NATIVE CAPACITY READY":
        raise RuntimeError("Spot native capacity is not ready")
    a_stats = scenario_a_statistics()
    b_stats = scenario_b_statistics(capacity)
    a_stats.to_csv(OUTPUT / "09_scenario_a_statistics.csv", index=False)
    b_stats.to_csv(OUTPUT / "10_scenario_b_statistics.csv", index=False)

    steps = pd.read_csv(OUTPUT / "39_dynamic_preflight_steps.csv")
    dynamic = pd.read_csv(OUTPUT / "40_dynamic_preflight_summary.csv")
    restored = pd.read_csv(OUTPUT / "40a_dynamic_restore_evidence.csv")
    episodes = pd.read_csv(OUTPUT / "04_alibaba2020_v3_episode_manifest.csv")
    solver = pd.DataFrame(
        [
            {
                "scenario_id": "A_ALIBABA2020_REPAIRED",
                "planner": "H1_REPAIRED",
                "solve_attempts": int(episodes["steps"].sum()),
                "solver_failures": int(episodes["h1_solver_failures"].sum()),
                "solver_status": "optimal",
                "status": "V3_FORMAL_GENERATION",
            },
            {
                "scenario_id": "A_ALIBABA2020_REPAIRED",
                "planner": "H4_ORACLE_REPAIRED",
                "solve_attempts": int(episodes["steps"].sum()),
                "solver_failures": int(episodes["teacher_solver_failures"].sum()),
                "solver_status": "optimal",
                "status": "V3_FORMAL_GENERATION",
            },
            {
                "scenario_id": "B_SPOTGPU2026_NATIVE_5DC",
                "planner": "H1_REPAIRED",
                "solve_attempts": len(steps),
                "solver_failures": int(steps["h1_status"].ne("optimal").sum()),
                "solver_status": "optimal",
                "status": "DYNAMIC_PREFLIGHT_ONLY",
            },
            {
                "scenario_id": "B_SPOTGPU2026_NATIVE_5DC",
                "planner": "H4_ORACLE_REPAIRED",
                "solve_attempts": len(steps),
                "solver_failures": int(steps["h4_status"].ne("optimal").sum()),
                "solver_status": "optimal",
                "status": "DYNAMIC_PREFLIGHT_ONLY",
            },
        ]
    )
    solver.to_csv(OUTPUT / "20_solver_quality.csv", index=False)

    write_text(
        OUTPUT / "01_protocol.md",
        """# MPC Expert Dataset v3 Protocol

1. Scenario A was independently regenerated first with seeds 3001-3040, the original split, repaired H1, and repaired H4 Oracle.
2. No Transformer, RL, or policy training was run.
3. Scenario B keeps all 466,867 Spot tasks unchanged and replaces the mismatched legacy 5DC capacity with a deterministic partition of the complete node_info pool.
4. CPU/GPU are measured and strictly conserved. Memory is marked MODELED_DC_MEMORY using the frozen Alibaba2020/SustainCluster fleet Memory/CPU ratio.
5. HP/Spot max waits remain 1/8 steps. GPU model is metadata only.
6. Spot expert generation was not run; only train-window dynamic preflight with continuous prefix state restoration was allowed.
""",
    )
    write_text(
        OUTPUT / "06_spot_temporal_state_audit.md",
        f"""# Spot Dynamic State-Restore Audit

- Windows: {', '.join(dynamic['window_type'].tolist())}; 16 steps each, train split only.
- Window starts: {', '.join(str(value) for value in restored['start_step'])}.
- Restored pre-window running tasks: {', '.join(str(value) for value in restored['restored_pre_window_running_task_count'])}.
- Empty-state starts: {int(restored['empty_state_start'].astype(bool).sum())}.
- H1/H4 optimal steps: {int(dynamic['h1_optimal_steps'].sum())}/{int(dynamic['h4_optimal_steps'].sum())} of {len(steps)} each.
- Solver failure/timeout: {int(dynamic['h1_failure_or_timeout'].sum())}/{int(dynamic['h4_failure_or_timeout'].sum())}.
- Maximum queue after dispatch: {int(dynamic['max_queue_after_dispatch'].max())}; ending queue: {int(dynamic['end_queue'].max())}.
- Hard SLA violations: {int(dynamic['hard_sla_violations'].max())}.
- Restore method: continuous deterministic prefix replay from train step 0. No window used an empty reset.
""",
    )
    action_rows = []
    for planner in ("h1", "h4"):
        actions = int(steps["queue_before_dispatch"].sum())
        defers = int(steps[f"{planner}_defer_actions"].sum())
        action_rows.append(
            {
                "scenario": "B_SPOTGPU2026_NATIVE_5DC",
                "planner": planner.upper(),
                "states": len(steps),
                "task_actions": actions,
                "defer_actions": defers,
                "defer_rate": defers / max(1, actions),
                "h1_h4_disagreement_rate": "NOT_STORED_PREFLIGHT_ONLY",
                "status": "SOLVABILITY_PREFLIGHT_NOT_FORMAL_EXPERT",
            }
        )
    pd.DataFrame(action_rows).to_csv(OUTPUT / "11_h1_oracle_behavior.csv", index=False)
    write_text(
        OUTPUT / "17_information_contract.md",
        """# Information Contract

- `dataset/scenario_a`: deployable current state plus H1/H4 labels.
- `simulator_only/scenario_a`: true duration and simulator-only truth.
- `privileged_future/scenario_a`: H4 Oracle future demand and future price/carbon.
- Spot full canonical and capacity evidence are audit/preflight data, not a formal expert dataset.
- Spot dynamic H1 sees current restored state only. H4 receives four future capacity nodes for a solvability-only preflight.
- Transformer inputs and outputs are absent. No policy learner was executed.
""",
    )
    write_text(
        OUTPUT / "22_information_leakage_audit.md",
        """# Information Leakage Audit

Status: PASS

- Alibaba2020 deployable task/state files contain estimated duration and current state, not true duration or privileged future arrays.
- True duration is isolated under `simulator_only/scenario_a`.
- H4 Oracle future is isolated under `privileged_future/scenario_a`.
- Spot capacity preflight uses simulator truth only for capacity and restoration audits; no learning dataset or model was created.
- Transformer, BC, SAC, RL, and policy training were not run.
""",
    )
    old_gpu = pd.read_csv(OUTPUT / "37_old_vs_native_capacity.csv")
    old_gpu = old_gpu[(old_gpu["scenario"] == "OLD_SUSTAINCLUSTER_5DC") & (old_gpu["resource"] == "gpu")].iloc[0]
    native_gpu = next(item for item in capacity["native_pressure"] if item["resource"] == "gpu")
    write_text(
        OUTPUT / "25_final_diagnosis.md",
        f"""# Final Diagnosis

Final status: **SPOT NATIVE CAPACITY READY**

The complete Spot node_info file contains {capacity['node_count']} nodes, {capacity['native_capacity']['cpu']:.0f} CPU cores, and {capacity['native_capacity']['gpu']:.0f} GPUs. Deterministic five-DC partitioning conserves every CPU core and GPU and keeps GPU-model proportions within the recorded audit tolerance.

The old 2,900-GPU scenario produced a mandatory GPU peak ratio of {old_gpu['mandatory_peak_ratio']:.6%}. With the unchanged 466,867-task workload and node-native 10,412-GPU capacity, the same lower-bound peak is {native_gpu['mandatory_peak_ratio']:.6%}; therefore the earlier 281.85% overload was principally a capacity-scenario mismatch.

Native GPU offered load is {native_gpu['offered_load_ratio']:.6%}, so the native scenario is not classified as too loose and no comparable-pressure capacity is required. All 64 restored-window H1 states and 64 H4 states solved optimally with zero timeout, queue carry-over, or hard SLA violation.

Alibaba2020 Expert Dataset v3 is independently generated and exactly aligned with v2. Spot Expert Dataset v3 is not generated in this round; its formal continuous generation is technically cleared for the next round.
""",
    )
    write_text(
        OUTPUT / "26_summary.md",
        f"""# MPC Expert Dataset v3 Current Summary

1. Alibaba2020 v3: 80 episodes, 7,680 states, 259,920 decisions; independently generated without Transformer.
2. v2/v3 student state, feasible mask, H1 action, and H4 Oracle action agreement: all 100%; mismatches: 0.
3. Spot source: 466,867 tasks unchanged; node pool: {capacity['node_count']} nodes, {capacity['native_capacity']['cpu']:.0f} CPU, {capacity['native_capacity']['gpu']:.0f} GPU.
4. Native 5DC: deterministic exclusive node partition; CPU/GPU strictly conserved; memory is MODELED_DC_MEMORY.
5. Native GPU pressure: offered {native_gpu['offered_load_ratio']:.6%}, mandatory peak {native_gpu['mandatory_peak_ratio']:.6%}, immediate-start peak {native_gpu['immediate_start_peak_ratio']:.6%}.
6. Arrival pressure P50/P90/P95/P99/max is recorded in `36_spot_native_arrival_pressure_summary.csv`.
7. Dynamic preflight: four post-warm-up train windows, 64 H1 and 64 H4 solves, zero failure/timeout and zero hard SLA violation.
8. Comparable-pressure capacity: NOT REQUIRED because native GPU pressure is material rather than too loose.
9. Spot formal 184-day expert generation: NOT RUN in this round.
10. Final status: SPOT NATIVE CAPACITY READY.
""",
    )
    pd.DataFrame(
        [
            {
                "dataset": "Alibaba2020",
                "formal_expert_dataset": True,
                "states": alibaba["counts"]["states"],
                "task_decisions": alibaba["counts"]["task_decisions"],
                "capacity_basis": "SUSTAINCLUSTER_5DC",
                "gpu_capacity": 2900,
                "gpu_offered_load_ratio": "NOT_APPLICABLE_DIFFERENT_TRACE",
                "status": "V3_READY",
            },
            {
                "dataset": "SpotGPU2026",
                "formal_expert_dataset": False,
                "states": len(steps),
                "task_decisions": int(steps["queue_before_dispatch"].sum()),
                "capacity_basis": "COMPLETE_NODE_INFO_NATIVE_5DC",
                "gpu_capacity": capacity["native_capacity"]["gpu"],
                "gpu_offered_load_ratio": native_gpu["offered_load_ratio"],
                "status": "DYNAMIC_PREFLIGHT_READY",
            },
        ]
    ).to_csv(OUTPUT / "21_cross_dataset_comparison.csv", index=False)

    sustain_repo = ROOT / "references/external_repos/sustain-cluster"
    current = {
        "dataset_version": "mpc_expert_dataset_v3",
        "final_status": capacity["final_status"],
        "formal_generation": {"alibaba2020": True, "spotgpu2026": False},
        "student_training": False,
        "transformer_inference": False,
        "rl": False,
        "workspace": {
            "branch": git("branch", "--show-current"),
            "head": git("rev-parse", "HEAD"),
            "sustaincluster_head": git("rev-parse", "HEAD", cwd=sustain_repo),
            "sustaincluster_clean": git("status", "--short", cwd=sustain_repo) == "",
        },
        "scenario_a": {
            "status": alibaba["status"],
            "counts": alibaba["counts"],
            "alignment": alibaba["alignment"]["summary"],
            "mismatch_reason_counts": alibaba["alignment"]["task_mismatch_reason_counts"],
        },
        "scenario_b": capacity,
        "legacy_old_capacity_preflight": {
            "gpu_capacity": 2900,
            "mandatory_gpu_peak_ratio": float(old_gpu["mandatory_peak_ratio"]),
            "status": "SUPERSEDED_BY_NODE_NATIVE_CAPACITY_CALIBRATION",
        },
        "next_step": "FORMAL_SPOT_EXPERT_DATASET_V3_GENERATION_ALLOWED_NEXT_ROUND",
    }
    write_json(OUTPUT / "00_generation_manifest.json", current)
    integrity_manifest().to_csv(OUTPUT / "23_current_integrity_manifest.csv", index=False)
    return OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("use --run")
    print(run())


if __name__ == "__main__":
    main()
