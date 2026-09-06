from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.forecast import run_forecast_aware_mpc_v1 as repaired_runner
from scripts.imitation import build_repaired_mpc_expert_dataset_v2 as frozen_builder
from sustaincluster_imitation.bc_v2_offline import (
    BCV2Arrays,
    choose_recommended_run,
    classification_metrics,
    infer,
    load_checkpoint,
    per_action_metrics,
    sha256_file,
    train_seed,
)
from sustaincluster_imitation.expert_dataset_v2 import (
    build_student_task_batch,
)
from sustaincluster_imitation.feature_encoder import SemanticActionSpace
from sustaincluster_imitation.structured_current import (
    ACTION_DIM,
    DC_IDS,
    INFORMATION_CLASS,
    StructuredArrayStore,
    TrainOnlyStandardizer,
    canonical_digest,
    dc_feature_names,
    dense_feature_names,
    extract_current_state,
    infer_structured,
    json_histogram,
    label_entropy,
    load_structured_checkpoint,
    model_parameter_count,
    running_feature_names,
    task_feature_names,
    train_structured_seed,
)
from sustaincluster_mpc.action_adapter import SustainClusterActionAdapter
from sustaincluster_mpc.rolling_horizon_optimizer import RollingHorizonOptimizer


AUDIT_NAME = "Structured Current-State Representation Repair v1"
DIAGNOSES = {
    "FLAT FEATURE INSUFFICIENCY",
    "STRUCTURAL CONTEXT BOTTLENECK",
    "H1 LABEL / DECISION-CONTEXT AMBIGUITY",
    "MIXED",
    "INCONCLUSIVE",
}


def load_config(path: Path) -> dict[str, Any]:
    return dict(
        yaml.safe_load(Path(path).read_text(encoding="utf-8"))[
            "structured_current_state_representation_v1"
        ]
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
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8", errors="replace"
    ).strip()


def verify_frozen_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    full = ROOT / str(config["expert_dataset_dir"]) / "expert_task_actions_full.parquet"
    manifest = json.loads(
        (ROOT / str(config["expert_dataset_manifest"])).read_text(encoding="utf-8")
    )
    generation = json.loads(
        (ROOT / str(config["expert_dataset_generation_config"])).read_text(
            encoding="utf-8"
        )
    )
    split = json.loads(
        (ROOT / str(config["split_manifest"])).read_text(encoding="utf-8")
    )
    actual_sha = sha256_file(full)
    if actual_sha != str(config["expected_dataset_sha256"]).upper():
        raise RuntimeError("Expert Dataset v2 SHA256 changed")
    if actual_sha != str(manifest["primary_dataset_sha256"]).upper():
        raise RuntimeError("Expert Dataset v2 manifest mismatch")
    if (manifest["episodes"], manifest["steps"], manifest["task_decisions"]) != (
        80,
        7680,
        259920,
    ):
        raise RuntimeError("Expert Dataset v2 size changed")
    if int(manifest["obs_dim"]) != 34 or int(manifest["action_dim"]) != 6:
        raise RuntimeError("Expert Dataset v2 dimensions changed")
    if generation["git_head"] != str(config["expected_git_head"]):
        raise RuntimeError("Expert Dataset provenance HEAD changed")
    if git_value(ROOT, "rev-parse", "HEAD") != str(config["expected_git_head"]):
        raise RuntimeError("current project HEAD differs from frozen baseline")
    sustain = ROOT / "references/external_repos/sustain-cluster"
    sustain_head = git_value(sustain, "rev-parse", "HEAD")
    if sustain_head != str(config["expected_sustaincluster_commit"]):
        raise RuntimeError("SustainCluster commit changed")
    if git_value(sustain, "status", "--short"):
        raise RuntimeError("SustainCluster worktree is not clean")
    assignments = split["episode_assignments"]
    if len(assignments) != 80:
        raise RuntimeError("frozen split assignment count changed")
    return {
        "dataset_sha256": actual_sha,
        "project_head": str(config["expected_git_head"]),
        "sustaincluster_commit": sustain_head,
        "split_manifest": str(config["split_manifest"]),
    }


def information_flow_rows() -> list[dict[str, Any]]:
    def row(
        group: str,
        name: str,
        usage: str,
        source: str,
        current34: str,
        dense: str,
        structured: str,
        notes: str,
    ) -> dict[str, Any]:
        return {
            "group": group,
            "planner_input_name": name,
            "planner_usage": usage,
            "source": source,
            "current34_representation": current34,
            "current_dense_representation": dense,
            "structured_representation": structured,
            "information_class": INFORMATION_CLASS,
            "notes": notes,
        }

    rows = [
        row("decision", "all pending tasks", "one MILP contains every task variable", "HorizonState.current.tasks", "focal task only", "queue aggregates", "pending set", "H1 is joint, not sequential"),
        row("decision", "task order", "MILP variable order and result alignment", "env.current_tasks list order -> original_index", "missing", "decision_position", "task position feature", "Python list order is stable under frozen replay"),
        row("task", "CPU demand", "capacity constraint and energy", "TaskSnapshot.cpu_cores", "exact index 5", "exact", "per-task", "current and deployable"),
        row("task", "GPU demand", "capacity constraint and energy", "TaskSnapshot.gpu_units", "exact index 6", "exact", "per-task", "current and deployable"),
        row("task", "memory demand", "capacity constraint and energy", "TaskSnapshot.memory_gb", "missing", "exact", "per-task", "critical missing Current34 field"),
        row("task", "estimated duration", "duration steps, energy and SLA", "controller_duration_minutes(..., deployable)", "exact index 7", "exact", "per-task", "never true duration"),
        row("task", "remaining SLA", "deadline and SLA-risk objective", "TaskSnapshot.remaining_sla_minutes", "exact index 8", "exact", "per-task", "current clock-derived slack"),
        row("task", "bandwidth", "transmission delay/cost construction", "TaskSnapshot.bandwidth_gb", "missing", "exact", "per-task", "critical missing Current34 field"),
        row("task", "origin DC", "network path and local/migration semantics", "TaskSnapshot.origin_dc_id", "exact index 4", "one-hot", "one-hot per task", "semantic DC identity"),
        row("task", "wait/defer state", "current scheduler context", "wait_intervals/was_deferred/scheduler_wait_intervals", "SLA only", "exact", "per-task", "not directly weighted by H1 dispatch-step cost at H1"),
        row("network", "destination transmission cost", "transmission objective", "TaskDestinationSnapshot.transmission_cost_usd", "missing", "5 destination pairs", "derived through focal and DC identity", "uses current static network matrix and bandwidth"),
        row("network", "destination transmission delay", "transfer steps and SLA feasibility", "TaskDestinationSnapshot.transmission_delay_seconds", "missing", "5 destination pairs", "recoverable from task/DC current attributes", "H1 clamps transfer to at least one step"),
        row("dc", "CPU/GPU/MEM total", "capacity normalization and bounds", "DataCenterSnapshot totals", "implicit ratios only", "exact per DC", "DC set", "static deployable capacity"),
        row("dc", "raw available CPU/GPU/MEM", "base current capacity", "DataCenterSnapshot available", "ratios only", "exact per DC", "DC set", "absolute magnitudes retained"),
        row("dc", "queued reservations", "subtracted from H1 h0 capacity", "dc.pending_tasks through known reservations", "aggregate ratio only", "exact reservation totals", "DC set and pending context", "assigned queue, distinct from current joint batch"),
        row("dc", "in-transit reservations", "subtracted when arrival_step is current", "env.in_transit_tasks", "aggregate ratio only", "per-DC aggregates", "DC set", "current known reservations"),
        row("dc", "running releases at h0", "added to H1 h0 capacity when release_step=0", "controller-visible finish_time", "aggregate ratio only", "running summaries and exact H1 capacity", "running set", "estimated/controller-visible, never true duration"),
        row("dc", "H1 current schedulable capacity", "joint CPU/GPU/MEM constraints", "HorizonDataCenterSnapshot.*_available[0]", "available ratio before full reservation semantics", "exact absolute values", "DC set", "direct H1 constraint RHS"),
        row("dc", "current electricity price", "energy objective", "HorizonDataCenterSnapshot.price[0]", "scaled per DC", "exact", "DC set", "current value only"),
        row("dc", "current carbon intensity", "carbon objective", "HorizonDataCenterSnapshot.carbon[0]", "scaled per DC", "exact", "DC set", "current value only"),
        row("context", "other pending task demands", "joint capacity competition", "all HorizonState.current.tasks", "missing", "sum/mean/max/p50/p90", "full pending set excluding focal in pooled context", "primary structural hypothesis"),
        row("context", "running task composition", "builds release-adjusted current capacity", "HorizonState.running_tasks", "missing", "per-DC summaries", "full running set", "estimated release step from current tasks"),
        row("context", "DC optimizer order", "deterministic epsilon ranks dc_index", "HorizonState.datacenters tuple order", "semantic ratios only", "optimizer_order_position", "DC identity plus order", "environment insertion order is DC3,DC1,DC4,DC2,DC5"),
        row("constraint", "feasible action mask", "student legality mask", "SustainClusterFeatureEncoder", "separate exact mask", "same exact mask", "same exact mask", "not part of feature normalization"),
        row("excluded", "future arrivals/signals", "not used by H1", "H1 no_future_arrivals, horizon=1", "absent", "absent", "absent", "Oracle and Transformer inputs prohibited"),
    ]
    return rows


def write_information_flow(output: Path) -> None:
    frame = pd.DataFrame(information_flow_rows())
    frame.to_csv(output / "02_h1_decision_information_flow.csv", index=False)
    write_text(
        output / "01_h1_decision_information_flow.md",
        """# H1 Decision Information Flow

## Decision process

H1 is a **joint current-batch MILP**. For the current `k_t` tasks, one call creates `k_t * 5 * 1 + k_t` binary variables, one exactly-once constraint per task, and shared CPU/GPU/Memory constraints across all tasks and DCs. It then decodes all first-step actions together. There is no within-call `env.step`, residual-capacity mutation, or sequential prior-action context.

Task order comes from the stable Python order of `env.current_tasks`; `SustainClusterStateAdapter.get_pending_tasks()` assigns that list position as `original_index`. The MILP variable layout follows this order. The explicit epsilon tie break ranks dispatch step and **optimizer DC tuple position**, but does not break symmetry between tasks competing for an equivalent slot. Therefore task order remains observable decision context and is retained by both repaired representations.

## Current-only boundary

Every new input is derived from the deployable H1 `HorizonState(horizon=1, forecast_mode=no_future_arrivals)`. CurrentDense keeps a fixed vector with focal task, destination network values, exact current DC constraint values, pending aggregates, running summaries, and in-transit summaries. StructuredCurrent keeps the focal task plus full pending, running, and DC sets. The pending pooled context excludes the focal task (`sum(phi(all pending)) - phi(focal)`).

No future arrival, Oracle signal, Transformer forecast, Teacher/H1 action, objective value, MILP variable, reward, or true duration is an input. Running release steps use controller-visible estimated finish times in deployable mode. H1 has horizon 1, so no `+15/+30/+45/+60` future release profile is invented.
""",
    )


def read_frozen_frame(config: Mapping[str, Any]) -> pd.DataFrame:
    path = ROOT / str(config["expert_dataset_dir"]) / "expert_task_actions_full.parquet"
    frame = pd.read_parquet(path).sort_values("row_id").reset_index(drop=True)
    if len(frame) != 259920 or not np.array_equal(frame["row_id"], np.arange(len(frame))):
        raise RuntimeError("frozen task row identity changed")
    if frame["h1_action_index"].isna().any():
        raise RuntimeError("frozen H1 labels are incomplete")
    return frame


def semantic_to_environment_actions(
    semantic_indices: Sequence[int],
    semantic: SemanticActionSpace,
    adapter: SustainClusterActionAdapter,
) -> list[int]:
    actions = []
    for value in semantic_indices:
        index = int(value)
        if index == int(semantic.defer_index):
            actions.append(int(adapter.mapping.defer_action))
        else:
            dc_id = semantic.dc_for_index(index)
            actions.append(int(adapter.mapping.dc_id_to_action[int(dc_id)]))
    return actions


def _state_table(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["feasible_count"] = working["feasible_action_mask"].map(
        lambda value: int(np.asarray(value, dtype=bool).sum())
    )
    state = (
        working.groupby(["episode_id", "step"], sort=False)
        .agg(
            split=("split", "first"),
            scenario=("scenario", "first"),
            seed=("seed", "first"),
            timestamp=("timestamp", "first"),
            risk=("deployable_risk_score", "first"),
            queue_size=("sample_id", "size"),
            feasible_mean=("feasible_count", "mean"),
            oracle_consensus=("exact_action_disagreement", lambda values: not bool(np.any(values))),
        )
        .reset_index()
    )
    state["state_id"] = state.apply(
        lambda row: f"{row.episode_id}::step_{int(row.step):03d}", axis=1
    )
    return state


def _representative_targets(state: pd.DataFrame) -> dict[str, tuple[str, int]]:
    risk = state["risk"].to_numpy(dtype=float)
    median = float(np.median(risk))
    rows = {
        "low_risk": state.loc[state["risk"].idxmin()],
        "median_risk": state.iloc[int(np.argmin(np.abs(risk - median)))],
        "high_risk": state.loc[state["risk"].idxmax()],
        "large_pending_batch": state.loc[state["queue_size"].idxmax()],
        "multiple_feasible_dcs": state.loc[state["feasible_mean"].idxmax()],
    }
    return {
        name: (str(row["episode_id"]), int(row["step"])) for name, row in rows.items()
    }


def replay_current_states(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, tuple[Any, SustainClusterActionAdapter]]]:
    episodes = pd.read_csv(ROOT / str(config["episode_manifest"]))
    optimizer_config = repaired_runner._optimizer_config(
        ROOT / str(config["optimizer_config"])
    )
    optimizer = RollingHorizonOptimizer()
    state_table = _state_table(frame)
    targets = _representative_targets(state_table)
    target_by_key = {value: name for name, value in targets.items()}
    candidates: dict[str, tuple[Any, SustainClusterActionAdapter]] = {}
    capacity_edge: tuple[float, Any, SustainClusterActionAdapter] | None = None
    pending_parts = []
    running_parts = []
    dc_parts = []
    dense_parts = []
    state_offsets = [0]
    running_offsets = [0]
    state_rows = []
    task_state_index = np.empty(len(frame), dtype=np.int32)
    replay_task_cursor = 0
    replay_h1_mismatches = 0
    replay_observation_mismatches = 0
    replay_mask_mismatches = 0

    for episode_number, episode in episodes.iterrows():
        episode_id = str(episode["episode_id"])
        episode_rows = frame.loc[frame["episode_id"] == episode_id]
        env = frozen_builder.build_env(
            pd.Timestamp(episode["environment_start"]),
            int(episode["steps"]),
            int(episode["seed"]),
            60.0,
        )
        adapter = SustainClusterActionAdapter.from_env(env)
        semantic = SemanticActionSpace(tuple(sorted(adapter.mapping.dc_id_to_action)), True)
        horizon_adapter = frozen_builder.make_horizon_adapter()
        try:
            for step in range(int(episode["steps"])):
                rows = episode_rows.loc[episode_rows["step"] == step].sort_values(
                    "task_position"
                )
                h1_state = horizon_adapter.build_horizon_state(
                    env, 1, "no_future_arrivals"
                )
                mask_state = horizon_adapter.build_horizon_state(
                    env, 5, "no_future_arrivals"
                )
                student = build_student_task_batch(env, mask_state, semantic)
                expected_ids = tuple(str(value) for value in rows["task_id"])
                saved_observations = np.stack(
                    rows["student_observation"].map(
                        lambda value: np.asarray(value, dtype=np.float32)
                    )
                )
                saved_masks = np.stack(
                    rows["feasible_action_mask"].map(
                        lambda value: np.asarray(value, dtype=bool)
                    )
                )
                replay_observation_mismatches += int(
                    expected_ids != student.task_ids
                    or not np.array_equal(saved_observations, student.observations)
                )
                replay_mask_mismatches += int(
                    not np.array_equal(saved_masks, student.feasible_action_mask)
                )
                result = optimizer.solve(h1_state, optimizer_config, adapter)
                if not result.feasible:
                    raise RuntimeError(f"H1 replay failed for {episode_id} step={step}")
                replay_labels = np.asarray(
                    [semantic.encode_decision(item) for item in result.first_step_decisions],
                    dtype=np.int64,
                )
                saved_labels = rows["h1_action_index"].to_numpy(dtype=np.int64)
                replay_h1_mismatches += int(not np.array_equal(replay_labels, saved_labels))
                representation = extract_current_state(h1_state)
                if len(rows) != len(representation.pending):
                    raise RuntimeError("replayed pending set does not align with frozen rows")
                state_index = len(state_rows)
                row_indices = rows["row_id"].to_numpy(dtype=np.int64)
                if not np.array_equal(
                    row_indices, np.arange(replay_task_cursor, replay_task_cursor + len(rows))
                ):
                    raise RuntimeError("replay row order differs from frozen dataset")
                task_state_index[row_indices] = state_index
                replay_task_cursor += len(rows)
                pending_parts.append(representation.pending)
                running_parts.append(representation.running)
                dc_parts.append(representation.datacenters)
                dense_parts.append(representation.dense)
                state_offsets.append(state_offsets[-1] + len(representation.pending))
                running_offsets.append(running_offsets[-1] + len(representation.running))
                state_id = f"{episode_id}::step_{step:03d}"
                state_rows.append(
                    {
                        "state_index": state_index,
                        "state_id": state_id,
                        "episode_id": episode_id,
                        "step": step,
                        "split": str(rows["split"].iloc[0]),
                        "scenario": str(rows["scenario"].iloc[0]),
                        "seed": int(rows["seed"].iloc[0]),
                        "timestamp": str(rows["timestamp"].iloc[0]),
                        "queue_size": len(rows),
                        "running_size": len(representation.running),
                        "resource_pressure": representation.resource_pressure,
                        "deployable_risk_score": float(rows["deployable_risk_score"].iloc[0]),
                        "oracle_consensus_state": not bool(rows["exact_action_disagreement"].any()),
                    }
                )
                key = (episode_id, step)
                if key in target_by_key:
                    candidates[target_by_key[key]] = (h1_state, adapter)
                if capacity_edge is None or representation.resource_pressure > capacity_edge[0]:
                    capacity_edge = (representation.resource_pressure, h1_state, adapter)
                teacher_actions = semantic_to_environment_actions(
                    rows["teacher_action_index"].to_numpy(dtype=np.int64),
                    semantic,
                    adapter,
                )
                env.step(teacher_actions)
        finally:
            env.close()
        print(
            f"replay episode={episode_id} states={(episode_number + 1) * int(episode['steps'])}/7680",
            flush=True,
        )
    if capacity_edge is None:
        raise RuntimeError("no capacity-edge representative state was found")
    candidates["capacity_edge"] = (capacity_edge[1], capacity_edge[2])
    if set(candidates) != {
        "low_risk",
        "median_risk",
        "high_risk",
        "large_pending_batch",
        "multiple_feasible_dcs",
        "capacity_edge",
    }:
        raise RuntimeError("representative H1 states are incomplete")
    if replay_task_cursor != len(frame):
        raise RuntimeError("replay did not cover every frozen task decision")
    if replay_h1_mismatches or replay_observation_mismatches or replay_mask_mismatches:
        raise RuntimeError(
            "frozen replay mismatch: "
            f"H1={replay_h1_mismatches}, obs={replay_observation_mismatches}, "
            f"mask={replay_mask_mismatches}"
        )
    return (
        {
            "pending_raw": np.concatenate(pending_parts).astype(np.float32),
            "running_raw": np.concatenate(running_parts).astype(np.float32),
            "dc_raw": np.stack(dc_parts).astype(np.float32),
            "dense_raw": np.concatenate(dense_parts).astype(np.float32),
            "state_offsets": np.asarray(state_offsets, dtype=np.int64),
            "running_offsets": np.asarray(running_offsets, dtype=np.int64),
            "state_frame": pd.DataFrame(state_rows),
            "task_state_index": task_state_index,
            "replay_h1_mismatches": replay_h1_mismatches,
            "replay_observation_mismatches": replay_observation_mismatches,
            "replay_mask_mismatches": replay_mask_mismatches,
        },
        candidates,
    )


def determinism_audit(
    candidates: Mapping[str, tuple[Any, SustainClusterActionAdapter]],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    optimizer = RollingHorizonOptimizer()
    optimizer_config = repaired_runner._optimizer_config(
        ROOT / str(config["optimizer_config"])
    )
    repetitions = int(config["determinism"]["repetitions"])
    rows = []
    for category, (state, adapter) in candidates.items():
        semantic = SemanticActionSpace(tuple(sorted(adapter.mapping.dc_id_to_action)), True)
        action_runs = []
        objectives = []
        for _ in range(repetitions):
            result = optimizer.solve(state, optimizer_config, adapter)
            if not result.feasible:
                raise RuntimeError("representative H1 determinism solve failed")
            action_runs.append(
                tuple(semantic.encode_decision(item) for item in result.first_step_decisions)
            )
            objectives.append(float(result.objective_value))
        baseline = action_runs[0]
        repeat_equal = all(run == baseline for run in action_runs[1:])
        reversed_tasks = tuple(
            replace(task, original_index=index)
            for index, task in enumerate(reversed(state.current.tasks))
        )
        reversed_current = replace(state.current, tasks=reversed_tasks)
        reversed_state = replace(state, current=reversed_current)
        reversed_result = optimizer.solve(reversed_state, optimizer_config, adapter)
        if not reversed_result.feasible:
            raise RuntimeError("task-order sensitivity solve failed")
        baseline_by_task = {
            task.task_id: action for task, action in zip(state.current.tasks, baseline)
        }
        reversed_by_task = {
            item.task_id: semantic.encode_decision(item)
            for item in reversed_result.first_step_decisions
        }
        changed = sum(
            baseline_by_task[task_id] != reversed_by_task[task_id]
            for task_id in baseline_by_task
        )
        rows.append(
            {
                "category": category,
                "state_id": f"{state.current.exogenous.current_time_utc}::{len(state.current.tasks)}tasks",
                "task_count": len(state.current.tasks),
                "repetitions": repetitions,
                "action_reproducibility": float(
                    np.mean(
                        [
                            np.mean(np.asarray(run) == np.asarray(baseline))
                            for run in action_runs
                        ]
                    )
                ),
                "full_multi_action_reproducibility": float(
                    np.mean([run == baseline for run in action_runs])
                ),
                "exact_repeat_deterministic": repeat_equal,
                "objective_range": max(objectives) - min(objectives),
                "reversed_order_changed_task_actions": changed,
                "reversed_order_same_by_task": changed == 0,
            }
        )
    return pd.DataFrame(rows)


def write_determinism_reports(output: Path, metrics: pd.DataFrame) -> None:
    metrics.to_csv(output / "04_h1_determinism_metrics.csv", index=False)
    exact_pass = bool(metrics["exact_repeat_deterministic"].all())
    order_sensitive = int((~metrics["reversed_order_same_by_task"]).sum())
    write_text(
        output / "03_h1_determinism_audit.md",
        f"""# H1 Determinism Audit

- Representative states: `{len(metrics)}` (low/median/high risk, large pending batch, multiple feasible DCs, and capacity edge).
- Repetitions per exact state: `{int(metrics['repetitions'].iloc[0])}`.
- Same exact state, task order, and feasible semantics reproducible: **{'PASS' if exact_pass else 'FAIL'}**.
- Objective range across repeated calls: `{float(metrics['objective_range'].max()):.12g}` maximum.
- States whose per-task action changed after reversing task order: `{order_sensitive}/{len(metrics)}`.

H1 is solved jointly in one MILP and performs no within-step sequential mutation. Exact repeated calls are the determinism criterion. Reversing task order is a separate sensitivity diagnostic because variable order can resolve task-symmetric optima even when every exact repeated call is deterministic. The repaired student representations retain `decision_position`; no teacher-forced prior-action context is used or needed.
""",
    )
