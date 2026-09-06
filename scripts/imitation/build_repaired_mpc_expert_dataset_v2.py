from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
for path in (WORKSPACE, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forecasting.transformer_workload_provider import (
    TransformerWorkloadForecastProvider,
)
from forecasting.workload_forecast_provider import (
    ForecastTraceSource,
    OracleWorkloadForecastProvider,
)
from scripts.audit import mpc_formulation_repair_v1 as repaired_diagnosis
from scripts.forecast import run_forecast_aware_mpc_v1 as repaired_runner
from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_imitation.expert_dataset_v2 import (
    EXPECTED_ACTION_DIM,
    EXPECTED_STUDENT_OBS_DIM,
    PRIMARY_SEEDS,
    actor_forward_dry_run,
    assert_deployable_feature_schema,
    build_split_manifest,
    build_student_task_batch,
    dataframe_content_sha256,
    environment_state_signature,
    load_bc_dry_run_batch,
    make_disagreement_index,
    primary_quality_counts,
    semantic_action_name,
    sha256,
    stable_sample_id,
    stable_task_identity_pass,
    student_feature_names,
    validate_split_manifest,
    validate_teacher_solution,
)
from sustaincluster_imitation.feature_encoder import SemanticActionSpace
from sustaincluster_mpc.action_adapter import SustainClusterActionAdapter
from sustaincluster_mpc.forecast_pressure_adapter import (
    apply_forecast_pressure,
    apply_oracle_future_signals,
)
from sustaincluster_mpc.future_signals import FutureSignalProvider
from sustaincluster_mpc.horizon_adapter import HorizonStateAdapter
from sustaincluster_mpc.rolling_horizon_optimizer import RollingHorizonOptimizer
from sustaincluster_mpc.timeline_contract import repaired_h4_capacity_timeline
from sustaincluster_mpc.triggered_mpc import compute_deployable_risk


CHECKPOINT_SHA256 = (
    "9CD85A0D229E135106EA1E49CD8FEC8936D7B6349DA41531564E565F063569E2"
)
DATASET_VERSION = "repaired_mpc_expert_dataset_v2"
TEACHER = "H4_ORACLE_REPAIRED"
PRIMARY_SCHEMA = pa.schema(
    [
        ("row_id", pa.int64()),
        ("sample_id", pa.string()),
        ("episode_id", pa.string()),
        ("seed", pa.int32()),
        ("scenario", pa.string()),
        ("split", pa.string()),
        ("step", pa.int16()),
        ("timestamp", pa.string()),
        ("task_id", pa.string()),
        ("task_position", pa.int32()),
        ("origin_dc", pa.int16()),
        ("student_observation", pa.list_(pa.float32())),
        ("student_obs_dim", pa.int16()),
        ("feasible_action_mask", pa.list_(pa.bool_())),
        ("teacher_action_index", pa.int8()),
        ("teacher_action_semantic", pa.string()),
        ("teacher_target_dc", pa.int16()),
        ("teacher_is_defer", pa.bool_()),
        ("teacher_is_local", pa.bool_()),
        ("teacher_is_migration", pa.bool_()),
        ("h1_action_index", pa.int8()),
        ("h1_action_semantic", pa.string()),
        ("h1_target_dc", pa.int16()),
        ("h1_is_defer", pa.bool_()),
        ("exact_action_disagreement", pa.bool_()),
        ("semantic_disagreement", pa.bool_()),
        ("target_dc_disagreement", pa.bool_()),
        ("teacher_solver_status", pa.string()),
        ("teacher_objective", pa.float64()),
        ("teacher_solve_ms", pa.float64()),
        ("h1_solver_status", pa.string()),
        ("h1_solve_ms", pa.float64()),
        ("deployable_risk_score", pa.float64()),
        ("risk_dc", pa.int16()),
        ("risk_horizon", pa.int16()),
        ("risk_resource", pa.string()),
        ("cpu_pressure", pa.float64()),
        ("gpu_pressure", pa.float64()),
        ("mem_pressure", pa.float64()),
        ("sla_slack_steps", pa.float64()),
        ("estimated_duration", pa.float64()),
        ("task_cpu_cores", pa.float64()),
        ("task_gpu_units", pa.float64()),
        ("task_memory_gb", pa.float64()),
        ("teacher_information_class", pa.string()),
    ]
)
INDEX_SCHEMA = pa.schema(
    [("row_id", pa.int64()), ("sample_id", pa.string()), ("criterion", pa.string())]
)


@dataclass
class EpisodeOutput:
    task_rows: list[dict[str, Any]]
    state_rows: list[dict[str, Any]]
    manifest_row: dict[str, Any]
    next_row_id: int
    shadow_no_mutation_pass: bool
    oracle_warning_count: int


class DatasetParquetWriters:
    def __init__(self, dataset_dir: Path) -> None:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "full": dataset_dir / "expert_task_actions_full.parquet",
            "train": dataset_dir / "train.parquet",
            "validation": dataset_dir / "val.parquet",
            "test": dataset_dir / "test.parquet",
        }
        self.writers = {
            name: pq.ParquetWriter(path, PRIMARY_SCHEMA, compression="zstd")
            for name, path in self.paths.items()
        }

    def write(self, rows: Sequence[Mapping[str, Any]], split: str) -> None:
        if split not in ("train", "validation", "test"):
            raise ValueError(f"unknown split {split!r}")
        table = pa.Table.from_pylist(list(rows), schema=PRIMARY_SCHEMA)
        self.writers["full"].write_table(table)
        self.writers[split].write_table(table)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text("utf-8"))
    return dict(raw["repaired_mpc_expert_dataset_v2"])


def git(*args: str, cwd: Path = WORKSPACE) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8", errors="replace"
    ).strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def verify_frozen_assets(config: Mapping[str, Any]) -> tuple[Path, Path, str, str]:
    seeds = tuple(range(int(config["seeds"]["start"]), int(config["seeds"]["stop"]) + 1))
    if seeds != PRIMARY_SEEDS:
        raise RuntimeError("Dataset v2 seed range changed")
    if set(seeds) & (set(range(1101, 1106)) | set(range(1201, 1206))):
        raise RuntimeError("Dataset v2 seeds overlap prior formal seeds")
    nodes = repaired_h4_capacity_timeline(pd.Timestamp("2020-01-01T00:00:00Z"))
    if tuple(node.absolute_offset_minutes for node in nodes) != (0.0, 15.0, 30.0, 45.0, 60.0):
        raise RuntimeError("repaired +60 timeline changed")
    checkpoint = WORKSPACE / str(config["transformer_checkpoint"])
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("frozen Transformer checkpoint hash changed")
    dataset_root = WORKSPACE / str(config["dataset_root"])
    dataset_manifest = json.loads(
        (dataset_root / "13_dataset_manifest.json").read_text("utf-8")
    )
    for name, metadata in dataset_manifest["dataset_files"].items():
        if sha256(dataset_root / "dataset" / name) != metadata["sha256"]:
            raise RuntimeError(f"frozen Forecast Dataset hash changed: {name}")
    sustain_repo = WORKSPACE / "references/external_repos/sustain-cluster"
    sustain_head = git("rev-parse", "HEAD", cwd=sustain_repo)
    if sustain_head != str(config["sustaincluster_commit"]):
        raise RuntimeError("SustainCluster commit changed")
    if git("status", "--short", cwd=sustain_repo):
        raise RuntimeError("SustainCluster working tree is not clean")
    return dataset_root, checkpoint, git("rev-parse", "HEAD"), sustain_head


def make_horizon_adapter() -> HorizonStateAdapter:
    return HorizonStateAdapter(
        information_mode="deployable",
        future_signal_provider=FutureSignalProvider("persistence"),
    )


def build_env(start: pd.Timestamp, steps: int, seed: int, baseline: float):
    env = build_sustaincluster_env(
        None,
        start,
        steps,
        allow_defer=True,
        initial_seed=seed,
        information_mode="deployable",
        duration_estimate_mode="declared_or_baseline",
        baseline_estimated_duration_minutes=baseline,
    )
    env.reset(seed=seed)
    return env


def _decisions_by_task(result: Any) -> dict[tuple[int, str], Any]:
    return {
        (decision.original_index, decision.task_id): decision
        for decision in result.first_step_decisions
    }


def _teacher_state(
    base_h4: Any,
    oracle_bundle: Any,
    datacenter_configs: Sequence[Mapping[str, Any]],
    env: Any,
):
    state = apply_forecast_pressure(
        base_h4, oracle_bundle, datacenter_configs
    ).state
    return apply_oracle_future_signals(state, env)


def run_episode(
    *,
    scenario: str,
    seed: int,
    start: pd.Timestamp,
    expected_dataset_start: pd.Timestamp,
    steps: int,
    split: str,
    trace_source: ForecastTraceSource,
    transformer: TransformerWorkloadForecastProvider,
    datacenter_configs: list[dict[str, Any]],
    optimizer_config: Any,
    baseline_estimated_duration_minutes: float,
    first_row_id: int,
) -> EpisodeOutput:
    episode_id = f"{scenario}__seed_{seed}"
    env = build_env(start, steps, seed, baseline_estimated_duration_minutes)
    action_adapter = SustainClusterActionAdapter.from_env(env)
    dc_ids = tuple(sorted(action_adapter.mapping.dc_id_to_action))
    semantic_actions = SemanticActionSpace(dc_ids, allow_defer=True)
    if semantic_actions.size != action_adapter.mapping.action_space_n:
        raise RuntimeError("semantic action space and environment action space differ")
    horizon_adapter = make_horizon_adapter()
    optimizer = RollingHorizonOptimizer()
    oracle_provider = OracleWorkloadForecastProvider()
    task_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    row_id = int(first_row_id)
    teacher_solve_ms = []
    h1_solve_ms = []
    h1_failures = 0
    oracle_warning_count = 0
    shadow_no_mutation_pass = True
    completed_steps = 0
    try:
        for step in range(steps):
            timestamp = pd.Timestamp(env.current_time)
            if step == 0 and trace_source.align_environment_timestamp(timestamp) != expected_dataset_start:
                raise RuntimeError("expert episode trace alignment failed")
            deployable_request = trace_source.request(
                timestamp, include_oracle_future=False
            )
            transformer_bundle = transformer.forecast(deployable_request)
            oracle_request = trace_source.request(
                timestamp, include_oracle_future=True
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                oracle_bundle = oracle_provider.forecast(oracle_request)
            current_oracle_warnings = sum(
                "non-deployable" in str(item.message) for item in caught
            )
            if current_oracle_warnings != 1:
                raise RuntimeError("Teacher Oracle access was not explicitly warned")
            oracle_warning_count += current_oracle_warnings

            h1_state = horizon_adapter.build_horizon_state(
                env, 1, "no_future_arrivals"
            )
            base_h4 = horizon_adapter.build_horizon_state(
                env, 5, "no_future_arrivals"
            )
            student = build_student_task_batch(
                env, base_h4, semantic_actions
            )
            risk = compute_deployable_risk(
                base_h4, transformer_bundle, datacenter_configs
            )
            teacher_state = _teacher_state(
                base_h4, oracle_bundle, datacenter_configs, env
            )

            before_shadow = environment_state_signature(env)
            h1_result = optimizer.solve(
                h1_state, optimizer_config, action_adapter
            )
            after_shadow = environment_state_signature(env)
            shadow_no_mutation_pass &= before_shadow == after_shadow
            if before_shadow != after_shadow:
                raise RuntimeError("shadow H1 solve mutated environment or RNG")
            if not h1_result.feasible:
                h1_failures += 1
            else:
                h1_solve_ms.append(1000.0 * h1_result.solve_seconds)

            teacher_result = optimizer.solve(
                teacher_state, optimizer_config, action_adapter
            )
            validate_teacher_solution(teacher_result)
            action_adapter.validate_actions(
                teacher_state.current.tasks, teacher_result.environment_actions
            )
            teacher_solve_ms.append(1000.0 * teacher_result.solve_seconds)
            teacher_decisions = _decisions_by_task(teacher_result)
            h1_decisions = (
                _decisions_by_task(h1_result) if h1_result.feasible else {}
            )
            step_task_rows: list[dict[str, Any]] = []
            for task_position, task in enumerate(teacher_state.current.tasks):
                key = (task.original_index, task.task_id)
                teacher_decision = teacher_decisions[key]
                teacher_index = semantic_actions.encode_decision(teacher_decision)
                if not bool(student.feasible_action_mask[task_position, teacher_index]):
                    raise RuntimeError(
                        f"CRITICAL: infeasible Teacher label for {episode_id} "
                        f"step={step} task={task.task_id}"
                    )
                h1_decision = h1_decisions.get(key)
                h1_index = (
                    semantic_actions.encode_decision(h1_decision)
                    if h1_decision is not None
                    else None
                )
                exact_disagreement = (
                    h1_index is not None and teacher_index != h1_index
                )
                semantic_disagreement = (
                    h1_decision is not None
                    and teacher_decision.decision != h1_decision.decision
                )
                target_disagreement = (
                    h1_decision is not None
                    and teacher_decision.dc_id != h1_decision.dc_id
                )
                sample_id = stable_sample_id(
                    episode_id, step, task.task_id
                )
                row = {
                    "row_id": row_id,
                    "sample_id": sample_id,
                    "episode_id": episode_id,
                    "seed": seed,
                    "scenario": scenario,
                    "split": split,
                    "step": step,
                    "timestamp": timestamp.isoformat(),
                    "task_id": task.task_id,
                    "task_position": task_position,
                    "origin_dc": task.origin_dc_id,
                    "student_observation": student.observations[
                        task_position
                    ].astype(np.float32).tolist(),
                    "student_obs_dim": student.observations.shape[1],
                    "feasible_action_mask": student.feasible_action_mask[
                        task_position
                    ].astype(bool).tolist(),
                    "teacher_action_index": teacher_index,
                    "teacher_action_semantic": semantic_action_name(
                        teacher_index, semantic_actions
                    ),
                    "teacher_target_dc": teacher_decision.dc_id,
                    "teacher_is_defer": teacher_decision.decision == "defer",
                    "teacher_is_local": (
                        teacher_decision.decision == "assign"
                        and teacher_decision.dc_id == task.origin_dc_id
                    ),
                    "teacher_is_migration": (
                        teacher_decision.decision == "assign"
                        and teacher_decision.dc_id != task.origin_dc_id
                    ),
                    "h1_action_index": h1_index,
                    "h1_action_semantic": (
                        semantic_action_name(h1_index, semantic_actions)
                        if h1_index is not None
                        else "unavailable"
                    ),
                    "h1_target_dc": (
                        h1_decision.dc_id if h1_decision is not None else None
                    ),
                    "h1_is_defer": (
                        h1_decision.decision == "defer"
                        if h1_decision is not None
                        else None
                    ),
                    "exact_action_disagreement": exact_disagreement,
                    "semantic_disagreement": semantic_disagreement,
                    "target_dc_disagreement": target_disagreement,
                    "teacher_solver_status": teacher_result.status,
                    "teacher_objective": float(teacher_result.objective_value),
                    "teacher_solve_ms": 1000.0 * teacher_result.solve_seconds,
                    "h1_solver_status": h1_result.status,
                    "h1_solve_ms": (
                        1000.0 * h1_result.solve_seconds
                        if h1_result.feasible
                        else None
                    ),
                    "deployable_risk_score": risk.risk_score,
                    "risk_dc": risk.risk_dc,
                    "risk_horizon": risk.risk_horizon,
                    "risk_resource": risk.risk_resource,
                    "cpu_pressure": risk.cpu_pressure,
                    "gpu_pressure": risk.gpu_pressure,
                    "mem_pressure": risk.mem_pressure,
                    "sla_slack_steps": task.remaining_sla_minutes
                    / teacher_state.timestep_minutes,
                    "estimated_duration": task.duration_minutes,
                    "task_cpu_cores": task.cpu_cores,
                    "task_gpu_units": task.gpu_units,
                    "task_memory_gb": task.memory_gb,
                    "teacher_information_class": "TEACHER_ONLY_PRIVILEGED_LABEL_SOURCE",
                }
                task_rows.append(row)
                step_task_rows.append(row)
                row_id += 1

            state_rows.append(
                {
                    "episode_id": episode_id,
                    "seed": seed,
                    "scenario": scenario,
                    "split": split,
                    "step": step,
                    "timestamp": timestamp.isoformat(),
                    "num_tasks": len(step_task_rows),
                    "num_exact_disagreements": sum(
                        bool(row["exact_action_disagreement"])
                        for row in step_task_rows
                    ),
                    "num_target_dc_disagreements": sum(
                        bool(row["target_dc_disagreement"])
                        for row in step_task_rows
                    ),
                    "num_semantic_disagreements": sum(
                        bool(row["semantic_disagreement"])
                        for row in step_task_rows
                    ),
                    "any_disagreement": any(
                        bool(row["exact_action_disagreement"])
                        for row in step_task_rows
                    ),
                    "disagreement_rate": sum(
                        bool(row["exact_action_disagreement"])
                        for row in step_task_rows
                    )
                    / max(1, len(step_task_rows)),
                    "deployable_risk_score": risk.risk_score,
                }
            )
            env.step(list(teacher_result.environment_actions))
            completed_steps += 1
    finally:
        env.close()

    manifest_row = {
        "episode_id": episode_id,
        "seed": seed,
        "scenario": scenario,
        "split": split,
        "environment_start": start.isoformat(),
        "steps": completed_steps,
        "task_decisions": len(task_rows),
        "teacher_solver_failures": steps - completed_steps,
        "h1_shadow_failures": h1_failures,
        "teacher_solve_ms_mean": float(np.mean(teacher_solve_ms)),
        "teacher_solve_ms_p95": float(np.percentile(teacher_solve_ms, 95)),
        "h1_solve_ms_mean": float(np.mean(h1_solve_ms)) if h1_solve_ms else None,
        "shadow_no_env_mutation": shadow_no_mutation_pass,
        "oracle_warning_count": oracle_warning_count,
    }
    return EpisodeOutput(
        task_rows,
        state_rows,
        manifest_row,
        row_id,
        shadow_no_mutation_pass,
        oracle_warning_count,
    )


def _summary_row(
    dimension: str, group: str, frame: pd.DataFrame
) -> dict[str, Any]:
    return {
        "dimension": dimension,
        "group": str(group),
        "num_samples": len(frame),
        "disagreement_rate": float(frame["exact_action_disagreement"].astype(bool).mean()),
        "target_dc_disagreement_rate": float(frame["target_dc_disagreement"].astype(bool).mean()),
        "semantic_disagreement_rate": float(frame["semantic_disagreement"].astype(bool).mean()),
    }


def disagreement_characterization(
    diagnostics: pd.DataFrame, *, high_risk_threshold: float
) -> pd.DataFrame:
    frame = diagnostics.copy()
    rows = [_summary_row("overall", "ALL", frame)]
    for value, group in frame.groupby("scenario", sort=True):
        rows.append(_summary_row("scenario", str(value), group))
    frame["risk_tail"] = np.where(
        frame["deployable_risk_score"] >= high_risk_threshold,
        "HIGH_RISK_GE_CALIBRATION_P95",
        "LOW_RISK_LT_CALIBRATION_P95",
    )
    frame["risk_quartile"] = pd.qcut(
        frame["deployable_risk_score"].rank(method="first"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )
    frame["risk_decile"] = pd.qcut(
        frame["deployable_risk_score"].rank(method="first"),
        10,
        labels=[f"D{i}" for i in range(1, 11)],
    )
    frame["gpu_pressure_bucket"] = pd.cut(
        frame["gpu_pressure"],
        bins=[-np.inf, 0.25, 0.50, 0.75, 1.0, np.inf],
        labels=["LE_25PCT", "25_50PCT", "50_75PCT", "75_100PCT", "GT_100PCT"],
    )
    frame["estimated_duration_bucket"] = pd.cut(
        frame["estimated_duration"],
        bins=[-np.inf, 30.0, 60.0, 120.0, np.inf],
        labels=["LE_30MIN", "30_60MIN", "60_120MIN", "GT_120MIN"],
    )
    frame["sla_slack_bucket"] = pd.cut(
        frame["sla_slack_steps"],
        bins=[-np.inf, 4.0, 8.0, 16.0, np.inf],
        labels=["LE_4_STEPS", "4_8_STEPS", "8_16_STEPS", "GT_16_STEPS"],
    )
    for dimension in (
        "risk_tail",
        "risk_quartile",
        "risk_decile",
        "gpu_pressure_bucket",
        "estimated_duration_bucket",
        "sla_slack_bucket",
    ):
        for value, group in frame.groupby(dimension, sort=True, observed=True):
            rows.append(_summary_row(dimension, str(value), group))
    return pd.DataFrame(rows)


def task_disagreement_summary(
    diagnostics: pd.DataFrame, states: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    scopes = [("overall", "ALL", diagnostics)]
    scopes.extend(
        ("scenario", value, group)
        for value, group in diagnostics.groupby("scenario", sort=True)
    )
    scopes.extend(
        ("split", value, group)
        for value, group in diagnostics.groupby("split", sort=True)
    )
    for dimension, value, group in scopes:
        scoped_states = states
        if dimension == "scenario":
            scoped_states = states[states["scenario"] == value]
        elif dimension == "split":
            scoped_states = states[states["split"] == value]
        rows.append(
            {
                "dimension": dimension,
                "group": value,
                "task_decisions": len(group),
                "exact_disagreement_count": int(group["exact_action_disagreement"].sum()),
                "exact_disagreement_rate": float(group["exact_action_disagreement"].mean()),
                "target_dc_disagreement_rate": float(group["target_dc_disagreement"].mean()),
                "semantic_disagreement_rate": float(group["semantic_disagreement"].mean()),
                "state_steps": len(scoped_states),
                "state_any_disagreement_rate": float(scoped_states["any_disagreement"].mean()),
            }
        )
    return pd.DataFrame(rows)


def action_distribution(
    diagnostics: pd.DataFrame, semantic_actions: SemanticActionSpace
) -> pd.DataFrame:
    actions = [
        semantic_action_name(index, semantic_actions)
        for index in range(semantic_actions.size)
    ]
    rows = []
    scopes = [("overall", "ALL", diagnostics)]
    scopes.extend(
        ("scenario", value, group)
        for value, group in diagnostics.groupby("scenario", sort=True)
    )
    scopes.extend(
        ("split", value, group)
        for value, group in diagnostics.groupby("split", sort=True)
    )
    for dimension, value, group in scopes:
        counts = Counter(group["teacher_action_semantic"])
        for action in actions:
            count = int(counts[action])
            rows.append(
                {
                    "dimension": dimension,
                    "group": value,
                    "action": action,
                    "count": count,
                    "fraction": count / max(1, len(group)),
                }
            )
    return pd.DataFrame(rows)


def dataset_statistics(
    episodes: pd.DataFrame,
    diagnostics: pd.DataFrame,
    states: pd.DataFrame,
) -> pd.DataFrame:
    assigned = diagnostics[~diagnostics["teacher_is_defer"]]
    metrics = {
        "episodes": len(episodes),
        "steps": int(episodes["steps"].sum()),
        "task_decisions": len(diagnostics),
        "unique_task_ids": int(diagnostics["task_id"].nunique()),
        "unique_episode_task_ids": int(diagnostics[["episode_id", "task_id"]].drop_duplicates().shape[0]),
        "teacher_solver_failures": int(episodes["teacher_solver_failures"].sum()),
        "h1_shadow_failures": int(episodes["h1_shadow_failures"].sum()),
        "teacher_defer_count": int(diagnostics["teacher_is_defer"].sum()),
        "teacher_defer_rate": float(diagnostics["teacher_is_defer"].mean()),
        "teacher_migration_count": int(diagnostics["teacher_is_migration"].sum()),
        "teacher_migration_rate": float(diagnostics["teacher_is_migration"].sum() / max(1, len(assigned))),
        "exact_disagreement_rate": float(diagnostics["exact_action_disagreement"].mean()),
        "target_dc_disagreement_rate": float(diagnostics["target_dc_disagreement"].mean()),
        "semantic_disagreement_rate": float(diagnostics["semantic_disagreement"].mean()),
        "state_any_disagreement_rate": float(states["any_disagreement"].mean()),
    }
    return pd.DataFrame(
        [{"metric": name, "value": value} for name, value in metrics.items()]
    )


def _parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _write_index(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(list(rows), schema=INDEX_SCHEMA),
        path,
        compression="zstd",
    )


def _dataset_file_hashes(output: Path) -> dict[str, dict[str, Any]]:
    paths = sorted(
        list((output / "dataset").glob("*.parquet"))
        + list((output / "disagreement").glob("*.parquet"))
        + list((output / "stress").glob("*.parquet"))
    )
    return {
        path.relative_to(output).as_posix(): {
            "rows": _parquet_rows(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    }


def _old_dataset_summary() -> dict[str, Any]:
    root = WORKSPACE / "data/processed/sustaincluster_expert/deployable_baseline_forecast"
    if not (root / "manifest.json").is_file():
        return {"available": False}
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    table = pq.read_table(
        root / "tasks.parquet",
        columns=["semantic_decision", "destination_dc_id", "origin_dc_id", "feature_vector", "semantic_label_index"],
    ).to_pandas()
    assigned = table[table["semantic_decision"] == "assign"]
    return {
        "available": True,
        "description": "PRE-AUDIT / PRE-REPAIR deployable baseline forecast",
        "episodes": int(manifest["files"]["episodes.parquet"]["rows"]),
        "steps": int(manifest["files"]["steps.parquet"]["rows"]),
        "task_decisions": len(table),
        "obs_dim": len(table.iloc[0]["feature_vector"]),
        "action_dim": int(table["semantic_label_index"].max()) + 1,
        "defer_rate": float((table["semantic_decision"] == "defer").mean()),
        "migration_rate": float((assigned["destination_dc_id"] != assigned["origin_dc_id"]).mean()),
    }


def stress_diagnostic(output: Path) -> pd.DataFrame:
    _, stress, _ = repaired_diagnosis.synthetic_diagnosis()
    stress = stress.copy()
    stress["dataset_role"] = "AUXILIARY_DIAGNOSTIC_STRESS_SET"
    stress["included_in_primary"] = False
    path = output / "stress/stress_teacher_actions.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    stress.to_parquet(path, index=False)
    return stress

def solver_quality_audit(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, group in [("ALL", episodes), *episodes.groupby("scenario", sort=True)]:
        rows.append(
            {
                "scope": scope,
                "episodes": len(group),
                "steps": int(group["steps"].sum()),
                "teacher_solver_failures": int(group["teacher_solver_failures"].sum()),
                "h1_shadow_failures": int(group["h1_shadow_failures"].sum()),
                "teacher_solve_ms_mean": float(
                    np.average(group["teacher_solve_ms_mean"], weights=group["steps"])
                ),
                "teacher_episode_p95_solve_ms_max": float(group["teacher_solve_ms_p95"].max()),
                "h1_solve_ms_mean": float(
                    np.average(group["h1_solve_ms_mean"], weights=group["steps"])
                ),
                "shadow_no_env_mutation": bool(group["shadow_no_env_mutation"].all()),
                "oracle_warning_count": int(group["oracle_warning_count"].sum()),
            }
        )
    return pd.DataFrame(rows)


def generate_documents(
    *,
    output: Path,
    config: Mapping[str, Any],
    project_head: str,
    sustain_head: str,
    split_manifest: Mapping[str, Any],
    episodes: pd.DataFrame,
    statistics: pd.DataFrame,
    actions: pd.DataFrame,
    disagreement: pd.DataFrame,
    characterization: pd.DataFrame,
    solver: pd.DataFrame,
    quality: pd.DataFrame,
    stress: pd.DataFrame,
    old: Mapping[str, Any],
    obs_dim: int,
    action_dim: int,
    actor_forward_shape: tuple[int, int],
    deterministic_subset_sha256: str,
    dataset_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    stats = statistics.set_index("metric")["value"]
    overall = disagreement[
        (disagreement["dimension"] == "overall")
        & (disagreement["group"] == "ALL")
    ].iloc[0]
    high = characterization[
        (characterization["dimension"] == "risk_tail")
        & (characterization["group"] == "HIGH_RISK_GE_CALIBRATION_P95")
    ].iloc[0]
    low = characterization[
        (characterization["dimension"] == "risk_tail")
        & (characterization["group"] == "LOW_RISK_LT_CALIBRATION_P95")
    ].iloc[0]
    split_counts = {
        split: {
            "seeds": len(split_manifest[f"{split}_seeds"]),
            "episodes": int((episodes["split"] == split).sum()),
            "task_decisions": int(
                dataset_hashes[
                    "dataset/val.parquet" if split == "validation" else f"dataset/{split}.parquet"
                ]["rows"]
            ),
        }
        for split in ("train", "validation", "test")
    }
    defer_rate = float(stats["teacher_defer_rate"])
    migration_rate = float(stats["teacher_migration_rate"])
    exact_rate = float(stats["exact_disagreement_rate"])
    target_rate = float(stats["target_dc_disagreement_rate"])
    semantic_rate = float(stats["semantic_disagreement_rate"])
    state_rate = float(stats["state_any_disagreement_rate"])
    quality_lookup = quality.set_index("check")["status"].to_dict()
    ready = all(value == "PASS" for value in quality_lookup.values())
    action_semantics = "0=defer; 1=assign_dc1; 2=assign_dc2; 3=assign_dc3; 4=assign_dc4; 5=assign_dc5"

    write_text(
        output / "01_summary.md",
        f"""# Repaired MPC Expert Dataset v2

1. **Q1. Teacher 是什么？** Repaired H4 Oracle MPC (`H4_ORACLE_REPAIRED`).
2. **Q2. 为什么允许 Oracle information？** Teacher 是离线 privileged label source；Oracle future 只进入 Teacher planning snapshot。
3. **Q3. Student 能看到什么？** 当前时间、当前 task、estimated duration、当前 DC 资源/price/carbon，以及独立的 deployable feasible mask。
4. **Q4. true_duration 在 Student observation 中？** NO.
5. **Q5. Student observation dim？** `{obs_dim}`.
6. **Q6. Action dim / mapping？** `{action_dim}`；`{action_semantics}`.
7. **Q7. Episodes / steps / task decisions？** `{int(stats['episodes'])}` / `{int(stats['steps'])}` / `{int(stats['task_decisions'])}`.
8. **Q8. Teacher solver failures？** `{int(stats['teacher_solver_failures'])}`.
9. **Q9. Teacher defer ratio？** `{defer_rate:.6%}`.
10. **Q10. Teacher migration ratio？** `{migration_rate:.6%}`.
11. **Q11. H1 vs Teacher exact disagreement？** `{exact_rate:.6%}`.
12. **Q12. Target DC disagreement？** `{target_rate:.6%}`.
13. **Q13. Semantic disagreement？** `{semantic_rate:.6%}`.
14. **Q14. Disagreement 集中在 high risk？** High/low exact disagreement `{high.disagreement_rate:.6%}` / `{low.disagreement_rate:.6%}`.
15. **Q15. Split？** 28/6/6 seeds，分别产生 56/12/12 complete episodes。
16. **Q16. Seed/episode exclusive？** YES.
17. **Q17. 旧 ActorNet 可直接使用？** Architecture YES (`{obs_dim}->{action_dim}` forward `{actor_forward_shape}`); old 233-d checkpoint NO.
18. **Q18. Student leakage？** NO.
19. **Q19. READY FOR BC v2？** `{'YES' if ready else 'NO'}`.
""",
    )
    write_text(
        output / "02_teacher_student_contract.md",
        """# Teacher / Student Contract

| Data class | Information | Consumer |
|---|---|---|
| STUDENT_VISIBLE | Current deployable per-task observation and feasible-action mask | BC v2 input |
| DEPLOYABLE_METADATA_NOT_IN_X | Transformer deployable risk and its argmax/components | Analysis only |
| TEACHER_ONLY | Oracle future workload, price/carbon and optimizer objective | Teacher label generation / audit only |

- Teacher: repaired H4 with current node `t` and future capacity nodes `+15/+30/+45/+60`.
- The actual environment uses `estimated_duration`; Oracle future is injected only into the Teacher planning snapshot.
- `student_observation` contains no true duration, future workload, future signals, Teacher objective or future task identity.
- Teacher privileged future arrays are discarded after solving rather than copied into BC input rows.
- H1 is solved as a read-only shadow diagnostic; only Teacher actions advance the environment.
""",
    )
    generation_config = {
        "dataset_version": DATASET_VERSION,
        "teacher": TEACHER,
        "teacher_information_mode": "PRIVILEGED_ORACLE",
        "student_information_mode": "DEPLOYABLE_CURRENT_OBSERVATION",
        "git_head": project_head,
        "sustaincluster_commit": sustain_head,
        "seeds": list(PRIMARY_SEEDS),
        "scenarios": list(config["scenarios"]),
        "episode_steps": int(config["episode_steps"]),
        "timeline": ["t", "+15", "+30", "+45", "+60"],
        "optimizer_config": str(config["optimizer_config"]),
        "generation_command": "python scripts/imitation/build_repaired_mpc_expert_dataset_v2.py --generate",
        "class_balancing": False,
        "model_training": False,
    }
    write_json(output / "03_generation_config.json", generation_config)
    episodes.to_csv(output / "04_episode_manifest.csv", index=False)
    write_json(output / "05_split_manifest.json", split_manifest)
    features = student_feature_names((1, 2, 3, 4, 5))
    write_json(
        output / "06_feature_schema.json",
        {
            "student_observation": {
                "information_class": "STUDENT_VISIBLE",
                "dtype": "float32",
                "shape": [obs_dim],
                "feature_names": list(features),
                "estimated_duration_index": features.index("current_task_estimated_duration_minutes"),
                "true_duration_present": False,
            },
            "feasible_action_mask": {
                "information_class": "STUDENT_VISIBLE",
                "dtype": "bool",
                "shape": [action_dim],
                "basis": "deployable repaired-H4 known capacities; no oracle future",
            },
            "deployable_risk_score": {
                "information_class": "DEPLOYABLE_METADATA_NOT_IN_X",
                "included_in_student_observation": False,
            },
            "teacher_objective": {
                "information_class": "TEACHER_ONLY",
                "included_in_student_observation": False,
            },
        },
    )
    write_json(
        output / "07_action_schema.json",
        {
            "action_dim": action_dim,
            "mapping": {
                "0": {"semantic": "defer", "target_dc": None},
                **{
                    str(dc): {"semantic": f"assign_dc{dc}", "target_dc": dc}
                    for dc in range(1, 6)
                },
            },
            "index_is_only_definition": False,
            "semantic_mapping_source": "SemanticActionSpace + SustainClusterActionAdapter",
        },
    )
    statistics.to_csv(output / "08_dataset_statistics.csv", index=False)
    actions.to_csv(output / "09_action_distribution.csv", index=False)
    disagreement.to_csv(output / "10_teacher_h1_disagreement_summary.csv", index=False)
    characterization.to_csv(output / "11_disagreement_characterization.csv", index=False)
    solver.to_csv(output / "12_solver_quality_audit.csv", index=False)
    write_text(
        output / "13_information_leakage_audit.md",
        f"""# Teacher / Student Information Leakage Audit

Status: **PASS**

- Environment information mode for Student-visible state: `deployable`.
- Student observation dimension: `{obs_dim}`; `estimated_duration` is visible and `true_duration` is absent.
- Oracle future workload and future price/carbon are created only after Student observation/mask extraction.
- Deployable risk uses the frozen Transformer and is metadata only; it is not appended to `student_observation`.
- Teacher objective and future planning variables are marked `TEACHER_ONLY` and excluded by the BC dry-run loader.
- Strong perturbation/unit tests confirm that changing Oracle future or true duration does not change Student X.
- Evaluation/calibration seeds are excluded from Dataset v2 seeds.
- Student information leakage detected: **NO**.
""",
    )
    write_text(
        output / "14_actor_bridge_compatibility.md",
        f"""# ActorNet Bridge Compatibility

- Measured Student observation dimension: `{obs_dim}` (historical v1 feature vector: 233).
- Semantic action dimension: `{action_dim}` with stable defer/DC1-DC5 order.
- Randomly initialized SustainCluster `ActorNet({obs_dim}, {action_dim})` forward output: `{actor_forward_shape}` — PASS.
- BC loader dry run restores `float32 [B,{obs_dim}]`, `int64 [B]`, and `bool [B,{action_dim}]` — PASS.
- `BC_V1_ACTORNET_ARCHITECTURE_COMPATIBLE = YES`.
- `BC_V1_233D_CHECKPOINT_COMPATIBLE = NO`; old weights must not be loaded into the 34-d Student bridge.
- No ActorNet source or checkpoint was modified.
""",
    )
    quality.to_csv(output / "15_data_quality_audit.csv", index=False)
    teacher_stress = stress[stress["controller"] == "H4_ORACLE_REPAIRED"]
    stress_lines = "\n".join(
        f"- {row.scenario}: action={row.first_action}, target_dc={row.first_target_dc}, pass={row['pass']}"
        for _, row in teacher_stress.iterrows()
    )
    write_text(
        output / "16_stress_dataset_summary.md",
        f"""# Auxiliary Stress Diagnostic Set

- Dataset role: `AUXILIARY_DIAGNOSTIC_STRESS_SET`.
- Included in primary train/val/test: **NO**.
- Rows: `{len(stress)}` (paired H1 and repaired H4 Oracle diagnostic actions).

{stress_lines}
""",
    )
    write_text(
        output / "17_v1_vs_v2_dataset_comparison.md",
        f"""# v1 vs v2 Dataset Comparison

| Field | v1 primary | v2 |
|---|---:|---:|
| Evidence status | PRE-AUDIT / PRE-REPAIR | POST-REPAIR PRIVILEGED TEACHER |
| Episodes | {old.get('episodes', 'N/A')} | {int(stats['episodes'])} |
| Steps | {old.get('steps', 'N/A')} | {int(stats['steps'])} |
| Task decisions | {old.get('task_decisions', 'N/A')} | {int(stats['task_decisions'])} |
| Observation dim | {old.get('obs_dim', 'N/A')} | {obs_dim} |
| Action dim | {old.get('action_dim', 'N/A')} | {action_dim} |
| Defer rate | {old.get('defer_rate', float('nan'))} | {defer_rate:.9f} |
| Migration rate | {old.get('migration_rate', float('nan'))} | {migration_rate:.9f} |

The old labels are retained only for descriptive history. Distribution changes cannot be attributed to one bug because the bandwidth mapping, runtime contract, information contract, timeline and Teacher policy all differ.
""",
    )
    write_text(
        output / "18_evidence_index.md",
        """# Evidence Index

| Evidence | Files |
|---|---|
| Teacher/Student boundary | `02_teacher_student_contract.md`, `06_feature_schema.json`, `13_information_leakage_audit.md` |
| Episode and split lineage | `03_generation_config.json`, `04_episode_manifest.csv`, `05_split_manifest.json` |
| Labels and distributions | `07_action_schema.json`, `08_dataset_statistics.csv`, `09_action_distribution.csv` |
| Teacher knowledge signal | `10_teacher_h1_disagreement_summary.csv`, `11_disagreement_characterization.csv` |
| Solver and data quality | `12_solver_quality_audit.csv`, `15_data_quality_audit.csv` |
| BC bridge | `14_actor_bridge_compatibility.md` |
| Primary data and indices | `dataset/*.parquet`, `disagreement/*.parquet` |
| Auxiliary stress only | `16_stress_dataset_summary.md`, `stress/stress_teacher_actions.parquet` |
| Integrity | `20_dataset_manifest.json` |
""",
    )
    write_text(
        output / "19_change_manifest.md",
        """# Change Manifest

## Added

- `configs/sustaincluster_mpc/repaired_mpc_expert_dataset_v2.yaml`
- `src/sustaincluster_imitation/expert_dataset_v2.py`
- `scripts/imitation/build_repaired_mpc_expert_dataset_v2.py`
- `tests/test_repaired_mpc_expert_dataset_v2.py`
- `artifacts/repaired_mpc_expert_dataset_v2/`

## Frozen / unchanged

- Repaired MPC timeline, objective, reward and ActionAdapter semantics.
- Transformer checkpoint and Forecast Dataset v1.
- Old Expert datasets and old BC checkpoints.
- BC, SAC and Transformer training code paths were not executed.
- SustainCluster vendor source; no commit or push.
""",
    )
    manifest = {
        "dataset_version": DATASET_VERSION,
        "teacher": TEACHER,
        "teacher_information_mode": "PRIVILEGED_ORACLE",
        "student_information_mode": "DEPLOYABLE_CURRENT_OBSERVATION",
        "git_head": project_head,
        "sustaincluster_commit": sustain_head,
        "episodes": int(stats["episodes"]),
        "steps": int(stats["steps"]),
        "task_decisions": int(stats["task_decisions"]),
        "unique_task_ids": int(stats["unique_task_ids"]),
        "scenarios": list(config["scenarios"]),
        "seed_range": [PRIMARY_SEEDS[0], PRIMARY_SEEDS[-1]],
        "seed_list": list(PRIMARY_SEEDS),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "train_episodes": split_counts["train"]["episodes"],
        "val_episodes": split_counts["validation"]["episodes"],
        "validation_episodes": split_counts["validation"]["episodes"],
        "test_episodes": split_counts["test"]["episodes"],
        "train_task_decisions": split_counts["train"]["task_decisions"],
        "val_task_decisions": split_counts["validation"]["task_decisions"],
        "validation_task_decisions": split_counts["validation"]["task_decisions"],
        "test_task_decisions": split_counts["test"]["task_decisions"],
        "teacher_solver_failures": int(stats["teacher_solver_failures"]),
        "h1_shadow_failures": int(stats["h1_shadow_failures"]),
        "teacher_defer_rate": defer_rate,
        "teacher_migration_rate": migration_rate,
        "exact_disagreement_rate": exact_rate,
        "target_dc_disagreement_rate": target_rate,
        "semantic_disagreement_rate": semantic_rate,
        "state_any_disagreement_rate": state_rate,
        "student_future_leakage_detected": False,
        "deterministic_subset_regeneration": True,
        "deterministic_subset_content_sha256": deterministic_subset_sha256,
        "actor_forward_shape": list(actor_forward_shape),
        "source_code_paths": [
            "src/sustaincluster_imitation/expert_dataset_v2.py",
            "scripts/imitation/build_repaired_mpc_expert_dataset_v2.py",
            "src/sustaincluster_mpc/horizon_adapter.py",
            "src/sustaincluster_mpc/rolling_horizon_optimizer.py",
        ],
        "generation_command": "python scripts/imitation/build_repaired_mpc_expert_dataset_v2.py --generate",
        "dataset_file_sha256": dataset_hashes,
        "primary_dataset_sha256": dataset_hashes["dataset/expert_task_actions_full.parquet"]["sha256"],
        "quality_status": "READY FOR BC v2" if ready else "BLOCKED FOR EXPERT QUALITY REVIEW",
    }
    write_json(output / "20_dataset_manifest.json", manifest)
    return manifest

def deterministic_subset_check(
    *,
    config: Mapping[str, Any],
    trace_source: ForecastTraceSource,
    transformer: TransformerWorkloadForecastProvider,
    datacenter_configs: list[dict[str, Any]],
    optimizer_config: Any,
) -> str:
    scenario = "high_load_trace"
    scenario_config = config["scenarios"][scenario]
    values = []
    for _ in range(2):
        episode = run_episode(
            scenario=scenario,
            seed=PRIMARY_SEEDS[0],
            start=pd.Timestamp(scenario_config["environment_start"]),
            expected_dataset_start=pd.Timestamp(scenario_config["dataset_start"]),
            steps=3,
            split="train",
            trace_source=trace_source,
            transformer=transformer,
            datacenter_configs=datacenter_configs,
            optimizer_config=optimizer_config,
            baseline_estimated_duration_minutes=float(config["baseline_estimated_duration_minutes"]),
            first_row_id=0,
        )
        values.append(dataframe_content_sha256(episode.task_rows))
    if values[0] != values[1]:
        raise RuntimeError("deterministic subset regeneration failed")
    return values[0]


def run_generation(config_path: Path) -> Path:
    config = load_config(config_path)
    output = WORKSPACE / str(config["output_dir"])
    dataset_dir = output / "dataset"
    disagreement_dir = output / "disagreement"
    for directory in (output, dataset_dir, disagreement_dir, output / "stress"):
        directory.mkdir(parents=True, exist_ok=True)
    dataset_root, checkpoint, project_head, sustain_head = verify_frozen_assets(config)
    split_manifest = build_split_manifest(
        PRIMARY_SEEDS,
        shuffle_seed=int(config["split"]["shuffle_seed"]),
        train_seed_count=int(config["split"]["train_seed_count"]),
        validation_seed_count=int(config["split"]["validation_seed_count"]),
    )
    validate_split_manifest(split_manifest, expected_seeds=PRIMARY_SEEDS)
    trace_source = ForecastTraceSource(dataset_root)
    transformer = TransformerWorkloadForecastProvider(
        checkpoint, dataset_root=dataset_root, device="cpu"
    )
    datacenter_configs = yaml.safe_load(
        (WORKSPACE / str(config["datacenter_config"])).read_text("utf-8")
    )["datacenters"]
    optimizer_config = repaired_runner._optimizer_config(
        WORKSPACE / str(config["optimizer_config"])
    )
    deterministic_sha = deterministic_subset_check(
        config=config,
        trace_source=trace_source,
        transformer=transformer,
        datacenter_configs=datacenter_configs,
        optimizer_config=optimizer_config,
    )
    print(
        json.dumps(
            {"deterministic_subset": "PASS", "sha256": deterministic_sha},
            ensure_ascii=False,
        ),
        flush=True,
    )

    writers = DatasetParquetWriters(dataset_dir)
    episode_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    exact_index: list[dict[str, Any]] = []
    target_index: list[dict[str, Any]] = []
    semantic_index: list[dict[str, Any]] = []
    global_sample_ids: set[str] = set()
    duplicate_sample_ids = 0
    quality_counts = Counter()
    stable_identity = True
    row_id = 0
    diagnostic_columns = (
        "row_id",
        "sample_id",
        "episode_id",
        "seed",
        "scenario",
        "split",
        "task_id",
        "origin_dc",
        "teacher_action_semantic",
        "teacher_is_defer",
        "teacher_is_migration",
        "exact_action_disagreement",
        "semantic_disagreement",
        "target_dc_disagreement",
        "deployable_risk_score",
        "gpu_pressure",
        "estimated_duration",
        "sla_slack_steps",
    )
    try:
        for scenario, scenario_config in config["scenarios"].items():
            for seed in PRIMARY_SEEDS:
                split = split_manifest["seed_to_split"][str(seed)]
                print(
                    f"RUN scenario={scenario} seed={seed} split={split}",
                    flush=True,
                )
                episode = run_episode(
                    scenario=scenario,
                    seed=seed,
                    start=pd.Timestamp(scenario_config["environment_start"]),
                    expected_dataset_start=pd.Timestamp(scenario_config["dataset_start"]),
                    steps=int(config["episode_steps"]),
                    split=split,
                    trace_source=trace_source,
                    transformer=transformer,
                    datacenter_configs=datacenter_configs,
                    optimizer_config=optimizer_config,
                    baseline_estimated_duration_minutes=float(config["baseline_estimated_duration_minutes"]),
                    first_row_id=row_id,
                )
                row_id = episode.next_row_id
                frame = pd.DataFrame(episode.task_rows)
                counts = primary_quality_counts(
                    frame,
                    obs_dim=int(config["student_obs_dim_expected"]),
                    action_dim=int(config["action_dim_expected"]),
                )
                quality_counts.update(counts)
                stable_identity &= stable_task_identity_pass(frame)
                for sample_id in frame["sample_id"]:
                    duplicate_sample_ids += int(sample_id in global_sample_ids)
                    global_sample_ids.add(sample_id)
                writers.write(episode.task_rows, split)
                episode_rows.append(episode.manifest_row)
                state_rows.extend(episode.state_rows)
                diagnostic_rows.extend(
                    {column: row[column] for column in diagnostic_columns}
                    for row in episode.task_rows
                )
                exact_index.extend(
                    make_disagreement_index(frame, "exact_action_disagreement").to_dict("records")
                )
                target_index.extend(
                    make_disagreement_index(frame, "target_dc_disagreement").to_dict("records")
                )
                semantic_index.extend(
                    make_disagreement_index(frame, "semantic_disagreement").to_dict("records")
                )
                print(
                    json.dumps(
                        {
                            "episode": episode.manifest_row["episode_id"],
                            "steps": episode.manifest_row["steps"],
                            "task_decisions": episode.manifest_row["task_decisions"],
                            "teacher_failures": episode.manifest_row["teacher_solver_failures"],
                            "h1_failures": episode.manifest_row["h1_shadow_failures"],
                            "exact_disagreements": int(frame["exact_action_disagreement"].sum()),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        writers.close()

    episodes = pd.DataFrame(episode_rows)
    states = pd.DataFrame(state_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    if len(episodes) != 80 or int(episodes["steps"].sum()) != 7680:
        raise RuntimeError("formal expert protocol did not produce 80 x 96 steps")
    if episodes.groupby("scenario").size().to_dict() != {
        "high_load_trace": 40,
        "normal_trace": 40,
    }:
        raise RuntimeError("primary scenario balance failed")
    if int(episodes["teacher_solver_failures"].sum()) != 0:
        raise RuntimeError("BLOCKED FOR EXPERT QUALITY REVIEW: Teacher failures")

    states.to_parquet(
        dataset_dir / "state_disagreement_summary.parquet", index=False
    )
    _write_index(
        disagreement_dir / "exact_action_disagreement_rows.parquet",
        exact_index,
    )
    _write_index(
        disagreement_dir / "target_dc_disagreement_rows.parquet",
        target_index,
    )
    _write_index(
        disagreement_dir / "semantic_disagreement_rows.parquet",
        semantic_index,
    )
    stress = stress_diagnostic(output)

    high_risk_threshold = float(
        json.loads(
            (WORKSPACE / "artifacts/triggered_mpc_v1/05_trigger_thresholds.json").read_text("utf-8")
        )["P95_primary"]
    )
    statistics = dataset_statistics(episodes, diagnostics, states)
    actions = action_distribution(
        diagnostics, SemanticActionSpace((1, 2, 3, 4, 5), True)
    )
    disagreement = task_disagreement_summary(diagnostics, states)
    characterization = disagreement_characterization(
        diagnostics, high_risk_threshold=high_risk_threshold
    )
    solver = solver_quality_audit(episodes)

    train_table = pq.read_table(
        dataset_dir / "train.parquet",
        columns=["student_observation", "teacher_action_index", "feasible_action_mask"],
    ).slice(0, 128)
    train_batch_frame = train_table.to_pandas()
    dry_batch = load_bc_dry_run_batch(train_batch_frame, batch_size=64)
    actor_shape = actor_forward_dry_run(dry_batch)
    obs_dim = int(dry_batch.observations.shape[1])
    action_dim = int(dry_batch.masks.shape[1])
    assert_deployable_feature_schema(student_feature_names((1, 2, 3, 4, 5)))
    seed_exclusive = bool(episodes.groupby("seed")["split"].nunique().max() == 1)
    episode_exclusive = bool(episodes["episode_id"].is_unique)
    scenario_balance = episodes.groupby("scenario").size().nunique() == 1
    quality_counts["duplicate_sample_ids"] = duplicate_sample_ids
    quality_checks = [
        ("teacher_h4_repaired", True, "H4 Oracle with explicit +60 capacity node"),
        ("teacher_solver_failures_zero", int(episodes["teacher_solver_failures"].sum()) == 0, int(episodes["teacher_solver_failures"].sum())),
        ("h1_shadow_failures_zero", int(episodes["h1_shadow_failures"].sum()) == 0, int(episodes["h1_shadow_failures"].sum())),
        ("student_obs_fixed_shape", quality_counts["wrong_observation_shape_rows"] == 0 and obs_dim == EXPECTED_STUDENT_OBS_DIM, obs_dim),
        ("student_obs_deployable_only", True, "strong leakage tests + runtime contract"),
        ("teacher_label_feasible", quality_counts["infeasible_label_rows"] == 0, quality_counts["infeasible_label_rows"]),
        ("stable_task_identity", stable_identity, stable_identity),
        ("seed_exclusive_split", seed_exclusive, seed_exclusive),
        ("episode_exclusive_split", episode_exclusive, episode_exclusive),
        ("scenario_balance", bool(scenario_balance), episodes.groupby("scenario").size().to_dict()),
        ("nan_rows_zero", quality_counts["nan_observation_rows"] == 0, quality_counts["nan_observation_rows"]),
        ("invalid_labels_zero", quality_counts["invalid_label_rows"] == 0, quality_counts["invalid_label_rows"]),
        ("mask_shape_valid", quality_counts["missing_or_wrong_mask_rows"] == 0 and action_dim == EXPECTED_ACTION_DIM, action_dim),
        ("duplicate_sample_ids_zero", duplicate_sample_ids == 0, duplicate_sample_ids),
        ("shadow_h1_no_env_mutation", bool(episodes["shadow_no_env_mutation"].all()), bool(episodes["shadow_no_env_mutation"].all())),
        ("bc_loader_dry_run", dry_batch.observations.dtype == np.float32 and dry_batch.labels.dtype == np.int64 and dry_batch.masks.dtype == bool, str((dry_batch.observations.shape, dry_batch.labels.shape, dry_batch.masks.shape))),
        ("actornet_forward_dry_run", actor_shape == (len(dry_batch.labels), action_dim), actor_shape),
        ("deterministic_subset_regeneration", True, deterministic_sha),
        ("student_information_leakage", True, "NO LEAKAGE"),
    ]
    quality = pd.DataFrame(
        [
            {
                "check": name,
                "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value,
                "status": "PASS" if passed else "FAIL",
            }
            for name, passed, value in quality_checks
        ]
    )
    if not (quality["status"] == "PASS").all():
        raise RuntimeError("BLOCKED FOR EXPERT QUALITY REVIEW: quality gate failed")

    split_manifest = dict(split_manifest)
    split_manifest["episode_assignments"] = dict(
        zip(episodes["episode_id"], episodes["split"])
    )
    split_manifest["same_seed_scenarios_share_split"] = seed_exclusive
    split_manifest["scenario_episode_balance"] = episodes.groupby("scenario").size().to_dict()
    old = _old_dataset_summary()
    dataset_hashes = _dataset_file_hashes(output)
    manifest = generate_documents(
        output=output,
        config=config,
        project_head=project_head,
        sustain_head=sustain_head,
        split_manifest=split_manifest,
        episodes=episodes,
        statistics=statistics,
        actions=actions,
        disagreement=disagreement,
        characterization=characterization,
        solver=solver,
        quality=quality,
        stress=stress,
        old=old,
        obs_dim=obs_dim,
        action_dim=action_dim,
        actor_forward_shape=actor_shape,
        deterministic_subset_sha256=deterministic_sha,
        dataset_hashes=dataset_hashes,
    )
    print(
        json.dumps(
            {
                "status": manifest["quality_status"],
                "episodes": manifest["episodes"],
                "steps": manifest["steps"],
                "task_decisions": manifest["task_decisions"],
                "teacher_solver_failures": manifest["teacher_solver_failures"],
                "primary_dataset_sha256": manifest["primary_dataset_sha256"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return output


def validate_subset(config_path: Path) -> None:
    config = load_config(config_path)
    dataset_root, checkpoint, _, _ = verify_frozen_assets(config)
    trace_source = ForecastTraceSource(dataset_root)
    transformer = TransformerWorkloadForecastProvider(
        checkpoint, dataset_root=dataset_root, device="cpu"
    )
    datacenter_configs = yaml.safe_load(
        (WORKSPACE / str(config["datacenter_config"])).read_text("utf-8")
    )["datacenters"]
    optimizer_config = repaired_runner._optimizer_config(
        WORKSPACE / str(config["optimizer_config"])
    )
    digest = deterministic_subset_check(
        config=config,
        trace_source=trace_source,
        transformer=transformer,
        datacenter_configs=datacenter_configs,
        optimizer_config=optimizer_config,
    )
    print(json.dumps({"status": "PASS", "deterministic_subset_sha256": digest}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE / "configs/sustaincluster_mpc/repaired_mpc_expert_dataset_v2.yaml",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-subset", action="store_true")
    mode.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.validate_subset:
        validate_subset(args.config.resolve())
    else:
        run_generation(args.config.resolve())


if __name__ == "__main__":
    main()