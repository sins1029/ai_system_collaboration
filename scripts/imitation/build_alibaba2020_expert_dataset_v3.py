from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from forecasting.workload_forecast_provider import (  # noqa: E402
    ForecastTraceSource,
    OracleWorkloadForecastProvider,
)
from scripts.forecast import run_forecast_aware_mpc_v1 as repaired_runner  # noqa: E402
from sustaincluster_imitation.environment_factory import (  # noqa: E402
    build_sustaincluster_env,
)
from sustaincluster_imitation.expert_dataset_v2 import (  # noqa: E402
    PRIMARY_SEEDS,
    build_split_manifest,
    build_student_task_batch,
    environment_state_signature,
    semantic_action_name,
    sha256,
    stable_sample_id,
    validate_split_manifest,
    validate_teacher_solution,
)
from sustaincluster_imitation.feature_encoder import SemanticActionSpace  # noqa: E402
from sustaincluster_mpc.action_adapter import SustainClusterActionAdapter  # noqa: E402
from sustaincluster_mpc.forecast_pressure_adapter import (  # noqa: E402
    apply_forecast_pressure,
    apply_oracle_future_signals,
    distribute_global_forecast,
)
from sustaincluster_mpc.future_signals import FutureSignalProvider  # noqa: E402
from sustaincluster_mpc.horizon_adapter import HorizonStateAdapter  # noqa: E402
from sustaincluster_mpc.rolling_horizon_optimizer import (  # noqa: E402
    RollingHorizonOptimizer,
)


CONFIG_PATH = ROOT / "configs/sustaincluster_mpc/mpc_expert_dataset_v3.yaml"
V2_CONFIG_PATH = ROOT / "configs/sustaincluster_mpc/repaired_mpc_expert_dataset_v2.yaml"
DATASET_VERSION = "alibaba2020_mpc_expert_dataset_v3"
H1_PROVENANCE = "H1_REPAIRED_CURRENT_ONLY"
H4_PROVENANCE = "H4_ORACLE_REPAIRED"
ACTION_AGREEMENT_GATE = 0.99


TASK_SCHEMA = pa.schema(
    [
        ("row_id", pa.int64()),
        ("state_id", pa.string()),
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
        ("sla_slack_steps", pa.float64()),
        ("estimated_duration", pa.float64()),
        ("task_cpu_cores", pa.float64()),
        ("task_gpu_units", pa.float64()),
        ("task_memory_gb", pa.float64()),
        ("h1_label_provenance", pa.string()),
        ("teacher_label_provenance", pa.string()),
        ("information_class", pa.string()),
    ]
)

TRUTH_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("state_id", pa.string()),
        ("episode_id", pa.string()),
        ("seed", pa.int32()),
        ("scenario", pa.string()),
        ("split", pa.string()),
        ("step", pa.int16()),
        ("timestamp", pa.string()),
        ("task_id", pa.string()),
        ("true_duration_minutes", pa.float64()),
        ("duration_source_type", pa.string()),
        ("sla_deadline_basis", pa.string()),
        ("information_class", pa.string()),
    ]
)

STATE_SCHEMA = pa.schema(
    [
        ("state_id", pa.string()),
        ("episode_id", pa.string()),
        ("seed", pa.int32()),
        ("scenario", pa.string()),
        ("split", pa.string()),
        ("step", pa.int16()),
        ("timestamp", pa.string()),
        ("num_tasks", pa.int32()),
        ("pending_task_ids_json", pa.string()),
        ("running_tasks_json", pa.string()),
        ("in_transit_tasks_json", pa.string()),
        ("datacenters_json", pa.string()),
        ("network_links_json", pa.string()),
        ("exogenous_json", pa.string()),
        ("h1_joint_action_indices_json", pa.string()),
        ("h4_joint_action_indices_json", pa.string()),
        ("num_exact_disagreements", pa.int32()),
        ("num_target_dc_disagreements", pa.int32()),
        ("num_semantic_disagreements", pa.int32()),
        ("any_disagreement", pa.bool_()),
        ("disagreement_rate", pa.float64()),
        ("current_state_sha256", pa.string()),
        ("information_class", pa.string()),
    ]
)

PRIVILEGED_SCHEMA = pa.schema(
    [
        ("state_id", pa.string()),
        ("episode_id", pa.string()),
        ("seed", pa.int32()),
        ("scenario", pa.string()),
        ("split", pa.string()),
        ("step", pa.int16()),
        ("timestamp", pa.string()),
        ("horizon_step", pa.int8()),
        ("horizon_minutes", pa.int16()),
        ("forecast_timestamp", pa.string()),
        ("dc_id", pa.int16()),
        ("origin_probability", pa.float64()),
        ("global_task_count", pa.float64()),
        ("global_cpu_demand", pa.float64()),
        ("global_gpu_demand", pa.float64()),
        ("global_memory_demand", pa.float64()),
        ("dc_expected_task_count", pa.float64()),
        ("dc_forecast_cpu_reservation", pa.float64()),
        ("dc_forecast_gpu_reservation", pa.float64()),
        ("dc_forecast_memory_reservation", pa.float64()),
        ("dc_cpu_available_after_forecast", pa.float64()),
        ("dc_gpu_available_after_forecast", pa.float64()),
        ("dc_memory_available_after_forecast", pa.float64()),
        ("electricity_price_usd_per_mwh", pa.float64()),
        ("carbon_intensity_gco2_per_kwh", pa.float64()),
        ("information_class", pa.string()),
    ]
)


class SplitParquetWriter:
    def __init__(self, root: Path, stem: str, schema: pa.Schema) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "full": root / f"{stem}_full.parquet",
            "train": root / f"{stem}_train.parquet",
            "validation": root / f"{stem}_val.parquet",
            "test": root / f"{stem}_test.parquet",
        }
        self.writers = {
            key: pq.ParquetWriter(path, schema, compression="zstd")
            for key, path in self.paths.items()
        }
        self.schema = schema

    def write(self, rows: Sequence[Mapping[str, Any]], split: str) -> None:
        if not rows:
            return
        table = pa.Table.from_pylist(list(rows), schema=self.schema)
        self.writers["full"].write_table(table)
        self.writers[split].write_table(table)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(json_compact(value).encode("utf-8")).hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8", errors="replace"
    ).strip()


def load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    v3 = yaml.safe_load(CONFIG_PATH.read_text("utf-8"))["mpc_expert_dataset_v3"]
    v2 = yaml.safe_load(V2_CONFIG_PATH.read_text("utf-8"))[
        "repaired_mpc_expert_dataset_v2"
    ]
    return dict(v3), dict(v2)


def verify_inputs(v3: Mapping[str, Any], v2: Mapping[str, Any]) -> dict[str, Any]:
    scenario = v3["scenario_a"]
    seeds = tuple(range(int(scenario["seeds"]["start"]), int(scenario["seeds"]["stop"]) + 1))
    if seeds != PRIMARY_SEEDS:
        raise RuntimeError("Alibaba2020 v3 seeds changed")
    if int(v2["episode_steps"]) != 96:
        raise RuntimeError("Alibaba2020 v3 episode length changed")
    expected = {
        ROOT / scenario["source_file"]: scenario["source_sha256"],
        ROOT / scenario["v2_root"] / scenario["v2_task_file"]: scenario["v2_task_sha256"],
        ROOT / scenario["v2_root"] / scenario["v2_state_file"]: scenario["v2_state_sha256"],
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise RuntimeError(f"frozen Alibaba input changed: {path}")
    sustain_repo = ROOT / "references/external_repos/sustain-cluster"
    sustain_head = git("rev-parse", "HEAD", cwd=sustain_repo)
    if sustain_head != str(v3["expected_sustaincluster_head"]):
        raise RuntimeError("SustainCluster commit changed")
    if git("status", "--short", cwd=sustain_repo):
        raise RuntimeError("SustainCluster working tree is not clean")
    dataset_root = ROOT / v2["dataset_root"]
    manifest = json.loads((dataset_root / "13_dataset_manifest.json").read_text("utf-8"))
    for name, metadata in manifest["dataset_files"].items():
        if sha256(dataset_root / "dataset" / name) != metadata["sha256"]:
            raise RuntimeError(f"Forecast trace source changed: {name}")
    return {
        "project_head": git("rev-parse", "HEAD"),
        "project_branch": git("branch", "--show-current"),
        "sustaincluster_head": sustain_head,
        "dataset_root": dataset_root,
    }


def build_env(start: pd.Timestamp, steps: int, seed: int, baseline: float) -> Any:
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


def make_horizon_adapter() -> HorizonStateAdapter:
    return HorizonStateAdapter(
        information_mode="deployable",
        future_signal_provider=FutureSignalProvider("persistence"),
    )


def decisions_by_task(result: Any) -> dict[tuple[int, str], Any]:
    return {
        (item.original_index, item.task_id): item
        for item in result.first_step_decisions
    }


def state_id(episode_id: str, step: int, timestamp: pd.Timestamp) -> str:
    raw = f"{episode_id}|{step}|{timestamp.isoformat()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def current_state_payload(state: Any) -> dict[str, Any]:
    return {
        "tasks": [asdict(item) for item in state.current.tasks],
        "datacenters": [asdict(item) for item in state.current.datacenters],
        "network_links": [asdict(item) for item in state.current.network_links],
        "task_destinations": [asdict(item) for item in state.current.task_destinations],
        "exogenous": asdict(state.current.exogenous),
        "allow_defer": state.current.allow_defer,
        "information_mode": state.current.information_mode,
        "running_tasks": [asdict(item) for item in state.running_tasks],
        "transit_tasks": [asdict(item) for item in state.transit_tasks],
    }


def privileged_rows(
    *,
    state_key: str,
    episode_id: str,
    seed: int,
    scenario: str,
    split: str,
    step: int,
    timestamp: pd.Timestamp,
    bundle: Any,
    teacher_state: Any,
    datacenter_configs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pressures = distribute_global_forecast(bundle, datacenter_configs)
    pressure_by_key = {(p.horizon_step, p.dc_id): p for p in pressures}
    point_by_step = {p.horizon_step: p for p in bundle.points}
    dc_by_id = {dc.dc_id: dc for dc in teacher_state.datacenters}
    rows: list[dict[str, Any]] = []
    for horizon_step in range(1, 5):
        point = point_by_step[horizon_step]
        capacity_index = horizon_step
        for dc_id in sorted(dc_by_id):
            dc = dc_by_id[dc_id]
            pressure = pressure_by_key[(horizon_step, dc_id)]
            rows.append(
                {
                    "state_id": state_key,
                    "episode_id": episode_id,
                    "seed": seed,
                    "scenario": scenario,
                    "split": split,
                    "step": step,
                    "timestamp": timestamp.isoformat(),
                    "horizon_step": horizon_step,
                    "horizon_minutes": point.horizon_minutes,
                    "forecast_timestamp": point.forecast_timestamp,
                    "dc_id": dc_id,
                    "origin_probability": pressure.origin_probability,
                    "global_task_count": point.global_task_count,
                    "global_cpu_demand": point.global_cpu_demand,
                    "global_gpu_demand": point.global_gpu_demand,
                    "global_memory_demand": point.global_memory_demand,
                    "dc_expected_task_count": pressure.expected_task_count,
                    "dc_forecast_cpu_reservation": dc.forecast_cpu_reservations[capacity_index],
                    "dc_forecast_gpu_reservation": dc.forecast_gpu_reservations[capacity_index],
                    "dc_forecast_memory_reservation": dc.forecast_memory_reservations[capacity_index],
                    "dc_cpu_available_after_forecast": dc.cpu_available_cores[capacity_index],
                    "dc_gpu_available_after_forecast": dc.gpu_available_units[capacity_index],
                    "dc_memory_available_after_forecast": dc.memory_available_gb[capacity_index],
                    "electricity_price_usd_per_mwh": dc.electricity_price_usd_per_mwh[capacity_index],
                    "carbon_intensity_gco2_per_kwh": dc.carbon_intensity_gco2_per_kwh[capacity_index],
                    "information_class": "PRIVILEGED_FUTURE_TEACHER_ONLY",
                }
            )
    return rows


def run_episode(
    *,
    scenario: str,
    seed: int,
    start: pd.Timestamp,
    expected_dataset_start: pd.Timestamp,
    steps: int,
    split: str,
    trace_source: ForecastTraceSource,
    datacenter_configs: list[dict[str, Any]],
    optimizer_config: Any,
    baseline_minutes: float,
    first_row_id: int,
) -> dict[str, Any]:
    episode_id = f"{scenario}__seed_{seed}"
    env = build_env(start, steps, seed, baseline_minutes)
    action_adapter = SustainClusterActionAdapter.from_env(env)
    dc_ids = tuple(sorted(action_adapter.mapping.dc_id_to_action))
    semantic_actions = SemanticActionSpace(dc_ids, allow_defer=True)
    horizon_adapter = make_horizon_adapter()
    optimizer = RollingHorizonOptimizer()
    oracle_provider = OracleWorkloadForecastProvider()
    task_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []
    row_id = int(first_row_id)
    teacher_ms: list[float] = []
    h1_ms: list[float] = []
    h1_failures = 0
    completed_steps = 0
    oracle_warning_count = 0
    try:
        for step in range(steps):
            timestamp = pd.Timestamp(env.current_time)
            if step == 0 and trace_source.align_environment_timestamp(timestamp) != expected_dataset_start:
                raise RuntimeError("Alibaba2020 v3 trace alignment failed")
            oracle_request = trace_source.request(timestamp, include_oracle_future=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                oracle_bundle = oracle_provider.forecast(oracle_request)
            warning_count = sum("non-deployable" in str(item.message) for item in caught)
            if warning_count != 1:
                raise RuntimeError("Oracle access warning contract changed")
            oracle_warning_count += warning_count

            h1_state = horizon_adapter.build_horizon_state(env, 1, "no_future_arrivals")
            base_h4 = horizon_adapter.build_horizon_state(env, 5, "no_future_arrivals")
            student = build_student_task_batch(env, base_h4, semantic_actions)
            teacher_state = apply_oracle_future_signals(
                apply_forecast_pressure(base_h4, oracle_bundle, datacenter_configs).state,
                env,
            )

            before = environment_state_signature(env)
            h1_result = optimizer.solve(h1_state, optimizer_config, action_adapter)
            if environment_state_signature(env) != before:
                raise RuntimeError("H1 shadow solve mutated environment")
            teacher_result = optimizer.solve(teacher_state, optimizer_config, action_adapter)
            if environment_state_signature(env) != before:
                raise RuntimeError("H4 teacher solve mutated environment")
            validate_teacher_solution(teacher_result)
            action_adapter.validate_actions(
                teacher_state.current.tasks, teacher_result.environment_actions
            )
            if h1_result.feasible:
                h1_ms.append(1000.0 * h1_result.solve_seconds)
            else:
                h1_failures += 1
            teacher_ms.append(1000.0 * teacher_result.solve_seconds)

            h1_decisions = decisions_by_task(h1_result) if h1_result.feasible else {}
            teacher_decisions = decisions_by_task(teacher_result)
            state_key = state_id(episode_id, step, timestamp)
            raw_by_position = list(env.current_tasks)
            step_rows: list[dict[str, Any]] = []
            h1_joint: list[int | None] = []
            h4_joint: list[int] = []
            for position, task in enumerate(teacher_state.current.tasks):
                raw_task = raw_by_position[position]
                if str(raw_task.job_name) != task.task_id:
                    raise RuntimeError("raw task order and scheduler snapshot diverged")
                key = (task.original_index, task.task_id)
                teacher_decision = teacher_decisions[key]
                teacher_index = semantic_actions.encode_decision(teacher_decision)
                if not bool(student.feasible_action_mask[position, teacher_index]):
                    raise RuntimeError("infeasible H4 Oracle label")
                h1_decision = h1_decisions.get(key)
                h1_index = (
                    semantic_actions.encode_decision(h1_decision)
                    if h1_decision is not None
                    else None
                )
                sample_id = stable_sample_id(episode_id, step, task.task_id)
                exact = h1_index is not None and h1_index != teacher_index
                semantic = (
                    h1_decision is not None
                    and h1_decision.decision != teacher_decision.decision
                )
                target = (
                    h1_decision is not None
                    and h1_decision.dc_id != teacher_decision.dc_id
                )
                row = {
                    "row_id": row_id,
                    "state_id": state_key,
                    "sample_id": sample_id,
                    "episode_id": episode_id,
                    "seed": seed,
                    "scenario": scenario,
                    "split": split,
                    "step": step,
                    "timestamp": timestamp.isoformat(),
                    "task_id": task.task_id,
                    "task_position": position,
                    "origin_dc": task.origin_dc_id,
                    "student_observation": student.observations[position].astype(np.float32).tolist(),
                    "student_obs_dim": int(student.observations.shape[1]),
                    "feasible_action_mask": student.feasible_action_mask[position].astype(bool).tolist(),
                    "teacher_action_index": teacher_index,
                    "teacher_action_semantic": semantic_action_name(teacher_index, semantic_actions),
                    "teacher_target_dc": teacher_decision.dc_id,
                    "teacher_is_defer": teacher_decision.decision == "defer",
                    "teacher_is_local": teacher_decision.decision == "assign" and teacher_decision.dc_id == task.origin_dc_id,
                    "teacher_is_migration": teacher_decision.decision == "assign" and teacher_decision.dc_id != task.origin_dc_id,
                    "h1_action_index": h1_index,
                    "h1_action_semantic": semantic_action_name(h1_index, semantic_actions) if h1_index is not None else "unavailable",
                    "h1_target_dc": h1_decision.dc_id if h1_decision is not None else None,
                    "h1_is_defer": h1_decision.decision == "defer" if h1_decision is not None else None,
                    "exact_action_disagreement": exact,
                    "semantic_disagreement": semantic,
                    "target_dc_disagreement": target,
                    "teacher_solver_status": teacher_result.status,
                    "teacher_objective": float(teacher_result.objective_value),
                    "teacher_solve_ms": 1000.0 * teacher_result.solve_seconds,
                    "h1_solver_status": h1_result.status,
                    "h1_solve_ms": 1000.0 * h1_result.solve_seconds if h1_result.feasible else None,
                    "sla_slack_steps": task.remaining_sla_minutes / teacher_state.timestep_minutes,
                    "estimated_duration": task.duration_minutes,
                    "task_cpu_cores": task.cpu_cores,
                    "task_gpu_units": task.gpu_units,
                    "task_memory_gb": task.memory_gb,
                    "h1_label_provenance": H1_PROVENANCE,
                    "teacher_label_provenance": H4_PROVENANCE,
                    "information_class": "DEPLOYABLE_CURRENT_PLUS_LABELS",
                }
                task_rows.append(row)
                step_rows.append(row)
                truth_rows.append(
                    {
                        "sample_id": sample_id,
                        "state_id": state_key,
                        "episode_id": episode_id,
                        "seed": seed,
                        "scenario": scenario,
                        "split": split,
                        "step": step,
                        "timestamp": timestamp.isoformat(),
                        "task_id": task.task_id,
                        "true_duration_minutes": float(raw_task.true_duration),
                        "duration_source_type": str(raw_task.duration_source_type),
                        "sla_deadline_basis": str(raw_task.sla_deadline_basis),
                        "information_class": "SIMULATOR_ONLY_TRUTH",
                    }
                )
                h1_joint.append(h1_index)
                h4_joint.append(teacher_index)
                row_id += 1

            payload = current_state_payload(base_h4)
            exact_count = sum(bool(row["exact_action_disagreement"]) for row in step_rows)
            target_count = sum(bool(row["target_dc_disagreement"]) for row in step_rows)
            semantic_count = sum(bool(row["semantic_disagreement"]) for row in step_rows)
            state_rows.append(
                {
                    "state_id": state_key,
                    "episode_id": episode_id,
                    "seed": seed,
                    "scenario": scenario,
                    "split": split,
                    "step": step,
                    "timestamp": timestamp.isoformat(),
                    "num_tasks": len(step_rows),
                    "pending_task_ids_json": json_compact([item.task_id for item in base_h4.current.tasks]),
                    "running_tasks_json": json_compact([asdict(item) for item in base_h4.running_tasks]),
                    "in_transit_tasks_json": json_compact([asdict(item) for item in base_h4.transit_tasks]),
                    "datacenters_json": json_compact([asdict(item) for item in base_h4.current.datacenters]),
                    "network_links_json": json_compact([asdict(item) for item in base_h4.current.network_links]),
                    "exogenous_json": json_compact(asdict(base_h4.current.exogenous)),
                    "h1_joint_action_indices_json": json_compact(h1_joint),
                    "h4_joint_action_indices_json": json_compact(h4_joint),
                    "num_exact_disagreements": exact_count,
                    "num_target_dc_disagreements": target_count,
                    "num_semantic_disagreements": semantic_count,
                    "any_disagreement": exact_count > 0,
                    "disagreement_rate": exact_count / max(1, len(step_rows)),
                    "current_state_sha256": content_hash(payload),
                    "information_class": "DEPLOYABLE_CURRENT_STATE",
                }
            )
            future_rows.extend(
                privileged_rows(
                    state_key=state_key,
                    episode_id=episode_id,
                    seed=seed,
                    scenario=scenario,
                    split=split,
                    step=step,
                    timestamp=timestamp,
                    bundle=oracle_bundle,
                    teacher_state=teacher_state,
                    datacenter_configs=datacenter_configs,
                )
            )
            env.step(list(teacher_result.environment_actions))
            completed_steps += 1
    finally:
        env.close()

    return {
        "task_rows": task_rows,
        "state_rows": state_rows,
        "truth_rows": truth_rows,
        "future_rows": future_rows,
        "next_row_id": row_id,
        "manifest": {
            "episode_id": episode_id,
            "seed": seed,
            "scenario": scenario,
            "split": split,
            "environment_start": start.isoformat(),
            "steps": completed_steps,
            "task_decisions": len(task_rows),
            "teacher_solver_failures": steps - completed_steps,
            "h1_solver_failures": h1_failures,
            "teacher_solve_ms_mean": float(np.mean(teacher_ms)),
            "teacher_solve_ms_p95": float(np.percentile(teacher_ms, 95)),
            "h1_solve_ms_mean": float(np.mean(h1_ms)) if h1_ms else None,
            "h1_solve_ms_p95": float(np.percentile(h1_ms, 95)) if h1_ms else None,
            "oracle_warning_count": oracle_warning_count,
        },
    }


def list_equal(left: Any, right: Any) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right))


def build_alignment(output: Path, v2_root: Path, v3_tasks: Path, v3_states: Path) -> dict[str, Any]:
    v2_tasks = pd.read_parquet(v2_root / "dataset/expert_task_actions_full.parquet")
    v3_task = pd.read_parquet(v3_tasks)
    v2_states = pd.read_parquet(v2_root / "dataset/state_disagreement_summary.parquet")
    v3_state = pd.read_parquet(v3_states)
    if v2_tasks["sample_id"].duplicated().any() or v3_task["sample_id"].duplicated().any():
        raise RuntimeError("sample_id is not unique")

    left = v2_tasks.set_index("sample_id", drop=False)
    right = v3_task.set_index("sample_id", drop=False)
    all_ids = left.index.union(right.index)
    mismatch_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    observation_equal = 0
    mask_equal = 0
    h1_equal = 0
    h4_equal = 0
    identity_equal = 0
    common = left.index.intersection(right.index)
    identity_columns = [
        "episode_id", "seed", "scenario", "split", "step", "timestamp",
        "task_id", "task_position", "origin_dc", "student_obs_dim",
        "estimated_duration", "task_cpu_cores", "task_gpu_units", "task_memory_gb",
    ]
    for sample_id in all_ids:
        reasons: list[str] = []
        if sample_id not in left.index:
            reasons.append("MISSING_IN_V2")
        elif sample_id not in right.index:
            reasons.append("MISSING_IN_V3")
        else:
            old = left.loc[sample_id]
            new = right.loc[sample_id]
            identity_ok = all(
                (pd.isna(old[column]) and pd.isna(new[column]))
                or old[column] == new[column]
                for column in identity_columns
            )
            obs_ok = list_equal(old["student_observation"], new["student_observation"])
            mask_ok = list_equal(old["feasible_action_mask"], new["feasible_action_mask"])
            h1_ok = (pd.isna(old["h1_action_index"]) and pd.isna(new["h1_action_index"])) or old["h1_action_index"] == new["h1_action_index"]
            h4_ok = old["teacher_action_index"] == new["teacher_action_index"]
            identity_equal += int(identity_ok)
            observation_equal += int(obs_ok)
            mask_equal += int(mask_ok)
            h1_equal += int(h1_ok)
            h4_equal += int(h4_ok)
            if not identity_ok:
                reasons.append("TASK_STATE_IDENTITY_CHANGED")
            if not obs_ok:
                reasons.append("STUDENT_OBSERVATION_CHANGED")
            if not mask_ok:
                reasons.append("FEASIBLE_MASK_CHANGED")
            if not h1_ok:
                reasons.append("H1_ACTION_CHANGED")
            if not h4_ok:
                reasons.append("H4_ORACLE_ACTION_CHANGED")
        if reasons:
            category = "+".join(reasons)
            reason_counts[category] += 1
            mismatch_rows.append({"sample_id": sample_id, "reason": category})

    state_keys = ["episode_id", "seed", "scenario", "split", "step"]
    state_compare = [
        "timestamp", "num_tasks", "num_exact_disagreements",
        "num_target_dc_disagreements", "num_semantic_disagreements",
        "any_disagreement", "disagreement_rate",
    ]
    old_states = v2_states.set_index(state_keys).sort_index()
    new_states = v3_state.set_index(state_keys).sort_index()
    all_state_keys = old_states.index.union(new_states.index)
    state_mismatches: list[dict[str, Any]] = []
    state_equal = 0
    for key in all_state_keys:
        reasons = []
        if key not in old_states.index:
            reasons.append("STATE_MISSING_IN_V2")
        elif key not in new_states.index:
            reasons.append("STATE_MISSING_IN_V3")
        else:
            old = old_states.loc[key]
            new = new_states.loc[key]
            for column in state_compare:
                if not np.isclose(old[column], new[column], equal_nan=True) if column == "disagreement_rate" else old[column] != new[column]:
                    reasons.append(f"STATE_FIELD_CHANGED:{column}")
            state_equal += int(not reasons)
        if reasons:
            state_mismatches.append(
                {
                    **dict(zip(state_keys, key if isinstance(key, tuple) else (key,))),
                    "reason": "+".join(reasons),
                }
            )

    denominator = len(common)
    metrics = [
        ("task_key_coverage", len(common), len(all_ids)),
        ("task_state_identity_agreement", identity_equal, denominator),
        ("student_observation_agreement", observation_equal, denominator),
        ("feasible_mask_agreement", mask_equal, denominator),
        ("h1_action_agreement", h1_equal, denominator),
        ("h4_oracle_action_agreement", h4_equal, denominator),
        ("state_summary_agreement", state_equal, len(all_state_keys)),
    ]
    summary = pd.DataFrame(
        [
            {
                "metric": metric,
                "matching": int(numerator),
                "total": int(total),
                "agreement_rate": float(numerator / max(1, total)),
            }
            for metric, numerator, total in metrics
        ]
    )
    summary.to_csv(output / "07_alibaba2020_v2_v3_alignment.csv", index=False)
    pd.DataFrame(
        [
            {"reason": key, "count": value, "share_of_all_ids": value / max(1, len(all_ids))}
            for key, value in sorted(reason_counts.items())
        ],
        columns=["reason", "count", "share_of_all_ids"],
    ).to_csv(output / "07a_alibaba2020_task_mismatch_reasons.csv", index=False)
    pd.DataFrame(mismatch_rows, columns=["sample_id", "reason"]).to_parquet(
        output / "dataset/scenario_a/v2_v3_task_mismatches.parquet", index=False
    )
    pd.DataFrame(state_mismatches).to_csv(
        output / "07b_alibaba2020_state_mismatches.csv", index=False
    )
    lookup = summary.set_index("metric")["agreement_rate"].to_dict()
    return {
        "summary": summary.to_dict("records"),
        "task_mismatch_reason_counts": dict(reason_counts),
        "task_mismatch_count": len(mismatch_rows),
        "state_mismatch_count": len(state_mismatches),
        "action_gate_pass": (
            lookup["h1_action_agreement"] >= ACTION_AGREEMENT_GATE
            and lookup["h4_oracle_action_agreement"] >= ACTION_AGREEMENT_GATE
        ),
    }


def output_hashes(output: Path) -> list[dict[str, Any]]:
    rows = []
    allowed_files = {
        "04_alibaba2020_v3_episode_manifest.csv",
        "05_alibaba2020_v3_split_manifest.json",
        "07_alibaba2020_v2_v3_alignment.csv",
        "07a_alibaba2020_task_mismatch_reasons.csv",
        "07b_alibaba2020_state_mismatches.csv",
        "08_alibaba2020_alignment_report.md",
    }
    allowed_prefixes = (
        "dataset/scenario_a/",
        "simulator_only/scenario_a/",
        "privileged_future/scenario_a/",
    )
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output).as_posix()
        if relative not in allowed_files and not relative.startswith(allowed_prefixes):
            continue
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def run_generation() -> Path:
    v3, v2 = load_configs()
    frozen = verify_inputs(v3, v2)
    output = ROOT / v3["output_dir"]
    dataset_dir = output / "dataset/scenario_a"
    truth_dir = output / "simulator_only/scenario_a"
    privileged_dir = output / "privileged_future/scenario_a"
    for directory in (dataset_dir, truth_dir, privileged_dir):
        directory.mkdir(parents=True, exist_ok=True)

    split_manifest = build_split_manifest(
        PRIMARY_SEEDS,
        shuffle_seed=int(v2["split"]["shuffle_seed"]),
        train_seed_count=int(v2["split"]["train_seed_count"]),
        validation_seed_count=int(v2["split"]["validation_seed_count"]),
    )
    validate_split_manifest(split_manifest, expected_seeds=PRIMARY_SEEDS)
    frozen_split = json.loads((ROOT / v3["scenario_a"]["split_manifest"]).read_text("utf-8"))
    decision_fields = (
        "split_unit",
        "shuffle_seed",
        "train_seeds",
        "validation_seeds",
        "test_seeds",
        "seed_to_split",
    )
    if any(split_manifest[field] != frozen_split[field] for field in decision_fields):
        raise RuntimeError("v3 split does not exactly match v2")

    trace_source = ForecastTraceSource(frozen["dataset_root"])
    datacenter_configs = yaml.safe_load(
        (ROOT / v2["datacenter_config"]).read_text("utf-8")
    )["datacenters"]
    optimizer_config = repaired_runner._optimizer_config(ROOT / v2["optimizer_config"])
    task_writer = SplitParquetWriter(dataset_dir, "expert_task_actions", TASK_SCHEMA)
    state_writer = SplitParquetWriter(dataset_dir, "current_states", STATE_SCHEMA)
    truth_writer = SplitParquetWriter(truth_dir, "task_truth", TRUTH_SCHEMA)
    future_writer = SplitParquetWriter(privileged_dir, "oracle_future", PRIVILEGED_SCHEMA)
    episode_rows: list[dict[str, Any]] = []
    row_id = 0
    try:
        for scenario, scenario_config in v2["scenarios"].items():
            for seed in PRIMARY_SEEDS:
                split = split_manifest["seed_to_split"][str(seed)]
                print(f"RUN_ALIBABA_V3 scenario={scenario} seed={seed} split={split}", flush=True)
                episode = run_episode(
                    scenario=scenario,
                    seed=seed,
                    start=pd.Timestamp(scenario_config["environment_start"]),
                    expected_dataset_start=pd.Timestamp(scenario_config["dataset_start"]),
                    steps=int(v2["episode_steps"]),
                    split=split,
                    trace_source=trace_source,
                    datacenter_configs=datacenter_configs,
                    optimizer_config=optimizer_config,
                    baseline_minutes=float(v2["baseline_estimated_duration_minutes"]),
                    first_row_id=row_id,
                )
                row_id = episode["next_row_id"]
                task_writer.write(episode["task_rows"], split)
                state_writer.write(episode["state_rows"], split)
                truth_writer.write(episode["truth_rows"], split)
                future_writer.write(episode["future_rows"], split)
                episode_rows.append(episode["manifest"])
                print(json.dumps(episode["manifest"], ensure_ascii=False), flush=True)
    finally:
        task_writer.close()
        state_writer.close()
        truth_writer.close()
        future_writer.close()

    episodes = pd.DataFrame(episode_rows)
    episodes.to_csv(output / "04_alibaba2020_v3_episode_manifest.csv", index=False)
    write_json(output / "05_alibaba2020_v3_split_manifest.json", frozen_split)
    expected = v3["scenario_a"]
    counts = {
        "episodes": len(episodes),
        "states": int(episodes["steps"].sum()),
        "task_decisions": int(episodes["task_decisions"].sum()),
        "teacher_solver_failures": int(episodes["teacher_solver_failures"].sum()),
        "h1_solver_failures": int(episodes["h1_solver_failures"].sum()),
    }
    if counts["episodes"] != int(expected["expected_episodes"]):
        raise RuntimeError("episode count mismatch")
    if counts["states"] != int(expected["expected_states"]):
        raise RuntimeError("state count mismatch")
    if counts["task_decisions"] != int(expected["expected_task_decisions"]):
        raise RuntimeError("task decision count mismatch")
    if counts["teacher_solver_failures"] or counts["h1_solver_failures"]:
        raise RuntimeError("Alibaba2020 v3 solver failures detected")

    alignment = build_alignment(
        output,
        ROOT / expected["v2_root"],
        dataset_dir / "expert_task_actions_full.parquet",
        dataset_dir / "current_states_full.parquet",
    )
    lookup = {item["metric"]: item["agreement_rate"] for item in alignment["summary"]}
    write_text(
        output / "08_alibaba2020_alignment_report.md",
        f"""# Alibaba2020 Expert Dataset v3 Alignment

- Independent regeneration: YES.
- Transformer instantiated or inferred: NO.
- Episodes / states / decisions: {counts['episodes']} / {counts['states']} / {counts['task_decisions']}.
- Comparable state summary agreement: {lookup['state_summary_agreement']:.9%}.
- Student observation agreement: {lookup['student_observation_agreement']:.9%}.
- Feasible mask agreement: {lookup['feasible_mask_agreement']:.9%}.
- Repaired H1 action agreement: {lookup['h1_action_agreement']:.9%}.
- Repaired H4 Oracle action agreement: {lookup['h4_oracle_action_agreement']:.9%}.
- Task mismatch count: {alignment['task_mismatch_count']}.
- State mismatch count: {alignment['state_mismatch_count']}.
- Action stability gate ({ACTION_AGREEMENT_GATE:.0%}): {'PASS' if alignment['action_gate_pass'] else 'FAIL'}.

All task-level mismatch combinations are enumerated in `07a_alibaba2020_task_mismatch_reasons.csv`; state-level field changes are enumerated in `07b_alibaba2020_state_mismatches.csv`.
""",
    )
    manifest = {
        "dataset_version": DATASET_VERSION,
        "status": "ALIBABA2020_V3_READY" if alignment["action_gate_pass"] else "TEACHER_ACTION_CHANGE_INVESTIGATION_REQUIRED",
        "transformer_used": False,
        "rl_used": False,
        "policy_training_used": False,
        "frozen": {
            key: (str(value.relative_to(ROOT)) if isinstance(value, Path) else value)
            for key, value in frozen.items()
        },
        "counts": counts,
        "alignment": alignment,
        "action_agreement_gate": ACTION_AGREEMENT_GATE,
        "information_regions": {
            "dataset/scenario_a": "DEPLOYABLE_CURRENT_PLUS_LABELS",
            "simulator_only/scenario_a": "SIMULATOR_ONLY_TRUTH",
            "privileged_future/scenario_a": "PRIVILEGED_FUTURE_TEACHER_ONLY",
        },
        "file_hashes": output_hashes(output),
    }
    write_json(output / "00_alibaba2020_v3_manifest.json", manifest)
    if not alignment["action_gate_pass"]:
        raise RuntimeError("teacher actions changed materially; stop before Spot capacity work")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if not args.generate:
        parser.error("use --generate")
    output = run_generation()
    print(json.dumps({"status": "ALIBABA2020_V3_READY", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
