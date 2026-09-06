from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from forecasting.workload_forecast_provider import (
    ForecastTraceSource,
    OracleWorkloadForecastProvider,
)
from scripts.forecast import run_forecast_aware_mpc_v1 as repaired_runner
from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_imitation.expert_dataset_v2 import (
    CALIBRATION_SEEDS,
    EVALUATION_SEEDS,
    EXPECTED_ACTION_DIM,
    EXPECTED_STUDENT_OBS_DIM,
    PRIMARY_SEEDS,
    actor_forward_dry_run,
    assert_deployable_feature_schema,
    build_split_manifest,
    build_state_disagreement_summary,
    build_student_task_batch,
    dataframe_content_sha256,
    environment_state_signature,
    load_bc_dry_run_batch,
    make_disagreement_index,
    primary_quality_counts,
    semantic_action_name,
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


WORKSPACE = Path(__file__).resolve().parents[1]
DATASET = WORKSPACE / "artifacts/forecast_dataset_v1"


@pytest.fixture(scope="module")
def real_snapshot():
    env = build_sustaincluster_env(
        None,
        pd.Timestamp("2023-02-13T00:00:00Z"),
        2,
        information_mode="deployable",
        duration_estimate_mode="declared_or_baseline",
        baseline_estimated_duration_minutes=60.0,
        initial_seed=3001,
    )
    env.reset(seed=3001)
    action_adapter = SustainClusterActionAdapter.from_env(env)
    semantic_actions = SemanticActionSpace(
        tuple(sorted(action_adapter.mapping.dc_id_to_action)), True
    )
    horizon_adapter = HorizonStateAdapter(
        information_mode="deployable",
        future_signal_provider=FutureSignalProvider("persistence"),
    )
    h1 = horizon_adapter.build_horizon_state(env, 1, "no_future_arrivals")
    base_h4 = horizon_adapter.build_horizon_state(env, 5, "no_future_arrivals")
    student = build_student_task_batch(env, base_h4, semantic_actions)
    trace = ForecastTraceSource(DATASET)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        oracle = OracleWorkloadForecastProvider().forecast(
            trace.request(env.current_time, include_oracle_future=True)
        )
    dc_configs = yaml.safe_load(
        (
            WORKSPACE
            / "references/external_repos/sustain-cluster/configs/env/datacenters.yaml"
        ).read_text("utf-8")
    )["datacenters"]
    teacher_state = apply_oracle_future_signals(
        apply_forecast_pressure(base_h4, oracle, dc_configs).state, env
    )
    optimizer = RollingHorizonOptimizer()
    optimizer_config = repaired_runner._optimizer_config(
        WORKSPACE / "configs/sustaincluster_mpc/h4_expert.yaml"
    )
    teacher = optimizer.solve(teacher_state, optimizer_config, action_adapter)
    h1_result = optimizer.solve(h1, optimizer_config, action_adapter)
    yield {
        "env": env,
        "adapter": action_adapter,
        "semantic": semantic_actions,
        "h1": h1,
        "base_h4": base_h4,
        "student": student,
        "teacher_state": teacher_state,
        "teacher": teacher,
        "h1_result": h1_result,
        "oracle_warnings": caught,
        "optimizer": optimizer,
        "optimizer_config": optimizer_config,
    }
    env.close()


def test_repaired_h4_teacher_timeline() -> None:
    nodes = repaired_h4_capacity_timeline(pd.Timestamp("2020-01-01T00:00:00Z"))
    assert [node.absolute_offset_minutes for node in nodes] == [0, 15, 30, 45, 60]


def test_privileged_teacher_oracle_access_is_explicit(real_snapshot) -> None:
    assert real_snapshot["teacher_state"].future_signal_mode == "oracle"
    assert real_snapshot["teacher_state"].forecast_mode == "oracle"
    assert any("non-deployable" in str(item.message) for item in real_snapshot["oracle_warnings"])


def test_deployable_student_does_not_change_after_teacher_build(real_snapshot) -> None:
    before = real_snapshot["student"].observations.copy()
    after = build_student_task_batch(
        real_snapshot["env"],
        real_snapshot["base_h4"],
        real_snapshot["semantic"],
    ).observations
    np.testing.assert_array_equal(before, after)


def test_student_observation_shape(real_snapshot) -> None:
    assert real_snapshot["student"].observations.shape[1] == EXPECTED_STUDENT_OBS_DIM


def test_estimated_duration_is_visible(real_snapshot) -> None:
    task = real_snapshot["env"].current_tasks[0]
    assert real_snapshot["student"].observations[0, 7] == pytest.approx(task.estimated_duration)


def test_true_duration_is_invisible(real_snapshot) -> None:
    env = real_snapshot["env"]
    task = env.current_tasks[0]
    before = build_student_task_batch(env, real_snapshot["base_h4"], real_snapshot["semantic"]).observations.copy()
    original = task.true_duration
    try:
        task.true_duration = original + 10000.0
        after = np.asarray(env._generate_per_task_obs_list(), dtype=np.float32)
    finally:
        task.true_duration = original
    np.testing.assert_array_equal(before, after)


def test_semantic_action_mapping_is_stable(real_snapshot) -> None:
    semantic = real_snapshot["semantic"]
    assert [semantic_action_name(i, semantic) for i in range(6)] == [
        "defer", "assign_dc1", "assign_dc2", "assign_dc3", "assign_dc4", "assign_dc5"
    ]


def test_teacher_labels_are_in_deployable_feasible_mask(real_snapshot) -> None:
    teacher = real_snapshot["teacher"]
    semantic = real_snapshot["semantic"]
    mask = real_snapshot["student"].feasible_action_mask
    for index, decision in enumerate(teacher.first_step_decisions):
        assert bool(mask[index, semantic.encode_decision(decision)])


def test_stable_task_ids_across_repeated_observation(real_snapshot) -> None:
    first = build_student_task_batch(real_snapshot["env"], real_snapshot["base_h4"], real_snapshot["semantic"])
    second = build_student_task_batch(real_snapshot["env"], real_snapshot["base_h4"], real_snapshot["semantic"])
    assert first.task_ids == second.task_ids


def test_sample_id_uses_episode_step_and_task() -> None:
    first = stable_sample_id("episode", 0, "task")
    second = stable_sample_id("episode", 1, "task")
    assert first != second


def test_shadow_h1_does_not_mutate_environment_or_rng(real_snapshot) -> None:
    before = environment_state_signature(real_snapshot["env"])
    real_snapshot["optimizer"].solve(
        real_snapshot["h1"],
        real_snapshot["optimizer_config"],
        real_snapshot["adapter"],
    )
    assert environment_state_signature(real_snapshot["env"]) == before


def test_state_level_multi_action_alignment(real_snapshot) -> None:
    state_ids = tuple(task.task_id for task in real_snapshot["teacher_state"].current.tasks)
    decision_ids = tuple(item.task_id for item in real_snapshot["teacher"].first_step_decisions)
    assert state_ids == decision_ids == real_snapshot["student"].task_ids


def test_parquet_observation_roundtrip(tmp_path: Path, real_snapshot) -> None:
    value = real_snapshot["student"].observations[0].astype(np.float32)
    path = tmp_path / "obs.parquet"
    pd.DataFrame({"student_observation": [value.tolist()]}).to_parquet(path, index=False)
    restored = np.asarray(pd.read_parquet(path).iloc[0]["student_observation"], dtype=np.float32)
    assert restored.shape == (EXPECTED_STUDENT_OBS_DIM,)
    np.testing.assert_allclose(restored, value)


def test_mask_roundtrip(tmp_path: Path, real_snapshot) -> None:
    value = real_snapshot["student"].feasible_action_mask[0]
    path = tmp_path / "mask.parquet"
    pd.DataFrame({"feasible_action_mask": [value.tolist()]}).to_parquet(path, index=False)
    restored = np.asarray(pd.read_parquet(path).iloc[0]["feasible_action_mask"], dtype=bool)
    np.testing.assert_array_equal(restored, value)


def test_episode_exclusive_split() -> None:
    split = build_split_manifest()
    assert len(split["seed_to_split"]) == 40
    assert set(split["seed_to_split"].values()) == {"train", "validation", "test"}


def test_seed_exclusive_split() -> None:
    split = build_split_manifest()
    validate_split_manifest(split, expected_seeds=PRIMARY_SEEDS)


def test_scenario_balance_protocol() -> None:
    episode_pairs = [(seed, scenario) for seed in PRIMARY_SEEDS for scenario in ("normal_trace", "high_load_trace")]
    assert len(episode_pairs) == 80
    assert Counter(scenario for _, scenario in episode_pairs) == {"normal_trace": 40, "high_load_trace": 40}


def test_solver_failure_cannot_become_expert_label() -> None:
    with pytest.raises(RuntimeError, match="fallback actions"):
        validate_teacher_solution(SimpleNamespace(status="failed", feasible=False))


def test_disagreement_index_consistency() -> None:
    frame = pd.DataFrame(
        {"row_id": [0, 1], "sample_id": ["a", "b"], "exact_action_disagreement": [False, True]}
    )
    index = make_disagreement_index(frame, "exact_action_disagreement")
    assert index.to_dict("records") == [
        {"row_id": 1, "sample_id": "b", "criterion": "exact_action_disagreement"}
    ]


def test_actornet_loader_and_forward_dry_run(real_snapshot) -> None:
    student = real_snapshot["student"]
    teacher = real_snapshot["teacher"]
    semantic = real_snapshot["semantic"]
    frame = pd.DataFrame(
        {
            "student_observation": student.observations[:8].tolist(),
            "teacher_action_index": [semantic.encode_decision(item) for item in teacher.first_step_decisions[:8]],
            "feasible_action_mask": student.feasible_action_mask[:8].tolist(),
        }
    )
    batch = load_bc_dry_run_batch(frame, batch_size=8)
    assert actor_forward_dry_run(batch) == (8, EXPECTED_ACTION_DIM)


def test_deterministic_content_regeneration(real_snapshot) -> None:
    student = real_snapshot["student"]
    teacher = real_snapshot["teacher"]
    semantic = real_snapshot["semantic"]
    rows = [
        {
            "sample_id": stable_sample_id("episode", 0, task_id),
            "student_observation": observation.tolist(),
            "feasible_action_mask": mask.tolist(),
            "teacher_action_index": semantic.encode_decision(decision),
        }
        for task_id, observation, mask, decision in zip(
            student.task_ids,
            student.observations,
            student.feasible_action_mask,
            teacher.first_step_decisions,
        )
    ]
    assert dataframe_content_sha256(rows) == dataframe_content_sha256([dict(row) for row in rows])


def test_frozen_reward_objective_and_horizon() -> None:
    config = repaired_runner._optimizer_config(WORKSPACE / "configs/sustaincluster_mpc/h4_expert.yaml")
    assert config.weights.waiting_defer == 100.0
    assert config.weights.sla_risk == 10.0
    assert config.weights.terminal_backlog == 100.0
    assert len(repaired_h4_capacity_timeline(pd.Timestamp("2020-01-01T00:00:00Z"))) == 5


def test_primary_seeds_exclude_prior_formal_seeds() -> None:
    assert set(PRIMARY_SEEDS).isdisjoint(CALIBRATION_SEEDS)
    assert set(PRIMARY_SEEDS).isdisjoint(EVALUATION_SEEDS)


def test_student_feature_schema_rejects_oracle_token() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        assert_deployable_feature_schema(["estimated_duration", "oracle_future_workload"])


def test_student_feature_schema_has_expected_dimension() -> None:
    names = student_feature_names((1, 2, 3, 4, 5))
    assert len(names) == EXPECTED_STUDENT_OBS_DIM
    assert_deployable_feature_schema(names)


def test_state_disagreement_summary_alignment() -> None:
    frame = pd.DataFrame(
        [
            {"episode_id": "e", "seed": 1, "scenario": "s", "split": "train", "step": 0, "timestamp": "t", "exact_action_disagreement": False, "semantic_disagreement": False, "target_dc_disagreement": False, "deployable_risk_score": 0.1},
            {"episode_id": "e", "seed": 1, "scenario": "s", "split": "train", "step": 0, "timestamp": "t", "exact_action_disagreement": True, "semantic_disagreement": False, "target_dc_disagreement": True, "deployable_risk_score": 0.1},
        ]
    )
    summary = build_state_disagreement_summary(frame).iloc[0]
    assert summary.num_tasks == 2
    assert summary.num_exact_disagreements == 1
    assert summary.disagreement_rate == pytest.approx(0.5)


def test_primary_quality_detects_no_errors(real_snapshot) -> None:
    student = real_snapshot["student"]
    teacher = real_snapshot["teacher"]
    semantic = real_snapshot["semantic"]
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(len(student.task_ids))],
            "student_observation": student.observations.tolist(),
            "feasible_action_mask": student.feasible_action_mask.tolist(),
            "teacher_action_index": [semantic.encode_decision(item) for item in teacher.first_step_decisions],
        }
    )
    assert set(primary_quality_counts(frame, obs_dim=34, action_dim=6).values()) == {0}


def test_stable_task_identity_audit() -> None:
    frame = pd.DataFrame(
        [
            {"episode_id": "e", "task_id": "t", "origin_dc": 1, "task_cpu_cores": 2.0, "task_gpu_units": 1.0, "task_memory_gb": 4.0, "estimated_duration": 60.0},
            {"episode_id": "e", "task_id": "t", "origin_dc": 1, "task_cpu_cores": 2.0, "task_gpu_units": 1.0, "task_memory_gb": 4.0, "estimated_duration": 60.0},
        ]
    )
    assert stable_task_identity_pass(frame)
