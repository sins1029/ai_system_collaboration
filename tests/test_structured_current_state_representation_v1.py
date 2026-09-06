from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


WORKSPACE = Path(__file__).resolve().parents[1]

from scripts.forecast import run_forecast_aware_mpc_v1 as repaired_runner
from scripts.imitation import build_repaired_mpc_expert_dataset_v2 as frozen_builder
from scripts.imitation import execute_structured_current_state_representation_v1 as pipeline
from scripts.imitation import launch_structured_current_state_representation_v1 as launcher
from scripts.imitation import run_structured_current_state_representation_v1 as audit
from sustaincluster_imitation.expert_dataset_v2 import build_student_task_batch
from sustaincluster_imitation.feature_encoder import SemanticActionSpace
from sustaincluster_imitation.structured_current import (
    ACTION_DIM,
    DC_IDS,
    StructuredArrayStore,
    StructuredCurrentPolicy,
    TrainOnlyStandardizer,
    dc_feature_names,
    dense_feature_names,
    extract_current_state,
    running_feature_names,
    task_feature_names,
)
from sustaincluster_mpc.action_adapter import SustainClusterActionAdapter
from sustaincluster_mpc.rolling_horizon_optimizer import RollingHorizonOptimizer


CONFIG_PATH = WORKSPACE / "configs/sustaincluster_imitation/structured_current_state_representation_v1.yaml"


@pytest.fixture(scope="module")
def config():
    return audit.load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def live_states(config):
    columns = list(pipeline.FRAME_COLUMNS)
    frame = pd.read_parquet(
        WORKSPACE
        / "artifacts/repaired_mpc_expert_dataset_v2/dataset/expert_task_actions_full.parquet",
        columns=columns,
    )
    episode = frame.loc[frame["episode_id"] == "high_load_trace__seed_3001"]
    env = frozen_builder.build_env(
        pd.Timestamp("2023-02-13T00:00:00Z"), 96, 3001, 60.0
    )
    adapter = SustainClusterActionAdapter.from_env(env)
    semantic = SemanticActionSpace(tuple(sorted(adapter.mapping.dc_id_to_action)), True)
    horizon = frozen_builder.make_horizon_adapter()
    optimizer = RollingHorizonOptimizer()
    optimizer_config = repaired_runner._optimizer_config(
        WORKSPACE / str(config["optimizer_config"])
    )
    rows0 = episode.loc[episode["step"] == 0].sort_values("task_position")
    h1_0 = horizon.build_horizon_state(env, 1, "no_future_arrivals")
    mask_0 = horizon.build_horizon_state(env, 5, "no_future_arrivals")
    student_0 = build_student_task_batch(env, mask_0, semantic)
    result_0 = optimizer.solve(h1_0, optimizer_config, adapter)
    env.step(
        audit.semantic_to_environment_actions(
            rows0["teacher_action_index"].to_numpy(dtype=np.int64),
            semantic,
            adapter,
        )
    )
    h1_1 = horizon.build_horizon_state(env, 1, "no_future_arrivals")
    representation_1 = extract_current_state(h1_1)
    yield {
        "env": env,
        "adapter": adapter,
        "semantic": semantic,
        "optimizer": optimizer,
        "optimizer_config": optimizer_config,
        "rows0": rows0,
        "h1_0": h1_0,
        "h1_1": h1_1,
        "student_0": student_0,
        "result_0": result_0,
        "representation_1": representation_1,
    }
    env.close()


def test_frozen_contract_and_split_identity(config) -> None:
    contract = audit.verify_frozen_contract(config)
    assert contract["dataset_sha256"] == config["expected_dataset_sha256"]
    split = pd.read_parquet(
        WORKSPACE / str(config["expert_dataset_dir"]) / "expert_task_actions_full.parquet",
        columns=["episode_id", "seed", "split"],
    )
    assert split.groupby("episode_id")["split"].nunique().max() == 1
    assert split.groupby("seed")["split"].nunique().max() == 1


def test_h1_label_provenance_replays_exactly(live_states) -> None:
    actual = np.asarray(
        [
            live_states["semantic"].encode_decision(item)
            for item in live_states["result_0"].first_step_decisions
        ],
        dtype=np.int64,
    )
    expected = live_states["rows0"]["h1_action_index"].to_numpy(dtype=np.int64)
    assert np.array_equal(actual, expected)


def test_current34_and_feasible_mask_replay_exactly(live_states) -> None:
    rows = live_states["rows0"]
    expected_obs = np.stack(rows["student_observation"].map(lambda x: np.asarray(x, np.float32)))
    expected_mask = np.stack(rows["feasible_action_mask"].map(lambda x: np.asarray(x, bool)))
    assert np.array_equal(live_states["student_0"].observations, expected_obs)
    assert np.array_equal(live_states["student_0"].feasible_action_mask, expected_mask)


def test_h1_is_joint_and_read_only(live_states) -> None:
    state = live_states["h1_0"]
    result = live_states["result_0"]
    assert len(result.first_step_decisions) == len(state.current.tasks) > 1
    assert result.integer_variable_count == len(state.current.tasks) * len(DC_IDS) + len(
        state.current.tasks
    )


def test_exact_h1_repeated_call_is_deterministic(live_states) -> None:
    state = live_states["h1_0"]
    values = []
    for _ in range(3):
        result = live_states["optimizer"].solve(
            state, live_states["optimizer_config"], live_states["adapter"]
        )
        values.append(tuple(item.dc_id if item.decision == "assign" else 0 for item in result.first_step_decisions))
    assert values[0] == values[1] == values[2]


def test_current_state_excludes_future_and_oracle(live_states) -> None:
    state = live_states["h1_1"]
    assert state.horizon == 1
    assert state.forecast_mode == "no_future_arrivals"
    assert state.information_mode == "deployable"
    assert state.future_arrivals == ()


def test_pending_set_is_exactly_current_tasks(live_states) -> None:
    state = live_states["h1_1"]
    representation = live_states["representation_1"]
    assert len(representation.pending) == len(state.current.tasks)
    assert representation.pending.shape[1] == len(task_feature_names())


def test_running_set_contains_only_current_running_tasks(live_states) -> None:
    state = live_states["h1_1"]
    expected = {
        str(task.job_name)
        for dc in live_states["env"].cluster_manager.datacenters.values()
        for task in dc.running_tasks
    }
    actual = {task.task_id for task in state.running_tasks}
    assert actual == expected
    assert live_states["representation_1"].running.shape[1] == len(running_feature_names())


def test_running_release_is_controller_visible_estimate(live_states) -> None:
    state = live_states["h1_1"]
    assert state.information_mode == "deployable"
    assert all(task.release_step >= 0 for task in state.running_tasks)
    forbidden = "true_duration"
    assert forbidden not in " ".join(running_feature_names()).lower()


def test_dense_schema_has_no_action_or_future_leakage() -> None:
    names = tuple(name.lower() for name in dense_feature_names())
    forbidden = (
        "teacher_action",
        "h1_action",
        "oracle",
        "future",
        "forecast",
        "objective",
        "reward",
        "true_duration",
        "risk_score",
        "trigger",
    )
    assert not any(token in name for name in names for token in forbidden)


def test_dense_feature_shape_matches_schema(live_states) -> None:
    representation = live_states["representation_1"]
    assert representation.dense.shape == (
        len(live_states["h1_1"].current.tasks),
        len(dense_feature_names()),
    )
    assert representation.datacenters.shape == (len(DC_IDS), len(dc_feature_names()))


def test_train_only_normalization_rejects_other_splits() -> None:
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    with pytest.raises(ValueError, match="train split"):
        TrainOnlyStandardizer.fit(values, split="validation")
    scaler = TrainOnlyStandardizer.fit(values, split="train")
    assert scaler.fitted_split == "train"
    assert np.isfinite(scaler.transform(values)).all()


def test_structured_store_offsets_and_masks_are_exact() -> None:
    store = StructuredArrayStore(
        pending_values=np.zeros((3, len(task_feature_names())), np.float32),
        state_offsets=np.asarray([0, 2, 3]),
        running_values=np.zeros((1, len(running_feature_names())), np.float32),
        running_offsets=np.asarray([0, 1, 1]),
        dc_values=np.zeros((2, len(DC_IDS), len(dc_feature_names())), np.float32),
        labels=np.asarray([0, 1, 2]),
        masks=np.ones((3, ACTION_DIM), bool),
        state_ids=("s0", "s1"),
        splits=("train", "test"),
    )
    assert store.state_arrays(0)[0].shape[0] == 2
    assert store.state_arrays(1)[0].shape[0] == 1
    assert np.array_equal(store.state_indices("test"), np.asarray([1]))


def test_pending_and_running_set_permutation_invariance() -> None:
    torch.manual_seed(4)
    model = StructuredCurrentPolicy(
        len(task_feature_names()), len(running_feature_names()), len(dc_feature_names()), 16
    ).eval()
    pending = torch.randn(7, len(task_feature_names()))
    running = torch.randn(5, len(running_feature_names()))
    dcs = torch.randn(5, len(dc_feature_names()))
    dc_ids = torch.tensor(DC_IDS)
    baseline = model.forward_state(pending, running, dcs, dc_ids)
    pending_order = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    running_order = torch.tensor([4, 1, 3, 0, 2])
    permuted = model.forward_state(
        pending[pending_order], running[running_order], dcs, dc_ids
    )
    assert torch.allclose(permuted, baseline[pending_order], atol=1e-6)


def test_dc_identity_is_preserved_under_row_permutation() -> None:
    torch.manual_seed(5)
    model = StructuredCurrentPolicy(
        len(task_feature_names()), len(running_feature_names()), len(dc_feature_names()), 16
    ).eval()
    pending = torch.randn(4, len(task_feature_names()))
    running = torch.randn(3, len(running_feature_names()))
    dcs = torch.randn(5, len(dc_feature_names()))
    ids = torch.tensor(DC_IDS)
    baseline = model.forward_state(pending, running, dcs, ids)
    order = torch.tensor([2, 4, 0, 3, 1])
    permuted = model.forward_state(pending, running, dcs[order], ids[order])
    assert torch.allclose(permuted, baseline, atol=1e-6)


def test_validation_only_run_selection() -> None:
    runs = [
        {"seed": 11, "best_val_loss": 0.4, "test_accuracy": 0.99},
        {"seed": 22, "best_val_loss": 0.3, "test_accuracy": 0.10},
    ]
    assert launcher.choose_best(runs)["seed"] == 22


def test_information_flow_declares_no_sequential_teacher_forcing() -> None:
    rows = audit.information_flow_rows()
    assert any(row["planner_input_name"] == "all pending tasks" for row in rows)
    assert all(row["information_class"] == "DEPLOYABLE_CURRENT_ONLY" for row in rows)
    assert "teacher forcing" not in " ".join(row["source"].lower() for row in rows)
