from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from sustaincluster_imitation.bc_v2_offline import (
    PREDICTION_COLUMNS,
    PRIVILEGED_OR_DIAGNOSTIC_COLUMNS,
    TRAINING_COLUMNS,
    BCV2Arrays,
    actor_architecture,
    build_actor,
    build_prediction_frame,
    choose_recommended_run,
    infer,
    load_checkpoint,
    load_config,
    load_training_arrays,
    masked_cross_entropy,
    masked_logits,
    recovery_metrics,
    save_checkpoint,
    set_deterministic_seed,
    sha256_file,
    state_level_recovery,
    three_way_categories,
    training_columns_are_deployable_only,
    validate_frozen_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/sustaincluster_imitation/bc_v2_offline.yaml"
DATASET_ROOT = ROOT / "artifacts/repaired_mpc_expert_dataset_v2"


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def test_arrays(config) -> BCV2Arrays:
    return load_training_arrays(
        ROOT / config["dataset_dir"] / "test.parquet", 34, 6
    )


@pytest.fixture
def tiny_config() -> dict:
    return {
        "obs_dim": 34,
        "action_dim": 6,
        "hidden_dim": 16,
        "use_layer_norm": True,
    }


@pytest.fixture
def tiny_arrays() -> BCV2Arrays:
    rng = np.random.default_rng(7)
    observations = rng.normal(size=(12, 34)).astype(np.float32)
    labels = np.asarray([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 3], dtype=np.int64)
    masks = np.ones((12, 6), dtype=bool)
    return BCV2Arrays(observations, labels, masks, tuple(f"s{i}" for i in range(12)))


def test_v2_dataset_hash(config) -> None:
    path = ROOT / config["dataset_dir"] / "expert_task_actions_full.parquet"
    assert sha256_file(path) == config["expected_dataset_sha256"]


def test_frozen_contract_obs_dim_is_34(config) -> None:
    assert validate_frozen_contract(config, ROOT)["obs_dim"] == 34


def test_frozen_contract_action_dim_is_6(config) -> None:
    assert validate_frozen_contract(config, ROOT)["action_dim"] == 6


def test_action_schema_has_frozen_semantics() -> None:
    schema = json.loads((DATASET_ROOT / "07_action_schema.json").read_text(encoding="utf-8"))
    assert [schema["mapping"][str(i)]["semantic"] for i in range(6)] == [
        "defer",
        "assign_dc1",
        "assign_dc2",
        "assign_dc3",
        "assign_dc4",
        "assign_dc5",
    ]


def test_actor_output_shape(tiny_config) -> None:
    assert build_actor(tiny_config)(torch.zeros(8, 34)).shape == (8, 6)


def test_actor_architecture_is_audited(tiny_config) -> None:
    audit = actor_architecture(build_actor(tiny_config), tiny_config)
    assert audit["input_dim"] == 34
    assert audit["output_dim"] == 6
    assert audit["normalization"] == "LayerNorm"
    assert audit["activation"] == "ReLU"
    assert audit["parameter_count"] > 0


def test_random_initialization_does_not_load_old_checkpoint(tiny_config) -> None:
    set_deterministic_seed(11)
    first = build_actor(tiny_config)
    set_deterministic_seed(22)
    second = build_actor(tiny_config)
    assert not torch.equal(next(first.parameters()), next(second.parameters()))
    assert "checkpoint" not in inspect.signature(build_actor).parameters


def test_feasible_mask_is_used_during_training(tiny_config, tiny_arrays) -> None:
    model = build_actor(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.from_numpy(tiny_arrays.observations)
    labels = torch.from_numpy(tiny_arrays.labels)
    masks = torch.from_numpy(tiny_arrays.masks)
    before = next(model.parameters()).detach().clone()
    loss = masked_cross_entropy(model(x), labels, masks)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, next(model.parameters()))


def test_infeasible_action_is_masked_at_inference() -> None:
    logits = torch.tensor([[100.0, 1.0, 0.0]])
    masks = torch.tensor([[False, True, True]])
    assert masked_logits(logits, masks).argmax(dim=-1).item() == 1


def test_teacher_label_must_be_feasible() -> None:
    logits = torch.zeros(1, 3)
    labels = torch.tensor([2])
    masks = torch.tensor([[True, True, False]])
    with pytest.raises(ValueError, match="infeasible"):
        masked_cross_entropy(logits, labels, masks)


def test_all_real_test_labels_are_feasible(test_arrays) -> None:
    assert test_arrays.masks[np.arange(len(test_arrays)), test_arrays.labels].all()


def test_masked_ce_matches_manual_cross_entropy() -> None:
    logits = torch.tensor([[1.0, 2.0, 9.0], [2.0, 0.0, 1.0]])
    labels = torch.tensor([1, 2])
    masks = torch.tensor([[True, True, False], [True, False, True]])
    expected = F.cross_entropy(
        torch.tensor([[1.0, 2.0, torch.finfo(torch.float32).min], [2.0, torch.finfo(torch.float32).min, 1.0]]),
        labels,
    )
    assert torch.allclose(masked_cross_entropy(logits, labels, masks), expected)


def test_train_val_test_seed_separation() -> None:
    split = json.loads((DATASET_ROOT / "05_split_manifest.json").read_text(encoding="utf-8"))
    train = set(split["train_seeds"])
    validation = set(split["validation_seeds"])
    test = set(split["test_seeds"])
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert len(train | validation | test) == 40


def test_training_loader_columns_are_deployable_only() -> None:
    assert training_columns_are_deployable_only()
    assert not set(TRAINING_COLUMNS) & PRIVILEGED_OR_DIAGNOSTIC_COLUMNS


def test_h1_action_is_excluded_from_x() -> None:
    assert "h1_action_index" not in TRAINING_COLUMNS


def test_risk_score_is_excluded_from_x() -> None:
    assert "deployable_risk_score" not in TRAINING_COLUMNS


def test_true_duration_is_excluded() -> None:
    schema = json.loads((DATASET_ROOT / "06_feature_schema.json").read_text(encoding="utf-8"))
    observation = schema["student_observation"]
    assert observation["true_duration_present"] is False
    assert all("true_duration" not in name for name in observation["feature_names"])
    assert "estimated_duration" in observation["feature_names"][7]


def test_privileged_metadata_change_does_not_change_training_x(test_arrays) -> None:
    before = test_arrays.observations.copy()
    metadata = pd.DataFrame({"h1_action_index": [1], "deployable_risk_score": [0.2]})
    metadata.loc[0, "h1_action_index"] = 5
    metadata.loc[0, "deployable_risk_score"] = 99.0
    np.testing.assert_array_equal(before, test_arrays.observations)


def test_checkpoint_save_load_equivalence(tmp_path, tiny_config) -> None:
    set_deterministic_seed(11)
    model = build_actor(tiny_config)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        tiny_config,
        {"initialization": "RANDOM", "old_bc_weights_loaded": False},
    )
    restored, payload = load_checkpoint(path)
    x = torch.randn(4, 34)
    assert torch.equal(model(x), restored(x))
    assert payload["old_bc_weights_loaded"] is False


def test_recommended_checkpoint_uses_validation_loss_only() -> None:
    runs = [
        {"seed": 11, "best_val_loss": 0.3, "test_accuracy": 0.99},
        {"seed": 22, "best_val_loss": 0.2, "test_accuracy": 0.10},
        {"seed": 33, "best_val_loss": 0.4, "test_accuracy": 1.00},
    ]
    assert choose_recommended_run(runs)["seed"] == 22


def test_consensus_disagreement_partition() -> None:
    teacher = np.asarray([1, 2, 3, 4])
    h1 = np.asarray([1, 5, 3, 2])
    consensus = teacher == h1
    assert consensus.tolist() == [True, False, True, False]
    assert int(consensus.sum() + (~consensus).sum()) == len(teacher)


def test_teacher_recovery_rate_computation() -> None:
    teacher = np.asarray([1, 2, 3, 4])
    h1 = np.asarray([1, 5, 2, 1])
    prediction = np.asarray([1, 2, 2, 0])
    metrics = recovery_metrics(teacher, h1, prediction)
    assert metrics["disagreement_samples"] == 3
    assert metrics["teacher_recovery_rate"] == pytest.approx(1 / 3)


def test_h1_fallback_rate_computation() -> None:
    teacher = np.asarray([1, 2, 3, 4])
    h1 = np.asarray([1, 5, 2, 1])
    prediction = np.asarray([1, 2, 2, 0])
    metrics = recovery_metrics(teacher, h1, prediction)
    assert metrics["h1_fallback_rate"] == pytest.approx(1 / 3)
    assert metrics["other_rate"] == pytest.approx(1 / 3)


def test_high_risk_threshold_is_frozen_calibration_p95(config) -> None:
    thresholds = json.loads((ROOT / config["risk_thresholds"]).read_text(encoding="utf-8"))
    assert thresholds["P95_primary"] == pytest.approx(0.5854637687956499)
    assert thresholds["calibration_seeds"] == [1101, 1102, 1103, 1104, 1105]


def test_three_way_categories_form_exact_partition() -> None:
    teacher = np.asarray([1, 1, 2, 3, 4])
    h1 = np.asarray([1, 1, 5, 1, 2])
    prediction = np.asarray([1, 2, 2, 1, 5])
    categories = three_way_categories(teacher, h1, prediction)
    assert categories.tolist() == [
        "A_TEACHER_EQ_H1_EQ_BC",
        "B_TEACHER_EQ_H1_BC_DIFFERS",
        "C_TEACHER_NE_H1_BC_EQ_TEACHER",
        "D_TEACHER_NE_H1_BC_EQ_H1",
        "E_TEACHER_NE_H1_BC_OTHER",
    ]


def test_state_level_recovery_aggregation() -> None:
    frame = pd.DataFrame(
        {
            "episode_id": ["e", "e", "e"],
            "step": [0, 0, 1],
            "teacher_action_index": [1, 2, 3],
            "h1_action_index": [4, 5, 3],
        }
    )
    result = state_level_recovery(frame, np.asarray([1, 5, 3]))
    assert len(result) == 1
    assert result.iloc[0]["num_disagreement_tasks"] == 2
    assert result.iloc[0]["num_recovered_tasks"] == 1
    assert bool(result.iloc[0]["any_recovered"])
    assert not bool(result.iloc[0]["all_recovered"])


def test_prediction_schema(tiny_arrays) -> None:
    evaluation = pd.DataFrame(
        {
            "sample_id": tiny_arrays.sample_ids,
            "episode_id": ["e"] * len(tiny_arrays),
            "seed": [3001] * len(tiny_arrays),
            "scenario": ["normal_trace"] * len(tiny_arrays),
            "step": list(range(len(tiny_arrays))),
            "task_id": [f"t{i}" for i in range(len(tiny_arrays))],
            "deployable_risk_score": np.linspace(0, 1, len(tiny_arrays)),
            "feasible_action_mask": tiny_arrays.masks.tolist(),
            "teacher_action_index": tiny_arrays.labels,
            "h1_action_index": np.ones(len(tiny_arrays), dtype=int),
        }
    )
    probabilities = np.full((len(tiny_arrays), 6), 1 / 6, dtype=np.float32)
    result = build_prediction_frame(
        evaluation, tiny_arrays.labels.copy(), probabilities, 0.5
    )
    assert list(result.columns) == list(PREDICTION_COLUMNS)
    assert len(result) == len(tiny_arrays)


def test_prediction_probabilities_sum_to_one(tiny_config, tiny_arrays) -> None:
    model = build_actor(tiny_config)
    _, probabilities, _ = infer(model, tiny_arrays, batch_size=5)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_deterministic_fixed_seed_inference(tiny_config, tiny_arrays) -> None:
    set_deterministic_seed(33)
    first = build_actor(tiny_config)
    first_prediction = infer(first, tiny_arrays, batch_size=4)
    set_deterministic_seed(33)
    second = build_actor(tiny_config)
    second_prediction = infer(second, tiny_arrays, batch_size=7)
    np.testing.assert_array_equal(first_prediction[0], second_prediction[0])
    np.testing.assert_allclose(first_prediction[1], second_prediction[1], atol=1e-7)


def test_standard_training_has_no_class_or_disagreement_weighting(config) -> None:
    assert config["class_weighting"] is False
    assert config["disagreement_weighting"] is False
    assert config["model_selection"] == "validation_masked_cross_entropy_only"


def test_real_test_loader_shape(test_arrays) -> None:
    assert test_arrays.observations.shape == (38988, 34)
    assert test_arrays.masks.shape == (38988, 6)
    assert test_arrays.labels.shape == (38988,)
