from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sustaincluster_imitation.bc_v2_offline import choose_recommended_run
from sustaincluster_imitation.identifiability_audit import (
    HORIZONS_MINUTES,
    PARITY_STATUS,
    RESOURCES,
    OracleFuturePressureBuilder,
    TrainOnlyStandardizer,
    build_audit_arrays,
    current_information_parity_rows,
    deterministic_feature_hash,
    future_feature_manifest,
    load_config,
    read_audit_frame,
    verify_frozen_inputs,
    verify_same_split_identity,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/sustaincluster_imitation/privileged_distillation_identifiability_v1.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def frames(config) -> dict[str, pd.DataFrame]:
    dataset = ROOT / config["expert_dataset_dir"]
    return {
        "train": read_audit_frame(dataset / "train.parquet"),
        "validation": read_audit_frame(dataset / "val.parquet"),
        "test": read_audit_frame(dataset / "test.parquet"),
    }


@pytest.fixture(scope="module")
def future_builder(config) -> OracleFuturePressureBuilder:
    return OracleFuturePressureBuilder(
        ROOT / config["forecast_dataset_root"],
        ROOT / config["datacenter_config"],
    )


def test_h1_label_provenance_is_saved_shadow_action(frames) -> None:
    for frame in frames.values():
        assert "h1_action_index" in frame
        assert not frame["h1_action_index"].isna().any()
        assert frame["h1_action_index"].between(0, 5).all()


def test_frozen_dataset_contract(config) -> None:
    contract = verify_frozen_inputs(config, ROOT)
    assert contract["dataset_sha256"] == config["expected_dataset_sha256"]
    assert contract["obs_dim"] == 34
    assert contract["action_dim"] == 6


def test_original_seed_split_identity(frames) -> None:
    assert verify_same_split_identity(frames)
    assert [len(frames[name]) for name in ("train", "validation", "test")] == [
        181944,
        38988,
        38988,
    ]


def test_h1_training_input_has_no_oracle_label_leakage(frames) -> None:
    frame = frames["train"].iloc[:32].copy()
    before = build_audit_arrays(frame, label_column="h1_action_index", action_dim=6)
    frame["teacher_action_index"] = (frame["teacher_action_index"] + 1) % 6
    after = build_audit_arrays(frame, label_column="h1_action_index", action_dim=6)
    np.testing.assert_array_equal(before.observations, after.observations)
    np.testing.assert_array_equal(before.labels, after.labels)


def test_future_feature_timeline_is_repaired() -> None:
    manifest = future_feature_manifest()
    assert len(manifest) == 60
    assert sorted({row["horizon_minutes"] for row in manifest}) == [15, 30, 45, 60]
    for row in manifest:
        assert row["timeline_semantics"] == (
            f"future node {row['horizon_step']} = +{row['horizon_minutes']} minutes"
        )


def test_future_feature_order_is_dc_horizon_resource() -> None:
    manifest = future_feature_manifest()
    expected = [
        (dc, horizon, resource)
        for dc in range(1, 6)
        for horizon in HORIZONS_MINUTES
        for resource in RESOURCES
    ]
    actual = [(row["dc"], row["horizon_minutes"], row["resource"]) for row in manifest]
    assert actual == expected
    assert [row["augmented_index"] for row in manifest] == list(range(34, 94))


def test_oracle_features_are_tagged_non_deployable() -> None:
    assert {
        row["information_class"] for row in future_feature_manifest()
    } == {"NON_DEPLOYABLE_ORACLE_DIAGNOSTIC"}


def test_future_features_exclude_actions_objective_risk_and_signals() -> None:
    names = " ".join(row["feature_name"] for row in future_feature_manifest()).lower()
    for forbidden in ("action", "objective", "reward", "risk", "trigger", "price", "carbon"):
        assert forbidden not in names


def test_oracle_future_construction_is_deterministic(frames, future_builder) -> None:
    timestamps = frames["test"]["timestamp"].iloc[:20].tolist()
    first = future_builder.matrix(timestamps)
    second = future_builder.matrix(timestamps)
    np.testing.assert_array_equal(first, second)
    sample_ids = frames["test"]["sample_id"].iloc[:20].astype(str).tolist()
    assert deterministic_feature_hash(sample_ids, first) == deterministic_feature_hash(
        sample_ids, second
    )


def test_same_timestamp_has_same_future_pressure(frames, future_builder) -> None:
    timestamp = frames["test"]["timestamp"].iloc[0]
    first = future_builder.vector(timestamp)
    second = future_builder.vector(pd.Timestamp(timestamp))
    np.testing.assert_array_equal(first, second)


def test_oracle_future_matrix_has_60_nonnegative_values(frames, future_builder) -> None:
    matrix = future_builder.matrix(frames["test"]["timestamp"].iloc[:8].tolist())
    assert matrix.shape == (8, 60)
    assert np.isfinite(matrix).all()
    assert (matrix >= 0).all()


def test_normalizer_can_only_fit_train() -> None:
    values = np.arange(180, dtype=float).reshape(3, 60)
    with pytest.raises(ValueError, match="train only"):
        TrainOnlyStandardizer.fit(values, split="validation")
    normalizer = TrainOnlyStandardizer.fit(values, split="train")
    np.testing.assert_allclose(normalizer.transform(values).mean(axis=0), 0.0, atol=1e-6)
    assert normalizer.fitted_split == "train"


def test_augmented_observation_is_94d(frames, future_builder) -> None:
    frame = frames["test"].iloc[:16]
    raw = future_builder.matrix(frame["timestamp"].tolist())
    normalizer = TrainOnlyStandardizer.fit(raw, split="train")
    arrays = build_audit_arrays(
        frame,
        label_column="teacher_action_index",
        action_dim=6,
        future_features=normalizer.transform(raw),
    )
    assert arrays.observations.shape == (16, 94)


def test_h1_and_oracle_labels_are_feasible(frames) -> None:
    frame = frames["test"].iloc[:1024]
    h1 = build_audit_arrays(frame, label_column="h1_action_index", action_dim=6)
    oracle = build_audit_arrays(
        frame, label_column="teacher_action_index", action_dim=6
    )
    assert h1.masks[np.arange(len(h1)), h1.labels].all()
    assert oracle.masks[np.arange(len(oracle)), oracle.labels].all()


def test_current34_parity_is_code_grounded_missing_critical_state() -> None:
    rows = current_information_parity_rows()
    assert PARITY_STATUS == "CURRENT34_MISSING_CRITICAL_STATE"
    missing = {
        row["planner_input_name"]
        for row in rows
        if row["exact_or_aggregate"] == "missing"
    }
    assert {
        "task_memory_demand",
        "task_bandwidth",
        "transmission_delay",
        "joint_pending_task_set",
        "in_transit_reservations",
    } <= missing


def test_checkpoint_selection_remains_validation_only() -> None:
    runs = [
        {"seed": 11, "best_val_loss": 0.4, "test_metric": 1.0},
        {"seed": 22, "best_val_loss": 0.3, "test_metric": 0.0},
        {"seed": 33, "best_val_loss": 0.5, "test_metric": 1.0},
    ]
    assert choose_recommended_run(runs)["seed"] == 22
