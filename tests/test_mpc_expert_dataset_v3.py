from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit import spotgpu2026_scenario_contract_v1 as spot
from scripts.imitation import build_mpc_expert_dataset_v3 as v3


OUTPUT = v3.OUTPUT


@pytest.fixture(scope="module")
def config() -> dict:
    return v3.load_config()


@pytest.fixture(scope="module")
def canonical() -> pd.DataFrame:
    return pd.read_parquet(OUTPUT / "preflight" / "spot_full_canonical.parquet")



def test_frozen_spot_contract_is_reused_exactly(config) -> None:
    scenario = config["scenario_b"]

    assert scenario["request_scope"] == "PER_JOB"
    assert scenario["time_step_seconds"] == 900
    assert scenario["task_order"] == ["submit_time", "original_index", "task_id"]
    assert scenario["no_shuffle"] is True
    assert scenario["duration_estimator"] == {
        "method": "hierarchical_conditional_median",
        "train_only": True,
        "minimum_support": 100,
    }
    assert scenario["memory"]["method"] == (
        "ALIBABA2020_TRAIN_ONLY_CONDITIONAL_MEDIAN"
    )
    assert scenario["bandwidth"]["value_gb"] == 0.020062580706
    assert scenario["origin"]["runtime_rng"] is False
    assert scenario["sla_profile"] == "medium"
    assert scenario["max_wait_steps"] == {"HP": 1, "Spot": 8}
    assert scenario["defer_allowed"] == {"HP": True, "Spot": True}
    assert scenario["gpu_heterogeneity"] == "METADATA_ONLY"


def test_arrival_boundaries_and_order_are_deterministic() -> None:
    np.testing.assert_array_equal(
        spot.arrival_step([899, 900, 1799, 1800]),
        np.asarray([0, 1, 1, 2]),
    )
    raw = pd.DataFrame(
        {
            "job_name": ["4", "3", "2", "1"],
            "submit_time": [1800, 900, 900, 900],
            "original_index": [3, 2, 1, 0],
        }
    )
    first = spot.stable_task_order(raw)
    second = spot.stable_task_order(raw)

    assert first["job_name"].tolist() == ["1", "2", "3", "4"]
    pd.testing.assert_frame_equal(first, second)


def test_full_canonical_has_no_drop_and_preserves_per_job_requests(canonical) -> None:
    assert len(canonical) == 466_867
    assert canonical["task_id"].is_unique
    assert canonical["raw_task_id"].notna().all()
    assert canonical["original_index"].is_unique
    np.testing.assert_array_equal(
        canonical["arrival_step"],
        canonical["submit_time"].to_numpy(dtype=np.int64) // 900,
    )
    np.testing.assert_array_equal(
        canonical["cpu_request_effective"], canonical["cpu_request_raw"]
    )
    np.testing.assert_array_equal(
        canonical["gpu_request_effective"], canonical["gpu_request_raw"]
    )
    assert set(canonical["request_scope"]) == {"PER_JOB"}
    assert np.all(canonical["bandwidth"].to_numpy() == 0.020062580706)


def test_task_order_has_no_shuffle_and_gpu_priority_metadata_is_preserved(
    canonical,
) -> None:
    key_columns = ["submit_time", "original_index", "task_id"]
    sorted_keys = canonical.sort_values(key_columns, kind="mergesort")[key_columns]
    pd.testing.assert_frame_equal(
        canonical[key_columns].reset_index(drop=True),
        sorted_keys.reset_index(drop=True),
    )
    assert canonical["gpu_model"].nunique() == 6
    assert set(canonical["priority"]) == {"HP", "Spot"}
    assert set(canonical["gpu_heterogeneity_mode"]) == {"METADATA_ONLY"}


def test_true_duration_and_privileged_fields_are_isolated(canonical) -> None:
    deployable = pd.read_parquet(
        OUTPUT / "preflight" / "deployable_current" / "spot_tasks.parquet"
    )
    truth = pd.read_parquet(
        OUTPUT / "preflight" / "simulator_only" / "spot_truth.parquet"
    )
    forbidden = ("true_duration", "future", "oracle", "teacher", "h1_", "h4_")

    assert "true_duration" in truth
    assert set(truth["task_id"]) == set(deployable["task_id"])
    assert not any(
        token in column.lower()
        for column in deployable.columns
        for token in forbidden
    )
    assert "estimated_duration" in deployable
    assert "estimated_duration_steps" in deployable


def test_train_only_models_and_exact_source_tags_are_preserved(canonical) -> None:
    audit = pd.read_csv(OUTPUT / "02_spot_full_canonical_audit.csv").set_index(
        "check"
    )

    assert int(audit.loc["duration_estimator_train_rows", "value"]) == 326_806
    assert set(canonical["estimated_duration_source"]) == {
        "MODELED_SPOT_TRAIN_ONLY_HIERARCHICAL_MEDIAN_V1"
    }
    assert set(canonical["memory_source"]) == {
        "MODELED_ALIBABA2020_CONDITIONAL_MEDIAN_V1"
    }
    assert set(canonical["memory_fallback_level"]) <= {
        "CPU_GPU_BIN",
        "CPU_BIN",
        "GLOBAL_TRAIN_MEDIAN",
    }
    assert set(canonical["bandwidth_source"]) == {
        "MODELED_ALIBABA2020_GLOBAL_MEDIAN_V1_FROZEN_12DP"
    }


def test_origin_and_medium_sla_are_exact_and_deterministic(config, canonical) -> None:
    small = pd.DataFrame(
        {
            "job_name": ["11", "12"],
            "submit_time": [0, 900],
        }
    )
    spot_config = spot.load_config()
    first = spot.deterministic_origins(small, spot_config)
    second = spot.deterministic_origins(small, spot_config)

    np.testing.assert_array_equal(first, second)
    assert set(canonical["origin_source"]) == {"MODELED_ORIGIN"}
    assert canonical.loc[canonical["priority"] == "HP", "max_wait_steps"].eq(1).all()
    assert canonical.loc[
        canonical["priority"] == "Spot", "max_wait_steps"
    ].eq(8).all()
    assert canonical["defer_allowed"].all()


def test_temporal_split_is_ordered_and_no_state_crosses_split(canonical) -> None:
    assert canonical.groupby("arrival_step")["temporal_split"].nunique().max() == 1
    split_order = {"train": 0, "validation": 1, "test": 2}
    encoded = canonical["temporal_split"].map(split_order).to_numpy()
    assert np.all(encoded[1:] >= encoded[:-1])
    assert canonical.groupby("temporal_split", sort=False).size().to_dict() == {
        "train": 326_842,
        "validation": 69_999,
        "test": 70_026,
    }


def test_joint_state_and_label_provenance_are_frozen(config) -> None:
    provenance = v3.field_provenance().set_index(
        ["scenario_id", "region", "field"]
    ).sort_index()
    scenario_a = "A_ALIBABA2020_REPAIRED"
    assert (scenario_a, "deployable_current", "pending_task_ids") in provenance.index
    assert (scenario_a, "deployable_current", "running_task_ids") in provenance.index
    assert (scenario_a, "deployable_current", "in_transit_task_ids") in provenance.index
    assert ("ALL", "labels", "joint_action_group") in provenance.index
    assert provenance.loc[("ALL", "labels", "h1_action"), "source_or_rule"] == (
        "H1_REPAIRED_CURRENT_ONLY"
    )
    assert (
        provenance.loc[
            ("ALL", "labels", "oracle_h4_action"), "source_or_rule"
        ]
        == "H4_ORACLE_REPAIRED"
    )
    assert config["planners"]["h4_oracle"]["privileged_offsets_minutes"] == [
        15,
        30,
        45,
        60,
    ]
    assert config["information_regions"]["privileged_future"][
        "default_loader_visible"
    ] is False
    assert config["information_regions"]["simulator_only"][
        "default_loader_visible"
    ] is False


def test_native_capacity_supersedes_old_capacity_blocker() -> None:
    manifest = json.loads(
        (OUTPUT / "43_spot_capacity_calibration_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    feasibility = pd.read_parquet(
        OUTPUT / "preflight" / "spot_static_feasibility.parquet"
    )
    native_gpu = next(
        item
        for item in manifest["native_pressure"]
        if item["resource"] == "gpu"
    )
    comparison = pd.read_csv(OUTPUT / "37_old_vs_native_capacity.csv")
    old_gpu = comparison[
        (comparison["scenario"] == "OLD_SUSTAINCLUSTER_5DC")
        & (comparison["resource"] == "gpu")
    ].iloc[0]

    assert feasibility["feasible"].all()
    assert manifest["static_feasible_rate"] == 1.0
    assert old_gpu["mandatory_peak_ratio"] == pytest.approx(
        2.8185068965517286
    )
    assert native_gpu["capacity"] == 10412
    assert native_gpu["mandatory_peak_ratio"] == pytest.approx(
        0.7850240107568203
    )
    assert manifest["final_status"] == "SPOT NATIVE CAPACITY READY"
    current = json.loads(
        (OUTPUT / "00_generation_manifest.json").read_text(encoding="utf-8")
    )
    assert current["timeline_complete"] is True
    assert current["task_decisions"] == 466867

def test_solver_failure_detection_rejects_fallback_labels() -> None:
    clean = pd.DataFrame({"solver_failures": [0, 0]})
    v3.validate_solver_failures(clean)

    failed = pd.DataFrame({"solver_failures": [0, 1]})
    with pytest.raises(RuntimeError, match="cannot become expert labels"):
        v3.validate_solver_failures(failed)


def test_alibaba_v2_baseline_is_hash_verified_and_v3_alignment_is_measured(
    config,
) -> None:
    manifest = json.loads(
        (OUTPUT / "00_alibaba2020_v3_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    root = v3.ROOT / config["scenario_a"]["v2_root"]
    tasks, states, episodes = v3.load_v2(config)

    assert v3.sha256(root / config["scenario_a"]["v2_task_file"]) == (
        config["scenario_a"]["v2_task_sha256"]
    )
    assert v3.sha256(root / config["scenario_a"]["v2_state_file"]) == (
        config["scenario_a"]["v2_state_sha256"]
    )
    assert (len(episodes), len(states), len(tasks)) == (80, 7_680, 259_920)
    measured = {
        row["metric"]: row["agreement_rate"]
        for row in manifest["alignment"]["summary"]
    }
    assert measured["student_observation_agreement"] == 1.0
    assert measured["feasible_mask_agreement"] == 1.0
    assert measured["h1_action_agreement"] == 1.0
    assert measured["h4_oracle_action_agreement"] == 1.0
    assert manifest["alignment"]["task_mismatch_reason_counts"] == {}

def test_same_canonical_content_hash_and_frozen_artifact_hashes_match(
    canonical,
) -> None:
    audit = pd.read_csv(OUTPUT / "02_spot_full_canonical_audit.csv").set_index(
        "check"
    )
    assert v3.dataframe_hash(canonical) == audit.loc[
        "canonical_content_sha256", "value"
    ]
    integrity = pd.read_csv(OUTPUT / "23_current_integrity_manifest.csv")
    superseded = {
        "00_generation_manifest.json",
        "24_tests.md",
        "dataset/scenario_b/BLOCKED.md",
        "privileged_future/scenario_b/BLOCKED.md",
    }
    for item in integrity.itertuples(index=False):
        if item.path in superseded:
            continue
        path = OUTPUT / item.path
        assert path.is_file()
        assert path.stat().st_size == item.bytes
        assert v3.sha256(path) == item.sha256

def test_required_reports_exist_and_status_is_ready() -> None:
    assert all((OUTPUT / name).is_file() for name in v3.EXPECTED_NUMBERED_OUTPUTS)
    assert (OUTPUT / "00_generation_manifest.json").is_file()
    tests = (OUTPUT / "24_tests.md").read_text(encoding="utf-8")
    diagnosis = (OUTPUT / "25_final_diagnosis.md").read_text(encoding="utf-8")

    assert "SPOT NATIVE CAPACITY READY" in diagnosis
    assert "| compileall |" in tests
    assert "FAIL" not in tests

