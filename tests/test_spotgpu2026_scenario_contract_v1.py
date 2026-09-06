from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit import spotgpu2026_scenario_contract_v1 as contract


ARTIFACT_DIR = contract.ROOT / "artifacts" / "spotgpu2026_scenario_contract_v1"


def _config() -> dict:
    config = deepcopy(contract.load_config())
    config["memory_model"]["minimum_support"] = 1
    config["bandwidth_model"]["minimum_support"] = 1
    return config


def _raw_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_name": ["103", "101", "102", "104"],
            "organization": ["o"] * 4,
            "gpu_model": ["A100", "A100", "V100", "V100"],
            "cpu_request": [8.0, 4.0, 16.0, 2.0],
            "gpu_request": [2.0, 1.0, 4.0, 1.0],
            "worker_num": [2, 1, 4, 1],
            "submit_time": [1800, 900, 900, 1799],
            "duration": [1000, 2000, 3000, 4000],
            "job_type": ["HP", "Spot", "HP", "Spot"],
            "original_index": [13, 10, 12, 11],
        }
    )


def _training_sample() -> pd.DataFrame:
    frame = _raw_sample().copy()
    frame["original_index"] = np.arange(len(frame))
    return frame


def _resource_training() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cpu": [2.0, 4.0, 8.0, 16.0, 32.0],
            "gpu": [1.0, 1.0, 2.0, 4.0, 8.0],
            "memory": [3.0, 6.0, 12.0, 24.0, 48.0],
            "bandwidth": [0.01, 0.02, 0.04, 0.08, 0.16],
        }
    )


def _models():
    config = _config()
    duration = contract.fit_duration_estimator(_training_sample(), support=1)
    resources = _resource_training()
    memory = contract.fit_memory_model(resources, config["memory_model"])
    bandwidth = contract.fit_bandwidth_models(
        resources, config["memory_model"], config["bandwidth_model"]
    )
    return config, duration, memory, bandwidth


def _canonical() -> pd.DataFrame:
    config, duration, memory, bandwidth = _models()
    return contract.build_canonical_sample(
        _raw_sample(), duration, memory, bandwidth, config
    )


def test_arrival_boundaries_and_task_order_are_deterministic() -> None:
    np.testing.assert_array_equal(
        contract.arrival_step([899, 900, 1799, 1800]),
        np.asarray([0, 1, 1, 2]),
    )

    ordered_once = contract.stable_task_order(_raw_sample())
    ordered_twice = contract.stable_task_order(_raw_sample())
    assert ordered_once["job_name"].tolist() == ["101", "102", "104", "103"]
    pd.testing.assert_frame_equal(ordered_once, ordered_twice)


def test_duration_estimator_is_train_only_and_does_not_read_query_truth() -> None:
    training = _training_sample()
    model = contract.fit_duration_estimator(training, support=1)
    query = _raw_sample()
    changed_truth = query.copy()
    changed_truth["duration"] = changed_truth["duration"] * 1_000_000

    prediction, levels = model.predict(query)
    repeated, repeated_levels = model.predict(changed_truth)

    assert model.train_rows == len(training)
    assert model.train_max_original_index == 3
    np.testing.assert_array_equal(prediction, repeated)
    np.testing.assert_array_equal(levels, repeated_levels)


def test_memory_and_bandwidth_models_use_only_supplied_training_rows() -> None:
    config = _config()
    training = _resource_training()
    memory = contract.fit_memory_model(training, config["memory_model"])
    bandwidth = contract.fit_bandwidth_models(
        training, config["memory_model"], config["bandwidth_model"]
    )
    cpu = np.asarray([4.0, 16.0])
    gpu = np.asarray([1.0, 4.0])

    first_memory, first_levels = memory.predict(cpu, gpu)
    second_memory, second_levels = memory.predict(cpu, gpu)
    first_bandwidth = bandwidth.candidates(cpu, gpu)
    second_bandwidth = bandwidth.candidates(cpu, gpu)

    assert memory.train_rows == len(training)
    assert bandwidth.train_rows == len(training)
    np.testing.assert_array_equal(first_memory, second_memory)
    np.testing.assert_array_equal(first_levels, second_levels)
    for candidate in first_bandwidth:
        np.testing.assert_array_equal(
            first_bandwidth[candidate], second_bandwidth[candidate]
        )


def test_canonical_conversion_preserves_scope_gpu_priority_and_is_repeatable() -> None:
    first = _canonical()
    second = _canonical()

    assert contract.frame_hash(first) == contract.frame_hash(second)
    np.testing.assert_array_equal(
        first["cpu_request_effective"], first["cpu_request_raw"]
    )
    np.testing.assert_array_equal(
        first["gpu_request_effective"], first["gpu_request_raw"]
    )
    assert set(first["request_scope"]) == {"PER_JOB"}
    assert first["gpu_model"].tolist() == ["A100", "V100", "V100", "A100"]
    assert first["priority"].tolist() == ["Spot", "HP", "Spot", "HP"]
    assert set(first["gpu_heterogeneity_mode"]) == {"METADATA_ONLY"}


def test_deployable_observation_excludes_truth_future_and_actions() -> None:
    deployable = contract.deployable_observation(_canonical())
    lowered = {column.lower() for column in deployable.columns}

    assert "true_duration" not in lowered
    for token in ("future", "oracle", "teacher", "h1_action", "h4_action"):
        assert not any(token in column for column in lowered)
    assert set(deployable.columns) == set(contract.DEPLOYABLE_COLUMNS)


def test_origin_and_provenance_contracts_are_complete() -> None:
    config = _config()
    raw = contract.stable_task_order(_raw_sample())
    first = contract.deterministic_origins(raw, config)
    second = contract.deterministic_origins(raw, config)
    canonical = _canonical()
    provenance = contract.provenance_contract()

    np.testing.assert_array_equal(first, second)
    assert set(provenance["field"]) == set(canonical.columns)
    assert provenance["field"].is_unique
    assert set(provenance["provenance_type"]) == {
        "DIRECT", "DERIVED", "MODELED", "SIMULATOR_ONLY", "CONSTANT"
    }
    truth = provenance.set_index("field").loc["true_duration"]
    assert truth["provenance_type"] == "SIMULATOR_ONLY"
    assert truth["visibility"] == "forbidden deployable"


def test_static_feasibility_detects_feasible_and_impossible_tasks() -> None:
    config = _config()
    canonical = _canonical().iloc[:2].copy()
    canonical.loc[canonical.index[1], "cpu_request_effective"] = 1_000_000
    result = contract.static_feasibility(canonical, config)

    assert bool(result.iloc[0]["feasible"])
    assert not bool(result.iloc[1]["feasible"])
    assert "CPU_EXCEEDS_ALL_DCS" in result.iloc[1]["infeasible_reason"]


def test_frozen_artifacts_are_complete_and_pass_release_gates() -> None:
    required = [
        f"{index:02d}_{name}"
        for index, name in enumerate(
            [
                "request_scope_audit.md",
                "request_scope_evidence.csv",
                "request_scope_sensitivity.csv",
                "arrival_contract.md",
                "duration_estimator_contract.md",
                "duration_estimator_support.csv",
                "duration_estimator_validation.csv",
                "memory_model_contract.md",
                "memory_model_validation.csv",
                "bandwidth_model_contract.md",
                "bandwidth_sensitivity.csv",
                "origin_dc_contract.md",
                "origin_dc_distribution.csv",
                "sla_defer_contract.md",
                "sla_scenario_table.csv",
                "gpu_type_contract.md",
                "field_provenance_contract.csv",
                "spot_canonical_schema.csv",
                "spot_canonical_schema.md",
                "spot_canonical_frozen_sample.csv",
                "resource_quality_audit.csv",
                "resource_quality_report.md",
                "static_feasibility_audit.csv",
                "static_feasibility_report.md",
                "frozen_scenario_contract.md",
                "information_leakage_audit.md",
                "tests.md",
                "final_diagnosis.md",
                "summary.md",
            ],
            start=1,
        )
    ]
    assert all((ARTIFACT_DIR / name).is_file() for name in required)

    sample = pd.read_csv(ARTIFACT_DIR / required[19])
    schema = pd.read_csv(ARTIFACT_DIR / required[17])
    feasibility = pd.read_csv(ARTIFACT_DIR / required[22])
    manifest = json.loads(
        (ARTIFACT_DIR / "audit_manifest.json").read_text(encoding="utf-8")
    )
