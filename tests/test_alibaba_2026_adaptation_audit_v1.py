from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.audit import alibaba_2026_adaptation_audit_v1 as audit


@pytest.fixture(scope="module")
def audited_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    return audit.validate_sources()


def test_frozen_sources_are_complete_and_task_time_is_recoverable(audited_sources) -> None:
    jobs, nodes = audited_sources

    assert jobs.shape == (466_867, 9)
    assert nodes.shape == (4_278, 4)
    assert jobs["job_name"].is_unique
    assert jobs["submit_time"].is_monotonic_increasing
    assert jobs["submit_time"].min() == 0
    assert jobs["submit_time"].max() == 15_902_470
    assert jobs.isna().sum().sum() == 0
    assert nodes.isna().sum().sum() == 0
    assert set(jobs["job_type"]) == {"HP", "Spot"}
    assert nodes["gpu_model"].nunique() == 6


def test_spot_canonical_sample_preserves_direct_fields_without_silent_fill(
    audited_sources,
) -> None:
    jobs, _ = audited_sources
    sample = audit.spot_canonical_sample(jobs, count=100)
    source = jobs.head(100).reset_index(drop=True)

    assert len(sample) == 100
    assert (sample["worker_num"] > 1).any()
    np.testing.assert_allclose(sample["cpu_request"], source["cpu_request"])
    np.testing.assert_allclose(sample["gpu_request"], source["gpu_request"])
    np.testing.assert_array_equal(sample["worker_num"], source["worker_num"])
    np.testing.assert_array_equal(sample["true_duration"], source["duration"])
    assert set(sample["memory_request"]) == {"MISSING"}
    assert set(sample["bandwidth"]) == {"MISSING"}
    assert set(sample["estimated_duration"]) == {"MISSING"}
    assert set(sample["origin_dc"]) == {"MISSING"}
    assert set(sample["sla"]) == {"MISSING"}
    assert sample["resource_scope"].str.contains("DOES NOT DISAMBIGUATE").all()
    assert sample["notes"].str.contains("No multiplication by worker_num").all()


def test_gpu2026_sample_is_explicitly_documentation_only() -> None:
    sample = audit.gpu2026_canonical_template()
    row = sample.iloc[0]

    assert row["sample_evidence"] == "DOCUMENTATION_LEVEL_ONLY"
    assert row["source_record_id"] == "NO_LOCAL_FACT_ROW"
    for field in (
        "task_id",
        "arrival_time",
        "cpu_request",
        "gpu_request",
        "memory_request",
        "bandwidth",
        "true_duration",
        "estimated_duration",
        "priority",
        "task_type",
        "gpu_type",
        "worker_num",
    ):
        assert row[field] == "MISSING"


def test_mapping_keeps_arrival_memory_network_and_duration_semantics_separate() -> None:
    mapping = audit.source_mapping().set_index("target_field")

    assert mapping.loc["arrival_time", "gpu2026_mapping"] == "MISSING"
    assert mapping.loc["arrival_time", "spotgpu2026_mapping"] == "DIRECT"
    assert mapping.loc["memory_request", "gpu2026_mapping"] == "MISSING"
    assert mapping.loc["memory_request", "spotgpu2026_mapping"] == "MISSING"
    assert mapping.loc["bandwidth", "gpu2026_mapping"] == "MISSING"
    assert mapping.loc["bandwidth", "spotgpu2026_mapping"] == "MISSING"
    assert mapping.loc["true_duration", "reliability"] == "HIGH"
    assert mapping.loc["true_duration", "gpu2026_mapping"] == "DERIVED"
    assert mapping.loc["sla", "gpu2026_mapping"] == "MODELED"
    assert mapping.loc["sla", "spotgpu2026_mapping"] == "MODELED"
    assert mapping.loc["estimated_duration", "spotgpu2026_mapping"] == "MODELED"


def test_compatibility_and_v3_schema_preserve_information_contract() -> None:
    compatibility = audit.compatibility_matrix().set_index("dimension")
    schema = audit.v3_schema().set_index(["level", "field"])

    assert compatibility.loc["task-level arrival", "gpu2026"] == "POOR"
    assert compatibility.loc["task-level arrival", "spotgpu2026"] == "GOOD"
    assert compatibility.loc["memory", "gpu2026"] == "MISSING"
    assert compatibility.loc["memory", "spotgpu2026"] == "MISSING"
    assert schema.loc[("task", "estimated_duration"), "information_class"] == (
        "DEPLOYABLE_CURRENT"
    )
    assert schema.loc[("task", "true_duration"), "information_class"] == (
        "SIMULATOR_GROUND_TRUTH"
    )
    assert schema.loc[("task", "h1_action"), "information_class"] == "LABEL_ONLY"
    assert schema.loc[
        ("privileged_future", "future_arrivals"), "information_class"
    ] == "PRIVILEGED_FUTURE"


def test_raw_manifest_contains_hashes_for_data_docs_and_repository_evidence() -> None:
    manifest = audit.raw_manifest().set_index("file")

    assert set(audit.EXPECTED_FILES).issubset(
        {audit.ROOT / path for path in manifest["local_path"]}
    )
    assert manifest.loc["job_info_df.csv", "status"] == "FULL_LOCAL_SOURCE_DATA"
    assert manifest.loc["gpu2026_schema.md", "status"] == (
        "OFFICIAL_DOCUMENTATION_SNAPSHOT"
    )
    assert manifest.loc["repository_metadata.json", "status"] == (
        "UPSTREAM_REPOSITORY_API_SNAPSHOT"
    )
    local_evidence = manifest[
        manifest["status"] != "NOT_DOWNLOADED_INTENTIONALLY"
    ]
    assert local_evidence["sha256"].str.fullmatch(r"[A-F0-9]{64}").all()
    not_downloaded = manifest[manifest["status"] == "NOT_DOWNLOADED_INTENTIONALLY"]
