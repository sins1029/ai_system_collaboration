from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.imitation import build_spotgpu2026_expert_dataset_v3 as spot_v3
from sustaincluster_imitation.spot_expert_v3_loader import load_spot_expert_v3


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/mpc_expert_dataset_v3"


@pytest.fixture(scope="module")
def inputs() -> dict[str, object]:
    return spot_v3.load_inputs()


def test_checkpoint_resume_exact_gate() -> None:
    path = (
        OUTPUT
        / "checkpoints/scenario_b/resume_exact_test_v1"
        / "checkpoint_resume_exact_test.json"
    )
    result = json.loads(path.read_text("utf-8"))
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["runtime_signature_a"] == result["runtime_signature_b"]
    assert (
        result["semantic_output_hashes_a"]
        == result["semantic_output_hashes_b"]
    )


def test_frozen_spot_contract_and_capacity(inputs: dict[str, object]) -> None:
    canonical = inputs["canonical"]
    dcs = inputs["dcs"]
    assert isinstance(canonical, pd.DataFrame)
    assert isinstance(dcs, pd.DataFrame)
    assert len(canonical) == 466867
    assert int(dcs["total_cores"].sum()) == 632636
    assert int(dcs["total_gpus"].sum()) == 10412
    assert dcs["total_gpus"].astype(int).tolist() == [
        1875,
        2291,
        2070,
        2615,
        1561,
    ]
    assert np.isclose(dcs["total_mem"].sum(), 542259.4285714285)
    assert set(dcs["memory_provenance"]) == {"MODELED_DC_MEMORY"}


def test_frozen_canonical_semantics(inputs: dict[str, object]) -> None:
    canonical = inputs["canonical"]
    assert isinstance(canonical, pd.DataFrame)
    assert np.array_equal(
        canonical["arrival_step"].to_numpy(),
        canonical["submit_time"].to_numpy() // 900,
    )
    assert canonical["cpu_request_effective"].equals(
        canonical["cpu_request_raw"]
    )
    assert canonical["gpu_request_effective"].equals(
        canonical["gpu_request_raw"]
    )
    assert set(canonical["request_scope"]) == {"PER_JOB"}
    assert bool(np.all(canonical["bandwidth"] == 0.020062580706))
    assert set(canonical["origin_source"]) == {"MODELED_ORIGIN"}
    assert set(canonical["max_wait_steps"].unique()) == {1, 8}
    assert set(canonical["estimated_duration_source"]) == {
        "MODELED_SPOT_TRAIN_ONLY_HIERARCHICAL_MEDIAN_V1"
    }
    ordered = canonical.sort_values(
        ["arrival_step", "submit_time", "original_index", "task_id"],
        kind="mergesort",
    )
    assert ordered.index.tolist() == canonical.index.tolist()


def test_energy_signal_provenance(inputs: dict[str, object]) -> None:
    dcs = inputs["dcs"]
    assert isinstance(dcs, pd.DataFrame)
    _, provenance = spot_v3.build_signal_trace(dcs, 8)
    assert set(provenance["provenance"]) == {"EXTERNAL_SCENARIO_SIGNAL"}
    assert not provenance["alibaba2026_measured"].any()
    assert set(provenance["signal_type"]) == {
        "carbon_intensity",
        "electricity_price",
    }


def test_formal_output_contract_when_generated(inputs: dict[str, object]) -> None:
    path = (
        OUTPUT
        / "dataset/scenario_b/deployable_current/states_full.parquet"
    )
    if not path.is_file():
        pytest.skip("formal Spot output has not been consolidated yet")
    checks = spot_v3.verify_formal_outputs(inputs)
    assert all(checks.values())


def test_formal_hashes_and_split_boundary_continuity() -> None:
    path = (
        OUTPUT
        / "dataset/scenario_b/deployable_current/states_full.parquet"
    )
    if not path.is_file():
        pytest.skip("formal Spot output has not been consolidated yet")
    manifest = pd.read_csv(OUTPUT / "60_spot_integrity_manifest.csv")
    formal = manifest[manifest["role"].str.startswith("formal_data:")]
    assert not formal.empty
    for row in formal.itertuples(index=False):
        artifact = ROOT / row.path
        assert artifact.is_file()
        assert spot_v3.sha256(artifact) == row.sha256

    states = pd.read_parquet(path).set_index("step")
    for left_step, right_step in ((10527, 10528), (13840, 13841)):
        assert states.at[left_step, "split"] != states.at[right_step, "split"]
        left_running = set(states.at[left_step, "running_task_ids"])
        right_running = set(states.at[right_step, "running_task_ids"])
        assert left_running & right_running
def test_default_loader_isolation_when_generated() -> None:
    path = (
        OUTPUT
        / "dataset/scenario_b/deployable_current/states_full.parquet"
    )
    if not path.is_file():
        pytest.skip("formal Spot output has not been consolidated yet")
    loaded = load_spot_expert_v3("test")
    assert loaded.labels is None
    assert loaded.simulator_truth is None
    assert loaded.privileged_future is None
    forbidden = {
        "true_duration_seconds",
        "true_completion_step",
        "h1_action",
        "h4_action",
    }
    assert not (forbidden & set(loaded.tasks.columns))
