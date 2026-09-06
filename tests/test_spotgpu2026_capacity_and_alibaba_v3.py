from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts.audit import spotgpu2026_capacity_calibration_v1 as capacity


OUTPUT = ROOT / "artifacts/mpc_expert_dataset_v3"


def test_complete_node_inventory_and_exact_capacity() -> None:
    _, contract = capacity.load_frozen_contract()
    nodes = capacity.load_nodes(contract)
    assert len(nodes) == 4278
    assert nodes["node_name"].nunique() == 4278
    assert int(nodes["cpu_num"].sum()) == 632636
    assert int(nodes["gpu_capacity_num"].sum()) == 10412
    assert nodes["gpu_model"].value_counts().to_dict() == {
        "A10": 2494,
        "GPU-series-1": 989,
        "A100-SXM4-80GB": 432,
        "H800": 219,
        "GPU-series-2": 122,
        "A800-SXM4-80GB": 22,
    }


def test_deterministic_partition_is_exclusive_and_conservative() -> None:
    _, contract = capacity.load_frozen_contract()
    nodes = capacity.load_nodes(contract)
    first = capacity.deterministic_partition(nodes)
    second = capacity.deterministic_partition(nodes)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == first["node_name"].nunique() == 4278
    assert sorted(first["dc_id"].unique()) == [1, 2, 3, 4, 5]
    assert int(first["cpu_num"].sum()) == 632636
    assert int(first["gpu_capacity_num"].sum()) == 10412
    _, _, _, dcs = capacity.capacity_tables(nodes, first)
    assert int(dcs["total_cores"].sum()) == 632636
    assert int(dcs["total_gpus"].sum()) == 10412
    assert set(dcs["memory_provenance"]) == {"MODELED_DC_MEMORY"}
    assert abs(dcs["total_mem"].sum() - 632636 * 330000 / 385000) < 1e-8
    assert capacity.model_by_dc(first)["absolute_share_deviation"].max() < 0.002


def test_native_capacity_contract_freezes_workload_and_sla() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/scenarios/spotgpu2026_native_capacity_v1.yaml").read_text("utf-8")
    )
    assert config["workload"]["expected_tasks"] == 466867
    assert not any(
        config["workload"][field]
        for field in (
            "allow_task_removal",
            "allow_sampling",
            "allow_submit_time_change",
            "allow_duration_change",
            "allow_request_change",
        )
    )
    assert config["sla"]["max_wait_steps"] == {"HP": 1, "Spot": 8}
    assert config["gpu_heterogeneity"]["mode"] == "METADATA_ONLY"
    assert config["memory"]["provenance"] == "MODELED_DC_MEMORY"


def test_alibaba2020_v3_is_complete_and_exactly_aligned() -> None:
    manifest = json.loads(
        (OUTPUT / "00_alibaba2020_v3_manifest.json").read_text("utf-8")
    )
    assert manifest["status"] == "ALIBABA2020_V3_READY"
    assert manifest["counts"] == {
        "episodes": 80,
        "states": 7680,
        "task_decisions": 259920,
        "teacher_solver_failures": 0,
        "h1_solver_failures": 0,
    }
    assert manifest["transformer_used"] is False
    assert manifest["rl_used"] is False
    assert manifest["policy_training_used"] is False
    rates = {
        row["metric"]: row["agreement_rate"]
        for row in manifest["alignment"]["summary"]
    }
    assert rates["student_observation_agreement"] == 1.0
    assert rates["feasible_mask_agreement"] == 1.0
    assert rates["h1_action_agreement"] == 1.0
    assert rates["h4_oracle_action_agreement"] == 1.0
    assert manifest["alignment"]["task_mismatch_count"] == 0
    assert manifest["alignment"]["state_mismatch_count"] == 0


def test_alibaba2020_v3_information_regions_and_counts() -> None:
    assert pq.ParquetFile(
        OUTPUT / "dataset/scenario_a/expert_task_actions_full.parquet"
    ).metadata.num_rows == 259920
    assert pq.ParquetFile(
        OUTPUT / "dataset/scenario_a/current_states_full.parquet"
    ).metadata.num_rows == 7680
    assert pq.ParquetFile(
        OUTPUT / "simulator_only/scenario_a/task_truth_full.parquet"
    ).metadata.num_rows == 259920
    assert pq.ParquetFile(
        OUTPUT / "privileged_future/scenario_a/oracle_future_full.parquet"
    ).metadata.num_rows == 7680 * 4 * 5
    task_columns = set(
        pq.ParquetFile(
            OUTPUT / "dataset/scenario_a/expert_task_actions_full.parquet"
        ).schema.names
    )
    assert "true_duration_minutes" not in task_columns
    assert "global_gpu_demand" not in task_columns


def test_native_capacity_pressure_and_decision() -> None:
    manifest = json.loads(
        (OUTPUT / "43_spot_capacity_calibration_manifest.json").read_text("utf-8")
    )
    assert manifest["final_status"] == "SPOT NATIVE CAPACITY READY"
    assert manifest["workload_task_count"] == 466867
    assert manifest["workload_modified"] is False
    assert manifest["native_capacity"]["cpu"] == 632636
    assert manifest["native_capacity"]["gpu"] == 10412
    assert manifest["gpu_capacity_multiplier"] == 10412 / 2900
    assert manifest["native_too_loose"] is False
    assert manifest["comparable_capacity_required"] is False
    pressure = {row["resource"]: row for row in manifest["native_pressure"]}
    assert pressure["gpu"]["offered_load_ratio"] > 0.60
    assert pressure["gpu"]["mandatory_peak_ratio"] < 1.0
    assert pressure["cpu"]["mandatory_peak_ratio"] < 1.0
    assert pressure["memory"]["mandatory_peak_ratio"] < 1.0


def test_dynamic_windows_restore_state_and_solve_stably() -> None:
    selected = pd.read_csv(OUTPUT / "38_dynamic_window_selection.csv")
    restored = pd.read_csv(OUTPUT / "40a_dynamic_restore_evidence.csv")
    summary = pd.read_csv(OUTPUT / "40_dynamic_preflight_summary.csv")
    assert set(selected["window_type"]) == {
        "LOW_LOAD",
        "MEDIUM_LOAD",
        "HIGH_LOAD",
        "LONG_TASK_DENSE",
    }
    assert selected["start_step"].ge(96).all()
    assert restored["empty_state_start"].eq(False).all()
    assert restored["restored_pre_window_running_task_count"].gt(0).all()
    assert summary["steps_checked"].eq(16).all()
    assert summary["h1_optimal_steps"].eq(16).all()
    assert summary["h4_optimal_steps"].eq(16).all()
    assert summary["h1_failure_or_timeout"].sum() == 0
    assert summary["h4_failure_or_timeout"].sum() == 0
    assert summary["max_queue_after_dispatch"].max() == 0
    assert summary["end_queue"].max() == 0
    assert summary["hard_sla_violations"].max() == 0


def test_current_manifest_has_one_authoritative_status() -> None:
    current = json.loads((OUTPUT / "00_generation_manifest.json").read_text("utf-8"))
    assert current["final_status"] == (
        "MPC EXPERT DATASET v3 READY WITH DOCUMENTED LIMITATIONS"
    )
    assert current["timeline_complete"] is True
    assert current["task_decisions"] == 466867
    assert current["h1_failures"] == 0
    assert current["h4_failures"] == 0
    assert current["policy_training_used"] is False
    assert current["transformer_used"] is False
    assert current["rl_used"] is False
