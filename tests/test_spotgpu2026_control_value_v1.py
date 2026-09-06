from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import evaluate_spotgpu2026_control_value_v1 as control
from scripts.imitation import build_spotgpu2026_expert_dataset_v3 as spot_v3


@pytest.fixture(scope="module")
def inputs():
    return spot_v3.load_inputs()


def test_frozen_contract_values(inputs):
    config = control.load_eval_config()
    canonical = inputs["canonical"]
    dcs = inputs["dcs"].sort_values("dc_id")
    assert len(canonical) == 466_867
    assert int(canonical["arrival_step"].max()) + 1 == 17_670
    assert int(dcs["total_cores"].sum()) == 632_636
    assert int(dcs["total_gpus"].sum()) == 10_412
    assert dcs["total_gpus"].astype(int).tolist() == [1875, 2291, 2070, 2615, 1561]
    assert np.isclose(float(dcs["total_mem"].sum()), 542259.4285714285)
    assert config["controllers"]["h4_oracle"]["future_offsets_minutes"] == [15, 30, 45, 60]


def test_preflight_checkpoint_resume_and_information_boundary():
    result = json.loads(
        (control.OUTPUT / "preflight/preflight_result.json").read_text("utf-8")
    )
    assert result["status"] == "PASS"
    assert all(result["shared_checks"].values())
    rows = {row["controller"]: row for row in result["controllers"]}
    assert rows["H1"]["runtime_signature_equal"]
    assert rows["H1"]["semantic_output_hashes_equal"]
    assert rows["H1"]["oracle_future_state_count"] == 0
    assert rows["H1"]["oracle_future_row_count"] == 0
    assert rows["H4_ORACLE"]["oracle_future_state_count"] == 100
    assert rows["H4_ORACLE"]["oracle_future_row_count"] == 100 * 4 * 5


def test_same_exogenous_inputs_and_continuous_timeline():
    h1 = pd.read_parquet(control.OUTPUT / "raw/h1/steps.parquet")
    h4 = pd.read_parquet(control.OUTPUT / "raw/h4_oracle/steps.parquet")
    left = h1.loc[h1["phase"] == "primary"].sort_values("step")
    right = h4.loc[h4["phase"] == "primary"].sort_values("step")
    assert left["step"].tolist() == list(range(17_670))
    assert right["step"].tolist() == list(range(17_670))
    columns = [
        "arrival_task_count",
        "arrival_cpu",
        "arrival_gpu",
        "arrival_memory",
        "timestamp_utc",
        "split",
    ]
    pd.testing.assert_frame_equal(
        left[columns].reset_index(drop=True),
        right[columns].reset_index(drop=True),
    )


def test_reward_and_stage_components_are_exact():
    components = [
        "stage_electricity",
        "stage_carbon",
        "stage_transmission",
        "stage_waiting_defer",
        "stage_sla_risk",
        "stage_terminal_backlog",
    ]
    for controller in ["h1", "h4_oracle"]:
        steps = pd.read_parquet(control.OUTPUT / f"raw/{controller}/steps.parquet")
        np.testing.assert_allclose(
            steps["stage_cost"].to_numpy(),
            steps[components].sum(axis=1).to_numpy(),
            rtol=1e-11,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            steps["reward"].to_numpy(),
            -steps["stage_cost"].to_numpy(),
            rtol=0,
            atol=0,
        )


def test_tail_drain_uses_common_rule_and_reaches_empty():
    tail = pd.read_csv(control.OUTPUT / "23_tail_drain_analysis.csv")
    assert set(tail["maximum_drain_steps"]) == {17_670}
    assert tail["drained_to_empty"].all()
    assert (tail[["tail_pending", "tail_running", "tail_in_transit"]] == 0).all().all()


def test_solver_retry_provenance_and_no_fallback():
    quality = pd.read_csv(control.OUTPUT / "24_solver_quality.csv")
    summary = quality.loc[quality["record_type"] == "SUMMARY"]
    assert (summary["failure"] == 0).all()
    assert (summary["timeout"] == 0).all()
    assert (summary["fallback"] == 0).all()
    steps = pd.concat(
        [
            pd.read_parquet(control.OUTPUT / "raw/h1/steps.parquet"),
            pd.read_parquet(control.OUTPUT / "raw/h4_oracle/steps.parquet"),
        ],
        ignore_index=True,
    )
    assert not steps["fallback_used"].any()
    assert set(steps.loc[steps["presolve_retry"], "solver_status"]) <= {"optimal"}


def test_lifecycle_metrics_and_frozen_wait_bounds():
    observed_violations = {}
    for controller in ["h1", "h4_oracle"]:
        life = pd.read_parquet(control.OUTPUT / f"raw/{controller}/lifecycle.parquet")
        assert len(life) == 466_867
        assert life["actual_execution_start_step"].notna().all()
        assert (life["waiting_steps"] >= 0).all()
        observed_violations[controller] = int(life["wait_bound_violation"].sum())
    assert observed_violations["h1"] == 0
    assert observed_violations["h4_oracle"] > 0


def test_fixed_state_gap_is_offline_and_tie_scale_is_frozen():
    summary = pd.read_csv(control.OUTPUT / "21_fixed_state_action_cost_gap.csv")
    all_row = summary.loc[
        (summary["stratum"] == "ALL") & (summary["group"] == "ALL")
    ].iloc[0]
    assert int(all_row["disagreeing_tasks"]) > 0
    assert all_row["tie_break_scale_threshold"] == pytest.approx(4e-9)
    assert 0 <= all_row["tie_break_scale_equivalent_fraction"] <= 1


def test_required_outputs_exist():
    names = [
        "01_protocol.md",
        "02_frozen_contract.md",
        "03_h1_closed_loop_summary.csv",
        "04_h4_closed_loop_summary.csv",
        "05_core_metric_comparison.csv",
        "06_daily_block_comparison.csv",
        "07_weekly_block_comparison.csv",
        "08_sla_analysis.csv",
        "09_waiting_analysis.csv",
        "10_completion_analysis.csv",
        "11_electricity_analysis.csv",
        "12_carbon_analysis.csv",
        "13_transmission_analysis.csv",
        "14_migration_analysis.csv",
        "15_backlog_analysis.csv",
        "16_priority_analysis.csv",
        "17_duration_group_analysis.csv",
        "18_duration_error_analysis.csv",
        "19_external_pressure_analysis.csv",
        "20_shadow_disagreement_vs_gain.csv",
        "21_fixed_state_action_cost_gap.csv",
        "22_near_equivalent_action_diagnostic.md",
        "23_tail_drain_analysis.csv",
        "24_solver_quality.csv",
        "25_information_leakage_audit.md",
        "26_tests.md",
        "27_final_diagnosis.md",
        "28_summary.md",
    ]
    assert all((control.OUTPUT / name).is_file() for name in names)
    assert len(list((control.OUTPUT / "plots").glob("*.png"))) >= 6


def test_no_learning_stack_or_tuning_is_imported():
    source = Path(control.__file__).read_text("utf-8")
    import_lines = [
        line.lower()
        for line in source.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]
    forbidden = ["transformer", "behavior_cloning", "bc_v2", "reinforcement", "stable_baselines"]
    assert not any(token in line for token in forbidden for line in import_lines)
    frozen = control.load_eval_config()["exclusions"]
    assert all(frozen.values())
