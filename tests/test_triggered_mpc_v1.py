from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting.transformer_workload_provider import (
    TransformerWorkloadForecastProvider,
)
from forecasting.workload_forecast_provider import ForecastTraceSource, build_bundle
from scripts.audit import mpc_control_authority_diagnosis_v1 as diagnosis
from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_mpc.forecast_pressure_adapter import (
    apply_forecast_pressure,
    distribute_global_forecast,
)
from sustaincluster_mpc.future_signals import FutureSignalProvider
from sustaincluster_mpc.horizon_adapter import (
    HorizonStateAdapter,
    RunningTaskHorizonSnapshot,
)
from sustaincluster_mpc.timeline_contract import repaired_h4_capacity_timeline
from sustaincluster_mpc.triggered_mpc import (
    CALIBRATION_SEEDS,
    EVALUATION_SEEDS,
    PRIMARY_TRIGGER_QUANTILE,
    TRIGGER_TRACE_COLUMNS,
    TriggeredTransformerMPC,
    calibrate_trigger_thresholds,
    compute_deployable_risk,
    safe_benefit_retention,
    select_planning_state,
    validate_trigger_trace_schema,
)


WORKSPACE = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    WORKSPACE
    / "artifacts/transformer_forecast_v1/checkpoints/transformer_seed_33_best.pt"
)
DATASET = WORKSPACE / "artifacts/forecast_dataset_v1"
ONE_DC = ({"dc_id": 1, "population_weight": 1.0, "timezone_shift": 0},)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _state(*, running: tuple[RunningTaskHorizonSnapshot, ...] = ()):
    base = diagnosis.make_state(
        (diagnosis.make_task(),),
        (diagnosis.make_horizon_dc(1, 5, total=100.0),),
    )
    return replace(base, running_tasks=running)


def _bundle(values: np.ndarray, provider: str = "transformer"):
    return build_bundle(
        provider=provider,  # type: ignore[arg-type]
        current_timestamp=pd.Timestamp("2023-02-13T12:00:00Z"),
        values=np.asarray(values, dtype=np.float64),
        history_available=True,
        fallback_used=False,
    )


def test_risk_score_formula_adds_existing_and_arrival_pressure() -> None:
    running = (
        RunningTaskHorizonSnapshot("running", 1, 3, 10.0, 20.0, 30.0),
    )
    values = np.zeros((4, 4))
    values[0] = [1.0, 20.0, 30.0, 40.0]
    risk = compute_deployable_risk(_state(running=running), _bundle(values), ONE_DC)
    assert risk.risk_score == pytest.approx((30.0 + 40.0) / 100.0)


def test_estimated_running_occupancy_respects_release_step() -> None:
    running = (
        RunningTaskHorizonSnapshot("running", 1, 3, 0.0, 40.0, 0.0),
    )
    risk = compute_deployable_risk(
        _state(running=running), _bundle(np.zeros((4, 4))), ONE_DC
    )
    gpu = [item for item in risk.components if item.resource == "gpu"]
    assert [item.predicted_existing_occupancy for item in gpu] == [40.0, 40.0, 0.0, 0.0]


def test_future_transformer_forecast_changes_risk() -> None:
    zero = compute_deployable_risk(_state(), _bundle(np.zeros((4, 4))), ONE_DC)
    values = np.zeros((4, 4))
    values[3, 2] = 80.0
    high = compute_deployable_risk(_state(), _bundle(values), ONE_DC)
    assert zero.risk_score == 0.0
    assert high.risk_score == pytest.approx(0.8)


def test_global_to_dc_distribution_conserves_forecast() -> None:
    values = np.repeat([[1.0, 20.0, 30.0, 40.0]], 4, axis=0)
    configs = (
        {"dc_id": 1, "population_weight": 0.4, "timezone_shift": 0},
        {"dc_id": 2, "population_weight": 0.6, "timezone_shift": 0},
    )
    rows = distribute_global_forecast(_bundle(values), configs)
    for step in range(1, 5):
        selected = [row for row in rows if row.horizon_step == step]
        assert sum(row.gpu_demand for row in selected) == pytest.approx(30.0)


def test_risk_takes_max_over_dc_horizon_and_resource() -> None:
    values = np.zeros((4, 4))
    values[2, 3] = 90.0
    risk = compute_deployable_risk(_state(), _bundle(values), ONE_DC)
    assert (risk.risk_dc, risk.risk_horizon, risk.risk_resource) == (1, 45, "memory")
    assert risk.risk_score == pytest.approx(0.9)


def test_calibration_uses_only_frozen_calibration_seeds() -> None:
    thresholds = calibrate_trigger_thresholds([0.1, 0.2, 0.3])
    assert thresholds.calibration_seeds == CALIBRATION_SEEDS


def test_evaluation_seeds_are_excluded_from_calibration() -> None:
    assert set(CALIBRATION_SEEDS).isdisjoint(EVALUATION_SEEDS)
    with pytest.raises(ValueError, match="disjoint"):
        calibrate_trigger_thresholds(
            [0.1], calibration_seeds=CALIBRATION_SEEDS, evaluation_seeds=CALIBRATION_SEEDS
        )


def test_threshold_quantiles_are_exact() -> None:
    values = np.arange(101, dtype=np.float64) / 100.0
    thresholds = calibrate_trigger_thresholds(values)
    assert thresholds.p90 == pytest.approx(np.percentile(values, 90))
    assert thresholds.p95_primary == pytest.approx(np.percentile(values, 95))
    assert thresholds.p99 == pytest.approx(np.percentile(values, 99))


def test_primary_threshold_is_predeclared_p95() -> None:
    assert PRIMARY_TRIGGER_QUANTILE == 0.95
    assert calibrate_trigger_thresholds([0.0, 1.0]).selection_rule == (
        "risk quantile only; no reward tuning"
    )


def test_trigger_is_deterministic() -> None:
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    first = compute_deployable_risk(_state(), _bundle(values), ONE_DC)
    second = compute_deployable_risk(_state(), _bundle(values), ONE_DC)
    assert first == second


def test_trigger_rejects_oracle_future_bundle() -> None:
    with pytest.raises(ValueError, match="oracle"):
        compute_deployable_risk(
            _state(), _bundle(np.zeros((4, 4)), provider="oracle"), ONE_DC
        )


def test_triggered_h1_path_returns_exact_h1_state() -> None:
    h1 = diagnosis.make_state(
        (diagnosis.make_task(),), (diagnosis.make_horizon_dc(1, 1),)
    )
    h4 = _state()
    selection = select_planning_state(h1, h4, 0.1, 0.2)
    assert selection.triggered is False
    assert selection.planning_state is h1
    assert diagnosis.solve(selection.planning_state).environment_actions == (
        diagnosis.solve(h1).environment_actions
    )


def test_triggered_h4_path_returns_exact_h4_state() -> None:
    h1 = diagnosis.make_state(
        (diagnosis.make_task(),), (diagnosis.make_horizon_dc(1, 1),)
    )
    h4 = _state()
    selection = select_planning_state(h1, h4, 0.2, 0.2)
    assert selection.triggered is True
    assert selection.planning_state is h4
    assert diagnosis.solve(selection.planning_state).environment_actions == (
        diagnosis.solve(h4).environment_actions
    )


def test_triggered_transformer_mpc_wraps_exact_paths() -> None:
    h1 = diagnosis.make_state(
        (diagnosis.make_task(),), (diagnosis.make_horizon_dc(1, 1),)
    )
    h4 = _state()
    bundle = _bundle(np.zeros((4, 4)))
    wrapper = TriggeredTransformerMPC(0.0, ONE_DC)
    risk, selection = wrapper.evaluate(h1, h4, h4, bundle)
    assert risk.risk_score == 0.0
    assert selection.triggered is True
    assert selection.planning_state is h4


def test_p95_controller_real_environment_smoke() -> None:
    env = build_sustaincluster_env(
        None,
        pd.Timestamp("2023-02-13T00:00:00Z"),
        1,
        information_mode="deployable",
        initial_seed=1201,
    )
    try:
        env.reset(seed=1201)
        trace = ForecastTraceSource(DATASET)
        transformer = TransformerWorkloadForecastProvider(
            CHECKPOINT, dataset_root=DATASET, device="cpu"
        )
        bundle = transformer.forecast(
            trace.request(env.current_time, include_oracle_future=False)
        )
        adapter = HorizonStateAdapter(
            information_mode="deployable",
            future_signal_provider=FutureSignalProvider("persistence"),
        )
        h1 = adapter.build_horizon_state(env, 1, "no_future_arrivals")
        base_h4 = adapter.build_horizon_state(env, 5, "no_future_arrivals")
        h4 = apply_forecast_pressure(base_h4, bundle, env.cluster_manager.get_config_list()).state
        risk = compute_deployable_risk(
            base_h4, bundle, env.cluster_manager.get_config_list()
        )
        wrapper = TriggeredTransformerMPC(
            risk.risk_score, env.cluster_manager.get_config_list()
        )
        wrapped_risk, selection = wrapper.evaluate(h1, h4, base_h4, bundle)
        assert wrapped_risk == risk
        assert diagnosis.solve(selection.planning_state).feasible is True
    finally:
        env.close()


def test_trigger_trace_schema_accepts_required_fields() -> None:
    validate_trigger_trace_schema(TRIGGER_TRACE_COLUMNS)


def test_trigger_trace_schema_rejects_missing_field() -> None:
    with pytest.raises(ValueError, match="missing"):
        validate_trigger_trace_schema(TRIGGER_TRACE_COLUMNS[:-1])


def test_shadow_h1_h4_solve_does_not_mutate_state() -> None:
    state = _state()
    before = state
    diagnosis.solve(state)
    diagnosis.solve(state)
    assert state == before


def test_benefit_retention_handles_invalid_direction_safely() -> None:
    assert safe_benefit_retention(10.0, 10.0, 10.0, direction="cost") is None
    assert safe_benefit_retention(10.0, 9.0, 9.5, direction="cost") == pytest.approx(0.5)
    assert safe_benefit_retention(10.0, 12.0, 11.0, direction="reward") == pytest.approx(0.5)


def test_transformer_checkpoint_hash_is_frozen() -> None:
    assert _sha256(CHECKPOINT) == (
        "9CD85A0D229E135106EA1E49CD8FEC8936D7B6349DA41531564E565F063569E2"
    )


def test_forecast_dataset_hashes_are_frozen() -> None:
    manifest = json.loads((DATASET / "13_dataset_manifest.json").read_text("utf-8"))
    for name, metadata in manifest["dataset_files"].items():
        assert _sha256(DATASET / "dataset" / name) == metadata["sha256"]


def test_repaired_timeline_still_contains_plus_60_node() -> None:
    nodes = repaired_h4_capacity_timeline(pd.Timestamp("2020-01-01T12:00:00Z"))
    assert nodes[4].absolute_offset_minutes == 60.0
    assert nodes[4].timestamp == pd.Timestamp("2020-01-01T13:00:00Z")


def test_arrival_flow_is_not_accumulated_across_future_nodes() -> None:
    values = np.zeros((4, 4))
    values[0, 2] = 50.0
    risk = compute_deployable_risk(_state(), _bundle(values), ONE_DC)
    gpu = [item for item in risk.components if item.resource == "gpu"]
    assert [item.predicted_future_arrival_demand for item in gpu] == [50.0, 0.0, 0.0, 0.0]
