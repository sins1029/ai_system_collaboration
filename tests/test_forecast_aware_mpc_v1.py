from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting.transformer_workload_provider import (
    TransformerWorkloadForecastProvider,
)
from forecasting.workload_forecast_provider import (
    ForecastRequest,
    ForecastTraceSource,
    OracleWorkloadForecastProvider,
    PersistenceWorkloadForecastProvider,
    build_bundle,
    clip_nonnegative,
)
from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_mpc.action_adapter import (
    ActionMapping,
    SustainClusterActionAdapter,
)
from sustaincluster_mpc.forecast_pressure_adapter import (
    apply_forecast_pressure,
    distribute_global_forecast,
    expected_origin_probabilities,
)
from sustaincluster_mpc.horizon_adapter import (
    HorizonDataCenterSnapshot,
    HorizonState,
)
from sustaincluster_mpc.rolling_horizon_optimizer import (
    RollingHorizonConfig,
    RollingHorizonOptimizer,
)
from sustaincluster_mpc.state_adapter import (
    DataCenterSnapshot,
    ExogenousSignalsSnapshot,
    NetworkLinkSnapshot,
    SchedulerState,
    TaskDestinationSnapshot,
    TaskSnapshot,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "artifacts/forecast_dataset_v1"
CHECKPOINT = (
    ROOT
    / "artifacts/transformer_forecast_v1/checkpoints/transformer_seed_33_best.pt"
)
DC_CONFIGS = [
    {"dc_id": 1, "population_weight": 0.4, "timezone_shift": -7},
    {"dc_id": 2, "population_weight": 0.6, "timezone_shift": 1},
]


def _request(*, oracle: float = 0.0) -> ForecastRequest:
    history = np.zeros((96, 8), dtype=np.float32)
    history[-1, :4] = [10.0, 20.0, 30.0, 40.0]
    return ForecastRequest(
        pd.Timestamp("1970-03-10T00:00:00Z"),
        history,
        np.asarray([10.0, 20.0, 30.0, 40.0]),
        np.full((4, 4), oracle, dtype=np.float64),
    )


def _task() -> TaskSnapshot:
    return TaskSnapshot(
        "task",
        0,
        1,
        1.0,
        1.0,
        1.0,
        60.0,
        60.0,
        "2023-02-13T00:00:00+00:00",
        "2023-02-13T03:00:00+00:00",
        180.0,
        1.0,
        0,
        False,
    )


def _current_dc(dc_id: int) -> DataCenterSnapshot:
    return DataCenterSnapshot(
        dc_id,
        f"DC{dc_id}",
        f"location-{dc_id}",
        100.0,
        100.0,
        0.0,
        100.0,
        1.0,
        100.0,
        100.0,
        0.0,
        100.0,
        1.0,
        100.0,
        100.0,
        0.0,
        100.0,
        1.0,
        0,
        0,
        0,
        (),
        10.0,
        10.0,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _horizon_dc(dc_id: int) -> HorizonDataCenterSnapshot:
    full = (100.0,) * 5
    zero = (0.0,) * 5
    return HorizonDataCenterSnapshot(
        dc_id,
        f"DC{dc_id}",
        f"location-{dc_id}",
        100.0,
        100.0,
        100.0,
        full,
        full,
        full,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        (10.0,) * 5,
        (10.0,) * 5,
    )


def _state() -> HorizonState:
    task = _task()
    current_dcs = (_current_dc(1), _current_dc(2))
    current = SchedulerState(
        (task,),
        current_dcs,
        tuple(
            NetworkLinkSnapshot(origin.dc_id, destination.dc_id, 0.0)
            for origin in current_dcs
            for destination in current_dcs
        ),
        tuple(
            TaskDestinationSnapshot(task.task_id, 0, dc.dc_id, 0.0, 0.0)
            for dc in current_dcs
        ),
        ExogenousSignalsSnapshot("2023-02-13T00:00:00+00:00", 15.0),
        True,
        "deployable",
    )
    return HorizonState(
        current,
        5,
        "no_future_arrivals",
        15.0,
        (_horizon_dc(1), _horizon_dc(2)),
        (),
        (),
        (),
        "deployable",
        "persistence",
    )


def _solve(bundle) -> object:
    applied = apply_forecast_pressure(_state(), bundle, DC_CONFIGS)
    adapter = SustainClusterActionAdapter(
        ActionMapping(((1, 1), (2, 2)), 0, 3)
    )
    return RollingHorizonOptimizer().solve(
        applied.state,
        RollingHorizonConfig(allow_defer=True),
        adapter,
    )


def test_transformer_checkpoint_loads_once() -> None:
    provider = TransformerWorkloadForecastProvider(
        CHECKPOINT, dataset_root=DATASET, device="cpu"
    )
    provider.forecast(_request())
    provider.forecast(_request())

    assert provider.checkpoint_load_count == 1
    assert provider.call_count == 2


def test_transformer_provider_output_shape() -> None:
    source = ForecastTraceSource(DATASET)
    request = source.request(
        pd.Timestamp("2023-02-13T00:00:00Z"), include_oracle_future=False
    )
    provider = TransformerWorkloadForecastProvider(
        CHECKPOINT, dataset_root=DATASET, device="cpu"
    )

    assert provider.forecast(request).as_array().shape == (4, 4)


def test_transformer_provider_uses_real_scale_inverse_transform() -> None:
    source = ForecastTraceSource(DATASET)
    request = source.request("2023-02-13T00:00:00Z", include_oracle_future=False)
    provider = TransformerWorkloadForecastProvider(
        CHECKPOINT, dataset_root=DATASET, device="cpu"
    )
    direct = np.maximum(provider._service.forecast(request.workload_history), 0.0)

    np.testing.assert_allclose(provider.forecast(request).as_array(), direct)


def test_negative_predictions_are_clipped_and_counted() -> None:
    values = np.asarray(
        [[-1.0, 2.0, -3.0, 4.0], [-1.0, -2.0, 3.0, 4.0]] * 2
    )

    clipped, counts = clip_nonnegative(values)

    assert clipped.min() == 0.0
    assert counts == (4, 2, 2, 0)


def test_persistence_repeats_current_workload() -> None:
    bundle = PersistenceWorkloadForecastProvider().forecast(_request(oracle=999.0))

    np.testing.assert_array_equal(
        bundle.as_array(),
        np.repeat([[10.0, 20.0, 30.0, 40.0]], 4, axis=0),
    )


def test_oracle_forecast_aligns_to_t_plus_one_through_four() -> None:
    source = ForecastTraceSource(DATASET)
    request = source.request("2023-02-13T00:00:00Z", include_oracle_future=True)
    with pytest.warns(RuntimeWarning, match="non-deployable"):
        bundle = OracleWorkloadForecastProvider().forecast(request)

    assert [point.horizon_step for point in bundle.points] == [1, 2, 3, 4]
    np.testing.assert_allclose(bundle.as_array(), request.oracle_future_workload)


def test_timestamp_horizon_alignment_is_exact() -> None:
    bundle = PersistenceWorkloadForecastProvider().forecast(_request())

    assert [point.horizon_minutes for point in bundle.points] == [15, 30, 45, 60]
    assert [point.forecast_timestamp for point in bundle.points] == [
        "1970-03-10T00:15:00+00:00",
        "1970-03-10T00:30:00+00:00",
        "1970-03-10T00:45:00+00:00",
        "1970-03-10T01:00:00+00:00",
    ]


def test_history_is_exactly_96_past_and_current_steps() -> None:
    source = ForecastTraceSource(DATASET)
    history, aligned = source.history("2023-02-13T00:00:00Z")

    assert aligned == pd.Timestamp("1970-03-10T00:00:00Z")
    assert history is not None and history.shape == (96, 8)


def test_missing_history_uses_only_persistence_fallback() -> None:
    provider = TransformerWorkloadForecastProvider(
        CHECKPOINT, dataset_root=DATASET, device="cpu"
    )
    request = replace(_request(), workload_history=None)

    bundle = provider.forecast(request)

    assert bundle.persistence_fallback_used is True
    assert bundle.forecast_history_available is False
    assert provider.fallback_count == 1


def test_origin_probabilities_sum_to_one() -> None:
    values = expected_origin_probabilities(DC_CONFIGS, "2023-02-13T00:15:00Z")

    assert sum(values.values()) == pytest.approx(1.0)
    assert set(values) == {1, 2}


def test_per_dc_resource_pressure_conserves_global_demand() -> None:
    bundle = PersistenceWorkloadForecastProvider().forecast(_request())
    rows = distribute_global_forecast(bundle, DC_CONFIGS)

    for step in range(1, 5):
        selected = [item for item in rows if item.horizon_step == step]
        assert sum(item.cpu_demand for item in selected) == pytest.approx(20.0)
        assert sum(item.gpu_demand for item in selected) == pytest.approx(30.0)
        assert sum(item.memory_demand for item in selected) == pytest.approx(40.0)


def test_origin_expectation_is_deterministic_without_rng() -> None:
    state = np.random.get_state()
    first = expected_origin_probabilities(DC_CONFIGS, "2023-02-13T00:15:00Z")
    second = expected_origin_probabilities(DC_CONFIGS, "2023-02-13T00:15:00Z")

    assert first == second
    assert np.array_equal(state[1], np.random.get_state()[1])


def test_deployable_providers_ignore_changed_oracle_future() -> None:
    persistence = PersistenceWorkloadForecastProvider()
    transformer = TransformerWorkloadForecastProvider(
        CHECKPOINT, dataset_root=DATASET, device="cpu"
    )
    low = _request(oracle=0.0)
    high = _request(oracle=1e9)

    np.testing.assert_array_equal(
        persistence.forecast(low).as_array(), persistence.forecast(high).as_array()
    )
    np.testing.assert_allclose(
        transformer.forecast(low).as_array(), transformer.forecast(high).as_array()
    )


def test_deployable_runtime_snapshot_uses_estimated_duration() -> None:
    env = build_sustaincluster_env(
        None,
        pd.Timestamp("2023-02-13T00:00:00Z"),
        4,
        information_mode="deployable",
        baseline_estimated_duration_minutes=60.0,
        initial_seed=1201,
    )
    try:
        env.reset(seed=1201)
        raw = env.current_tasks[0]
        assert raw.estimated_duration == 60.0
        assert raw.true_duration != raw.estimated_duration
    finally:
        env.close()


def test_paired_environments_have_identical_initial_arrival_trace() -> None:
    def fingerprint() -> tuple:
        env = build_sustaincluster_env(
            None,
            pd.Timestamp("2023-02-13T00:00:00Z"),
            4,
            information_mode="deployable",
            initial_seed=1201,
        )
        try:
            env.reset(seed=1201)
            return tuple(
                (
                    task.source_job_name,
                    task.origin_dc_id,
                    task.cores_req,
                    task.gpu_req,
                    task.mem_req,
                )
                for task in env.current_tasks
            )
        finally:
            env.close()

    assert fingerprint() == fingerprint()


def test_h4_transformer_pressure_smoke_is_feasible() -> None:
    provider = TransformerWorkloadForecastProvider(
        CHECKPOINT, dataset_root=DATASET, device="cpu"
    )

    result = _solve(provider.forecast(_request()))

    assert result.feasible
    assert len(result.environment_actions) == 1


def test_h4_persistence_pressure_smoke_is_feasible() -> None:
    result = _solve(PersistenceWorkloadForecastProvider().forecast(_request()))

    assert result.feasible
    assert len(result.environment_actions) == 1


def test_oracle_mode_emits_explicit_upper_bound_warning() -> None:
    with pytest.warns(RuntimeWarning, match="non-deployable"):
        OracleWorkloadForecastProvider().forecast(_request(oracle=1.0))


def test_pressure_bridge_keeps_future_task_objects_empty() -> None:
    bundle = PersistenceWorkloadForecastProvider().forecast(_request())

    applied = apply_forecast_pressure(_state(), bundle, DC_CONFIGS)

    assert applied.state.future_arrivals == ()
    assert len(applied.pressures) == 8
