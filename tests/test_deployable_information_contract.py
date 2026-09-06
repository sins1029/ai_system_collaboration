from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sustaincluster_contract.runtime import RuntimeInformationContract
from sustaincluster_contract.workload import (
    WORKLOAD_FIELD_INDEX,
    WORKLOAD_TASK_FIELDS,
    extract_tasks_from_row,
)
from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_imitation.paths import (
    prepare_sustaincluster_imports,
    resolve_sustaincluster_root,
)
from sustaincluster_mpc.future_signals import FutureSignalProvider
from sustaincluster_mpc.horizon_adapter import HorizonStateAdapter


SUSTAINCLUSTER_REPO = resolve_sustaincluster_root()
ARRIVAL = pd.Timestamp("2023-08-01T05:00:00Z")
SYNTHETIC_TASK_DATA = [
    "test_job",
    100,
    200,
    pd.Timestamp("2020-01-01T00:00:00Z"),
    60.0,
    100.0,
    40.0,
    2.0,
    123.456,
    7.89,
    "Wednesday",
    2,
]


def _extract(contract: RuntimeInformationContract):
    prepare_sustaincluster_imports(SUSTAINCLUSTER_REPO)
    return extract_tasks_from_row(
        {"tasks_matrix": [SYNTHETIC_TASK_DATA]},
        datacenter_configs=[
            {
                "dc_id": 1,
                "population_weight": 1.0,
                "timezone_shift": 0,
            }
        ],
        current_time_utc=ARRIVAL,
        task_scale=5,
        contract=contract,
    )[0]


def test_workload_field_mapping_contract_and_bandwidth_regression() -> None:
    assert len(WORKLOAD_TASK_FIELDS) == 12
    assert WORKLOAD_FIELD_INDEX["duration_min"] == 4
    assert WORKLOAD_FIELD_INDEX["avg_gpu_wrk_mem"] == 8
    assert WORKLOAD_FIELD_INDEX["bandwidth_gb"] == 9

    task = _extract(
        RuntimeInformationContract.for_mode(
            "deployable",
            baseline_estimated_duration_minutes=45.0,
        )
    )

    assert task.source_job_name == "test_job"
    assert task.arrival_time == ARRIVAL
    assert task.duration == pytest.approx(60.0)
    assert task.true_duration == pytest.approx(60.0)
    assert task.estimated_duration == pytest.approx(45.0)
    assert task.cores_req == pytest.approx(5.0)
    assert task.gpu_req == pytest.approx(2.0)
    assert task.mem_req == pytest.approx(10.0)
    assert task.bandwidth_gb == pytest.approx(7.89)
    assert task.bandwidth_gb != pytest.approx(123.456)
    assert task.origin_dc_id == 1
    assert task.sla_deadline == ARRIVAL + pd.Timedelta(minutes=67.5)
    assert task.sla_deadline_basis == "estimated_duration_at_arrival"


def test_schema_matches_workload_generator_source_order() -> None:
    source_path = (
        SUSTAINCLUSTER_REPO
        / "data/workload/alibaba_2020_dataset/extract_dataset_from_dfas.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    generated_orders = [
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "columns_of_interest"
            for target in node.targets
        )
    ]

    assert generated_orders
    assert tuple(generated_orders[-1]) == WORKLOAD_TASK_FIELDS


def test_runtime_information_modes_are_explicit() -> None:
    oracle = _extract(RuntimeInformationContract.for_mode("oracle"))
    assert oracle.true_duration == pytest.approx(60.0)
    assert oracle.estimated_duration == pytest.approx(60.0)
    assert oracle.duration == pytest.approx(60.0)
    assert oracle.sla_deadline == ARRIVAL + pd.Timedelta(minutes=90.0)
    assert oracle.sla_deadline_basis == "true_duration_oracle"

    observed_requests = []

    def estimator(request):
        observed_requests.append(request)
        return 33.0

    external = _extract(
        RuntimeInformationContract.for_mode(
            "deployable",
            duration_estimate_mode="externally_supplied",
            external_duration_estimator=estimator,
        )
    )
    assert external.true_duration == pytest.approx(60.0)
    assert external.estimated_duration == pytest.approx(33.0)
    assert external.estimated_duration_source == "EXTERNAL_RUNTIME_ESTIMATOR"
    assert len(observed_requests) == 1
    request = observed_requests[0]
    assert request.source_job_name == "test_job"
    assert request.arrival_time == ARRIVAL
    assert request.cores_req == pytest.approx(5.0)
    assert request.gpu_req == pytest.approx(2.0)
    assert request.mem_req == pytest.approx(10.0)
    assert request.bandwidth_gb == pytest.approx(7.89)
    assert not hasattr(request, "true_duration")
    assert not hasattr(request, "duration_min")
    assert not hasattr(request, "end_time")

    with pytest.raises(ValueError, match="cannot use true duration"):
        RuntimeInformationContract.for_mode(
            "deployable", duration_estimate_mode="oracle"
        )


class _PriceManager:
    prices = np.asarray([10.0, 20.0, 30.0])
    index = 1

    @staticmethod
    def get_current_price() -> float:
        return 20.0


class _CarbonManager:
    carbon_smooth = np.asarray([100.0, 200.0, 300.0])
    time_step = 1

    @staticmethod
    def get_current_ci(norm: bool = False) -> float:
        assert norm is False
        return 200.0


def _signal_dc():
    return SimpleNamespace(
        dc_id=7,
        price_manager=_PriceManager(),
        ci_manager=_CarbonManager(),
    )


def test_future_signal_provider_oracle_persistence_and_external() -> None:
    dc = _signal_dc()
    oracle = FutureSignalProvider("oracle")
    assert oracle.electricity_price(dc, ARRIVAL, 4) == (
        20.0,
        30.0,
        10.0,
        20.0,
    )
    assert oracle.carbon_intensity(dc, ARRIVAL, 4) == (
        200.0,
        300.0,
        100.0,
        200.0,
    )

    persistence = FutureSignalProvider("persistence")
    assert persistence.electricity_price(dc, ARRIVAL, 3) == (20.0,) * 3
    assert persistence.carbon_intensity(dc, ARRIVAL, 3) == (200.0,) * 3

    calls = []

    def external_source(signal, dc_id, current_time, horizon, current):
        calls.append((signal, dc_id, current_time, horizon, current))
        return [current + offset for offset in range(horizon)]

    external = FutureSignalProvider("external", external_source)
    assert external.electricity_price(dc, ARRIVAL, 2) == (20.0, 21.0)
    assert calls == [("electricity_price", 7, ARRIVAL, 2, 20.0)]


def test_deployable_capacity_release_uses_estimated_duration() -> None:
    task = SimpleNamespace(
        job_name="running",
        start_time=ARRIVAL,
        finish_time=ARRIVAL + pd.Timedelta(minutes=60),
        true_duration=60.0,
        estimated_duration=13.0,
        cores_req=4.0,
        gpu_req=1.0,
        mem_req=8.0,
    )
    dc = SimpleNamespace(dc_id=1, running_tasks=[task])
    env = SimpleNamespace(
        current_time=ARRIVAL,
        cluster_manager=SimpleNamespace(datacenters={"DC1": dc}),
    )
    adapter = HorizonStateAdapter()

    oracle = adapter._snapshot_running_tasks(env, 15.0, "oracle")
    deployable = adapter._snapshot_running_tasks(env, 15.0, "deployable")

    assert oracle[0].release_step == 4
    assert deployable[0].release_step == 1


def test_real_environment_rl_and_mpc_respect_deployable_contract() -> None:
    env = build_sustaincluster_env(
        SUSTAINCLUSTER_REPO,
        ARRIVAL,
        2,
        allow_defer=True,
        initial_seed=501,
        information_mode="deployable",
        baseline_estimated_duration_minutes=13.0,
    )
    try:
        env.reset(seed=501)
        assert env.current_tasks
        task = env.current_tasks[0]
        assert task.duration == task.true_duration
        assert task.estimated_duration == pytest.approx(13.0)
        assert task.true_duration >= 15.0

        observations = env._generate_per_task_obs_list()
        assert len(observations) == len(env.current_tasks)
        assert all(obs.shape == (env.obs_dim_per_task,) for obs in observations)
        assert observations[0][7] == pytest.approx(13.0)

        h1 = HorizonStateAdapter().build_horizon_state(
            env, 1, "no_future_arrivals"
        )
        assert h1.current.tasks[0].duration_minutes == pytest.approx(13.0)
        assert h1.information_mode == "deployable"
        assert h1.future_signal_mode == "persistence"

        h4 = HorizonStateAdapter().build_horizon_state(
            env, 4, "no_future_arrivals"
        )
        assert not h4.future_arrivals
        for dc_state in h4.datacenters:
            assert len(set(dc_state.electricity_price_usd_per_mwh)) == 1
            assert len(set(dc_state.carbon_intensity_gco2_per_kwh)) == 1

        with pytest.raises(ValueError, match="no_future_arrivals"):
            HorizonStateAdapter().build_horizon_state(env, 4, "oracle")
        with pytest.raises(ValueError, match="oracle future"):
            HorizonStateAdapter(
                future_signal_provider=FutureSignalProvider("oracle")
            ).build_horizon_state(env, 4, "no_future_arrivals")
    finally:
        env.close()
