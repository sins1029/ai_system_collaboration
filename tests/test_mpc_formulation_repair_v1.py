from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts.audit import mpc_control_authority_diagnosis_v1 as diagnosis
from scripts.forecast.run_forecast_aware_mpc_v1 import _arrival_signature
from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_imitation.feature_encoder import (
    SemanticActionSpace,
    SustainClusterFeatureEncoder,
)
from sustaincluster_mpc.action_adapter import SustainClusterActionAdapter
from sustaincluster_mpc.forecast_pressure_adapter import apply_forecast_pressure
from sustaincluster_mpc.future_signals import FutureSignalProvider
from sustaincluster_mpc.horizon_adapter import HorizonStateAdapter
from sustaincluster_mpc.timeline_contract import (
    REPAIRED_H4_CAPACITY_NODES,
    forecast_step_to_capacity_index,
    repaired_h4_capacity_timeline,
)
from sustaincluster_mpc.waiting_metrics import waiting_statistics


WORKSPACE = Path(__file__).resolve().parents[1]
CURRENT_TIME = pd.Timestamp("2020-01-01T12:00:00Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


@pytest.fixture(scope="module")
def sentinel_application() -> Any:
    state = diagnosis.make_state(
        (diagnosis.make_task(),),
        (diagnosis.make_horizon_dc(1, REPAIRED_H4_CAPACITY_NODES, total=1000.0),),
    )
    values = np.asarray(
        [
            [1.0, 101.0, 11.0, 201.0],
            [2.0, 102.0, 22.0, 202.0],
            [3.0, 103.0, 33.0, 203.0],
            [4.0, 104.0, 44.0, 204.0],
        ]
    )
    return apply_forecast_pressure(
        state,
        diagnosis.global_bundle(values),
        ({"dc_id": 1, "population_weight": 1.0, "timezone_shift": 0},),
    )


@pytest.fixture(scope="module")
def defer_sequence() -> dict[str, Any]:
    env = build_sustaincluster_env(
        None,
        pd.Timestamp("2023-02-13T00:00:00Z"),
        5,
        allow_defer=True,
        initial_seed=1201,
        information_mode="deployable",
        baseline_estimated_duration_minutes=60.0,
    )
    try:
        env.reset(seed=1201)
        selected = env.current_tasks[0]
        env.current_tasks = [selected]
        selected.sla_deadline = env.current_time + pd.Timedelta(hours=6)
        deadline = selected.sla_deadline
        _, _, _, _, info1 = env.step([0])
        state1 = HorizonStateAdapter(
            information_mode="deployable",
            future_signal_provider=FutureSignalProvider("persistence"),
        ).build_horizon_state(env, REPAIRED_H4_CAPACITY_NODES, "no_future_arrivals")

        actions2 = [
            0
            if task is selected
            else SustainClusterActionAdapter.from_env(env).mapping.dc_id_to_action[
                int(task.origin_dc_id)
            ]
            for task in env.current_tasks
        ]
        _, _, _, _, info2 = env.step(actions2)
        state2 = HorizonStateAdapter(
            information_mode="deployable",
            future_signal_provider=FutureSignalProvider("persistence"),
        ).build_horizon_state(env, REPAIRED_H4_CAPACITY_NODES, "no_future_arrivals")

        selected_snapshot = next(
            task for task in state2.current.tasks if task.task_id == selected.job_name
        )
        action_space = SemanticActionSpace(
            tuple(sorted(dc.dc_id for dc in state2.current.datacenters))
        )
        encoded = SustainClusterFeatureEncoder(
            action_space, REPAIRED_H4_CAPACITY_NODES
        ).encode(state2)
        selected_row = state2.current.tasks.index(selected_snapshot)
        observation_wait = float(encoded.features[selected_row, 7] * 100.0)

        actions3 = [
            SustainClusterActionAdapter.from_env(env).mapping.dc_id_to_action[
                int(task.origin_dc_id)
            ]
            for task in env.current_tasks
        ]
        env.step(actions3)
        accepted = (
            any(task is selected for _, task, _ in env.in_transit_tasks)
            or any(
                task is selected
                for dc in env.cluster_manager.datacenters.values()
                for task in tuple(dc.pending_tasks) + tuple(dc.running_tasks)
            )
        )
        return {
            "selected": selected,
            "state1": state1,
            "state2": state2,
            "info1": info1,
            "info2": info2,
            "deadline": deadline,
            "observation_wait": observation_wait,
            "accepted": accepted,
        }
    finally:
        env.close()


def test_old_h1_is_current_capacity_node_only() -> None:
    state = diagnosis.make_state(
        (diagnosis.make_task(duration=60.0),),
        (diagnosis.make_horizon_dc(1, 1),),
    )
    result = diagnosis.solve(state)
    assert state.horizon == 1
    assert result.plans[0].dispatch_step == 0
    assert result.datacenter_allocations[0].gpu_units == (40.0,)


def test_repaired_h4_has_current_plus_four_future_nodes() -> None:
    nodes = repaired_h4_capacity_timeline(CURRENT_TIME)
    assert len(nodes) == 5
    assert [node.timestamp for node in nodes] == [
        CURRENT_TIME + pd.Timedelta(minutes=offset)
        for offset in (0, 15, 30, 45, 60)
    ]


def test_forecast_plus_15_uses_1215(sentinel_application: Any) -> None:
    dc = sentinel_application.state.datacenters[0]
    assert forecast_step_to_capacity_index(1, 15) == 1
    assert dc.forecast_gpu_reservations[1] == pytest.approx(11.0)
    assert repaired_h4_capacity_timeline(CURRENT_TIME)[1].timestamp.hour == 12
    assert repaired_h4_capacity_timeline(CURRENT_TIME)[1].timestamp.minute == 15


def test_forecast_plus_30_uses_1230(sentinel_application: Any) -> None:
    dc = sentinel_application.state.datacenters[0]
    assert forecast_step_to_capacity_index(2, 30) == 2
    assert dc.forecast_gpu_reservations[2] == pytest.approx(22.0)
    assert repaired_h4_capacity_timeline(CURRENT_TIME)[2].timestamp.minute == 30


def test_forecast_plus_45_uses_1245(sentinel_application: Any) -> None:
    dc = sentinel_application.state.datacenters[0]
    assert forecast_step_to_capacity_index(3, 45) == 3
    assert dc.forecast_gpu_reservations[3] == pytest.approx(33.0)
    assert repaired_h4_capacity_timeline(CURRENT_TIME)[3].timestamp.minute == 45


def test_forecast_plus_60_uses_1300(sentinel_application: Any) -> None:
    dc = sentinel_application.state.datacenters[0]
    assert forecast_step_to_capacity_index(4, 60) == 4
    assert dc.forecast_gpu_reservations[4] == pytest.approx(44.0)
    assert repaired_h4_capacity_timeline(CURRENT_TIME)[4].timestamp == pd.Timestamp(
        "2020-01-01T13:00:00Z"
    )


def test_cpu_gpu_memory_sentinels_share_one_timeline(
    sentinel_application: Any,
) -> None:
    dc = sentinel_application.state.datacenters[0]
    assert dc.forecast_cpu_reservations == (0.0, 101.0, 102.0, 103.0, 104.0)
    assert dc.forecast_gpu_reservations == (0.0, 11.0, 22.0, 33.0, 44.0)
    assert dc.forecast_memory_reservations == (0.0, 201.0, 202.0, 203.0, 204.0)


def test_price_timeline_current_then_four_future_values() -> None:
    manager = SimpleNamespace(
        prices=(100.0, 11.0, 22.0, 33.0, 44.0),
        index=0,
        get_current_price=lambda: 100.0,
    )
    dc = SimpleNamespace(dc_id=1, price_manager=manager)
    values = FutureSignalProvider("oracle").electricity_price(dc, CURRENT_TIME, 5)
    assert list(zip((node.timestamp for node in repaired_h4_capacity_timeline(CURRENT_TIME)), values))[-1] == (
        pd.Timestamp("2020-01-01T13:00:00Z"),
        44.0,
    )
    assert values == (100.0, 11.0, 22.0, 33.0, 44.0)


def test_carbon_timeline_current_then_four_future_values() -> None:
    manager = SimpleNamespace(
        carbon_smooth=(100.0, 11.0, 22.0, 33.0, 44.0),
        time_step=0,
        get_current_ci=lambda norm=False: 100.0,
    )
    dc = SimpleNamespace(dc_id=1, ci_manager=manager)
    values = FutureSignalProvider("oracle").carbon_intensity(dc, CURRENT_TIME, 5)
    assert values == (100.0, 11.0, 22.0, 33.0, 44.0)


def test_current_action_is_dispatch_stage_zero() -> None:
    state = diagnosis.make_state(
        (diagnosis.make_task(),),
        (diagnosis.make_horizon_dc(1, REPAIRED_H4_CAPACITY_NODES),),
    )
    result = diagnosis.solve(state)
    assert result.plans[0].dispatch_step == 0
    assert result.plans[0].execution_start_step == 1
    assert result.environment_actions == (1,)


def test_long_task_occupancy_uses_1215_through_1300() -> None:
    state = diagnosis.make_state(
        (diagnosis.make_task(duration=60.0),),
        (diagnosis.make_horizon_dc(1, REPAIRED_H4_CAPACITY_NODES),),
    )
    allocation = diagnosis.solve(state).datacenter_allocations[0].gpu_units
    occupied = [
        node.timestamp
        for node, value in zip(repaired_h4_capacity_timeline(CURRENT_TIME), allocation)
        if value > 0
    ]
    assert occupied == [
        pd.Timestamp("2020-01-01T12:15:00Z"),
        pd.Timestamp("2020-01-01T12:30:00Z"),
        pd.Timestamp("2020-01-01T12:45:00Z"),
        pd.Timestamp("2020-01-01T13:00:00Z"),
    ]


def test_old_four_node_bridge_is_rejected() -> None:
    state = diagnosis.make_state(
        (diagnosis.make_task(),), (diagnosis.make_horizon_dc(1, 4),)
    )
    with pytest.raises(ValueError, match="current capacity node"):
        apply_forecast_pressure(
            state,
            diagnosis.global_bundle(np.zeros((4, 4))),
            ({"dc_id": 1, "population_weight": 1.0, "timezone_shift": 0},),
        )


def test_defer_increments_wait_once(defer_sequence: dict[str, Any]) -> None:
    selected = defer_sequence["selected"]
    snapshot = next(
        task
        for task in defer_sequence["state1"].current.tasks
        if task.task_id == selected.job_name
    )
    assert snapshot.wait_intervals == 1
    assert defer_sequence["info1"]["scheduler_wait_intervals_added"] == 1


def test_repeated_defer_accumulates_waiting(defer_sequence: dict[str, Any]) -> None:
    selected = defer_sequence["selected"]
    snapshot = next(
        task
        for task in defer_sequence["state2"].current.tasks
        if task.task_id == selected.job_name
    )
    assert snapshot.wait_intervals == 2
    assert selected.scheduler_wait_intervals == 2


def test_scheduler_defer_has_no_double_increment(defer_sequence: dict[str, Any]) -> None:
    assert defer_sequence["info1"]["scheduler_wait_intervals_added"] == 1
    assert defer_sequence["info2"]["scheduler_wait_intervals_added"] == 1
    assert defer_sequence["selected"].wait_intervals == 2


def test_waiting_observation_matches_task_state(defer_sequence: dict[str, Any]) -> None:
    assert defer_sequence["observation_wait"] == pytest.approx(2.0)


def test_waiting_metric_matches_task_state(defer_sequence: dict[str, Any]) -> None:
    stats = waiting_statistics([defer_sequence["selected"].wait_intervals])
    assert stats.average == pytest.approx(2.0)
    assert stats.p50 == pytest.approx(2.0)
    assert stats.p95 == pytest.approx(2.0)
    assert stats.maximum == pytest.approx(2.0)


def test_deadline_is_fixed_through_defer(defer_sequence: dict[str, Any]) -> None:
    assert defer_sequence["selected"].sla_deadline == defer_sequence["deadline"]


def test_task_executes_after_two_defers(defer_sequence: dict[str, Any]) -> None:
    assert defer_sequence["accepted"] is True
    assert defer_sequence["selected"].temporarily_deferred is False


def test_deployable_mode_rejects_oracle_future_signals() -> None:
    env = build_sustaincluster_env(
        None,
        pd.Timestamp("2023-02-13T00:00:00Z"),
        1,
        information_mode="deployable",
    )
    try:
        env.reset(seed=1201)
        adapter = HorizonStateAdapter(
            information_mode="deployable",
            future_signal_provider=FutureSignalProvider("oracle"),
        )
        with pytest.raises(ValueError, match="cannot read oracle"):
            adapter.build_horizon_state(env, 5, "no_future_arrivals")
    finally:
        env.close()


def test_same_seed_produces_same_real_arrival_trace() -> None:
    environments = [
        build_sustaincluster_env(
            None,
            pd.Timestamp("2023-02-13T00:00:00Z"),
            3,
            information_mode="deployable",
            initial_seed=1201,
        )
        for _ in range(2)
    ]
    traces: list[list[str]] = []
    try:
        for env in environments:
            env.reset(seed=1201)
            signatures = []
            for _ in range(3):
                signatures.append(_arrival_signature(env))
                mapping = SustainClusterActionAdapter.from_env(env).mapping
                env.step(
                    [
                        mapping.dc_id_to_action[int(task.origin_dc_id)]
                        for task in env.current_tasks
                    ]
                )
            traces.append(signatures)
    finally:
        for env in environments:
            env.close()
    assert traces[0] == traces[1]


def test_transformer_checkpoint_hash_is_frozen() -> None:
    checkpoint = (
        WORKSPACE
        / "artifacts/transformer_forecast_v1/checkpoints/transformer_seed_33_best.pt"
    )
    assert _sha256(checkpoint) == (
        "9CD85A0D229E135106EA1E49CD8FEC8936D7B6349DA41531564E565F063569E2"
    )


def test_forecast_dataset_hashes_are_frozen() -> None:
    root = WORKSPACE / "artifacts/forecast_dataset_v1"
    manifest = json.loads((root / "13_dataset_manifest.json").read_text("utf-8"))
    assert manifest["forecast_horizon"] == 4
    assert manifest["history_length"] == 96
    for name, metadata in manifest["dataset_files"].items():
        assert _sha256(root / "dataset" / name) == metadata["sha256"]


def test_repaired_h4_real_environment_smoke() -> None:
    env = build_sustaincluster_env(
        None,
        pd.Timestamp("2023-02-13T00:00:00Z"),
        1,
        information_mode="deployable",
        initial_seed=1201,
    )
    try:
        env.reset(seed=1201)
        state = HorizonStateAdapter(
            information_mode="deployable",
            future_signal_provider=FutureSignalProvider("persistence"),
        ).build_horizon_state(env, 5, "no_future_arrivals")
        result = diagnosis.solve(state)
        assert state.horizon == 5
        assert all(len(dc.cpu_available_cores) == 5 for dc in state.datacenters)
        assert result.feasible is True
    finally:
        env.close()
