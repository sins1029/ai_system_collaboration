from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sustaincluster_imitation.bc_policy import (
    BCPolicy,
    BCPolicyConfig,
    weighted_masked_cross_entropy,
)
from sustaincluster_imitation.dataset_reader import ExpertDatasetReader
from sustaincluster_imitation.dataset_writer import ExpertDatasetWriter
from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_imitation.feature_encoder import (
    SemanticActionSpace,
    SustainClusterFeatureEncoder,
)
from sustaincluster_imitation.forecast_baseline import HistoricalArrivalForecaster
from sustaincluster_imitation.paths import resolve_sustaincluster_root
from sustaincluster_imitation.sac_weight_bridge import initialize_sac_actor_from_bc
from sustaincluster_imitation.split_dataset import build_episode_split
from sustaincluster_mpc import (
    ActionMapping,
    AssignmentDecision,
    DataCenterSnapshot,
    ExogenousSignalsSnapshot,
    HorizonDataCenterSnapshot,
    HorizonState,
    SchedulerState,
    TaskDestinationSnapshot,
    TaskSnapshot,
    SustainClusterActionAdapter,
    HorizonStateAdapter,
)


WORKSPACE = Path(__file__).resolve().parents[1]
SUSTAINCLUSTER_REPO = resolve_sustaincluster_root()


def _task() -> TaskSnapshot:
    return TaskSnapshot(
        task_id="transfer-two-steps",
        original_index=0,
        origin_dc_id=1,
        cpu_cores=8.0,
        gpu_units=2.0,
        memory_gb=16.0,
        duration_minutes=15.0,
        remaining_duration_minutes=15.0,
        arrival_time_utc="2023-08-01T05:00:00+00:00",
        sla_deadline_utc="2023-08-01T07:00:00+00:00",
        remaining_sla_minutes=120.0,
        bandwidth_gb=1.0,
        wait_intervals=0,
        was_deferred=False,
    )


def _current_dc() -> DataCenterSnapshot:
    return DataCenterSnapshot(
        dc_id=1,
        dc_name="DC1",
        location="test",
        cpu_total_cores=100.0,
        cpu_available_cores=0.0,
        cpu_reserved_cores=100.0,
        cpu_schedulable_cores=0.0,
        cpu_available_ratio=0.0,
        gpu_total_units=10.0,
        gpu_available_units=0.0,
        gpu_reserved_units=10.0,
        gpu_schedulable_units=0.0,
        gpu_available_ratio=0.0,
        memory_total_gb=100.0,
        memory_available_gb=0.0,
        memory_reserved_gb=100.0,
        memory_schedulable_gb=0.0,
        memory_available_ratio=0.0,
        running_task_count=1,
        queued_task_count=0,
        in_transit_task_count=0,
        resource_release_times_utc=(),
        electricity_price_usd_per_mwh=50.0,
        carbon_intensity_gco2_per_kwh=400.0,
        total_power_kw=0.0,
        it_power_kw=0.0,
        cooling_power_kw=0.0,
        internal_temperature_c=22.0,
        ambient_temperature_c=20.0,
        crac_setpoint_c=22.0,
    )


def _state() -> HorizonState:
    task = _task()
    current = SchedulerState(
        tasks=(task,),
        datacenters=(_current_dc(),),
        network_links=(),
        task_destinations=(
            TaskDestinationSnapshot(
                task_id=task.task_id,
                original_index=task.original_index,
                destination_dc_id=1,
                transmission_cost_usd=0.0,
                transmission_delay_seconds=1800.0,
            ),
        ),
        exogenous=ExogenousSignalsSnapshot(
            current_time_utc="2023-08-01T05:00:00+00:00",
            timestep_minutes=15.0,
        ),
        allow_defer=True,
    )
    horizon_dc = HorizonDataCenterSnapshot(
        dc_id=1,
        dc_name="DC1",
        location="test",
        cpu_total_cores=100.0,
        gpu_total_units=10.0,
        memory_total_gb=100.0,
        cpu_available_cores=(0.0, 0.0, 100.0, 100.0),
        gpu_available_units=(0.0, 0.0, 10.0, 10.0),
        memory_available_gb=(0.0, 0.0, 100.0, 100.0),
        known_cpu_reservations=(100.0, 100.0, 0.0, 0.0),
        known_gpu_reservations=(10.0, 10.0, 0.0, 0.0),
        known_memory_reservations=(100.0, 100.0, 0.0, 0.0),
        forecast_cpu_reservations=(0.0, 0.0, 0.0, 0.0),
        forecast_gpu_reservations=(0.0, 0.0, 0.0, 0.0),
        forecast_memory_reservations=(0.0, 0.0, 0.0, 0.0),
        electricity_price_usd_per_mwh=(50.0, 50.0, 50.0, 50.0),
        carbon_intensity_gco2_per_kwh=(400.0, 400.0, 400.0, 400.0),
    )
    return HorizonState(
        current=current,
        horizon=4,
        forecast_mode="no_future_arrivals",
        timestep_minutes=15.0,
        datacenters=(horizon_dc,),
        running_tasks=(),
        transit_tasks=(),
        future_arrivals=(),
    )


def test_feasible_mask_uses_destination_transfer_delay() -> None:
    encoder = SustainClusterFeatureEncoder(SemanticActionSpace((1,)), horizon=4)

    encoded = encoder.encode(_state())

    assert encoded.feasible_action_mask.tolist() == [[True, True]]


def test_semantic_dc_label_is_independent_of_environment_action_mapping() -> None:
    task = _task()
    decision = AssignmentDecision(task.task_id, task.original_index, "assign", 1)
    first = SustainClusterActionAdapter(ActionMapping(((1, 1), (2, 2)), 0, 3))
    shuffled = SustainClusterActionAdapter(ActionMapping(((2, 1), (1, 2)), 0, 3))

    assert SemanticActionSpace((1, 2)).encode_decision(decision) == 1
    assert first.encode_assignments((task,), (decision,)) == [1]
    assert shuffled.encode_assignments((task,), (decision,)) == [2]


def test_empty_tasks_produce_empty_features_and_actions() -> None:
    state = _state()
    empty_current = replace(
        state.current, tasks=(), task_destinations=()
    )
    empty_state = replace(state, current=empty_current)
    encoder = SustainClusterFeatureEncoder(SemanticActionSpace((1,)), horizon=4)

    encoded = encoder.encode(empty_state)

    assert encoded.features.shape == (0, encoder.feature_dim)
    assert encoded.feasible_action_mask.shape == (0, 2)
    adapter = SustainClusterActionAdapter(ActionMapping(((1, 1),), 0, 2))
    assert adapter.encode_assignments((), ()) == []


def test_padding_tasks_do_not_enter_bc_loss() -> None:
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    task_mask = torch.tensor([True, False])
    feasible = torch.tensor([[True, True], [False, False]])

    loss = weighted_masked_cross_entropy(
        logits, labels, task_mask, feasible
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.equal(logits.grad[1], torch.zeros(2))


def test_infeasible_expert_action_is_rejected_by_bc_loss() -> None:
    with pytest.raises(ValueError, match="不可行"):
        weighted_masked_cross_entropy(
            torch.zeros((1, 2)),
            torch.tensor([1]),
            torch.tensor([True]),
            torch.tensor([[True, False]]),
        )


def test_baseline_forecast_is_deterministic_and_causal() -> None:
    first = HistoricalArrivalForecaster(horizon=4, history_window=4, seed=7)
    second = HistoricalArrivalForecaster(horizon=4, history_window=4, seed=7)
    observed = (_task(),)
    first.observe(observed)
    second.observe(observed)

    assert first.predict(15.0) == second.predict(15.0)
    assert first.observed_steps == 1


def test_episode_split_is_seed_exclusive_and_oracle_free(tmp_path: Path) -> None:
    episodes = [
        {
            "episode_id": f"deployable-{seed}",
            "seed": seed,
            "burst_intensity": 1.5 if seed >= 9 else 1.0,
        }
        for seed in range(1, 11)
    ]

    manifest = build_episode_split(episodes, tmp_path / "split.json")

    groups = [set(manifest["seeds"][name]) for name in ("train", "validation", "test")]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
    assert manifest["leakage_checks"]["oracle_in_primary_train"] is False
    assert manifest["leakage_checks"]["test_has_unseen_burst_intensity"] is True


def test_dataset_writer_reader_round_trip(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    manifest = ExpertDatasetWriter(dataset).write(
        [{"episode_id": "episode-1", "seed": 1}],
        [{"episode_id": "episode-1", "step_index": 0}],
        [{"episode_id": "episode-1", "semantic_decision": "defer"}],
        {"variant": "test"},
    )

    reader = ExpertDatasetReader(dataset)

    assert manifest["files"]["episodes.parquet"]["rows"] == 1
    assert reader.read_episodes()[0]["episode_id"] == "episode-1"
    assert reader.read_tasks()[0]["semantic_decision"] == "defer"


def test_bc_checkpoint_loads_and_bridges_to_sac_actor(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bc.pt"
    policy = BCPolicy(BCPolicyConfig(233, (1, 2, 3, 4, 5), 256, True))
    policy.save_checkpoint(checkpoint, {"test": True})

    loaded = BCPolicy.load_checkpoint(checkpoint)
    actor, config, result = initialize_sac_actor_from_bc(checkpoint)

    assert loaded.config == policy.config
    assert config == policy.config
    assert result.exact_key_match is True
    assert tuple(actor.state_dict()) == tuple(policy.actor.state_dict())


def test_primary_split_contains_no_oracle_episode() -> None:
    split_path = (
        WORKSPACE
        / "data/processed/sustaincluster_expert/split_manifest.json"
    )
    if not split_path.exists():
        pytest.skip("generated expert split is not available")
    split = json.loads(split_path.read_text(encoding="utf-8"))

    assert not any(
        "oracle_upper_bound" in episode_id
        for episode_id in split["episode_assignments"]
    )


def test_bc_policy_continuously_drives_real_multi_action_environment() -> None:
    checkpoint = (
        WORKSPACE / "artifacts/sustaincluster_imitation/bc_actor_seed_11.pt"
    )
    if not checkpoint.exists():
        pytest.skip("trained BC checkpoint is not available")
    policy = BCPolicy.load_checkpoint(checkpoint)
    encoder = SustainClusterFeatureEncoder(
        SemanticActionSpace((1, 2, 3, 4, 5)), horizon=4
    )
    env = build_sustaincluster_env(
        SUSTAINCLUSTER_REPO,
        pd.Timestamp("2023-08-01T05:00:00Z"),
        3,
        allow_defer=True,
        initial_seed=9191,
    )
    env.reset(seed=9191)
    action_adapter = SustainClusterActionAdapter.from_env(env)
    state_adapter = HorizonStateAdapter()
    forecaster = HistoricalArrivalForecaster(4, 16, 9191)

    for _ in range(3):
        state = state_adapter.build_horizon_state(
            env, 4, "no_future_arrivals"
        )
        forecaster.observe(state.current.tasks)
        forecast = forecaster.predict(state.timestep_minutes)
        state = forecaster.apply(state, forecast)
        encoded = encoder.encode(state, forecast.uncertainties)
        decisions = policy.predict(encoded)
        actions = action_adapter.encode_assignments(
            state.current.tasks, decisions
        )
        assert len(actions) == len(state.current.tasks)
        _, _, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            break


def test_environment_factory_restores_working_directory() -> None:
    before = Path.cwd()
    env = build_sustaincluster_env(
        None,
        pd.Timestamp("2023-08-01T05:00:00Z"),
        1,
        allow_defer=True,
        initial_seed=9191,
    )
    try:
        assert Path.cwd() == before
    finally:
        env.close()
