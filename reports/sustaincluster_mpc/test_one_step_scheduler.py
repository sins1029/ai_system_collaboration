from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest
import yaml

from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_imitation.paths import prepare_sustaincluster_imports
from sustaincluster_mpc import (
    ActionMapping,
    AssignmentDecision,
    DataCenterSnapshot,
    ExogenousSignalsSnapshot,
    NetworkLinkSnapshot,
    ObjectiveWeights,
    OneStepOptimizer,
    OptimizationConfig,
    SchedulerState,
    SustainClusterActionAdapter,
    SustainClusterStateAdapter,
    TaskDestinationSnapshot,
    TaskSnapshot,
)


ZERO_WEIGHTS = ObjectiveWeights(0.0, 0.0, 0.0, 0.0, 0.0)


def make_task(
    task_id: str,
    index: int,
    *,
    origin: int = 1,
    cpu: float = 1.0,
    gpu: float = 0.0,
    memory: float = 1.0,
    duration: float = 10.0,
    remaining_sla: float = 120.0,
    bandwidth: float = 1.0,
) -> TaskSnapshot:
    return TaskSnapshot(
        task_id=task_id,
        original_index=index,
        origin_dc_id=origin,
        cpu_cores=cpu,
        gpu_units=gpu,
        memory_gb=memory,
        duration_minutes=duration,
        remaining_duration_minutes=duration,
        arrival_time_utc="2023-08-01T05:00:00+00:00",
        sla_deadline_utc="2023-08-01T07:00:00+00:00",
        remaining_sla_minutes=remaining_sla,
        bandwidth_gb=bandwidth,
        wait_intervals=0,
        was_deferred=False,
    )


def make_dc(
    dc_id: int,
    *,
    cpu: float = 100.0,
    gpu: float = 100.0,
    memory: float = 100.0,
    price: float = 10.0,
    carbon: float = 10.0,
) -> DataCenterSnapshot:
    return DataCenterSnapshot(
        dc_id=dc_id,
        dc_name=f"DC{dc_id}",
        location=f"location-{dc_id}",
        cpu_total_cores=cpu,
        cpu_available_cores=cpu,
        cpu_reserved_cores=0.0,
        cpu_schedulable_cores=cpu,
        cpu_available_ratio=1.0,
        gpu_total_units=max(gpu, 1.0),
        gpu_available_units=gpu,
        gpu_reserved_units=0.0,
        gpu_schedulable_units=gpu,
        gpu_available_ratio=gpu / max(gpu, 1.0),
        memory_total_gb=max(memory, 1.0),
        memory_available_gb=memory,
        memory_reserved_gb=0.0,
        memory_schedulable_gb=memory,
        memory_available_ratio=memory / max(memory, 1.0),
        running_task_count=0,
        queued_task_count=0,
        in_transit_task_count=0,
        resource_release_times_utc=(),
        electricity_price_usd_per_mwh=price,
        carbon_intensity_gco2_per_kwh=carbon,
        total_power_kw=0.0,
        it_power_kw=0.0,
        cooling_power_kw=0.0,
        internal_temperature_c=22.0,
        ambient_temperature_c=20.0,
        crac_setpoint_c=22.0,
    )


def make_state(
    tasks: tuple[TaskSnapshot, ...],
    dcs: tuple[DataCenterSnapshot, ...],
    *,
    allow_defer: bool,
    transfer_costs: dict[tuple[int, int], float] | None = None,
) -> SchedulerState:
    transfer_costs = transfer_costs or {}
    links = tuple(
        NetworkLinkSnapshot(
            origin_dc_id=origin.dc_id,
            destination_dc_id=destination.dc_id,
            transmission_cost_usd_per_gb=transfer_costs.get(
                (origin.dc_id, destination.dc_id), 0.0
            ),
        )
        for origin in dcs
        for destination in dcs
    )
    destinations = tuple(
        TaskDestinationSnapshot(
            task_id=task.task_id,
            original_index=task.original_index,
            destination_dc_id=dc.dc_id,
            transmission_cost_usd=transfer_costs.get(
                (task.origin_dc_id, dc.dc_id), 0.0
            )
            * task.bandwidth_gb,
            transmission_delay_seconds=0.0,
        )
        for task in tasks
        for dc in dcs
    )
    return SchedulerState(
        tasks=tasks,
        datacenters=dcs,
        network_links=links,
        task_destinations=destinations,
        exogenous=ExogenousSignalsSnapshot(
            current_time_utc="2023-08-01T05:00:00+00:00",
            timestep_minutes=15.0,
        ),
        allow_defer=allow_defer,
    )


def make_action_adapter(
    dcs: tuple[DataCenterSnapshot, ...], allow_defer: bool
) -> SustainClusterActionAdapter:
    offset = 1 if allow_defer else 0
    return SustainClusterActionAdapter(
        ActionMapping(
            dc_id_to_action_items=tuple(
                (dc.dc_id, index + offset) for index, dc in enumerate(dcs)
            ),
            defer_action=0 if allow_defer else None,
            action_space_n=len(dcs) + offset,
        )
    )


def solve(
    state: SchedulerState,
    *,
    weights: ObjectiveWeights = ZERO_WEIGHTS,
):
    adapter = make_action_adapter(state.datacenters, state.allow_defer)
    optimizer = OneStepOptimizer(
        OptimizationConfig(allow_defer=state.allow_defer, weights=weights)
    )
    return optimizer.solve(state, adapter)


def test_no_tasks_returns_empty_actions() -> None:
    dcs = (make_dc(1),)
    result = solve(make_state((), dcs, allow_defer=True))
    assert result.status == "optimal"
    assert result.assignments == ()
    assert result.environment_actions == ()


def test_single_task_uses_dynamic_action_encoding() -> None:
    tasks = (make_task("task-1", 0, origin=5),)
    dcs = (make_dc(5), make_dc(2))
    result = solve(make_state(tasks, dcs, allow_defer=False))
    assert result.status == "optimal"
    assert result.assignments[0].dc_id == 5
    assert result.environment_actions == (0,)


def test_multiple_tasks_with_capacity_are_all_assigned() -> None:
    tasks = tuple(make_task(f"task-{i}", i) for i in range(4))
    dcs = (make_dc(1), make_dc(2))
    result = solve(make_state(tasks, dcs, allow_defer=True))
    assert result.status == "optimal"
    assert all(item.decision == "assign" for item in result.assignments)
    assert len(result.environment_actions) == len(tasks)


def test_gpu_shortage_uses_another_datacenter() -> None:
    tasks = (make_task("gpu-task", 0, gpu=2.0),)
    dcs = (make_dc(1, gpu=1.0), make_dc(2, gpu=3.0))
    result = solve(make_state(tasks, dcs, allow_defer=True))
    assert result.assignments[0].dc_id == 2


def test_memory_shortage_never_overallocates() -> None:
    tasks = (
        make_task("memory-a", 0, memory=8.0),
        make_task("memory-b", 1, memory=8.0),
    )
    dcs = (make_dc(1, memory=8.0), make_dc(2, memory=8.0))
    result = solve(make_state(tasks, dcs, allow_defer=False))
    assert result.status == "optimal"
    assert all(allocation.memory_gb <= 8.0 for allocation in result.datacenter_allocations)


def test_all_capacity_shortage_defers_when_allowed() -> None:
    tasks = (make_task("large", 0, cpu=10.0),)
    dcs = (make_dc(1, cpu=1.0), make_dc(2, cpu=1.0))
    result = solve(make_state(tasks, dcs, allow_defer=True))
    assert result.status == "optimal"
    assert result.assignments[0].decision == "defer"
    assert result.environment_actions == (0,)


def test_all_capacity_shortage_is_infeasible_without_defer() -> None:
    tasks = (make_task("large", 0, cpu=10.0),)
    dcs = (make_dc(1, cpu=1.0), make_dc(2, cpu=1.0))
    result = solve(make_state(tasks, dcs, allow_defer=False))
    assert result.status == "infeasible"
    assert result.assignments == ()
    assert result.environment_actions == ()


def test_high_transfer_cost_prefers_origin() -> None:
    tasks = (make_task("data-heavy", 0, origin=1, bandwidth=10.0),)
    dcs = (make_dc(1), make_dc(2))
    weights = ObjectiveWeights(0.0, 0.0, 1.0, 100.0, 0.0)
    state = make_state(
        tasks,
        dcs,
        allow_defer=True,
        transfer_costs={(1, 1): 0.0, (1, 2): 100.0},
    )
    assert solve(state, weights=weights).assignments[0].dc_id == 1


def test_electricity_weight_prefers_lower_price() -> None:
    tasks = (make_task("compute", 0, origin=1, cpu=20.0),)
    dcs = (make_dc(1, price=100.0), make_dc(2, price=1.0))
    weights = ObjectiveWeights(1.0, 0.0, 0.0, 100.0, 0.0)
    assert solve(make_state(tasks, dcs, allow_defer=True), weights=weights).assignments[0].dc_id == 2


def test_carbon_weight_prefers_lower_carbon() -> None:
    tasks = (make_task("compute", 0, origin=1, cpu=20.0),)
    dcs = (make_dc(1, carbon=900.0), make_dc(2, carbon=10.0))
    weights = ObjectiveWeights(0.0, 1.0, 0.0, 100.0, 0.0)
    assert solve(make_state(tasks, dcs, allow_defer=True), weights=weights).assignments[0].dc_id == 2


def test_sla_urgent_task_has_higher_defer_penalty() -> None:
    tasks = (
        make_task("loose", 0, cpu=1.0, duration=1.0, remaining_sla=100.0),
        make_task("urgent", 1, cpu=1.0, duration=1.0, remaining_sla=5.0),
    )
    dcs = (make_dc(1, cpu=1.0),)
    weights = ObjectiveWeights(0.0, 0.0, 0.0, 0.0, 100.0)
    result = solve(make_state(tasks, dcs, allow_defer=True), weights=weights)
    decisions = {item.task_id: item.decision for item in result.assignments}
    assert decisions == {"loose": "defer", "urgent": "assign"}


def test_task_assignment_action_order_stays_aligned() -> None:
    tasks = (make_task("first", 0), make_task("second", 1))
    adapter = SustainClusterActionAdapter(
        ActionMapping(((9, 1), (4, 2)), defer_action=0, action_space_n=3)
    )
    assignments = (
        AssignmentDecision("first", 0, "assign", 4),
        AssignmentDecision("second", 1, "assign", 9),
    )
    assert adapter.encode_assignments(tasks, assignments) == [2, 1]
    with pytest.raises(ValueError, match="未对齐"):
        adapter.encode_assignments(tasks, tuple(reversed(assignments)))


def test_empty_nonempty_switch_has_no_adapter_residue() -> None:
    task = make_task("task", 0)
    adapter = SustainClusterActionAdapter(
        ActionMapping(((1, 1),), defer_action=0, action_space_n=2)
    )
    assignment = AssignmentDecision("task", 0, "assign", 1)
    assert adapter.encode_assignments((task,), (assignment,)) == [1]
    assert adapter.encode_assignments((), ()) == []
    assert adapter.encode_assignments((task,), (assignment,)) == [1]


def test_defer_rejected_when_disabled_with_task_context() -> None:
    task = make_task("task", 0)
    adapter = SustainClusterActionAdapter(
        ActionMapping(((1, 0),), defer_action=None, action_space_n=1)
    )
    with pytest.raises(ValueError, match="task_id='task'.*环境动作映射已禁用延后动作"):
        adapter.encode_assignments(
            (task,), (AssignmentDecision("task", 0, "defer"),)
        )


def test_real_adapter_is_read_only_and_empty_env_step_succeeds() -> None:
    env = build_real_env(prepare_sustaincluster_imports(), allow_defer=True)
    env.reset(seed=123)
    before = env_fingerprint(env)
    state = SustainClusterStateAdapter(env).build_scheduler_state()
    assert env_fingerprint(env) == before
    assert tuple(task.original_index for task in state.tasks) == tuple(
        range(len(state.tasks))
    )
    with pytest.raises(FrozenInstanceError):
        state.tasks[0].cpu_cores = 0.0

    env.current_tasks = []
    empty_state = SustainClusterStateAdapter(env).build_scheduler_state()
    mapping = SustainClusterActionAdapter.from_env(env)
    result = OneStepOptimizer(
        OptimizationConfig(allow_defer=True)
    ).solve(empty_state, mapping)
    _, reward, terminated, truncated, info = env.step(
        list(result.environment_actions)
    )
    assert result.environment_actions == ()
    assert reward == 0.0
    assert info["scheduled_tasks_this_step"] == 0
    assert not terminated
    assert not truncated


def build_real_env(repo: Path, allow_defer: bool):
    return build_sustaincluster_env(
        repo,
        pd.Timestamp("2023-08-01T05:00:00Z"),
        96,
        allow_defer=allow_defer,
        initial_seed=123,
    )

def env_fingerprint(env) -> tuple:
    task_values = tuple(
        (
            task.job_name,
            task.origin_dc_id,
            task.dest_dc_id,
            task.temporarily_deferred,
            task.wait_intervals,
        )
        for task in env.current_tasks
    )
    dc_values = tuple(
        (
            name,
            dc.dc_id,
            dc.available_cores,
            dc.available_gpus,
            dc.available_mem,
            len(dc.running_tasks),
            len(dc.pending_tasks),
        )
        for name, dc in env.cluster_manager.datacenters.items()
    )
    transit_values = tuple(
        (str(arrival), task.job_name, dc_name)
        for arrival, task, dc_name in env.in_transit_tasks
    )
    return task_values, dc_values, transit_values
