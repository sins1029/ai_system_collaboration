from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest

from sustaincluster_imitation.paths import resolve_sustaincluster_root
from sustaincluster_mpc import (
    ActionMapping,
    DataCenterSnapshot,
    ExogenousSignalsSnapshot,
    ForecastNoiseConfig,
    FutureArrivalAggregate,
    HorizonDataCenterSnapshot,
    HorizonState,
    HorizonStateAdapter,
    NetworkLinkSnapshot,
    ObjectiveWeights,
    OneStepOptimizer,
    OptimizationConfig,
    RollingHorizonConfig,
    RollingHorizonOptimizer,
    RollingObjectiveWeights,
    SchedulerState,
    SustainClusterActionAdapter,
    TaskDestinationSnapshot,
    TaskSnapshot,
    TransitTaskHorizonSnapshot,
)


REPO = resolve_sustaincluster_root()



def make_task(
    task_id: str,
    index: int,
    *,
    origin: int = 1,
    cpu: float = 1.0,
    gpu: float = 0.0,
    memory: float = 1.0,
    duration: float = 15.0,
    remaining_sla: float = 180.0,
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
        sla_deadline_utc="2023-08-01T08:00:00+00:00",
        remaining_sla_minutes=remaining_sla,
        bandwidth_gb=bandwidth,
        wait_intervals=0,
        was_deferred=False,
    )


def make_current_dc(
    dc_id: int,
    *,
    cpu: float = 100.0,
    gpu: float = 100.0,
    memory: float = 100.0,
    price: float = 10.0,
    carbon: float = 10.0,
) -> DataCenterSnapshot:
    cpu_total = max(cpu, 100.0)
    gpu_total = max(gpu, 100.0)
    memory_total = max(memory, 100.0)
    return DataCenterSnapshot(
        dc_id=dc_id,
        dc_name=f"DC{dc_id}",
        location=f"location-{dc_id}",
        cpu_total_cores=cpu_total,
        cpu_available_cores=cpu,
        cpu_reserved_cores=0.0,
        cpu_schedulable_cores=cpu,
        cpu_available_ratio=cpu / cpu_total,
        gpu_total_units=gpu_total,
        gpu_available_units=gpu,
        gpu_reserved_units=0.0,
        gpu_schedulable_units=gpu,
        gpu_available_ratio=gpu / gpu_total,
        memory_total_gb=memory_total,
        memory_available_gb=memory,
        memory_reserved_gb=0.0,
        memory_schedulable_gb=memory,
        memory_available_ratio=memory / memory_total,
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


def make_horizon_dc(
    dc_id: int,
    horizon: int,
    *,
    cpu: tuple[float, ...] | None = None,
    gpu: tuple[float, ...] | None = None,
    memory: tuple[float, ...] | None = None,
    prices: tuple[float, ...] | None = None,
    carbon: tuple[float, ...] | None = None,
    forecast_gpu: tuple[float, ...] | None = None,
) -> HorizonDataCenterSnapshot:
    cpu = cpu or (100.0,) * horizon
    gpu = gpu or (100.0,) * horizon
    memory = memory or (100.0,) * horizon
    prices = prices or (10.0,) * horizon
    carbon = carbon or (10.0,) * horizon
    zeros = (0.0,) * horizon
    forecast_gpu = forecast_gpu or zeros
    return HorizonDataCenterSnapshot(
        dc_id=dc_id,
        dc_name=f"DC{dc_id}",
        location=f"location-{dc_id}",
        cpu_total_cores=max(100.0, *cpu),
        gpu_total_units=max(100.0, *gpu),
        memory_total_gb=max(100.0, *memory),
        cpu_available_cores=cpu,
        gpu_available_units=gpu,
        memory_available_gb=memory,
        known_cpu_reservations=zeros,
        known_gpu_reservations=zeros,
        known_memory_reservations=zeros,
        forecast_cpu_reservations=zeros,
        forecast_gpu_reservations=forecast_gpu,
        forecast_memory_reservations=zeros,
        electricity_price_usd_per_mwh=prices,
        carbon_intensity_gco2_per_kwh=carbon,
    )


def make_horizon_state(
    tasks: tuple[TaskSnapshot, ...],
    dcs: tuple[HorizonDataCenterSnapshot, ...],
    *,
    allow_defer: bool = True,
    forecast_mode: str = "no_future_arrivals",
    future_arrivals: tuple[FutureArrivalAggregate, ...] = (),
    transit_tasks: tuple[TransitTaskHorizonSnapshot, ...] = (),
    transfer_delays: dict[tuple[int, int], float] | None = None,
    transfer_costs: dict[tuple[int, int], float] | None = None,
) -> HorizonState:
    transfer_delays = transfer_delays or {}
    transfer_costs = transfer_costs or {}
    current_dcs = tuple(
        make_current_dc(
            dc.dc_id,
            cpu=dc.cpu_available_cores[0],
            gpu=dc.gpu_available_units[0],
            memory=dc.memory_available_gb[0],
            price=dc.electricity_price_usd_per_mwh[0],
            carbon=dc.carbon_intensity_gco2_per_kwh[0],
        )
        for dc in dcs
    )
    links = tuple(
        NetworkLinkSnapshot(origin.dc_id, destination.dc_id, 0.0)
        for origin in dcs
        for destination in dcs
    )
    destinations = tuple(
        TaskDestinationSnapshot(
            task_id=task.task_id,
            original_index=task.original_index,
            destination_dc_id=dc.dc_id,
            transmission_cost_usd=transfer_costs.get(
                (task.original_index, dc.dc_id), 0.0
            ),
            transmission_delay_seconds=transfer_delays.get(
                (task.original_index, dc.dc_id), 0.0
            ),
        )
        for task in tasks
        for dc in dcs
    )
    current = SchedulerState(
        tasks=tasks,
        datacenters=current_dcs,
        network_links=links,
        task_destinations=destinations,
        exogenous=ExogenousSignalsSnapshot(
            current_time_utc="2023-08-01T05:00:00+00:00",
            timestep_minutes=15.0,
        ),
        allow_defer=allow_defer,
    )
    return HorizonState(
        current=current,
        horizon=len(dcs[0].cpu_available_cores),
        forecast_mode=forecast_mode,
        timestep_minutes=15.0,
        datacenters=dcs,
        running_tasks=(),
        transit_tasks=transit_tasks,
        future_arrivals=future_arrivals,
    )


def make_action_adapter(
    dcs: tuple[HorizonDataCenterSnapshot, ...], allow_defer: bool = True
) -> SustainClusterActionAdapter:
    offset = int(allow_defer)
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
    state: HorizonState,
    *,
    weights: RollingObjectiveWeights | None = None,
    allow_defer: bool | None = None,
) -> object:
    enabled = state.current.allow_defer if allow_defer is None else allow_defer
    config = RollingHorizonConfig(
        allow_defer=enabled,
        weights=weights
        or RollingObjectiveWeights(
            electricity=0.0,
            carbon=0.0,
            transmission=0.0,
            waiting_defer=1.0,
            sla_risk=10.0,
            terminal_backlog=100.0,
        ),
    )
    return RollingHorizonOptimizer().solve(
        state, config, make_action_adapter(state.datacenters, enabled)
    )


def test_h1_matches_one_step_for_loose_feasible_task() -> None:
    task = make_task("task", 0)
    hdc = make_horizon_dc(7, 1)
    horizon_state = make_horizon_state((task,), (hdc,), allow_defer=True)
    rolling = solve(horizon_state)
    one_step = OneStepOptimizer(
        OptimizationConfig(
            allow_defer=True,
            weights=ObjectiveWeights(0.0, 0.0, 0.0, 100.0, 10.0),
        )
    ).solve(
        horizon_state.current,
        make_action_adapter(horizon_state.datacenters, True),
    )
    assert rolling.first_step_decisions[0].decision == "assign"
    assert rolling.first_step_decisions[0].dc_id == one_step.assignments[0].dc_id
    assert rolling.environment_actions == one_step.environment_actions


def test_sufficient_capacity_dispatches_now() -> None:
    state = make_horizon_state(
        (make_task("task", 0, gpu=2.0),),
        (make_horizon_dc(1, 4, gpu=(4.0,) * 4),),
    )
    result = solve(state)
    assert result.plans[0].dispatch_step == 0
    assert result.environment_actions == (1,)


def test_future_capacity_release_causes_one_step_defer() -> None:
    state = make_horizon_state(
        (make_task("task", 0, gpu=2.0),),
        (make_horizon_dc(1, 4, gpu=(0.0, 0.0, 4.0, 4.0)),),
    )
    result = solve(state)
    assert result.plans[0].dispatch_step == 1
    assert result.first_step_decisions[0].decision == "defer"


def test_forecast_reservation_changes_current_decision() -> None:
    task = make_task("low-priority", 0, gpu=4.0)
    no_forecast = make_horizon_state(
        (task,), (make_horizon_dc(1, 4, gpu=(4.0,) * 4),)
    )
    reserved_dc = make_horizon_dc(
        1,
        4,
        gpu=(4.0, 0.0, 4.0, 4.0),
        forecast_gpu=(0.0, 4.0, 0.0, 0.0),
    )
    future = (
        FutureArrivalAggregate(1, 1, 1, 1.0, 4.0, 1.0, 1, 3),
    )
    oracle = make_horizon_state(
        (task,),
        (reserved_dc,),
        forecast_mode="oracle",
        future_arrivals=future,
    )
    assert solve(no_forecast).plans[0].dispatch_step == 0
    assert solve(oracle).first_step_decisions[0].decision == "defer"


def test_low_future_price_delays_loose_task() -> None:
    state = make_horizon_state(
        (make_task("flexible", 0, gpu=1.0),),
        (
            make_horizon_dc(
                1, 4, prices=(1000.0, 1000.0, 0.0, 0.0)
            ),
        ),
    )
    weights = RollingObjectiveWeights(10.0, 0.0, 0.0, 0.01, 0.0, 100.0)
    result = solve(state, weights=weights)
    assert result.plans[0].dispatch_step == 1
    assert result.first_step_decisions[0].decision == "defer"


def test_urgent_task_does_not_wait_for_low_price() -> None:
    task = make_task("urgent", 0, gpu=1.0, remaining_sla=30.0)
    state = make_horizon_state(
        (task,),
        (make_horizon_dc(1, 4, prices=(1000.0, 1000.0, 0.0, 0.0)),),
    )
    weights = RollingObjectiveWeights(10.0, 0.0, 0.0, 0.0, 10.0, 100.0)
    result = solve(state, weights=weights)
    assert result.plans[0].dispatch_step == 0
    assert result.projected_sla_violations == 0


def test_duration_occupancy_respects_every_horizon_capacity() -> None:
    tasks = (
        make_task("a", 0, cpu=6.0, duration=30.0),
        make_task("b", 1, cpu=6.0, duration=30.0),
    )
    state = make_horizon_state(
        tasks,
        (make_horizon_dc(1, 4, cpu=(10.0,) * 4),),
    )
    result = solve(state)
    allocation = result.datacenter_allocations[0]
    assert max(allocation.cpu_cores) <= 10.0
    assert {plan.dispatch_step for plan in result.plans} == {0, 2}


def test_in_transit_task_is_reserved_but_not_assigned_twice() -> None:
    transit = (
        TransitTaskHorizonSnapshot("already-routed", 1, 1, 2, 2.0, 1.0, 4.0),
    )
    state = make_horizon_state(
        (), (make_horizon_dc(1, 4, gpu=(2.0, 1.0, 1.0, 2.0)),),
        transit_tasks=transit,
    )
    result = solve(state)
    assert result.plans == ()
    assert result.environment_actions == ()
    assert result.integer_variable_count == 0


def test_terminal_backlog_cost_prevents_horizon_pushing() -> None:
    task = make_task("energy-heavy", 0, gpu=1.0)
    state = make_horizon_state(
        (task,), (make_horizon_dc(1, 4, prices=(1000.0,) * 4),)
    )
    cheap_backlog = RollingObjectiveWeights(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    protected = RollingObjectiveWeights(1.0, 0.0, 0.0, 0.0, 0.0, 10.0)
    assert solve(state, weights=cheap_backlog).terminal_backlog_count == 1
    assert solve(state, weights=protected).terminal_backlog_count == 0


@pytest.mark.parametrize("horizon", [2, 4, 8])
def test_supported_rolling_horizons_solve(horizon: int) -> None:
    state = make_horizon_state(
        (make_task("task", 0),), (make_horizon_dc(1, horizon),)
    )
    result = solve(state)
    assert result.status == "optimal"
    assert result.integer_variable_count == horizon + 1


def test_empty_task_set_returns_empty_actions() -> None:
    state = make_horizon_state((), (make_horizon_dc(1, 4),))
    result = solve(state)
    assert result.status == "optimal"
    assert result.environment_actions == ()


def test_short_horizon_reports_terminal_backlog_and_sla_risk() -> None:
    task = make_task("impossible", 0, duration=60.0, remaining_sla=30.0)
    state = make_horizon_state((task,), (make_horizon_dc(1, 2),))
    result = solve(state)
    assert result.status == "optimal"
    assert result.terminal_backlog_count == 1
    assert result.projected_sla_violations == 1
    assert result.first_step_decisions[0].decision == "defer"


def test_dynamic_datacenter_order_is_preserved_in_actions() -> None:
    task = make_task("task", 0)
    dcs = (make_horizon_dc(9, 2), make_horizon_dc(3, 2))
    state = make_horizon_state((task,), dcs)
    result = solve(state)
    assert result.plans[0].destination_dc_id == 9
    assert result.environment_actions == (1,)


def test_long_transmission_delay_blocks_late_dispatch() -> None:
    task = make_task("remote", 0, remaining_sla=60.0)
    dc = make_horizon_dc(1, 4)
    state = make_horizon_state(
        (task,),
        (dc,),
        transfer_delays={(0, 1): 30.0 * 60.0},
    )
    result = solve(state)
    assert result.plans[0].dispatch_step == 0
    assert result.plans[0].execution_start_step == 2


def test_multi_step_plan_never_executes_beyond_terminal_boundary() -> None:
    task = make_task("terminal-boundary", 0, remaining_sla=240.0)
    dc = make_horizon_dc(1, 4)
    state = make_horizon_state((task,), (dc,))
    result = solve(state)
    assert result.plans[0].decision == "dispatch"
    assert result.plans[0].execution_start_step < state.horizon


def _numpy_rng_equal(left: tuple, right: tuple) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_real_forecast_modes_are_read_only_and_noisy_oracle_is_reproducible() -> None:
    from run_one_step_closed_loop import build_env, env_fingerprint

    previous_cwd = Path.cwd()
    try:
        env = build_env(REPO, allow_defer=True)
        env.reset(seed=123)
        before_env = env_fingerprint(env)
        before_python = random.getstate()
        before_numpy = np.random.get_state()
        adapter = HorizonStateAdapter(
            ForecastNoiseConfig(
                demand_relative_std=0.2,
                duration_relative_std=0.2,
                arrival_step_std=0.5,
                seed=456,
            )
        )
        no_future = adapter.build_horizon_state(env, 4, "no_future_arrivals")
        oracle = adapter.build_horizon_state(env, 4, "oracle")
        noisy_a = adapter.build_horizon_state(env, 4, "noisy_oracle")
        noisy_b = adapter.build_horizon_state(env, 4, "noisy_oracle")
        assert no_future.future_arrivals == ()
        assert oracle.forecast_mode == "oracle"
        assert noisy_a == noisy_b
        assert env_fingerprint(env) == before_env
        assert random.getstate() == before_python
        assert _numpy_rng_equal(np.random.get_state(), before_numpy)
    finally:
        os.chdir(previous_cwd)


def test_invalid_horizon_is_rejected_before_environment_access() -> None:
    with pytest.raises(ValueError, match="horizon"):
        HorizonStateAdapter().build_horizon_state(object(), 3, "oracle")
