from __future__ import annotations

from sustaincluster_mpc.action_adapter import (
    ActionMapping,
    SustainClusterActionAdapter,
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
from sustaincluster_mpc.terminal_h60_optimizer import (
    TerminalH60Config,
    TerminalH60DataCenter,
    TerminalH60Optimizer,
)


def _state(duration_minutes: float = 90.0) -> HorizonState:
    task = TaskSnapshot(
        task_id="task-1",
        original_index=0,
        origin_dc_id=1,
        cpu_cores=10.0,
        gpu_units=1.0,
        memory_gb=10.0,
        duration_minutes=duration_minutes,
        remaining_duration_minutes=duration_minutes,
        arrival_time_utc="2026-01-01T00:00:00+00:00",
        sla_deadline_utc="2026-01-02T00:00:00+00:00",
        remaining_sla_minutes=1440.0,
        bandwidth_gb=0.0,
        wait_intervals=0,
        was_deferred=False,
    )
    current_dcs = tuple(
        DataCenterSnapshot(
            dc_id=dc_id,
            dc_name=f"DC{dc_id}",
            location=f"L{dc_id}",
            cpu_total_cores=100.0,
            cpu_available_cores=100.0,
            cpu_reserved_cores=0.0,
            cpu_schedulable_cores=100.0,
            cpu_available_ratio=1.0,
            gpu_total_units=10.0,
            gpu_available_units=10.0,
            gpu_reserved_units=0.0,
            gpu_schedulable_units=10.0,
            gpu_available_ratio=1.0,
            memory_total_gb=100.0,
            memory_available_gb=100.0,
            memory_reserved_gb=0.0,
            memory_schedulable_gb=100.0,
            memory_available_ratio=1.0,
            running_task_count=0,
            queued_task_count=0,
            in_transit_task_count=0,
            resource_release_times_utc=(),
            electricity_price_usd_per_mwh=0.0,
            carbon_intensity_gco2_per_kwh=0.0,
            total_power_kw=None,
            it_power_kw=None,
            cooling_power_kw=None,
            internal_temperature_c=None,
            ambient_temperature_c=None,
            crac_setpoint_c=None,
        )
        for dc_id in (1, 2)
    )
    horizon_dcs = tuple(
        HorizonDataCenterSnapshot(
            dc_id=dc_id,
            dc_name=f"DC{dc_id}",
            location=f"L{dc_id}",
            cpu_total_cores=100.0,
            gpu_total_units=10.0,
            memory_total_gb=100.0,
            cpu_available_cores=(100.0,),
            gpu_available_units=(10.0,),
            memory_available_gb=(100.0,),
            known_cpu_reservations=(0.0,),
            known_gpu_reservations=(0.0,),
            known_memory_reservations=(0.0,),
            forecast_cpu_reservations=(0.0,),
            forecast_gpu_reservations=(0.0,),
            forecast_memory_reservations=(0.0,),
            electricity_price_usd_per_mwh=(0.0,),
            carbon_intensity_gco2_per_kwh=(0.0,),
        )
        for dc_id in (1, 2)
    )
    links = tuple(
        NetworkLinkSnapshot(origin, destination, 0.0)
        for origin in (1, 2)
        for destination in (1, 2)
    )
    destinations = tuple(
        TaskDestinationSnapshot("task-1", 0, dc_id, 0.0, 0.0)
        for dc_id in (1, 2)
    )
    current = SchedulerState(
        tasks=(task,),
        datacenters=current_dcs,
        network_links=links,
        task_destinations=destinations,
        exogenous=ExogenousSignalsSnapshot(
            "2026-01-01T00:00:00+00:00", 15.0
        ),
        allow_defer=True,
        information_mode="deployable",
    )
    return HorizonState(
        current=current,
        horizon=1,
        forecast_mode="no_future_arrivals",
        timestep_minutes=15.0,
        datacenters=horizon_dcs,
        running_tasks=(),
        transit_tasks=(),
        future_arrivals=(),
        information_mode="deployable",
        future_signal_mode="persistence",
    )


def _terminal() -> tuple[TerminalH60DataCenter, ...]:
    return (
        TerminalH60DataCenter(
            1,
            100.0,
            10.0,
            100.0,
            90.0,
            9.0,
            90.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        TerminalH60DataCenter(
            2,
            100.0,
            10.0,
            100.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    )


def _adapter() -> SustainClusterActionAdapter:
    return SustainClusterActionAdapter(
        ActionMapping(((1, 1), (2, 2)), 0, 3)
    )


def test_lambda_zero_matches_current_only_optimizer() -> None:
    state = _state()
    config = RollingHorizonConfig(allow_defer=True)
    expected = RollingHorizonOptimizer().solve(state, config, _adapter())
    actual = TerminalH60Optimizer().solve(
        state,
        _terminal(),
        current_config=config,
        terminal_config=TerminalH60Config(lambda_terminal=0.0),
        action_adapter=_adapter(),
    )
    assert actual.status == "optimal"
    assert actual.environment_actions == expected.environment_actions
    assert actual.terminal_contribution == 0.0


def test_long_task_can_avoid_high_terminal_pressure() -> None:
    result = TerminalH60Optimizer().solve(
        _state(),
        _terminal(),
        current_config=RollingHorizonConfig(allow_defer=True),
        terminal_config=TerminalH60Config(),
        action_adapter=_adapter(),
    )
    assert result.status == "optimal"
    assert result.environment_actions == (2,)
    assert result.terminal_contribution > 0.0


def test_task_finishing_before_t60_has_no_terminal_occupancy() -> None:
    result = TerminalH60Optimizer().solve(
        _state(duration_minutes=30.0),
        _terminal(),
        current_config=RollingHorizonConfig(allow_defer=True),
        terminal_config=TerminalH60Config(),
        action_adapter=_adapter(),
    )
    assert result.status == "optimal"
    assert result.environment_actions == (1,)
    assert result.terminal_costs.resource_pressure == 0.0


def test_identical_input_is_deterministic() -> None:
    optimizer = TerminalH60Optimizer()
    kwargs = {
        "current_config": RollingHorizonConfig(allow_defer=True),
        "terminal_config": TerminalH60Config(),
        "action_adapter": _adapter(),
    }
    first = optimizer.solve(_state(), _terminal(), **kwargs)
    second = optimizer.solve(_state(), _terminal(), **kwargs)
    assert first.environment_actions == second.environment_actions
    assert first.objective_value == second.objective_value


def test_negative_terminal_electricity_price_is_valid() -> None:
    first, second = _terminal()
    negative_price = TerminalH60DataCenter(
        dc_id=first.dc_id,
        cpu_total_cores=first.cpu_total_cores,
        gpu_total_units=first.gpu_total_units,
        memory_total_gb=first.memory_total_gb,
        estimated_existing_cpu_cores=first.estimated_existing_cpu_cores,
        estimated_existing_gpu_units=first.estimated_existing_gpu_units,
        estimated_existing_memory_gb=first.estimated_existing_memory_gb,
        arriving_cpu_demand=first.arriving_cpu_demand,
        arriving_gpu_demand=first.arriving_gpu_demand,
        arriving_memory_demand=first.arriving_memory_demand,
        electricity_price_usd_per_mwh=-25.0,
        carbon_intensity_gco2_per_kwh=first.carbon_intensity_gco2_per_kwh,
    )
    result = TerminalH60Optimizer().solve(
        _state(),
        (negative_price, second),
        current_config=RollingHorizonConfig(allow_defer=True),
        terminal_config=TerminalH60Config(),
        action_adapter=_adapter(),
    )
    assert result.status == "optimal"
