from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forecasting.transformer_workload_provider import (
    TransformerWorkloadForecastProvider,
)
from forecasting.workload_forecast_provider import (
    ForecastRequest,
    PersistenceWorkloadForecastProvider,
)
from scripts.audit import mpc_control_authority_diagnosis_v1 as diagnosis
from sustaincluster_mpc.action_adapter import AssignmentDecision
from sustaincluster_mpc.forecast_pressure_adapter import apply_forecast_pressure


@pytest.fixture(scope="module")
def defer_transition() -> dict:
    return diagnosis.real_defer_transition()


def test_defer_semantic_action_is_feasible() -> None:
    task = diagnosis.make_task()
    dc = diagnosis.make_horizon_dc(1)
    adapter = diagnosis.action_adapter((dc,))

    actions = adapter.encode_assignments(
        (task,),
        (AssignmentDecision(task.task_id, task.original_index, "defer"),),
    )

    assert actions == [0]


def test_defer_environment_transition_retains_task(defer_transition: dict) -> None:
    assert defer_transition["remains_external_pending"] is True
    assert defer_transition["accepted_assignment_next_step"] is True


def test_defer_waiting_increment_is_repaired(
    defer_transition: dict,
) -> None:
    assert defer_transition["waiting_increment"] == 1


def test_defer_sla_clock_evolves_one_timestep(defer_transition: dict) -> None:
    assert defer_transition["sla_clock_delta_minutes"] == pytest.approx(15.0)


def test_future_workload_perturbation_reaches_optimizer() -> None:
    task = diagnosis.make_task()
    state = diagnosis.make_state(
        (task,), (diagnosis.make_horizon_dc(1, 5), diagnosis.make_horizon_dc(2, 5))
    )
    configs = (
        {"dc_id": 1, "population_weight": 0.8, "timezone_shift": 0},
        {"dc_id": 2, "population_weight": 0.2, "timezone_shift": 0},
    )
    zero = np.zeros((4, 4), dtype=float)
    high = np.repeat([[10.0, 90.0, 90.0, 90.0]], 4, axis=0)

    without_pressure = apply_forecast_pressure(
        state, diagnosis.global_bundle(zero), configs
    ).state
    with_pressure = apply_forecast_pressure(
        state, diagnosis.global_bundle(high), configs
    ).state

    assert with_pressure.datacenters[0].gpu_available_units[1] < without_pressure.datacenters[0].gpu_available_units[1]
    assert diagnosis.first_action(diagnosis.solve(without_pressure))[1] == 1
    assert diagnosis.first_action(diagnosis.solve(with_pressure))[1] == 2


def test_future_price_perturbation_reaches_optimizer() -> None:
    task = replace(
        diagnosis.make_task(), cpu_cores=20.0, gpu_units=20.0, memory_gb=20.0
    )
    low = diagnosis.make_state(
        (task,),
        (diagnosis.make_horizon_dc(1, price=(1000.0, 200.0, 200.0, 200.0)),),
    )
    high = diagnosis.make_state(
        (task,),
        (diagnosis.make_horizon_dc(1, price=(1000.0, 2000.0, 2000.0, 2000.0)),),
    )

    low_result = diagnosis.solve(low)
    high_result = diagnosis.solve(high)

    assert low_result.costs.electricity < high_result.costs.electricity
    assert diagnosis.first_action(low_result)[0] == "assign"
    assert diagnosis.first_action(high_result)[0] == "assign"


def test_long_running_task_occupies_future_capacity() -> None:
    result = diagnosis.solve(
        diagnosis.make_state(
            (diagnosis.make_task(duration=60.0),),
            (diagnosis.make_horizon_dc(1),),
        )
    )

    assert result.datacenter_allocations[0].gpu_units == (0.0, 40.0, 40.0, 40.0)


@pytest.mark.parametrize(
    ("estimated_duration", "expected_future_steps"),
    ((15.0, 1), (30.0, 2), (60.0, 3)),
)
def test_estimated_duration_controls_horizon_occupancy(
    estimated_duration: float, expected_future_steps: int
) -> None:
    task = diagnosis.make_task(duration=estimated_duration)
    result = diagnosis.solve(
        diagnosis.make_state((task,), (diagnosis.make_horizon_dc(1),))
    )

    assert sum(
        value > 0 for value in result.datacenter_allocations[0].gpu_units[1:]
    ) == expected_future_steps


def test_swapped_future_pressure_reverses_first_placement() -> None:
    task = diagnosis.make_task()
    dc1_high = diagnosis.make_state(
        (task,),
        (
            diagnosis.make_horizon_dc(1, gpu=(100.0, 20.0, 20.0, 20.0)),
            diagnosis.make_horizon_dc(2),
        ),
    )
    dc2_high = diagnosis.make_state(
        (task,),
        (
            diagnosis.make_horizon_dc(1),
            diagnosis.make_horizon_dc(2, gpu=(100.0, 20.0, 20.0, 20.0)),
        ),
    )

    assert diagnosis.first_action(diagnosis.solve(dc1_high))[1] == 2
    assert diagnosis.first_action(diagnosis.solve(dc2_high))[1] == 1


def test_per_dc_future_pressure_changes_capacity_rhs() -> None:
    state = diagnosis.make_state(
        (diagnosis.make_task(),),
        (diagnosis.make_horizon_dc(1, 5), diagnosis.make_horizon_dc(2, 5)),
    )
    configs = (
        {"dc_id": 1, "population_weight": 0.75, "timezone_shift": 0},
        {"dc_id": 2, "population_weight": 0.25, "timezone_shift": 0},
    )
    values = np.repeat([[0.0, 20.0, 40.0, 60.0]], 4, axis=0)

    application = apply_forecast_pressure(
        state, diagnosis.global_bundle(values), configs
    )

    assert application.state.datacenters[0].cpu_available_cores[0] == pytest.approx(100.0)
    assert application.state.datacenters[0].cpu_available_cores[1] == pytest.approx(85.0)
    assert application.state.datacenters[0].gpu_available_units[0] == pytest.approx(100.0)
    assert application.state.datacenters[0].gpu_available_units[1] == pytest.approx(70.0)
    assert application.state.datacenters[0].memory_available_gb[0] == pytest.approx(100.0)
    assert application.state.datacenters[0].memory_available_gb[1] == pytest.approx(55.0)


def test_action_mapping_consistently_encodes_assign_and_defer() -> None:
    task = diagnosis.make_task()
    dcs = (diagnosis.make_horizon_dc(9), diagnosis.make_horizon_dc(3))
    adapter = diagnosis.action_adapter(dcs)

    assign = adapter.encode_assignments(
        (task,),
        (AssignmentDecision(task.task_id, task.original_index, "assign", 3),),
    )
    defer = adapter.encode_assignments(
        (task,),
        (AssignmentDecision(task.task_id, task.original_index, "defer"),),
    )

    assert assign == [2]
    assert defer == [0]


def test_deployable_forecasts_ignore_oracle_future() -> None:
    history = np.zeros((96, 8), dtype=np.float32)
    history[-1, :4] = [10.0, 20.0, 30.0, 40.0]
    low = ForecastRequest(
        pd.Timestamp("1970-03-10T00:00:00Z"),
        history,
        history[-1, :4],
        np.zeros((4, 4)),
    )
    high = replace(low, oracle_future_workload=np.full((4, 4), 1e9))
    persistence = PersistenceWorkloadForecastProvider()
    transformer = TransformerWorkloadForecastProvider(
        diagnosis.WORKSPACE / "artifacts/transformer_forecast_v1/checkpoints/transformer_seed_33_best.pt",
        dataset_root=diagnosis.WORKSPACE / "artifacts/forecast_dataset_v1",
        device="cpu",
    )

    np.testing.assert_array_equal(
        persistence.forecast(low).as_array(), persistence.forecast(high).as_array()
    )
    np.testing.assert_allclose(
        transformer.forecast(low).as_array(), transformer.forecast(high).as_array()
    )


def test_bridge_h1_forecast_maps_to_future_capacity_index_one() -> None:
    state = diagnosis.make_state(
        (diagnosis.make_task(),), (diagnosis.make_horizon_dc(1, 5),)
    )
    configs = ({"dc_id": 1, "population_weight": 1.0, "timezone_shift": 0},)
    values = np.zeros((4, 4), dtype=float)
    values[0, 2] = 25.0

    application = apply_forecast_pressure(
        state, diagnosis.global_bundle(values), configs
    )

    assert application.state.datacenters[0].gpu_available_units == (
        100.0,
        75.0,
        100.0,
        100.0,
        100.0,
    )


def test_low_future_price_is_dominated_by_frozen_waiting_cost() -> None:
    task = replace(
        diagnosis.make_task(), cpu_cores=20.0, gpu_units=20.0, memory_gb=20.0
    )
    state = diagnosis.make_state(
        (task,),
        (diagnosis.make_horizon_dc(1, price=(1000.0, 1000.0, 200.0, 200.0)),),
    )
    costs = {
        row["forced_option"]: row
        for row in diagnosis.forced_cost_rows(state, diagnosis.optimizer_config())
    }

    assert costs["DEFER_ONE_STEP"]["waiting_defer"] == pytest.approx(100.0)
    assert costs["DEFER_ONE_STEP"]["total"] > costs["EXECUTE_NOW"]["total"]
    assert diagnosis.first_action(diagnosis.solve(state))[0] == "assign"
