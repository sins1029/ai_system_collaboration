from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.h60_dataset import H60_TARGET_NAMES
from forecasting.h60_models import denormalize_h60, h60_metric_summary
from scripts.competition.run_mpc_h60_night2 import build_terminal_occupancy_cache
from sustaincluster_mpc.h60_controllers import MpcH60LearnedController
from sustaincluster_mpc.terminal_h60_optimizer import TerminalH60DataCenter


def test_terminal_occupancy_cache_reproduces_visible_t60_interval() -> None:
    states = pd.DataFrame({"step": np.arange(12)})
    tasks = pd.DataFrame(
        {
            "task_id": ["task-a"],
            "cpu_cores": [10.0],
            "gpu_units": [2.0],
            "memory_gb": [20.0],
        }
    )
    lifecycle = pd.DataFrame(
        {
            "task_id": ["task-a"],
            "destination_dc": [2],
            "dispatch_step": [1],
            "planned_execution_start_step": [2],
            "actual_execution_start_step": [3],
            "planned_estimated_completion_step": [10],
            "start_estimated_completion_step": [11],
            "observed_true_completion_step": [9],
        }
    )

    cache = build_terminal_occupancy_cache(states, tasks, lifecycle)

    # Visible from t=2 and estimated active at t+4 while t < 7.
    expected = np.zeros((12, 5, 3))
    expected[2:7, 1] = [10.0, 2.0, 20.0]
    np.testing.assert_allclose(cache, expected)


def test_h60_metrics_report_normalized_and_stable_raw_units() -> None:
    scaler = {
        "parameters": {
            name: {"mean": float(index + 1), "scale": float(index + 2)}
            for index, name in enumerate(H60_TARGET_NAMES)
        }
    }
    truth = np.zeros((2, 1, 4), dtype=np.float32)
    prediction = np.ones((2, 1, 4), dtype=np.float32)

    metrics = h60_metric_summary(truth, prediction, scaler=scaler)
    raw = denormalize_h60(prediction, scaler)

    assert metrics["normalized_macro_mae"] == 1.0
    assert metrics["normalized_rmse"] == 1.0
    np.testing.assert_allclose(raw[0, 0], [3.0, 5.0, 7.0, 9.0])


def test_learned_controller_rejects_oracle_terminal_input_before_solve() -> None:
    oracle = TerminalH60DataCenter(
        dc_id=1,
        cpu_total_cores=100.0,
        gpu_total_units=10.0,
        memory_total_gb=100.0,
        estimated_existing_cpu_cores=0.0,
        estimated_existing_gpu_units=0.0,
        estimated_existing_memory_gb=0.0,
        arriving_cpu_demand=1.0,
        arriving_gpu_demand=1.0,
        arriving_memory_demand=1.0,
        electricity_price_usd_per_mwh=100.0,
        carbon_intensity_gco2_per_kwh=300.0,
        source="PRIVILEGED_ORACLE_T60",
    )

    controller = MpcH60LearnedController()
    try:
        controller.solve(None, (oracle,), current_config=None, terminal_config=None, action_adapter=None)
    except ValueError as exc:
        assert "DEPLOYABLE_LEARNED_T60" in str(exc)
    else:
        raise AssertionError("Learned controller accepted Oracle terminal input")
