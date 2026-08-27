from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from sustaincluster_mpc import (
    FutureArrivalAggregate,
    HorizonDataCenterSnapshot,
    HorizonState,
    TaskSnapshot,
)


@dataclass(frozen=True)
class ForecastUncertainty:
    arrival_step: int
    origin_dc_id: int
    task_count_std: float
    cpu_cores_std: float
    gpu_units_std: float
    memory_gb_std: float


@dataclass(frozen=True)
class BaselineForecast:
    aggregates: tuple[FutureArrivalAggregate, ...]
    uncertainties: tuple[ForecastUncertainty, ...]
    history_steps: int


class HistoricalArrivalForecaster:
    """仅使用已观测任务快照的因果滚动均值预测。"""

    def __init__(
        self,
        horizon: int = 4,
        history_window: int = 16,
        seed: int = 123,
    ) -> None:
        if horizon < 1 or history_window < 1:
            raise ValueError("horizon 和 history_window 必须为正数")
        self.horizon = horizon
        self.history_window = history_window
        self.seed = seed
        self._history: deque[dict[int, np.ndarray]] = deque(
            maxlen=history_window
        )

    @property
    def observed_steps(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        self._history.clear()

    def observe(self, tasks: Iterable[TaskSnapshot]) -> None:
        by_origin: dict[int, np.ndarray] = {}
        durations: dict[int, list[float]] = {}
        deadlines: dict[int, list[float]] = {}
        for task in tasks:
            value = by_origin.setdefault(task.origin_dc_id, np.zeros(5))
            value += np.array(
                [1.0, task.cpu_cores, task.gpu_units, task.memory_gb, 0.0]
            )
            durations.setdefault(task.origin_dc_id, []).append(
                task.remaining_duration_minutes
            )
            deadlines.setdefault(task.origin_dc_id, []).append(
                task.remaining_sla_minutes
            )
        for origin, value in by_origin.items():
            value[4] = max(durations[origin], default=15.0)
            deadline = min(deadlines[origin], default=60.0)
            value.resize(6, refcheck=False)
            value[5] = deadline
        self._history.append({key: value.copy() for key, value in by_origin.items()})

    def predict(self, timestep_minutes: float) -> BaselineForecast:
        if not self._history:
            return BaselineForecast((), (), 0)
        origins = sorted({key for step in self._history for key in step})
        aggregates: list[FutureArrivalAggregate] = []
        uncertainties: list[ForecastUncertainty] = []
        for origin in origins:
            matrix = np.vstack(
                [step.get(origin, np.zeros(6)) for step in self._history]
            )
            average = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            task_count = int(round(max(0.0, average[0])))
            if task_count == 0 and average[0] > 0:
                task_count = 1
            for arrival_step in range(1, self.horizon):
                aggregates.append(
                    FutureArrivalAggregate(
                        arrival_step=arrival_step,
                        origin_dc_id=origin,
                        task_count=task_count,
                        cpu_cores=max(0.0, float(average[1])),
                        gpu_units=max(0.0, float(average[2])),
                        memory_gb=max(0.0, float(average[3])),
                        maximum_duration_steps=max(
                            1, int(math.ceil(max(0.0, average[4]) / timestep_minutes))
                        ),
                        minimum_deadline_step=max(
                            arrival_step,
                            int(math.floor(max(0.0, average[5]) / timestep_minutes)),
                        ),
                    )
                )
                uncertainties.append(
                    ForecastUncertainty(
                        arrival_step,
                        origin,
                        float(std[0]),
                        float(std[1]),
                        float(std[2]),
                        float(std[3]),
                    )
                )
        return BaselineForecast(
            tuple(aggregates), tuple(uncertainties), len(self._history)
        )

    def apply(
        self, state: HorizonState, forecast: BaselineForecast
    ) -> HorizonState:
        datacenters = tuple(
            self._reserve_datacenter(dc, forecast.aggregates, state.horizon)
            for dc in state.datacenters
        )
        return replace(
            state,
            forecast_mode="baseline_history",  # type: ignore[arg-type]
            datacenters=datacenters,
            future_arrivals=forecast.aggregates,
        )

    @staticmethod
    def _reserve_datacenter(
        dc: HorizonDataCenterSnapshot,
        arrivals: tuple[FutureArrivalAggregate, ...],
        horizon: int,
    ) -> HorizonDataCenterSnapshot:
        cpu = np.array(dc.cpu_available_cores, dtype=float)
        gpu = np.array(dc.gpu_available_units, dtype=float)
        memory = np.array(dc.memory_available_gb, dtype=float)
        reserve_cpu = np.array(dc.forecast_cpu_reservations, dtype=float)
        reserve_gpu = np.array(dc.forecast_gpu_reservations, dtype=float)
        reserve_memory = np.array(dc.forecast_memory_reservations, dtype=float)
        for arrival in arrivals:
            if arrival.origin_dc_id != dc.dc_id:
                continue
            stop = min(
                horizon,
                arrival.arrival_step + arrival.maximum_duration_steps,
            )
            section = slice(arrival.arrival_step, stop)
            reserve_cpu[section] += arrival.cpu_cores
            reserve_gpu[section] += arrival.gpu_units
            reserve_memory[section] += arrival.memory_gb
            cpu[section] = np.maximum(0.0, cpu[section] - arrival.cpu_cores)
            gpu[section] = np.maximum(0.0, gpu[section] - arrival.gpu_units)
            memory[section] = np.maximum(
                0.0, memory[section] - arrival.memory_gb
            )
        return replace(
            dc,
            cpu_available_cores=tuple(float(x) for x in cpu),
            gpu_available_units=tuple(float(x) for x in gpu),
            memory_available_gb=tuple(float(x) for x in memory),
            forecast_cpu_reservations=tuple(float(x) for x in reserve_cpu),
            forecast_gpu_reservations=tuple(float(x) for x in reserve_gpu),
            forecast_memory_reservations=tuple(
                float(x) for x in reserve_memory
            ),
        )
