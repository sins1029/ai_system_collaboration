from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from sustaincluster_contract.runtime import (
    InformationMode,
    controller_duration_minutes,
    controller_finish_time,
)
from sustaincluster_mpc.future_signals import (
    FutureSignalMode,
    FutureSignalProvider,
)
from sustaincluster_mpc.state_adapter import (
    DataCenterSnapshot,
    SchedulerState,
    SustainClusterStateAdapter,
)


ForecastMode = Literal["no_future_arrivals", "oracle", "noisy_oracle"]
SUPPORTED_HORIZONS = frozenset({1, 2, 4, 5, 8})


@dataclass(frozen=True)
class ForecastNoiseConfig:
    demand_relative_std: float = 0.10
    duration_relative_std: float = 0.10
    arrival_step_std: float = 0.50
    seed: int = 123

    def __post_init__(self) -> None:
        for name in (
            "demand_relative_std",
            "duration_relative_std",
            "arrival_step_std",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须为有限非负数")


@dataclass(frozen=True)
class RunningTaskHorizonSnapshot:
    task_id: str
    dc_id: int
    release_step: int
    cpu_cores: float
    gpu_units: float
    memory_gb: float


@dataclass(frozen=True)
class TransitTaskHorizonSnapshot:
    task_id: str
    destination_dc_id: int
    arrival_step: int
    duration_steps: int
    cpu_cores: float
    gpu_units: float
    memory_gb: float


@dataclass(frozen=True)
class FutureArrivalAggregate:
    arrival_step: int
    origin_dc_id: int
    task_count: int
    cpu_cores: float
    gpu_units: float
    memory_gb: float
    maximum_duration_steps: int
    minimum_deadline_step: int


@dataclass(frozen=True)
class HorizonDataCenterSnapshot:
    dc_id: int
    dc_name: str
    location: str
    cpu_total_cores: float
    gpu_total_units: float
    memory_total_gb: float
    cpu_available_cores: tuple[float, ...]
    gpu_available_units: tuple[float, ...]
    memory_available_gb: tuple[float, ...]
    known_cpu_reservations: tuple[float, ...]
    known_gpu_reservations: tuple[float, ...]
    known_memory_reservations: tuple[float, ...]
    forecast_cpu_reservations: tuple[float, ...]
    forecast_gpu_reservations: tuple[float, ...]
    forecast_memory_reservations: tuple[float, ...]
    electricity_price_usd_per_mwh: tuple[float, ...]
    carbon_intensity_gco2_per_kwh: tuple[float, ...]


@dataclass(frozen=True)
class HorizonState:
    current: SchedulerState
    horizon: int
    forecast_mode: ForecastMode
    timestep_minutes: float
    datacenters: tuple[HorizonDataCenterSnapshot, ...]
    running_tasks: tuple[RunningTaskHorizonSnapshot, ...]
    transit_tasks: tuple[TransitTaskHorizonSnapshot, ...]
    future_arrivals: tuple[FutureArrivalAggregate, ...]
    information_mode: InformationMode = "oracle"
    future_signal_mode: FutureSignalMode = "oracle"


@dataclass(frozen=True)
class _ForecastTask:
    task_id: str
    arrival_step: int
    origin_dc_id: int
    duration_steps: int
    deadline_step: int
    cpu_cores: float
    gpu_units: float
    memory_gb: float


class HorizonStateAdapter:
    """从当前环境构建指定长度的只读滚动时域状态."""

    def __init__(
        self,
        noise_config: ForecastNoiseConfig | None = None,
        *,
        information_mode: InformationMode | None = None,
        future_signal_provider: FutureSignalProvider | None = None,
    ) -> None:
        self._noise = noise_config or ForecastNoiseConfig()
        self._configured_information_mode = information_mode
        self._future_signal_provider = future_signal_provider

    def build_horizon_state(
        self,
        env: Any,
        horizon: int,
        forecast_mode: ForecastMode,
    ) -> HorizonState:
        if horizon not in SUPPORTED_HORIZONS:
            raise ValueError(
                f"horizon 必须是以下值之一 {sorted(SUPPORTED_HORIZONS)}, got {horizon}"
            )
        if forecast_mode not in (
            "no_future_arrivals",
            "oracle",
            "noisy_oracle",
        ):
            raise ValueError(f"不支持 forecast_mode={forecast_mode!r}")

        env_mode = getattr(env, "information_mode", "oracle")
        if (
            self._configured_information_mode is not None
            and hasattr(env, "information_mode")
            and self._configured_information_mode != env_mode
        ):
            raise ValueError(
                "Horizon adapter information_mode must match the environment"
            )
        information_mode = self._configured_information_mode or env_mode
        if information_mode not in ("oracle", "deployable"):
            raise ValueError(
                f"unsupported information_mode={information_mode!r}"
            )
        signal_provider = self._future_signal_provider or FutureSignalProvider(
            "oracle" if information_mode == "oracle" else "persistence"
        )
        if information_mode == "deployable" and signal_provider.mode == "oracle":
            raise ValueError(
                "deployable mode cannot read oracle future price/carbon traces"
            )
        if information_mode == "deployable" and forecast_mode != "no_future_arrivals":
            raise ValueError(
                "deployable mode requires no_future_arrivals until an external "
                "arrival forecast provider is implemented"
            )

        current = SustainClusterStateAdapter(
            env, information_mode
        ).build_scheduler_state()
        step_minutes = current.exogenous.timestep_minutes
        running = self._snapshot_running_tasks(
            env, step_minutes, information_mode
        )
        transit = self._snapshot_transit_tasks(
            env, step_minutes, information_mode
        )
        forecast_tasks = self._forecast_tasks(
            env, horizon, forecast_mode, step_minutes, information_mode
        )
        future_arrivals = self._aggregate_future_arrivals(forecast_tasks)
        datacenters = self._build_datacenter_timelines(
            env,
            current.datacenters,
            running,
            transit,
            forecast_tasks,
            horizon,
            step_minutes,
            information_mode,
            signal_provider,
        )
        return HorizonState(
            current=current,
            horizon=horizon,
            forecast_mode=forecast_mode,
            timestep_minutes=step_minutes,
            datacenters=datacenters,
            running_tasks=running,
            transit_tasks=transit,
            future_arrivals=future_arrivals,
            information_mode=information_mode,
            future_signal_mode=signal_provider.mode,
        )

    def _snapshot_running_tasks(
        self,
        env: Any,
        step_minutes: float,
        information_mode: InformationMode,
    ) -> tuple[RunningTaskHorizonSnapshot, ...]:
        values: list[RunningTaskHorizonSnapshot] = []
        for dc in env.cluster_manager.datacenters.values():
            for task in tuple(dc.running_tasks):
                finish_time = getattr(task, "finish_time", None)
                if finish_time is None:
                    raise ValueError(
                        f"running task {task.job_name} 没有 finish_time"
                    )
                release_step = self._ceil_steps(
                    self._minutes_between(
                        controller_finish_time(task, information_mode),
                        env.current_time,
                    ),
                    step_minutes,
                    minimum=0,
                )
                values.append(
                    RunningTaskHorizonSnapshot(
                        task_id=str(task.job_name),
                        dc_id=int(dc.dc_id),
                        release_step=release_step,
                        cpu_cores=self._nonnegative(task.cores_req, "running CPU"),
                        gpu_units=self._nonnegative(task.gpu_req, "running GPU"),
                        memory_gb=self._nonnegative(task.mem_req, "running memory"),
                    )
                )
        return tuple(values)

    def _snapshot_transit_tasks(
        self,
        env: Any,
        step_minutes: float,
        information_mode: InformationMode,
    ) -> tuple[TransitTaskHorizonSnapshot, ...]:
        values: list[TransitTaskHorizonSnapshot] = []
        for arrival_time, task, dc_name in tuple(env.in_transit_tasks):
            dc = env.cluster_manager.datacenters.get(dc_name)
            if dc is None:
                raise ValueError(f"传输中任务包含未知目标 {dc_name}")
            arrival_step = self._ceil_steps(
                self._minutes_between(arrival_time, env.current_time),
                step_minutes,
                minimum=0,
            )
            values.append(
                TransitTaskHorizonSnapshot(
                    task_id=str(task.job_name),
                    destination_dc_id=int(dc.dc_id),
                    arrival_step=arrival_step,
                    duration_steps=self._ceil_steps(
                        controller_duration_minutes(task, information_mode),
                        step_minutes,
                        minimum=1,
                    ),
                    cpu_cores=self._nonnegative(task.cores_req, "transit CPU"),
                    gpu_units=self._nonnegative(task.gpu_req, "transit GPU"),
                    memory_gb=self._nonnegative(task.mem_req, "transit memory"),
                )
            )
        return tuple(values)

    def _forecast_tasks(
        self,
        env: Any,
        horizon: int,
        mode: ForecastMode,
        step_minutes: float,
        information_mode: InformationMode,
    ) -> tuple[_ForecastTask, ...]:
        if mode == "no_future_arrivals" or horizon == 1:
            return ()

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        predicted: list[_ForecastTask] = []
        try:
            for arrival_step in range(1, horizon):
                future_time = env.current_time + arrival_step * env.time_step
                tasks = env.cluster_manager.get_tasks_for_timestep(future_time)
                for task in tasks:
                    duration_steps = self._ceil_steps(
                        controller_duration_minutes(task, information_mode),
                        step_minutes,
                        minimum=1,
                    )
                    remaining_deadline_minutes = self._minutes_between(
                        task.sla_deadline, env.current_time
                    )
                    predicted.append(
                        _ForecastTask(
                            task_id=str(task.job_name),
                            arrival_step=arrival_step,
                            origin_dc_id=int(task.origin_dc_id),
                            duration_steps=duration_steps,
                            deadline_step=max(
                                arrival_step,
                                int(
                                    math.floor(
                                        remaining_deadline_minutes / step_minutes
                                    )
                                ),
                            ),
                            cpu_cores=self._nonnegative(
                                task.cores_req, "forecast CPU"
                            ),
                            gpu_units=self._nonnegative(
                                task.gpu_req, "forecast GPU"
                            ),
                            memory_gb=self._nonnegative(
                                task.mem_req, "forecast memory"
                            ),
                        )
                    )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

        if mode == "noisy_oracle":
            predicted = self._apply_noise(
                predicted, horizon, int(getattr(env, "global_step", 0))
            )
        return tuple(predicted)

    def _apply_noise(
        self,
        tasks: Sequence[_ForecastTask],
        horizon: int,
        global_step: int,
    ) -> list[_ForecastTask]:
        rng = np.random.default_rng(
            self._noise.seed + global_step * 1009 + horizon * 9176
        )
        values: list[_ForecastTask] = []
        for task in tasks:
            arrival_shift = int(
                np.rint(rng.normal(0.0, self._noise.arrival_step_std))
            )
            arrival_step = min(
                horizon - 1, max(1, task.arrival_step + arrival_shift)
            )
            demand_factor = max(
                0.0, rng.normal(1.0, self._noise.demand_relative_std)
            )
            duration_factor = max(
                0.0, rng.normal(1.0, self._noise.duration_relative_std)
            )
            duration_steps = max(
                1, int(math.ceil(task.duration_steps * duration_factor))
            )
            deadline_slack = max(0, task.deadline_step - task.arrival_step)
            values.append(
                _ForecastTask(
                    task_id=task.task_id,
                    arrival_step=arrival_step,
                    origin_dc_id=task.origin_dc_id,
                    duration_steps=duration_steps,
                    deadline_step=arrival_step + deadline_slack,
                    cpu_cores=task.cpu_cores * demand_factor,
                    gpu_units=task.gpu_units * demand_factor,
                    memory_gb=task.memory_gb * demand_factor,
                )
            )
        return values

    def _aggregate_future_arrivals(
        self, tasks: Sequence[_ForecastTask]
    ) -> tuple[FutureArrivalAggregate, ...]:
        groups: dict[tuple[int, int], list[_ForecastTask]] = {}
        for task in tasks:
            groups.setdefault((task.arrival_step, task.origin_dc_id), []).append(task)
        values = []
        for (arrival_step, origin_dc_id), group in sorted(groups.items()):
            values.append(
                FutureArrivalAggregate(
                    arrival_step=arrival_step,
                    origin_dc_id=origin_dc_id,
                    task_count=len(group),
                    cpu_cores=sum(task.cpu_cores for task in group),
                    gpu_units=sum(task.gpu_units for task in group),
                    memory_gb=sum(task.memory_gb for task in group),
                    maximum_duration_steps=max(task.duration_steps for task in group),
                    minimum_deadline_step=min(task.deadline_step for task in group),
                )
            )
        return tuple(values)

    def _build_datacenter_timelines(
        self,
        env: Any,
        current_dcs: Sequence[DataCenterSnapshot],
        running: Sequence[RunningTaskHorizonSnapshot],
        transit: Sequence[TransitTaskHorizonSnapshot],
        forecast_tasks: Sequence[_ForecastTask],
        horizon: int,
        step_minutes: float,
        information_mode: InformationMode,
        signal_provider: FutureSignalProvider,
    ) -> tuple[HorizonDataCenterSnapshot, ...]:
        running_by_dc: dict[int, list[RunningTaskHorizonSnapshot]] = {}
        transit_by_dc: dict[int, list[TransitTaskHorizonSnapshot]] = {}
        forecast_by_dc: dict[int, list[_ForecastTask]] = {}
        for task in running:
            running_by_dc.setdefault(task.dc_id, []).append(task)
        for task in transit:
            transit_by_dc.setdefault(task.destination_dc_id, []).append(task)
        for task in forecast_tasks:
            forecast_by_dc.setdefault(task.origin_dc_id, []).append(task)

        raw_dcs = {
            int(dc.dc_id): dc for dc in env.cluster_manager.datacenters.values()
        }
        results: list[HorizonDataCenterSnapshot] = []
        for current in current_dcs:
            raw = raw_dcs[current.dc_id]
            release_cpu = np.zeros(horizon)
            release_gpu = np.zeros(horizon)
            release_memory = np.zeros(horizon)
            known_cpu = np.zeros(horizon)
            known_gpu = np.zeros(horizon)
            known_memory = np.zeros(horizon)
            forecast_cpu = np.zeros(horizon)
            forecast_gpu = np.zeros(horizon)
            forecast_memory = np.zeros(horizon)

            for task in running_by_dc.get(current.dc_id, ()):
                if task.release_step < horizon:
                    release_cpu[task.release_step:] += task.cpu_cores
                    release_gpu[task.release_step:] += task.gpu_units
                    release_memory[task.release_step:] += task.memory_gb

            for task in tuple(raw.pending_tasks):
                duration_steps = self._ceil_steps(
                    controller_duration_minutes(task, information_mode),
                    step_minutes,
                    minimum=1,
                )
                self._reserve(
                    known_cpu,
                    known_gpu,
                    known_memory,
                    0,
                    duration_steps,
                    float(task.cores_req),
                    float(task.gpu_req),
                    float(task.mem_req),
                )
            for task in transit_by_dc.get(current.dc_id, ()):
                self._reserve(
                    known_cpu,
                    known_gpu,
                    known_memory,
                    task.arrival_step,
                    task.duration_steps,
                    task.cpu_cores,
                    task.gpu_units,
                    task.memory_gb,
                )
            for task in forecast_by_dc.get(current.dc_id, ()):
                self._reserve(
                    forecast_cpu,
                    forecast_gpu,
                    forecast_memory,
                    task.arrival_step,
                    task.duration_steps,
                    task.cpu_cores,
                    task.gpu_units,
                    task.memory_gb,
                )

            cpu_available = np.clip(
                current.cpu_available_cores
                + release_cpu
                - known_cpu
                - forecast_cpu,
                0.0,
                current.cpu_total_cores,
            )
            gpu_available = np.clip(
                current.gpu_available_units
                + release_gpu
                - known_gpu
                - forecast_gpu,
                0.0,
                current.gpu_total_units,
            )
            memory_available = np.clip(
                current.memory_available_gb
                + release_memory
                - known_memory
                - forecast_memory,
                0.0,
                current.memory_total_gb,
            )
            prices = signal_provider.electricity_price(
                raw, env.current_time, horizon
            )
            carbon = signal_provider.carbon_intensity(
                raw, env.current_time, horizon
            )
            results.append(
                HorizonDataCenterSnapshot(
                    dc_id=current.dc_id,
                    dc_name=current.dc_name,
                    location=current.location,
                    cpu_total_cores=current.cpu_total_cores,
                    gpu_total_units=current.gpu_total_units,
                    memory_total_gb=current.memory_total_gb,
                    cpu_available_cores=tuple(float(x) for x in cpu_available),
                    gpu_available_units=tuple(float(x) for x in gpu_available),
                    memory_available_gb=tuple(float(x) for x in memory_available),
                    known_cpu_reservations=tuple(float(x) for x in known_cpu),
                    known_gpu_reservations=tuple(float(x) for x in known_gpu),
                    known_memory_reservations=tuple(float(x) for x in known_memory),
                    forecast_cpu_reservations=tuple(float(x) for x in forecast_cpu),
                    forecast_gpu_reservations=tuple(float(x) for x in forecast_gpu),
                    forecast_memory_reservations=tuple(
                        float(x) for x in forecast_memory
                    ),
                    electricity_price_usd_per_mwh=prices,
                    carbon_intensity_gco2_per_kwh=carbon,
                )
            )
        return tuple(results)

    @staticmethod
    def _reserve(
        cpu: np.ndarray,
        gpu: np.ndarray,
        memory: np.ndarray,
        start_step: int,
        duration_steps: int,
        cpu_value: float,
        gpu_value: float,
        memory_value: float,
    ) -> None:
        start = max(0, int(start_step))
        stop = min(len(cpu), start + max(1, int(duration_steps)))
        if start >= len(cpu):
            return
        cpu[start:stop] += cpu_value
        gpu[start:stop] += gpu_value
        memory[start:stop] += memory_value

    @classmethod
    def _cyclic_values(
        cls,
        values: Sequence[float],
        index: int,
        horizon: int,
        nonnegative: bool = False,
    ) -> tuple[float, ...]:
        if len(values) == 0:
            raise ValueError("预测源序列为空")
        result = []
        for offset in range(horizon):
            value = cls._finite(values[(int(index) + offset) % len(values)])
            if nonnegative and value < 0:
                raise ValueError("碳强度预测必须为非负数")
            result.append(value)
        return tuple(result)

    @staticmethod
    def _ceil_steps(value_minutes: float, step_minutes: float, minimum: int) -> int:
        if not math.isfinite(value_minutes):
            raise ValueError("时间间隔必须为有限数")
        return max(minimum, int(math.ceil(max(0.0, value_minutes) / step_minutes)))

    @staticmethod
    def _minutes_between(later: Any, earlier: Any) -> float:
        delta = later - earlier
        return float(delta.total_seconds()) / 60.0

    @staticmethod
    def _finite(value: Any) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"值必须为有限数，实际为 {number}")
        return number

    @classmethod
    def _nonnegative(cls, value: Any, label: str) -> float:
        number = cls._finite(value)
        if number < 0:
            raise ValueError(f"{label} 必须为非负数")
        return number
