from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from sustaincluster_contract.runtime import (
    InformationMode,
    controller_duration_minutes,
    controller_finish_time,
)


@dataclass(frozen=True)
class TaskSnapshot:
    task_id: str
    original_index: int
    origin_dc_id: int
    cpu_cores: float
    gpu_units: float
    memory_gb: float
    duration_minutes: float
    remaining_duration_minutes: float
    arrival_time_utc: str
    sla_deadline_utc: str
    remaining_sla_minutes: float
    bandwidth_gb: float
    wait_intervals: int
    was_deferred: bool
    scheduler_wait_intervals: int = 0


@dataclass(frozen=True)
class DataCenterSnapshot:
    dc_id: int
    dc_name: str
    location: str
    cpu_total_cores: float
    cpu_available_cores: float
    cpu_reserved_cores: float
    cpu_schedulable_cores: float
    cpu_available_ratio: float
    gpu_total_units: float
    gpu_available_units: float
    gpu_reserved_units: float
    gpu_schedulable_units: float
    gpu_available_ratio: float
    memory_total_gb: float
    memory_available_gb: float
    memory_reserved_gb: float
    memory_schedulable_gb: float
    memory_available_ratio: float
    running_task_count: int
    queued_task_count: int
    in_transit_task_count: int
    resource_release_times_utc: tuple[str, ...]
    electricity_price_usd_per_mwh: float
    carbon_intensity_gco2_per_kwh: float
    total_power_kw: Optional[float]
    it_power_kw: Optional[float]
    cooling_power_kw: Optional[float]
    internal_temperature_c: Optional[float]
    ambient_temperature_c: Optional[float]
    crac_setpoint_c: Optional[float]


@dataclass(frozen=True)
class NetworkLinkSnapshot:
    origin_dc_id: int
    destination_dc_id: int
    transmission_cost_usd_per_gb: float


@dataclass(frozen=True)
class TaskDestinationSnapshot:
    task_id: str
    original_index: int
    destination_dc_id: int
    transmission_cost_usd: float
    transmission_delay_seconds: float


@dataclass(frozen=True)
class ExogenousSignalsSnapshot:
    current_time_utc: str
    timestep_minutes: float


@dataclass(frozen=True)
class SchedulerState:
    tasks: tuple[TaskSnapshot, ...]
    datacenters: tuple[DataCenterSnapshot, ...]
    network_links: tuple[NetworkLinkSnapshot, ...]
    task_destinations: tuple[TaskDestinationSnapshot, ...]
    exogenous: ExogenousSignalsSnapshot
    allow_defer: bool
    information_mode: InformationMode = "oracle"


class SustainClusterStateAdapter:
    """构建不可变调度快照，不修改 SustainCluster。"""

    def __init__(
        self,
        env: Any,
        information_mode: InformationMode | None = None,
    ) -> None:
        if env is None:
            raise ValueError("env 不能为 None")
        if not hasattr(env, "cluster_manager") or not hasattr(env, "current_tasks"):
            raise TypeError("env 必须提供 cluster_manager 和 current_tasks")
        self._env = env
        self.information_mode = information_mode or getattr(
            env, "information_mode", "oracle"
        )
        if self.information_mode not in ("oracle", "deployable"):
            raise ValueError(
                f"unsupported information_mode={self.information_mode!r}"
            )

    def get_pending_tasks(self) -> tuple[TaskSnapshot, ...]:
        current_time = self._required_attribute(self._env, "current_time", "env")
        dc_ids = {
            int(self._required_attribute(dc, "dc_id", "datacenter"))
            for dc in self._env.cluster_manager.datacenters.values()
        }
        snapshots: list[TaskSnapshot] = []
        for index, task in enumerate(tuple(self._env.current_tasks)):
            task_id = str(self._required_attribute(task, "job_name", f"task[{index}]"))
            origin_dc_id = int(
                self._required_attribute(task, "origin_dc_id", f"task {task_id}")
            )
            if origin_dc_id not in dc_ids:
                raise ValueError(
                    f"task {task_id} 包含未知 origin_dc_id={origin_dc_id}"
                )

            duration = self._nonnegative(
                controller_duration_minutes(task, self.information_mode),
                f"task {task_id} duration_minutes",
            )
            finish_time = getattr(task, "finish_time", None)
            if finish_time is None:
                remaining_duration = duration
            else:
                remaining_duration = max(
                    0.0,
                    self._minutes_between(
                        controller_finish_time(task, self.information_mode),
                        current_time,
                    ),
                )

            sla_deadline = self._required_attribute(
                task, "sla_deadline", f"task {task_id}"
            )
            arrival_time = self._required_attribute(
                task, "arrival_time", f"task {task_id}"
            )
            wait_intervals = int(getattr(task, "wait_intervals", 0))
            scheduler_wait_intervals = int(
                getattr(task, "scheduler_wait_intervals", 0)
            )
            if wait_intervals < 0:
                raise ValueError(
                    f"task {task_id} wait_intervals 必须为非负数"
                )
            if not 0 <= scheduler_wait_intervals <= wait_intervals:
                raise ValueError(
                    f"task {task_id} scheduler waiting 与 total waiting 不一致"
                )

            snapshots.append(
                TaskSnapshot(
                    task_id=task_id,
                    original_index=index,
                    origin_dc_id=origin_dc_id,
                    cpu_cores=self._nonnegative(
                        self._required_attribute(
                            task, "cores_req", f"task {task_id}"
                        ),
                        f"task {task_id} cpu_cores",
                    ),
                    gpu_units=self._nonnegative(
                        self._required_attribute(
                            task, "gpu_req", f"task {task_id}"
                        ),
                        f"task {task_id} gpu_units",
                    ),
                    memory_gb=self._nonnegative(
                        self._required_attribute(
                            task, "mem_req", f"task {task_id}"
                        ),
                        f"task {task_id} memory_gb",
                    ),
                    duration_minutes=duration,
                    remaining_duration_minutes=remaining_duration,
                    arrival_time_utc=self._iso_timestamp(
                        arrival_time, f"task {task_id} arrival_time"
                    ),
                    sla_deadline_utc=self._iso_timestamp(
                        sla_deadline, f"task {task_id} sla_deadline"
                    ),
                    remaining_sla_minutes=self._finite(
                        self._minutes_between(sla_deadline, current_time),
                        f"task {task_id} remaining_sla_minutes",
                    ),
                    bandwidth_gb=self._nonnegative(
                        self._required_attribute(
                            task, "bandwidth_gb", f"task {task_id}"
                        ),
                        f"task {task_id} bandwidth_gb",
                    ),
                    wait_intervals=wait_intervals,
                    was_deferred=bool(
                        getattr(task, "temporarily_deferred", False)
                    ),
                    scheduler_wait_intervals=scheduler_wait_intervals,
                )
            )
        return tuple(snapshots)

    def get_datacenter_states(self) -> tuple[DataCenterSnapshot, ...]:
        reservations = self._reservation_totals()
        snapshots: list[DataCenterSnapshot] = []
        for dc_name, dc in tuple(self._env.cluster_manager.datacenters.items()):
            dc_id = int(self._required_attribute(dc, "dc_id", dc_name))
            total_cpu = self._positive(dc.total_cores, f"{dc_name} total_cores")
            total_gpu = self._positive(dc.total_gpus, f"{dc_name} total_gpus")
            total_memory = self._positive(
                dc.total_mem_GB, f"{dc_name} total_mem_GB"
            )
            available_cpu = self._capacity(
                dc.available_cores, total_cpu, f"{dc_name} available_cores"
            )
            available_gpu = self._capacity(
                dc.available_gpus, total_gpu, f"{dc_name} available_gpus"
            )
            available_memory = self._capacity(
                dc.available_mem, total_memory, f"{dc_name} available_mem"
            )
            reserved = reservations.get(dc_name, (0.0, 0.0, 0.0, 0))
            reserved_cpu, reserved_gpu, reserved_memory, in_transit_count = reserved
            queued_tasks = tuple(dc.pending_tasks)
            for task in queued_tasks:
                reserved_cpu += self._nonnegative(
                    task.cores_req, f"{dc_name} queued task CPU"
                )
                reserved_gpu += self._nonnegative(
                    task.gpu_req, f"{dc_name} queued task GPU"
                )
                reserved_memory += self._nonnegative(
                    task.mem_req, f"{dc_name} queued task memory"
                )

            dc_info = getattr(dc, "dc_info", None)
            if dc_info is None:
                dc_info = {}
            if not isinstance(dc_info, dict):
                raise TypeError(f"{dc_name} dc_info 必须是字典")
            running_tasks = tuple(dc.running_tasks)
            release_times = tuple(
                sorted(
                    self._iso_timestamp(
                        controller_finish_time(task, self.information_mode),
                        "running task controller-visible finish_time",
                    )
                    for task in running_tasks
                    if getattr(task, "finish_time", None) is not None
                )
            )

            snapshots.append(
                DataCenterSnapshot(
                    dc_id=dc_id,
                    dc_name=str(dc_name),
                    location=str(
                        self._required_attribute(dc, "location", dc_name)
                    ),
                    cpu_total_cores=total_cpu,
                    cpu_available_cores=available_cpu,
                    cpu_reserved_cores=reserved_cpu,
                    cpu_schedulable_cores=max(0.0, available_cpu - reserved_cpu),
                    cpu_available_ratio=available_cpu / total_cpu,
                    gpu_total_units=total_gpu,
                    gpu_available_units=available_gpu,
                    gpu_reserved_units=reserved_gpu,
                    gpu_schedulable_units=max(0.0, available_gpu - reserved_gpu),
                    gpu_available_ratio=available_gpu / total_gpu,
                    memory_total_gb=total_memory,
                    memory_available_gb=available_memory,
                    memory_reserved_gb=reserved_memory,
                    memory_schedulable_gb=max(
                        0.0, available_memory - reserved_memory
                    ),
                    memory_available_ratio=available_memory / total_memory,
                    running_task_count=len(running_tasks),
                    queued_task_count=len(queued_tasks),
                    in_transit_task_count=int(in_transit_count),
                    resource_release_times_utc=release_times,
                    electricity_price_usd_per_mwh=self._finite(
                        dc.price_manager.get_current_price(),
                        f"{dc_name} electricity price",
                    ),
                    carbon_intensity_gco2_per_kwh=self._nonnegative(
                        dc.ci_manager.get_current_ci(norm=False),
                        f"{dc_name} carbon intensity",
                    ),
                    total_power_kw=self._optional_finite(
                        dc_info.get("dc_total_power_kW"),
                        f"{dc_name} total power",
                    ),
                    it_power_kw=self._optional_finite(
                        dc_info.get("dc_ITE_total_power_kW"),
                        f"{dc_name} IT power",
                    ),
                    cooling_power_kw=self._optional_finite(
                        dc_info.get("dc_HVAC_total_power_kW"),
                        f"{dc_name} cooling power",
                    ),
                    internal_temperature_c=self._optional_finite(
                        dc_info.get("dc_int_temperature"),
                        f"{dc_name} internal temperature",
                    ),
                    ambient_temperature_c=self._optional_finite(
                        dc_info.get("dc_exterior_ambient_temp"),
                        f"{dc_name} ambient temperature",
                    ),
                    crac_setpoint_c=self._optional_finite(
                        getattr(dc, "current_crac_setpoint", None),
                        f"{dc_name} CRAC setpoint",
                    ),
                )
            )
        return tuple(snapshots)

    def get_network_state(self) -> tuple[NetworkLinkSnapshot, ...]:
        mapper = importlib.import_module("utils.transmission_region_mapper")
        provider = str(self._env.cluster_manager.cloud_provider)
        matrix = self._env.cluster_manager.transmission_matrix
        datacenters = tuple(self._env.cluster_manager.datacenters.values())
        links: list[NetworkLinkSnapshot] = []
        for origin in datacenters:
            origin_region = mapper.map_location_to_region(origin.location, provider)
            for destination in datacenters:
                destination_region = mapper.map_location_to_region(
                    destination.location, provider
                )
                try:
                    raw_cost = matrix.loc[origin_region, destination_region]
                except KeyError as exc:
                    raise ValueError(
                        "缺少传输成本："
                        f"{origin_region}->{destination_region}"
                    ) from exc
                links.append(
                    NetworkLinkSnapshot(
                        origin_dc_id=int(origin.dc_id),
                        destination_dc_id=int(destination.dc_id),
                        transmission_cost_usd_per_gb=self._nonnegative(
                            raw_cost,
                            f"network cost {origin.dc_id}->{destination.dc_id}",
                        ),
                    )
                )
        return tuple(links)

    def get_exogenous_signals(self) -> ExogenousSignalsSnapshot:
        time_step = self._required_attribute(self._env, "time_step", "env")
        if hasattr(time_step, "total_seconds"):
            timestep_minutes = float(time_step.total_seconds()) / 60.0
        else:
            timestep_minutes = self._positive(time_step, "env timestep_minutes")
        return ExogenousSignalsSnapshot(
            current_time_utc=self._iso_timestamp(
                self._required_attribute(self._env, "current_time", "env"),
                "env current_time",
            ),
            timestep_minutes=self._positive(
                timestep_minutes, "env timestep_minutes"
            ),
        )

    def build_scheduler_state(self) -> SchedulerState:
        tasks = self.get_pending_tasks()
        datacenters = self.get_datacenter_states()
        links = self.get_network_state()
        task_destinations = self._get_task_destination_states(
            tasks, datacenters, links
        )
        return SchedulerState(
            tasks=tasks,
            datacenters=datacenters,
            network_links=links,
            task_destinations=task_destinations,
            exogenous=self.get_exogenous_signals(),
            allow_defer=not bool(self._env.disable_defer_action),
            information_mode=self.information_mode,
        )

    def _get_task_destination_states(
        self,
        tasks: Sequence[TaskSnapshot],
        datacenters: Sequence[DataCenterSnapshot],
        links: Sequence[NetworkLinkSnapshot],
    ) -> tuple[TaskDestinationSnapshot, ...]:
        delay_module = importlib.import_module("data.network_cost.network_delay")
        provider = str(self._env.cluster_manager.cloud_provider)
        dc_by_id = {dc.dc_id: dc for dc in datacenters}
        link_cost = {
            (link.origin_dc_id, link.destination_dc_id):
            link.transmission_cost_usd_per_gb
            for link in links
        }
        values: list[TaskDestinationSnapshot] = []
        for task in tasks:
            origin = dc_by_id[task.origin_dc_id]
            for destination in datacenters:
                delay = delay_module.get_transmission_delay(
                    origin.location,
                    destination.location,
                    provider,
                    task.bandwidth_gb,
                )
                values.append(
                    TaskDestinationSnapshot(
                        task_id=task.task_id,
                        original_index=task.original_index,
                        destination_dc_id=destination.dc_id,
                        transmission_cost_usd=self._nonnegative(
                            link_cost[(task.origin_dc_id, destination.dc_id)]
                            * task.bandwidth_gb,
                            f"task {task.task_id} transmission cost",
                        ),
                        transmission_delay_seconds=self._nonnegative(
                            delay, f"task {task.task_id} transmission delay"
                        ),
                    )
                )
        return tuple(values)

    def _reservation_totals(self) -> dict[str, tuple[float, float, float, int]]:
        totals: dict[str, list[float]] = {}
        for item in tuple(getattr(self._env, "in_transit_tasks", ())):
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError("in_transit_tasks 包含无效条目")
            _, task, dc_name = item
            if dc_name not in self._env.cluster_manager.datacenters:
                raise ValueError(f"传输中任务包含未知目标 {dc_name}")
            value = totals.setdefault(str(dc_name), [0.0, 0.0, 0.0, 0.0])
            value[0] += self._nonnegative(task.cores_req, "in-transit CPU")
            value[1] += self._nonnegative(task.gpu_req, "in-transit GPU")
            value[2] += self._nonnegative(task.mem_req, "in-transit memory")
            value[3] += 1
        return {
            name: (value[0], value[1], value[2], int(value[3]))
            for name, value in totals.items()
        }

    @staticmethod
    def _required_attribute(obj: Any, name: str, owner: str) -> Any:
        if not hasattr(obj, name):
            raise AttributeError(f"{owner} is missing required field {name}")
        value = getattr(obj, name)
        if value is None:
            raise ValueError(f"{owner}.{name} 不能为 None")
        return value

    @staticmethod
    def _finite(value: Any, label: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label} 必须为数值，实际为 {value!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label} 必须为有限数，实际为 {number}")
        return number

    @classmethod
    def _nonnegative(cls, value: Any, label: str) -> float:
        number = cls._finite(value, label)
        if number < 0:
            raise ValueError(f"{label} 必须为非负数，实际为 {number}")
        return number

    @classmethod
    def _positive(cls, value: Any, label: str) -> float:
        number = cls._finite(value, label)
        if number <= 0:
            raise ValueError(f"{label} 必须为正数，实际为 {number}")
        return number

    @classmethod
    def _capacity(cls, value: Any, total: float, label: str) -> float:
        number = cls._nonnegative(value, label)
        tolerance = max(1.0, total) * 1e-9
        if number > total + tolerance:
            raise ValueError(f"{label}={number} 超过总容量 {total}")
        return min(number, total)

    @classmethod
    def _optional_finite(cls, value: Any, label: str) -> Optional[float]:
        if value is None:
            return None
        return cls._finite(value, label)

    @staticmethod
    def _iso_timestamp(value: Any, label: str) -> str:
        if not hasattr(value, "isoformat"):
            raise TypeError(f"{label} 必须提供 isoformat()")
        return str(value.isoformat())

    @staticmethod
    def _minutes_between(later: Any, earlier: Any) -> float:
        delta = later - earlier
        if not hasattr(delta, "total_seconds"):
            raise TypeError("时间戳相减未返回时间间隔")
        return float(delta.total_seconds()) / 60.0
