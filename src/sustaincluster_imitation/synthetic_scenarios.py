from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sustaincluster_mpc import (
    AssignmentDecision,
    DataCenterSnapshot,
    ExogenousSignalsSnapshot,
    HorizonDataCenterSnapshot,
    HorizonState,
    NetworkLinkSnapshot,
    SchedulerState,
    TaskDestinationSnapshot,
    TaskSnapshot,
)


@dataclass(frozen=True)
class SyntheticTaskSpec:
    task_id: str
    arrival_step: int
    deadline_step: int
    duration_steps: int
    origin_dc_id: int
    cpu_cores: float
    gpu_units: float
    memory_gb: float
    bandwidth_gb: float = 1.0


@dataclass(frozen=True)
class SyntheticDataCenterSpec:
    dc_id: int
    cpu_capacity: tuple[float, ...]
    gpu_capacity: tuple[float, ...]
    memory_capacity: tuple[float, ...]
    prices: tuple[float, ...]
    carbon: tuple[float, ...]


@dataclass(frozen=True)
class SyntheticScenario:
    name: str
    steps: int
    datacenters: tuple[SyntheticDataCenterSpec, ...]
    tasks: tuple[SyntheticTaskSpec, ...]
    burst_intensity: float = 1.0


@dataclass
class PendingTask:
    spec: SyntheticTaskSpec
    wait_steps: int = 0


@dataclass(frozen=True)
class ScheduledTask:
    spec: SyntheticTaskSpec
    dc_id: int
    dispatch_step: int
    start_step: int
    completion_step: int
    wait_steps: int


def build_synthetic_scenarios(
    steps: int = 96, burst_intensity: float = 1.0
) -> tuple[SyntheticScenario, ...]:
    if steps < 8:
        raise ValueError("合成场景至少需要八个步骤")
    cycle = 12

    def repeated(pattern: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(pattern[index % len(pattern)]) for index in range(steps + 20))

    def tasks_for_cycle(kind: str) -> list[SyntheticTaskSpec]:
        tasks: list[SyntheticTaskSpec] = []
        for base in range(0, steps, cycle):
            if kind == "gpu_burst":
                for index in range(4):
                    tasks.append(
                        SyntheticTaskSpec(
                            f"gpu-flex-{base}-{index}",
                            base,
                            base + 10,
                            3,
                            1,
                            1.0,
                            2.0 * burst_intensity,
                            4.0,
                        )
                    )
                for index in range(2):
                    tasks.append(
                        SyntheticTaskSpec(
                            f"gpu-urgent-{base}-{index}",
                            base + 1,
                            base + 4,
                            2,
                            1,
                            1.0,
                            4.0 * burst_intensity,
                            4.0,
                        )
                    )
            elif kind == "future_low_price":
                for index in range(3):
                    tasks.append(
                        SyntheticTaskSpec(
                            f"price-flex-{base}-{index}",
                            base,
                            base + 7,
                            1,
                            1,
                            5.0,
                            1.0,
                            4.0,
                        )
                    )
            elif kind == "capacity_release":
                tasks.append(
                    SyntheticTaskSpec(
                        f"release-{base}",
                        base,
                        base + 8,
                        2,
                        1,
                        2.0,
                        4.0,
                        4.0,
                    )
                )
            elif kind == "sla_conflict":
                tasks.append(
                    SyntheticTaskSpec(
                        f"urgent-{base}",
                        base,
                        base + 2,
                        1,
                        1,
                        8.0,
                        2.0,
                        4.0,
                    )
                )
            elif kind == "center_heterogeneity":
                for index in range(3):
                    tasks.append(
                        SyntheticTaskSpec(
                            f"placement-{base}-{index}",
                            base,
                            base + 7,
                            2,
                            1,
                            6.0,
                            2.0,
                            8.0,
                        )
                    )
        return tasks

    common_cpu = repeated([32.0])
    common_memory = repeated([128.0])
    common_carbon = repeated([300.0])
    gpu_burst = SyntheticScenario(
        "gpu_burst",
        steps,
        (
            SyntheticDataCenterSpec(
                1,
                common_cpu,
                repeated([8.0 * max(1.0, burst_intensity)]),
                common_memory,
                repeated([100.0]),
                common_carbon,
            ),
        ),
        tuple(tasks_for_cycle("gpu_burst")),
        burst_intensity,
    )
    low_price = SyntheticScenario(
        "future_low_price",
        steps,
        (
            SyntheticDataCenterSpec(
                1,
                repeated([20.0]),
                repeated([4.0]),
                common_memory,
                repeated([1000.0, 1000.0, 20.0, 20.0, 20.0, 100.0]),
                common_carbon,
            ),
        ),
        tuple(tasks_for_cycle("future_low_price")),
    )
    capacity_release = SyntheticScenario(
        "capacity_release",
        steps,
        (
            SyntheticDataCenterSpec(
                1,
                common_cpu,
                repeated([0.0, 0.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]),
                common_memory,
                repeated([100.0]),
                common_carbon,
            ),
        ),
        tuple(tasks_for_cycle("capacity_release")),
    )
    sla_conflict = SyntheticScenario(
        "sla_conflict",
        steps,
        (
            SyntheticDataCenterSpec(
                1,
                common_cpu,
                repeated([8.0]),
                common_memory,
                repeated([1000.0, 1000.0, 10.0, 10.0, 10.0, 100.0]),
                common_carbon,
            ),
        ),
        tuple(tasks_for_cycle("sla_conflict")),
    )
    center = SyntheticScenario(
        "center_heterogeneity",
        steps,
        (
            SyntheticDataCenterSpec(
                1,
                repeated([8.0]),
                repeated([2.0]),
                repeated([32.0]),
                repeated([20.0]),
                repeated([100.0]),
            ),
            SyntheticDataCenterSpec(
                2,
                repeated([32.0]),
                repeated([8.0]),
                repeated([128.0]),
                repeated([300.0]),
                repeated([500.0]),
            ),
        ),
        tuple(tasks_for_cycle("center_heterogeneity")),
    )
    balancing_tasks = []
    for step in range(steps):
        for index in range(12):
            balancing_tasks.extend(
                (
                    SyntheticTaskSpec(
                        f"cpu-specialist-{step}-{index}",
                        step,
                        step + 3,
                        1,
                        1,
                        20.0,
                        0.0,
                        0.1,
                    ),
                    SyntheticTaskSpec(
                        f"gpu-specialist-{step}-{index}",
                        step,
                        step + 3,
                        1,
                        1,
                        1.0,
                        2.0,
                        0.1,
                    ),
                    SyntheticTaskSpec(
                        f"memory-specialist-{step}-{index}",
                        step,
                        step + 3,
                        1,
                        1,
                        1.0,
                        0.0,
                        20.0,
                    ),
                )
            )
    zero = repeated([0.0])
    class_balance = SyntheticScenario(
        "class_balance_heterogeneity",
        steps,
        (
            SyntheticDataCenterSpec(1, zero, zero, zero, repeated([100.0]), common_carbon),
            SyntheticDataCenterSpec(2, repeated([300.0]), zero, repeated([10.0]), repeated([10.0]), common_carbon),
            SyntheticDataCenterSpec(3, zero, zero, zero, repeated([100.0]), common_carbon),
            SyntheticDataCenterSpec(4, repeated([20.0]), repeated([30.0]), repeated([10.0]), repeated([20.0]), common_carbon),
            SyntheticDataCenterSpec(5, repeated([20.0]), zero, repeated([300.0]), repeated([30.0]), common_carbon),
        ),
        tuple(balancing_tasks),
    )
    return gpu_burst, low_price, capacity_release, sla_conflict, center, class_balance


class SyntheticEpisodeSimulator:
    def __init__(self, scenario: SyntheticScenario) -> None:
        self.scenario = scenario
        self.current_step = 0
        self.pending: list[PendingTask] = []
        self.scheduled: list[ScheduledTask] = []

    def reset(self) -> None:
        self.current_step = 0
        self.pending.clear()
        self.scheduled.clear()

    def admit_current_arrivals(self) -> None:
        self.pending.extend(
            PendingTask(task)
            for task in self.scenario.tasks
            if task.arrival_step == self.current_step
        )

    def build_state(self, horizon: int) -> HorizonState:
        h_dcs = []
        current_dcs = []
        for dc in self.scenario.datacenters:
            available = [
                self._available(dc, self.current_step + offset)
                for offset in range(horizon)
            ]
            zeros = (0.0,) * horizon
            h_dcs.append(
                HorizonDataCenterSnapshot(
                    dc.dc_id,
                    f"DC{dc.dc_id}",
                    f"synthetic-{dc.dc_id}",
                    max(dc.cpu_capacity),
                    max(max(dc.gpu_capacity), 1.0),
                    max(max(dc.memory_capacity), 1.0),
                    tuple(item[0] for item in available),
                    tuple(item[1] for item in available),
                    tuple(item[2] for item in available),
                    zeros,
                    zeros,
                    zeros,
                    zeros,
                    zeros,
                    zeros,
                    tuple(self._at(dc.prices, self.current_step + h) for h in range(horizon)),
                    tuple(self._at(dc.carbon, self.current_step + h) for h in range(horizon)),
                )
            )
            current_dcs.append(self._current_dc(dc, available[0]))
        tasks = tuple(
            TaskSnapshot(
                item.spec.task_id,
                index,
                item.spec.origin_dc_id,
                item.spec.cpu_cores,
                item.spec.gpu_units,
                item.spec.memory_gb,
                item.spec.duration_steps * 15.0,
                item.spec.duration_steps * 15.0,
                f"step:{item.spec.arrival_step}",
                f"step:{item.spec.deadline_step}",
                (item.spec.deadline_step - self.current_step) * 15.0,
                item.spec.bandwidth_gb,
                item.wait_steps,
                item.wait_steps > 0,
            )
            for index, item in enumerate(self.pending)
        )
        links = tuple(
            NetworkLinkSnapshot(origin.dc_id, destination.dc_id, 0.0)
            for origin in self.scenario.datacenters
            for destination in self.scenario.datacenters
        )
        destinations = tuple(
            TaskDestinationSnapshot(task.task_id, task.original_index, dc.dc_id, 0.0, 0.0)
            for task in tasks
            for dc in self.scenario.datacenters
        )
        current = SchedulerState(
            tasks,
            tuple(current_dcs),
            links,
            destinations,
            ExogenousSignalsSnapshot(f"step:{self.current_step}", 15.0),
            True,
        )
        return HorizonState(
            current,
            horizon,
            "no_future_arrivals",
            15.0,
            tuple(h_dcs),
            (),
            (),
            (),
        )

    def apply(self, decisions: Sequence[AssignmentDecision]) -> None:
        if len(decisions) != len(self.pending):
            raise ValueError("合成决策数量不匹配")
        by_dc = {dc.dc_id: dc for dc in self.scenario.datacenters}
        kept = []
        for pending, decision in zip(self.pending, decisions):
            if decision.decision == "defer":
                pending.wait_steps += 1
                kept.append(pending)
                continue
            dc = by_dc[int(decision.dc_id)]
            start = self._find_slot(pending.spec, dc, self.current_step + 1)
            self.scheduled.append(
                ScheduledTask(
                    pending.spec,
                    dc.dc_id,
                    self.current_step,
                    start,
                    start + pending.spec.duration_steps,
                    pending.wait_steps + max(0, start - self.current_step - 1),
                )
            )
        self.pending = kept
        self.current_step += 1

    def oracle_future_tasks(self, horizon: int) -> tuple[SyntheticTaskSpec, ...]:
        return tuple(
            task
            for task in self.scenario.tasks
            if self.current_step < task.arrival_step < self.current_step + horizon
        )

    def summary(self) -> dict[str, float | int]:
        violations = sum(
            task.completion_step > task.spec.deadline_step for task in self.scheduled
        ) + len(self.pending)
        waits = [task.wait_steps for task in self.scheduled] + [
            item.wait_steps for item in self.pending
        ]
        return {
            "completed_tasks": sum(
                task.completion_step <= self.scenario.steps
                for task in self.scheduled
            ),
            "sla_violations": violations,
            "remaining_backlog": len(self.pending),
            "average_wait_steps": sum(waits) / max(1, len(waits)),
            "p95_wait_steps": sorted(waits)[int(0.95 * (len(waits) - 1))]
            if waits
            else 0.0,
        }

    def _available(
        self, dc: SyntheticDataCenterSpec, step: int
    ) -> tuple[float, float, float]:
        base = (
            self._at(dc.cpu_capacity, step),
            self._at(dc.gpu_capacity, step),
            self._at(dc.memory_capacity, step),
        )
        active = [
            task
            for task in self.scheduled
            if task.dc_id == dc.dc_id
            and task.start_step <= step < task.completion_step
        ]
        used = (
            sum(task.spec.cpu_cores for task in active),
            sum(task.spec.gpu_units for task in active),
            sum(task.spec.memory_gb for task in active),
        )
        return tuple(max(0.0, total - value) for total, value in zip(base, used))

    def _find_slot(
        self, task: SyntheticTaskSpec, dc: SyntheticDataCenterSpec, earliest: int
    ) -> int:
        for start in range(earliest, self.scenario.steps + 20):
            if all(
                task.cpu_cores <= self._available(dc, step)[0] + 1e-9
                and task.gpu_units <= self._available(dc, step)[1] + 1e-9
                and task.memory_gb <= self._available(dc, step)[2] + 1e-9
                for step in range(start, start + task.duration_steps)
            ):
                return start
        return self.scenario.steps + 20

    def _current_dc(
        self,
        dc: SyntheticDataCenterSpec,
        available: tuple[float, float, float],
    ) -> DataCenterSnapshot:
        cpu_total = max(dc.cpu_capacity)
        gpu_total = max(max(dc.gpu_capacity), 1.0)
        memory_total = max(max(dc.memory_capacity), 1.0)
        return DataCenterSnapshot(
            dc.dc_id,
            f"DC{dc.dc_id}",
            f"synthetic-{dc.dc_id}",
            cpu_total,
            available[0],
            0.0,
            available[0],
            available[0] / max(cpu_total, 1.0),
            gpu_total,
            available[1],
            0.0,
            available[1],
            available[1] / gpu_total,
            memory_total,
            available[2],
            0.0,
            available[2],
            available[2] / memory_total,
            0,
            0,
            0,
            (),
            self._at(dc.prices, self.current_step),
            self._at(dc.carbon, self.current_step),
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def _at(values: tuple[float, ...], step: int) -> float:
        return float(values[min(max(0, step), len(values) - 1)])
