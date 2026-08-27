from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from sustaincluster_mpc.action_adapter import (
    AssignmentDecision,
    SustainClusterActionAdapter,
)
from sustaincluster_mpc.state_adapter import (
    DataCenterSnapshot,
    SchedulerState,
    TaskSnapshot,
)


@dataclass(frozen=True)
class ObjectiveWeights:
    electricity: float = 1.0
    carbon: float = 1.0
    transmission: float = 1.0
    defer: float = 100.0
    sla_risk: float = 10.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"weight {name} 必须为有限非负数")


@dataclass(frozen=True)
class OptimizationConfig:
    allow_defer: bool
    weights: ObjectiveWeights = ObjectiveWeights()
    cpu_power_w_per_core: float = 6.0
    gpu_power_w_per_unit: float = 500.0
    memory_power_w_per_gb: float = 2.5
    defer_base_penalty: float = 1.0
    deterministic_tie_break_epsilon: float = 1e-9
    solver_time_limit_seconds: Optional[float] = 10.0

    def __post_init__(self) -> None:
        nonnegative = (
            "cpu_power_w_per_core",
            "gpu_power_w_per_unit",
            "memory_power_w_per_gb",
            "defer_base_penalty",
            "deterministic_tie_break_epsilon",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须为有限非负数")
        if self.solver_time_limit_seconds is not None:
            if (
                not math.isfinite(self.solver_time_limit_seconds)
                or self.solver_time_limit_seconds <= 0
            ):
                raise ValueError("solver_time_limit_seconds 必须为正数")


@dataclass(frozen=True)
class CostBreakdown:
    electricity: float
    carbon: float
    transmission: float
    defer: float
    sla_risk: float

    @property
    def total(self) -> float:
        return (
            self.electricity
            + self.carbon
            + self.transmission
            + self.defer
            + self.sla_risk
        )


@dataclass(frozen=True)
class DataCenterAllocation:
    dc_id: int
    task_count: int
    cpu_cores: float
    gpu_units: float
    memory_gb: float


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    message: str
    solve_seconds: float
    objective_value: Optional[float]
    costs: CostBreakdown
    assignments: tuple[AssignmentDecision, ...]
    environment_actions: tuple[int, ...]
    datacenter_allocations: tuple[DataCenterAllocation, ...]

    @property
    def feasible(self) -> bool:
        return self.status == "optimal"


class OneStepOptimizer:
    """求解当前单时间步分配 MILP；这不是滚动时域 MPC。"""

    def __init__(self, config: OptimizationConfig) -> None:
        self._config = config

    def solve(
        self,
        state: SchedulerState,
        action_adapter: SustainClusterActionAdapter,
    ) -> OptimizationResult:
        self._validate_state(state, action_adapter)
        if not state.tasks:
            actions = action_adapter.encode_assignments((), ())
            return OptimizationResult(
                status="optimal",
                message="No pending tasks",
                solve_seconds=0.0,
                objective_value=0.0,
                costs=self._zero_costs(),
                assignments=(),
                environment_actions=tuple(actions),
                datacenter_allocations=self._empty_allocations(state.datacenters),
            )

        tasks = state.tasks
        datacenters = state.datacenters
        task_count = len(tasks)
        dc_count = len(datacenters)
        assignment_variables = task_count * dc_count
        defer_variables = task_count if self._config.allow_defer else 0
        variable_count = assignment_variables + defer_variables

        component_vectors = self._component_vectors(state)
        objective = sum(component_vectors.values(), np.zeros(variable_count))
        objective += self._tie_break_vector(
            task_count, dc_count, variable_count
        )
        constraints = self._build_constraints(tasks, datacenters, variable_count)
        options: dict[str, float | bool] = {"presolve": True}
        if self._config.solver_time_limit_seconds is not None:
            options["time_limit"] = self._config.solver_time_limit_seconds

        started = time.perf_counter()
        try:
            result = milp(
                c=objective,
                integrality=np.ones(variable_count, dtype=np.int8),
                bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
                constraints=constraints,
                options=options,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            return self._failed_result(
                state, "solver_error", str(exc), elapsed
            )
        elapsed = time.perf_counter() - started

        if result.status != 0 or result.x is None:
            status = {
                1: "limit_reached",
                2: "infeasible",
                3: "unbounded",
                4: "solver_error",
            }.get(int(result.status), "solver_error")
            return self._failed_result(
                state, status, str(result.message), elapsed
            )

        assignments = self._decode_assignments(
            tasks, datacenters, np.asarray(result.x), dc_count
        )
        actions = action_adapter.encode_assignments(tasks, assignments)
        costs = CostBreakdown(
            electricity=self._selected_component(
                component_vectors["electricity"], result.x
            ),
            carbon=self._selected_component(
                component_vectors["carbon"], result.x
            ),
            transmission=self._selected_component(
                component_vectors["transmission"], result.x
            ),
            defer=self._selected_component(
                component_vectors["defer"], result.x
            ),
            sla_risk=self._selected_component(
                component_vectors["sla_risk"], result.x
            ),
        )
        allocations = self._build_allocations(tasks, datacenters, assignments)
        self._validate_allocations(datacenters, allocations)
        return OptimizationResult(
            status="optimal",
            message=str(result.message),
            solve_seconds=elapsed,
            objective_value=costs.total,
            costs=costs,
            assignments=assignments,
            environment_actions=tuple(actions),
            datacenter_allocations=allocations,
        )

    def _component_vectors(self, state: SchedulerState) -> dict[str, np.ndarray]:
        tasks = state.tasks
        datacenters = state.datacenters
        dc_count = len(datacenters)
        assignment_variables = len(tasks) * dc_count
        variable_count = assignment_variables + (
            len(tasks) if self._config.allow_defer else 0
        )
        vectors = {
            name: np.zeros(variable_count, dtype=float)
            for name in (
                "electricity",
                "carbon",
                "transmission",
                "defer",
                "sla_risk",
            )
        }
        destination_data = {
            (item.original_index, item.destination_dc_id): item
            for item in state.task_destinations
        }
        step_minutes = state.exogenous.timestep_minutes

        for task_index, task in enumerate(tasks):
            energy_kwh = self._task_energy_kwh(task)
            for dc_index, dc in enumerate(datacenters):
                variable = task_index * dc_count + dc_index
                destination = destination_data[(task.original_index, dc.dc_id)]
                vectors["electricity"][variable] = (
                    self._config.weights.electricity
                    * energy_kwh
                    * dc.electricity_price_usd_per_mwh
                    / 1000.0
                )
                vectors["carbon"][variable] = (
                    self._config.weights.carbon
                    * energy_kwh
                    * dc.carbon_intensity_gco2_per_kwh
                    / 1000.0
                )
                vectors["transmission"][variable] = (
                    self._config.weights.transmission
                    * destination.transmission_cost_usd
                )
                completion_minutes = (
                    task.remaining_duration_minutes
                    + destination.transmission_delay_seconds / 60.0
                )
                assignment_risk = max(
                    0.0,
                    completion_minutes - task.remaining_sla_minutes,
                ) / step_minutes
                vectors["sla_risk"][variable] = (
                    self._config.weights.sla_risk * assignment_risk
                )

            if self._config.allow_defer:
                variable = assignment_variables + task_index
                vectors["defer"][variable] = (
                    self._config.weights.defer
                    * self._config.defer_base_penalty
                )
                vectors["sla_risk"][variable] = (
                    self._config.weights.sla_risk
                    * self._defer_sla_risk(task, step_minutes)
                )
        return vectors

    def _build_constraints(
        self,
        tasks: Sequence[TaskSnapshot],
        datacenters: Sequence[DataCenterSnapshot],
        variable_count: int,
    ) -> tuple[LinearConstraint, LinearConstraint]:
        task_count = len(tasks)
        dc_count = len(datacenters)
        assignment_variables = task_count * dc_count

        assignment_matrix = lil_matrix((task_count, variable_count), dtype=float)
        for task_index in range(task_count):
            for dc_index in range(dc_count):
                assignment_matrix[task_index, task_index * dc_count + dc_index] = 1.0
            if self._config.allow_defer:
                assignment_matrix[
                    task_index, assignment_variables + task_index
                ] = 1.0
        assignment_constraint = LinearConstraint(
            assignment_matrix.tocsr(),
            np.ones(task_count),
            np.ones(task_count),
        )

        capacity_matrix = lil_matrix((dc_count * 3, variable_count), dtype=float)
        capacity_upper = np.zeros(dc_count * 3, dtype=float)
        for dc_index, dc in enumerate(datacenters):
            capacity_upper[dc_index * 3] = dc.cpu_schedulable_cores
            capacity_upper[dc_index * 3 + 1] = dc.gpu_schedulable_units
            capacity_upper[dc_index * 3 + 2] = dc.memory_schedulable_gb
            for task_index, task in enumerate(tasks):
                variable = task_index * dc_count + dc_index
                capacity_matrix[dc_index * 3, variable] = task.cpu_cores
                capacity_matrix[dc_index * 3 + 1, variable] = task.gpu_units
                capacity_matrix[dc_index * 3 + 2, variable] = task.memory_gb
        capacity_constraint = LinearConstraint(
            capacity_matrix.tocsr(),
            np.full(dc_count * 3, -np.inf),
            capacity_upper,
        )
        return assignment_constraint, capacity_constraint

    def _decode_assignments(
        self,
        tasks: Sequence[TaskSnapshot],
        datacenters: Sequence[DataCenterSnapshot],
        solution: np.ndarray,
        dc_count: int,
    ) -> tuple[AssignmentDecision, ...]:
        assignment_variables = len(tasks) * dc_count
        decisions: list[AssignmentDecision] = []
        for task_index, task in enumerate(tasks):
            assigned = [
                dc_index
                for dc_index in range(dc_count)
                if solution[task_index * dc_count + dc_index] > 0.5
            ]
            is_deferred = (
                self._config.allow_defer
                and solution[assignment_variables + task_index] > 0.5
            )
            if len(assigned) + int(is_deferred) != 1:
                raise RuntimeError(
                    f"求解器返回无效决策，任务 {task.task_id}"
                )
            if is_deferred:
                decisions.append(
                    AssignmentDecision(
                        task_id=task.task_id,
                        original_index=task.original_index,
                        decision="defer",
                    )
                )
            else:
                decisions.append(
                    AssignmentDecision(
                        task_id=task.task_id,
                        original_index=task.original_index,
                        decision="assign",
                        dc_id=datacenters[assigned[0]].dc_id,
                    )
                )
        return tuple(decisions)

    def _build_allocations(
        self,
        tasks: Sequence[TaskSnapshot],
        datacenters: Sequence[DataCenterSnapshot],
        assignments: Sequence[AssignmentDecision],
    ) -> tuple[DataCenterAllocation, ...]:
        task_by_key = {
            (task.original_index, task.task_id): task for task in tasks
        }
        totals = {
            dc.dc_id: [0.0, 0.0, 0.0, 0.0] for dc in datacenters
        }
        for assignment in assignments:
            if assignment.decision != "assign":
                continue
            task = task_by_key[(assignment.original_index, assignment.task_id)]
            value = totals[int(assignment.dc_id)]
            value[0] += 1
            value[1] += task.cpu_cores
            value[2] += task.gpu_units
            value[3] += task.memory_gb
        return tuple(
            DataCenterAllocation(
                dc_id=dc.dc_id,
                task_count=int(totals[dc.dc_id][0]),
                cpu_cores=totals[dc.dc_id][1],
                gpu_units=totals[dc.dc_id][2],
                memory_gb=totals[dc.dc_id][3],
            )
            for dc in datacenters
        )

    @staticmethod
    def _validate_allocations(
        datacenters: Sequence[DataCenterSnapshot],
        allocations: Sequence[DataCenterAllocation],
    ) -> None:
        dc_by_id = {dc.dc_id: dc for dc in datacenters}
        tolerance = 1e-7
        for allocation in allocations:
            dc = dc_by_id[allocation.dc_id]
            if allocation.cpu_cores > dc.cpu_schedulable_cores + tolerance:
                raise RuntimeError(f"CPU 容量超限，dc_id={dc.dc_id}")
            if allocation.gpu_units > dc.gpu_schedulable_units + tolerance:
                raise RuntimeError(f"GPU 容量超限，dc_id={dc.dc_id}")
            if allocation.memory_gb > dc.memory_schedulable_gb + tolerance:
                raise RuntimeError(f"内存容量超限，dc_id={dc.dc_id}")

    def _validate_state(
        self,
        state: SchedulerState,
        action_adapter: SustainClusterActionAdapter,
    ) -> None:
        if state.allow_defer != self._config.allow_defer:
            raise ValueError(
                "优化器 allow_defer 与调度状态不一致"
            )
        if action_adapter.mapping.allow_defer != self._config.allow_defer:
            raise ValueError(
                "优化器 allow_defer 与动作映射不一致"
            )
        dc_ids = tuple(dc.dc_id for dc in state.datacenters)
        if not dc_ids or len(set(dc_ids)) != len(dc_ids):
            raise ValueError("调度状态中的数据中心必须唯一")
        if set(action_adapter.mapping.dc_id_to_action) != set(dc_ids):
            raise ValueError("动作映射中的数据中心与调度状态不一致")
        if tuple(task.original_index for task in state.tasks) != tuple(
            range(len(state.tasks))
        ):
            raise ValueError("任务 original_index 必须连续且有序")
        expected_destinations = len(state.tasks) * len(state.datacenters)
        if len(state.task_destinations) != expected_destinations:
            raise ValueError(
                "调度状态的任务-目标网络数据不完整"
            )

    def _failed_result(
        self,
        state: SchedulerState,
        status: str,
        message: str,
        elapsed: float,
    ) -> OptimizationResult:
        return OptimizationResult(
            status=status,
            message=message,
            solve_seconds=elapsed,
            objective_value=None,
            costs=self._zero_costs(),
            assignments=(),
            environment_actions=(),
            datacenter_allocations=self._empty_allocations(state.datacenters),
        )

    @staticmethod
    def _empty_allocations(
        datacenters: Sequence[DataCenterSnapshot],
    ) -> tuple[DataCenterAllocation, ...]:
        return tuple(
            DataCenterAllocation(dc.dc_id, 0, 0.0, 0.0, 0.0)
            for dc in datacenters
        )

    @staticmethod
    def _zero_costs() -> CostBreakdown:
        return CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0)

    def _task_energy_kwh(self, task: TaskSnapshot) -> float:
        power_watts = (
            task.cpu_cores * self._config.cpu_power_w_per_core
            + task.gpu_units * self._config.gpu_power_w_per_unit
            + task.memory_gb * self._config.memory_power_w_per_gb
        )
        return power_watts / 1000.0 * task.remaining_duration_minutes / 60.0

    @staticmethod
    def _defer_sla_risk(task: TaskSnapshot, step_minutes: float) -> float:
        remaining = task.remaining_sla_minutes
        if remaining <= 0:
            return 1.0 + abs(remaining) / step_minutes
        urgency = step_minutes / max(remaining, step_minutes)
        projected_lateness = max(
            0.0,
            task.remaining_duration_minutes + step_minutes - remaining,
        ) / step_minutes
        return urgency + projected_lateness

    def _tie_break_vector(
        self, task_count: int, dc_count: int, variable_count: int
    ) -> np.ndarray:
        epsilon = self._config.deterministic_tie_break_epsilon
        values = np.zeros(variable_count, dtype=float)
        for task_index in range(task_count):
            for dc_index in range(dc_count):
                values[task_index * dc_count + dc_index] = epsilon * (
                    dc_index + 1
                )
            if self._config.allow_defer:
                values[task_count * dc_count + task_index] = epsilon * (
                    dc_count + 1
                )
        return values

    @staticmethod
    def _selected_component(component: np.ndarray, solution: np.ndarray) -> float:
        value = float(np.dot(component, np.rint(solution)))
        if abs(value) < 1e-12:
            return 0.0
        return value
