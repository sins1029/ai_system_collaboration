from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from sustaincluster_mpc.action_adapter import (
    AssignmentDecision,
    SustainClusterActionAdapter,
)
from sustaincluster_mpc.horizon_adapter import (
    HorizonDataCenterSnapshot,
    HorizonState,
)
from sustaincluster_mpc.state_adapter import TaskSnapshot


@dataclass(frozen=True)
class RollingObjectiveWeights:
    electricity: float = 1.0
    carbon: float = 1.0
    transmission: float = 1.0
    waiting_defer: float = 1.0
    sla_risk: float = 20.0
    terminal_backlog: float = 50.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"weight {name} 必须为有限非负数")


@dataclass(frozen=True)
class RollingHorizonConfig:
    allow_defer: bool
    weights: RollingObjectiveWeights = RollingObjectiveWeights()
    cpu_power_w_per_core: float = 6.0
    gpu_power_w_per_unit: float = 500.0
    memory_power_w_per_gb: float = 2.5
    waiting_cost_per_step: float = 1.0
    terminal_backlog_base_cost: float = 1.0
    deterministic_tie_break_epsilon: float = 1e-9
    solver_time_limit_seconds: Optional[float] = 20.0

    def __post_init__(self) -> None:
        for name in (
            "cpu_power_w_per_core",
            "gpu_power_w_per_unit",
            "memory_power_w_per_gb",
            "waiting_cost_per_step",
            "terminal_backlog_base_cost",
            "deterministic_tie_break_epsilon",
        ):
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
class HorizonTaskPlan:
    task_id: str
    original_index: int
    decision: Literal["dispatch", "terminal_backlog"]
    destination_dc_id: int | None
    dispatch_step: int | None
    execution_start_step: int | None
    duration_steps: int
    deadline_step: int


@dataclass(frozen=True)
class RollingCostBreakdown:
    electricity: float
    carbon: float
    transmission: float
    waiting_defer: float
    sla_risk: float
    terminal_backlog: float

    @property
    def total(self) -> float:
        return sum(self.__dict__.values())


@dataclass(frozen=True)
class HorizonDataCenterAllocation:
    dc_id: int
    cpu_cores: tuple[float, ...]
    gpu_units: tuple[float, ...]
    memory_gb: tuple[float, ...]
    task_starts: tuple[int, ...]


@dataclass(frozen=True)
class RollingHorizonResult:
    status: str
    message: str
    solve_seconds: float
    objective_value: float | None
    costs: RollingCostBreakdown
    first_step_costs: RollingCostBreakdown
    plans: tuple[HorizonTaskPlan, ...]
    first_step_decisions: tuple[AssignmentDecision, ...]
    environment_actions: tuple[int, ...]
    datacenter_allocations: tuple[HorizonDataCenterAllocation, ...]
    integer_variable_count: int
    terminal_backlog_count: int
    projected_sla_violations: int

    @property
    def feasible(self) -> bool:
        return self.status == "optimal"


class RollingHorizonOptimizer:
    """求解滚动时域任务分配，并只向环境提交当前首步动作。"""
    def solve(
        self,
        horizon_state: HorizonState,
        config: RollingHorizonConfig,
        action_adapter: SustainClusterActionAdapter,
    ) -> RollingHorizonResult:
        self._validate(horizon_state, config, action_adapter)
        tasks = horizon_state.current.tasks
        datacenters = horizon_state.datacenters
        horizon = horizon_state.horizon
        if not tasks:
            return self._empty_result(horizon_state, action_adapter)

        task_count = len(tasks)
        dc_count = len(datacenters)
        dispatch_variables = task_count * dc_count * horizon
        variable_count = dispatch_variables + task_count
        components, metadata = self._component_vectors(
            horizon_state, config, variable_count
        )
        objective = sum(components.values(), np.zeros(variable_count))
        objective += self._tie_break_vector(
            task_count, dc_count, horizon, variable_count, config
        )
        lower = np.zeros(variable_count)
        upper = self._variable_upper_bounds(
            horizon_state, config, metadata, variable_count
        )
        constraints = self._constraints(
            horizon_state, metadata, variable_count
        )
        options: dict[str, float | bool] = {"presolve": True}
        if config.solver_time_limit_seconds is not None:
            options["time_limit"] = config.solver_time_limit_seconds

        started = time.perf_counter()
        try:
            result = milp(
                c=objective,
                integrality=np.ones(variable_count, dtype=np.int8),
                bounds=Bounds(lower, upper),
                constraints=constraints,
                options=options,
            )
        except Exception as exc:
            return self._failed_result(
                horizon_state,
                "solver_error",
                str(exc),
                time.perf_counter() - started,
                variable_count,
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
                horizon_state,
                status,
                str(result.message),
                elapsed,
                variable_count,
            )

        solution = np.asarray(result.x)
        plans = self._decode_plans(horizon_state, metadata, solution)
        # 只执行滚动计划的第一个时刻；下一环境步读取最新状态后重新求解。
        first_decisions = self._first_step_decisions(plans)
        actions = action_adapter.encode_assignments(tasks, first_decisions)
        costs = self._selected_costs(components, solution)
        first_costs = self._first_step_costs(
            horizon_state, config, plans, metadata
        )
        allocations = self._allocations(horizon_state, metadata, plans)
        self._validate_allocations(horizon_state, allocations)
        terminal_count = sum(
            plan.decision == "terminal_backlog" for plan in plans
        )
        projected_violations = self._projected_sla_violations(
            plans, horizon_state.horizon
        )
        return RollingHorizonResult(
            status="optimal",
            message=str(result.message),
            solve_seconds=elapsed,
            objective_value=costs.total,
            costs=costs,
            first_step_costs=first_costs,
            plans=plans,
            first_step_decisions=first_decisions,
            environment_actions=tuple(actions),
            datacenter_allocations=allocations,
            integer_variable_count=variable_count,
            terminal_backlog_count=terminal_count,
            projected_sla_violations=projected_violations,
        )

    def _component_vectors(
        self,
        state: HorizonState,
        config: RollingHorizonConfig,
        variable_count: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        tasks = state.current.tasks
        datacenters = state.datacenters
        horizon = state.horizon
        dc_count = len(datacenters)
        dispatch_variables = len(tasks) * dc_count * horizon
        components = {
            name: np.zeros(variable_count)
            for name in (
                "electricity",
                "carbon",
                "transmission",
                "waiting_defer",
                "sla_risk",
                "terminal_backlog",
            )
        }
        destination_data = {
            (item.original_index, item.destination_dc_id): item
            for item in state.current.task_destinations
        }
        duration_steps: list[int] = []
        deadline_steps: list[int] = []
        transfer_steps = np.zeros((len(tasks), dc_count), dtype=int)

        for task_index, task in enumerate(tasks):
            duration = self._ceil_steps(
                task.remaining_duration_minutes, state.timestep_minutes, 1
            )
            deadline = max(
                0,
                int(
                    math.floor(
                        task.remaining_sla_minutes / state.timestep_minutes
                    )
                ),
            )
            duration_steps.append(duration)
            deadline_steps.append(deadline)
            energy_kwh = self._task_energy_kwh(task, config)
            for dc_index, dc in enumerate(datacenters):
                destination = destination_data[(task.original_index, dc.dc_id)]
                transfer = max(
                    1,
                    self._ceil_steps(
                        destination.transmission_delay_seconds / 60.0,
                        state.timestep_minutes,
                        0,
                    ),
                )
                transfer_steps[task_index, dc_index] = transfer
                for dispatch_step in range(horizon):
                    variable = self._x_index(
                        task_index, dc_index, dispatch_step, dc_count, horizon
                    )
                    execution_step = dispatch_step + transfer
                    cost_step = min(execution_step, horizon - 1)
                    components["electricity"][variable] = (
                        config.weights.electricity
                        * energy_kwh
                        * dc.electricity_price_usd_per_mwh[cost_step]
                        / 1000.0
                    )
                    components["carbon"][variable] = (
                        config.weights.carbon
                        * energy_kwh
                        * dc.carbon_intensity_gco2_per_kwh[cost_step]
                        / 1000.0
                    )
                    components["transmission"][variable] = (
                        config.weights.transmission
                        * destination.transmission_cost_usd
                    )
                    components["waiting_defer"][variable] = (
                        config.weights.waiting_defer
                        * config.waiting_cost_per_step
                        * dispatch_step
                    )
                    slack = deadline - (execution_step + duration)
                    components["sla_risk"][variable] = (
                        config.weights.sla_risk / max(1.0, float(slack + 1))
                    )

            backlog_variable = dispatch_variables + task_index
            components["waiting_defer"][backlog_variable] = (
                config.weights.waiting_defer
                * config.waiting_cost_per_step
                * horizon
            )
            components["terminal_backlog"][backlog_variable] = (
                config.weights.terminal_backlog
                * config.terminal_backlog_base_cost
            )
            overdue_steps = max(0, horizon + duration - deadline)
            components["sla_risk"][backlog_variable] = (
                config.weights.sla_risk
                * (1.0 + overdue_steps / max(1, horizon))
            )

        return components, {
            "duration_steps": tuple(duration_steps),
            "deadline_steps": tuple(deadline_steps),
            "transfer_steps": transfer_steps,
        }

    def _variable_upper_bounds(
        self,
        state: HorizonState,
        config: RollingHorizonConfig,
        metadata: dict[str, object],
        variable_count: int,
    ) -> np.ndarray:
        tasks = state.current.tasks
        dc_count = len(state.datacenters)
        horizon = state.horizon
        dispatch_variables = len(tasks) * dc_count * horizon
        upper = np.ones(variable_count)
        durations = metadata["duration_steps"]
        deadlines = metadata["deadline_steps"]
        transfers = metadata["transfer_steps"]
        for task_index in range(len(tasks)):
            for dc_index in range(dc_count):
                for dispatch_step in range(horizon):
                    variable = self._x_index(
                        task_index, dc_index, dispatch_step, dc_count, horizon
                    )
                    execution_step = dispatch_step + int(
                        transfers[task_index, dc_index]
                    )
                    completion_step = (
                        execution_step
                        + int(durations[task_index])
                    )
                    if horizon > 1 and execution_step >= horizon:
                        upper[variable] = 0.0
                    if completion_step > int(deadlines[task_index]):
                        upper[variable] = 0.0
                    if not config.allow_defer and dispatch_step > 0:
                        upper[variable] = 0.0
            if not config.allow_defer:
                upper[dispatch_variables + task_index] = 0.0
        return upper

    def _constraints(
        self,
        state: HorizonState,
        metadata: dict[str, object],
        variable_count: int,
    ) -> tuple[LinearConstraint, LinearConstraint]:
        tasks = state.current.tasks
        datacenters = state.datacenters
        horizon = state.horizon
        task_count = len(tasks)
        dc_count = len(datacenters)
        dispatch_variables = task_count * dc_count * horizon
        durations = metadata["duration_steps"]
        transfers = metadata["transfer_steps"]

        once = lil_matrix((task_count, variable_count), dtype=float)
        for task_index in range(task_count):
            for dc_index in range(dc_count):
                for dispatch_step in range(horizon):
                    once[
                        task_index,
                        self._x_index(
                            task_index,
                            dc_index,
                            dispatch_step,
                            dc_count,
                            horizon,
                        ),
                    ] = 1.0
            once[task_index, dispatch_variables + task_index] = 1.0
        once_constraint = LinearConstraint(
            once.tocsr(), np.ones(task_count), np.ones(task_count)
        )

        capacity = lil_matrix((dc_count * horizon * 3, variable_count))
        upper = np.zeros(dc_count * horizon * 3)
        for dc_index, dc in enumerate(datacenters):
            for time_step in range(horizon):
                row = (dc_index * horizon + time_step) * 3
                upper[row] = dc.cpu_available_cores[time_step]
                upper[row + 1] = dc.gpu_available_units[time_step]
                upper[row + 2] = dc.memory_available_gb[time_step]
            for task_index, task in enumerate(tasks):
                duration = int(durations[task_index])
                transfer = int(transfers[task_index, dc_index])
                for dispatch_step in range(horizon):
                    execution_step = dispatch_step + transfer
                    effective_start = min(execution_step, horizon - 1)
                    effective_stop = min(horizon, effective_start + duration)
                    variable = self._x_index(
                        task_index,
                        dc_index,
                        dispatch_step,
                        dc_count,
                        horizon,
                    )
                    for time_step in range(effective_start, effective_stop):
                        row = (dc_index * horizon + time_step) * 3
                        capacity[row, variable] = task.cpu_cores
                        capacity[row + 1, variable] = task.gpu_units
                        capacity[row + 2, variable] = task.memory_gb
        capacity_constraint = LinearConstraint(
            capacity.tocsr(),
            np.full(len(upper), -np.inf),
            upper,
        )
        return once_constraint, capacity_constraint

    def _decode_plans(
        self,
        state: HorizonState,
        metadata: dict[str, object],
        solution: np.ndarray,
    ) -> tuple[HorizonTaskPlan, ...]:
        tasks = state.current.tasks
        dcs = state.datacenters
        horizon = state.horizon
        dc_count = len(dcs)
        dispatch_variables = len(tasks) * dc_count * horizon
        durations = metadata["duration_steps"]
        deadlines = metadata["deadline_steps"]
        transfers = metadata["transfer_steps"]
        plans: list[HorizonTaskPlan] = []
        for task_index, task in enumerate(tasks):
            selected: list[tuple[int, int]] = []
            for dc_index in range(dc_count):
                for dispatch_step in range(horizon):
                    variable = self._x_index(
                        task_index,
                        dc_index,
                        dispatch_step,
                        dc_count,
                        horizon,
                    )
                    if solution[variable] > 0.5:
                        selected.append((dc_index, dispatch_step))
            backlog = solution[dispatch_variables + task_index] > 0.5
            if len(selected) + int(backlog) != 1:
                raise RuntimeError(
                    f"滚动优化返回无效决策，任务 {task.task_id}"
                )
            if backlog:
                plans.append(
                    HorizonTaskPlan(
                        task_id=task.task_id,
                        original_index=task.original_index,
                        decision="terminal_backlog",
                        destination_dc_id=None,
                        dispatch_step=None,
                        execution_start_step=None,
                        duration_steps=int(durations[task_index]),
                        deadline_step=int(deadlines[task_index]),
                    )
                )
            else:
                dc_index, dispatch_step = selected[0]
                plans.append(
                    HorizonTaskPlan(
                        task_id=task.task_id,
                        original_index=task.original_index,
                        decision="dispatch",
                        destination_dc_id=dcs[dc_index].dc_id,
                        dispatch_step=dispatch_step,
                        execution_start_step=dispatch_step
                        + int(transfers[task_index, dc_index]),
                        duration_steps=int(durations[task_index]),
                        deadline_step=int(deadlines[task_index]),
                    )
                )
        return tuple(plans)

    @staticmethod
    def _first_step_decisions(
        plans: Sequence[HorizonTaskPlan],
    ) -> tuple[AssignmentDecision, ...]:
        values = []
        for plan in plans:
            if plan.decision == "dispatch" and plan.dispatch_step == 0:
                values.append(
                    AssignmentDecision(
                        task_id=plan.task_id,
                        original_index=plan.original_index,
                        decision="assign",
                        dc_id=plan.destination_dc_id,
                    )
                )
            else:
                values.append(
                    AssignmentDecision(
                        task_id=plan.task_id,
                        original_index=plan.original_index,
                        decision="defer",
                    )
                )
        return tuple(values)

    def _allocations(
        self,
        state: HorizonState,
        metadata: dict[str, object],
        plans: Sequence[HorizonTaskPlan],
    ) -> tuple[HorizonDataCenterAllocation, ...]:
        horizon = state.horizon
        tasks = {
            (task.original_index, task.task_id): task
            for task in state.current.tasks
        }
        cpu = {dc.dc_id: np.zeros(horizon) for dc in state.datacenters}
        gpu = {dc.dc_id: np.zeros(horizon) for dc in state.datacenters}
        memory = {dc.dc_id: np.zeros(horizon) for dc in state.datacenters}
        starts = {dc.dc_id: np.zeros(horizon, dtype=int) for dc in state.datacenters}
        for plan in plans:
            if plan.decision != "dispatch":
                continue
            dc_id = int(plan.destination_dc_id)
            task = tasks[(plan.original_index, plan.task_id)]
            effective_start = min(int(plan.execution_start_step), horizon - 1)
            stop = min(horizon, effective_start + plan.duration_steps)
            cpu[dc_id][effective_start:stop] += task.cpu_cores
            gpu[dc_id][effective_start:stop] += task.gpu_units
            memory[dc_id][effective_start:stop] += task.memory_gb
            starts[dc_id][int(plan.dispatch_step)] += 1
        return tuple(
            HorizonDataCenterAllocation(
                dc_id=dc.dc_id,
                cpu_cores=tuple(float(x) for x in cpu[dc.dc_id]),
                gpu_units=tuple(float(x) for x in gpu[dc.dc_id]),
                memory_gb=tuple(float(x) for x in memory[dc.dc_id]),
                task_starts=tuple(int(x) for x in starts[dc.dc_id]),
            )
            for dc in state.datacenters
        )

    @staticmethod
    def _validate_allocations(
        state: HorizonState,
        allocations: Sequence[HorizonDataCenterAllocation],
    ) -> None:
        dcs = {dc.dc_id: dc for dc in state.datacenters}
        tolerance = 1e-7
        for allocation in allocations:
            dc = dcs[allocation.dc_id]
            for step in range(state.horizon):
                if allocation.cpu_cores[step] > dc.cpu_available_cores[step] + tolerance:
                    raise RuntimeError(f"CPU 容量超限，dc_id={dc.dc_id}, h={step}")
                if allocation.gpu_units[step] > dc.gpu_available_units[step] + tolerance:
                    raise RuntimeError(f"GPU 容量超限，dc_id={dc.dc_id}, h={step}")
                if allocation.memory_gb[step] > dc.memory_available_gb[step] + tolerance:
                    raise RuntimeError(f"内存容量超限，dc_id={dc.dc_id}, h={step}")

    def _first_step_costs(
        self,
        state: HorizonState,
        config: RollingHorizonConfig,
        plans: Sequence[HorizonTaskPlan],
        metadata: dict[str, object],
    ) -> RollingCostBreakdown:
        tasks = {(task.original_index, task.task_id): task for task in state.current.tasks}
        dcs = {dc.dc_id: dc for dc in state.datacenters}
        destinations = {
            (item.original_index, item.destination_dc_id): item
            for item in state.current.task_destinations
        }
        values = {name: 0.0 for name in RollingCostBreakdown.__dataclass_fields__}
        for plan in plans:
            task = tasks[(plan.original_index, plan.task_id)]
            if plan.decision == "dispatch" and plan.dispatch_step == 0:
                dc = dcs[int(plan.destination_dc_id)]
                destination = destinations[(plan.original_index, dc.dc_id)]
                energy = self._task_energy_kwh(task, config)
                cost_step = min(int(plan.execution_start_step), state.horizon - 1)
                values["electricity"] += config.weights.electricity * energy * dc.electricity_price_usd_per_mwh[cost_step] / 1000.0
                values["carbon"] += config.weights.carbon * energy * dc.carbon_intensity_gco2_per_kwh[cost_step] / 1000.0
                values["transmission"] += config.weights.transmission * destination.transmission_cost_usd
                slack = plan.deadline_step - (int(plan.execution_start_step) + plan.duration_steps)
                values["sla_risk"] += config.weights.sla_risk / max(1.0, float(slack + 1))
            else:
                values["waiting_defer"] += config.weights.waiting_defer * config.waiting_cost_per_step
                urgency = state.timestep_minutes / max(
                    state.timestep_minutes,
                    max(0.0, task.remaining_sla_minutes),
                )
                values["sla_risk"] += config.weights.sla_risk * urgency
        return RollingCostBreakdown(**values)

    @staticmethod
    def _selected_costs(
        components: dict[str, np.ndarray], solution: np.ndarray
    ) -> RollingCostBreakdown:
        selected = np.rint(solution)
        return RollingCostBreakdown(
            **{
                name: float(np.dot(vector, selected))
                for name, vector in components.items()
            }
        )

    @staticmethod
    def _projected_sla_violations(
        plans: Sequence[HorizonTaskPlan], horizon: int
    ) -> int:
        count = 0
        for plan in plans:
            if plan.decision == "terminal_backlog":
                if horizon + plan.duration_steps > plan.deadline_step:
                    count += 1
            elif int(plan.execution_start_step) + plan.duration_steps > plan.deadline_step:
                count += 1
        return count

    @staticmethod
    def _validate(
        state: HorizonState,
        config: RollingHorizonConfig,
        action_adapter: SustainClusterActionAdapter,
    ) -> None:
        if state.current.allow_defer != config.allow_defer:
            raise ValueError("配置 allow_defer 与当前状态不一致")
        if action_adapter.mapping.allow_defer != config.allow_defer:
            raise ValueError("动作适配器延后设置与配置不一致")
        if tuple(task.original_index for task in state.current.tasks) != tuple(
            range(len(state.current.tasks))
        ):
            raise ValueError("当前任务的 original_index 必须连续")
        dc_ids = {dc.dc_id for dc in state.datacenters}
        if dc_ids != set(action_adapter.mapping.dc_id_to_action):
            raise ValueError("horizon 数据中心与动作映射不一致")
        for dc in state.datacenters:
            sequences = (
                dc.cpu_available_cores,
                dc.gpu_available_units,
                dc.memory_available_gb,
                dc.electricity_price_usd_per_mwh,
                dc.carbon_intensity_gco2_per_kwh,
            )
            if any(len(values) != state.horizon for values in sequences):
                raise ValueError(f"dc_id={dc.dc_id} 的 horizon 长度无效")

    def _empty_result(
        self,
        state: HorizonState,
        action_adapter: SustainClusterActionAdapter,
    ) -> RollingHorizonResult:
        actions = action_adapter.encode_assignments((), ())
        allocations = tuple(
            HorizonDataCenterAllocation(
                dc.dc_id,
                (0.0,) * state.horizon,
                (0.0,) * state.horizon,
                (0.0,) * state.horizon,
                (0,) * state.horizon,
            )
            for dc in state.datacenters
        )
        zero = self._zero_costs()
        return RollingHorizonResult(
            "optimal",
            "No pending tasks",
            0.0,
            0.0,
            zero,
            zero,
            (),
            (),
            tuple(actions),
            allocations,
            0,
            0,
            0,
        )

    def _failed_result(
        self,
        state: HorizonState,
        status: str,
        message: str,
        elapsed: float,
        variable_count: int,
    ) -> RollingHorizonResult:
        zero = self._zero_costs()
        return RollingHorizonResult(
            status=status,
            message=message,
            solve_seconds=elapsed,
            objective_value=None,
            costs=zero,
            first_step_costs=zero,
            plans=(),
            first_step_decisions=(),
            environment_actions=(),
            datacenter_allocations=(),
            integer_variable_count=variable_count,
            terminal_backlog_count=0,
            projected_sla_violations=0,
        )

    @staticmethod
    def _zero_costs() -> RollingCostBreakdown:
        return RollingCostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def _x_index(
        task: int, dc: int, step: int, dc_count: int, horizon: int
    ) -> int:
        return (task * dc_count + dc) * horizon + step

    @staticmethod
    def _ceil_steps(value_minutes: float, step_minutes: float, minimum: int) -> int:
        return max(minimum, int(math.ceil(max(0.0, value_minutes) / step_minutes)))

    @staticmethod
    def _task_energy_kwh(
        task: TaskSnapshot, config: RollingHorizonConfig
    ) -> float:
        watts = (
            task.cpu_cores * config.cpu_power_w_per_core
            + task.gpu_units * config.gpu_power_w_per_unit
            + task.memory_gb * config.memory_power_w_per_gb
        )
        return watts / 1000.0 * task.remaining_duration_minutes / 60.0

    @staticmethod
    def _tie_break_vector(
        task_count: int,
        dc_count: int,
        horizon: int,
        variable_count: int,
        config: RollingHorizonConfig,
    ) -> np.ndarray:
        values = np.zeros(variable_count)
        dispatch_variables = task_count * dc_count * horizon
        epsilon = config.deterministic_tie_break_epsilon
        for task in range(task_count):
            for dc in range(dc_count):
                for step in range(horizon):
                    values[(task * dc_count + dc) * horizon + step] = epsilon * (
                        step * dc_count + dc + 1
                    )
            values[dispatch_variables + task] = epsilon * (
                horizon * dc_count + 1
            )
        return values
