from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, milp

from sustaincluster_mpc.action_adapter import (
    AssignmentDecision,
    SustainClusterActionAdapter,
)
from sustaincluster_mpc.horizon_adapter import HorizonState
from sustaincluster_mpc.rolling_horizon_optimizer import (
    HorizonDataCenterAllocation,
    RollingCostBreakdown,
    RollingHorizonConfig,
    RollingHorizonOptimizer,
)


@dataclass(frozen=True)
class TerminalH60DataCenter:
    dc_id: int
    cpu_total_cores: float
    gpu_total_units: float
    memory_total_gb: float
    estimated_existing_cpu_cores: float
    estimated_existing_gpu_units: float
    estimated_existing_memory_gb: float
    arriving_cpu_demand: float
    arriving_gpu_demand: float
    arriving_memory_demand: float
    electricity_price_usd_per_mwh: float
    carbon_intensity_gco2_per_kwh: float
    source: str = "PRIVILEGED_ORACLE_T60"

    def __post_init__(self) -> None:
        if self.dc_id <= 0:
            raise ValueError("dc_id must be positive")
        for name in (
            "cpu_total_cores",
            "gpu_total_units",
            "memory_total_gb",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "estimated_existing_cpu_cores",
            "estimated_existing_gpu_units",
            "estimated_existing_memory_gb",
            "arriving_cpu_demand",
            "arriving_gpu_demand",
            "arriving_memory_demand",
            "carbon_intensity_gco2_per_kwh",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(float(self.electricity_price_usd_per_mwh)):
            raise ValueError("electricity_price_usd_per_mwh must be finite")

    @property
    def base_pressure(self) -> tuple[float, float, float]:
        return (
            (self.estimated_existing_cpu_cores + self.arriving_cpu_demand)
            / self.cpu_total_cores,
            (self.estimated_existing_gpu_units + self.arriving_gpu_demand)
            / self.gpu_total_units,
            (self.estimated_existing_memory_gb + self.arriving_memory_demand)
            / self.memory_total_gb,
        )


@dataclass(frozen=True)
class TerminalH60Weights:
    resource_pressure: float = 100.0
    electricity: float = 1.0
    carbon: float = 1.0
    terminal_backlog: float = 100.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"terminal weight {name} must be finite and nonnegative")


@dataclass(frozen=True)
class TerminalH60Config:
    target_offset_steps: int = 4
    lambda_terminal: float = 1.0
    weights: TerminalH60Weights = TerminalH60Weights()
    solver_time_limit_seconds: Optional[float] = 20.0

    def __post_init__(self) -> None:
        if self.target_offset_steps != 4:
            raise ValueError("MPC-H60 requires target_offset_steps=4 at 15 min/step")
        if not math.isfinite(float(self.lambda_terminal)) or self.lambda_terminal < 0:
            raise ValueError("lambda_terminal must be finite and nonnegative")
        if self.solver_time_limit_seconds is not None and (
            not math.isfinite(float(self.solver_time_limit_seconds))
            or self.solver_time_limit_seconds <= 0
        ):
            raise ValueError("solver_time_limit_seconds must be positive")


@dataclass(frozen=True)
class TerminalH60CostBreakdown:
    resource_pressure: float
    electricity: float
    carbon: float
    terminal_backlog: float

    @property
    def total(self) -> float:
        return sum(self.__dict__.values())


@dataclass(frozen=True)
class TerminalH60Projection:
    dc_id: int
    cpu_pressure: float
    gpu_pressure: float
    memory_pressure: float
    selected_active_task_count: int


@dataclass(frozen=True)
class TerminalH60Result:
    status: str
    message: str
    solve_seconds: float
    objective_value: float | None
    current_costs: RollingCostBreakdown
    terminal_costs: TerminalH60CostBreakdown
    terminal_contribution: float
    assignments: tuple[AssignmentDecision, ...]
    environment_actions: tuple[int, ...]
    datacenter_allocations: tuple[HorizonDataCenterAllocation, ...]
    terminal_projections: tuple[TerminalH60Projection, ...]
    presolve_retry: bool

    @property
    def feasible(self) -> bool:
        return self.status == "optimal"


class TerminalH60Optimizer:
    """Current-only MILP plus one explicit t+60 terminal-cost node."""

    def __init__(self) -> None:
        self._current_optimizer = RollingHorizonOptimizer()

    def solve(
        self,
        current_state: HorizonState,
        terminal_datacenters: Sequence[TerminalH60DataCenter],
        *,
        current_config: RollingHorizonConfig,
        terminal_config: TerminalH60Config,
        action_adapter: SustainClusterActionAdapter,
    ) -> TerminalH60Result:
        self._validate(
            current_state,
            terminal_datacenters,
            current_config,
            terminal_config,
            action_adapter,
        )
        if not current_state.current.tasks:
            return self._empty_result(current_state, terminal_datacenters)

        tasks = current_state.current.tasks
        dc_count = len(current_state.datacenters)
        task_count = len(tasks)
        variable_count = task_count * dc_count + task_count
        current_components, metadata = self._current_optimizer._component_vectors(
            current_state, current_config, variable_count
        )
        terminal_components = self._terminal_component_vectors(
            current_state,
            terminal_datacenters,
            terminal_config,
            current_config,
            metadata,
            variable_count,
        )
        objective = sum(current_components.values(), np.zeros(variable_count))
        objective += terminal_config.lambda_terminal * sum(
            terminal_components.values(), np.zeros(variable_count)
        )
        objective += self._current_optimizer._tie_break_vector(
            task_count,
            dc_count,
            1,
            variable_count,
            current_config,
        )
        upper = self._current_optimizer._variable_upper_bounds(
            current_state,
            current_config,
            metadata,
            variable_count,
        )
        constraints = self._current_optimizer._constraints(
            current_state, metadata, variable_count
        )

        raw, elapsed, retried = self._run_solver(
            objective,
            upper,
            constraints,
            terminal_config.solver_time_limit_seconds,
        )
        if raw.status != 0 or raw.x is None:
            status = {
                1: "limit_reached",
                2: "infeasible",
                3: "unbounded",
                4: "solver_error",
            }.get(int(raw.status), "solver_error")
            return self._failed_result(
                status,
                str(raw.message),
                elapsed,
                terminal_datacenters,
                retried,
            )

        solution = np.asarray(raw.x)
        plans = self._current_optimizer._decode_plans(
            current_state, metadata, solution
        )
        assignments = self._current_optimizer._first_step_decisions(plans)
        actions = action_adapter.encode_assignments(tasks, assignments)
        current_costs = self._current_optimizer._selected_costs(
            current_components, solution
        )
        terminal_costs = self._selected_terminal_costs(
            terminal_components, solution
        )
        allocations = self._current_optimizer._allocations(
            current_state, metadata, plans
        )
        self._current_optimizer._validate_allocations(
            current_state, allocations
        )
        projections = self._terminal_projections(
            current_state,
            terminal_datacenters,
            assignments,
            metadata,
        )
        terminal_contribution = terminal_config.lambda_terminal * terminal_costs.total
        return TerminalH60Result(
            status="optimal",
            message=str(raw.message),
            solve_seconds=elapsed,
            objective_value=current_costs.total + terminal_contribution,
            current_costs=current_costs,
            terminal_costs=terminal_costs,
            terminal_contribution=terminal_contribution,
            assignments=assignments,
            environment_actions=tuple(actions),
            datacenter_allocations=allocations,
            terminal_projections=projections,
            presolve_retry=retried,
        )

    @staticmethod
    def _run_solver(
        objective: np.ndarray,
        upper: np.ndarray,
        constraints: object,
        time_limit: float | None,
    ) -> tuple[object, float, bool]:
        options: dict[str, float | bool] = {"presolve": True}
        if time_limit is not None:
            options["time_limit"] = time_limit
        started = time.perf_counter()
        try:
            raw = milp(
                c=objective,
                integrality=np.ones(len(objective), dtype=np.int8),
                bounds=Bounds(np.zeros(len(objective)), upper),
                constraints=constraints,
                options=options,
            )
        except Exception as exc:
            class SolverError:
                status = 4
                x = None
                message = str(exc)

            return SolverError(), time.perf_counter() - started, False
        elapsed = time.perf_counter() - started
        if raw.status != 2:
            return raw, elapsed, False

        retry_options = dict(options)
        retry_options["presolve"] = False
        retry_started = time.perf_counter()
        try:
            retry = milp(
                c=objective,
                integrality=np.ones(len(objective), dtype=np.int8),
                bounds=Bounds(np.zeros(len(objective)), upper),
                constraints=constraints,
                options=retry_options,
            )
        except Exception:
            return raw, elapsed + time.perf_counter() - retry_started, True
        return retry, elapsed + time.perf_counter() - retry_started, True

    def _terminal_component_vectors(
        self,
        state: HorizonState,
        terminal_datacenters: Sequence[TerminalH60DataCenter],
        terminal_config: TerminalH60Config,
        current_config: RollingHorizonConfig,
        metadata: dict[str, object],
        variable_count: int,
    ) -> dict[str, np.ndarray]:
        vectors = {
            name: np.zeros(variable_count, dtype=float)
            for name in TerminalH60CostBreakdown.__dataclass_fields__
        }
        dc_count = len(state.datacenters)
        dispatch_variables = len(state.current.tasks) * dc_count
        terminal_by_id = {item.dc_id: item for item in terminal_datacenters}
        durations = metadata["duration_steps"]
        transfers = metadata["transfer_steps"]

        for task_index, task in enumerate(state.current.tasks):
            for dc_index, dc in enumerate(state.datacenters):
                variable = task_index * dc_count + dc_index
                execution_step = int(transfers[task_index, dc_index])
                active_at_target = (
                    execution_step <= terminal_config.target_offset_steps
                    < execution_step + int(durations[task_index])
                )
                if not active_at_target:
                    continue
                terminal = terminal_by_id[dc.dc_id]
                shares = (
                    task.cpu_cores / terminal.cpu_total_cores,
                    task.gpu_units / terminal.gpu_total_units,
                    task.memory_gb / terminal.memory_total_gb,
                )
                vectors["resource_pressure"][variable] = (
                    terminal_config.weights.resource_pressure
                    * sum(
                        share * (1.0 + pressure)
                        for share, pressure in zip(shares, terminal.base_pressure)
                    )
                )
                energy_kwh = self._terminal_slot_energy_kwh(
                    task.cpu_cores,
                    task.gpu_units,
                    task.memory_gb,
                    state.timestep_minutes,
                    current_config,
                )
                vectors["electricity"][variable] = (
                    terminal_config.weights.electricity
                    * energy_kwh
                    * terminal.electricity_price_usd_per_mwh
                    / 1000.0
                )
                vectors["carbon"][variable] = (
                    terminal_config.weights.carbon
                    * energy_kwh
                    * terminal.carbon_intensity_gco2_per_kwh
                    / 1000.0
                )
            vectors["terminal_backlog"][dispatch_variables + task_index] = (
                terminal_config.weights.terminal_backlog
            )
        return vectors

    @staticmethod
    def _terminal_slot_energy_kwh(
        cpu_cores: float,
        gpu_units: float,
        memory_gb: float,
        timestep_minutes: float,
        config: RollingHorizonConfig,
    ) -> float:
        watts = (
            cpu_cores * config.cpu_power_w_per_core
            + gpu_units * config.gpu_power_w_per_unit
            + memory_gb * config.memory_power_w_per_gb
        )
        return watts / 1000.0 * timestep_minutes / 60.0

    @staticmethod
    def _selected_terminal_costs(
        components: dict[str, np.ndarray],
        solution: np.ndarray,
    ) -> TerminalH60CostBreakdown:
        selected = np.rint(solution)
        return TerminalH60CostBreakdown(
            **{
                name: float(np.dot(vector, selected))
                for name, vector in components.items()
            }
        )

    @staticmethod
    def _terminal_projections(
        state: HorizonState,
        terminal_datacenters: Sequence[TerminalH60DataCenter],
        assignments: Sequence[AssignmentDecision],
        metadata: dict[str, object],
    ) -> tuple[TerminalH60Projection, ...]:
        task_by_key = {
            (task.original_index, task.task_id): task
            for task in state.current.tasks
        }
        dc_position = {
            dc.dc_id: index for index, dc in enumerate(state.datacenters)
        }
        durations = metadata["duration_steps"]
        transfers = metadata["transfer_steps"]
        added = {
            item.dc_id: np.zeros(3, dtype=float)
            for item in terminal_datacenters
        }
        counts = {item.dc_id: 0 for item in terminal_datacenters}
        for assignment in assignments:
            if assignment.decision != "assign":
                continue
            task = task_by_key[(assignment.original_index, assignment.task_id)]
            dc_id = int(assignment.dc_id)
            dc_index = dc_position[dc_id]
            execution_step = int(transfers[task.original_index, dc_index])
            if not (
                execution_step <= 4
                < execution_step + int(durations[task.original_index])
            ):
                continue
            added[dc_id] += np.asarray(
                [task.cpu_cores, task.gpu_units, task.memory_gb], dtype=float
            )
            counts[dc_id] += 1
        values = []
        for item in terminal_datacenters:
            base = np.asarray(
                [
                    item.estimated_existing_cpu_cores + item.arriving_cpu_demand,
                    item.estimated_existing_gpu_units + item.arriving_gpu_demand,
                    item.estimated_existing_memory_gb + item.arriving_memory_demand,
                ],
                dtype=float,
            )
            totals = np.asarray(
                [item.cpu_total_cores, item.gpu_total_units, item.memory_total_gb],
                dtype=float,
            )
            pressure = (base + added[item.dc_id]) / totals
            values.append(
                TerminalH60Projection(
                    dc_id=item.dc_id,
                    cpu_pressure=float(pressure[0]),
                    gpu_pressure=float(pressure[1]),
                    memory_pressure=float(pressure[2]),
                    selected_active_task_count=counts[item.dc_id],
                )
            )
        return tuple(values)

    def _validate(
        self,
        state: HorizonState,
        terminal_datacenters: Sequence[TerminalH60DataCenter],
        current_config: RollingHorizonConfig,
        terminal_config: TerminalH60Config,
        action_adapter: SustainClusterActionAdapter,
    ) -> None:
        if state.horizon != 1:
            raise ValueError("MPC-H60 current state must use exactly one current node")
        if not math.isclose(state.timestep_minutes, 15.0):
            raise ValueError("MPC-H60 requires a 15-minute timestep")
        self._current_optimizer._validate(
            state, current_config, action_adapter
        )
        state_ids = tuple(dc.dc_id for dc in state.datacenters)
        terminal_ids = tuple(item.dc_id for item in terminal_datacenters)
        if state_ids != terminal_ids:
            raise ValueError("terminal datacenters must match current state order")
        if len(set(terminal_ids)) != len(terminal_ids):
            raise ValueError("terminal datacenter ids must be unique")
        if terminal_config.target_offset_steps * state.timestep_minutes != 60.0:
            raise ValueError("terminal target must be exactly 60 minutes")

    @staticmethod
    def _zero_current_costs() -> RollingCostBreakdown:
        return RollingCostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def _zero_terminal_costs() -> TerminalH60CostBreakdown:
        return TerminalH60CostBreakdown(0.0, 0.0, 0.0, 0.0)

    def _empty_result(
        self,
        state: HorizonState,
        terminal_datacenters: Sequence[TerminalH60DataCenter],
    ) -> TerminalH60Result:
        projections = tuple(
            TerminalH60Projection(
                dc_id=item.dc_id,
                cpu_pressure=item.base_pressure[0],
                gpu_pressure=item.base_pressure[1],
                memory_pressure=item.base_pressure[2],
                selected_active_task_count=0,
            )
            for item in terminal_datacenters
        )
        return TerminalH60Result(
            status="optimal",
            message="No pending tasks",
            solve_seconds=0.0,
            objective_value=0.0,
            current_costs=self._zero_current_costs(),
            terminal_costs=self._zero_terminal_costs(),
            terminal_contribution=0.0,
            assignments=(),
            environment_actions=(),
            datacenter_allocations=(),
            terminal_projections=projections,
            presolve_retry=False,
        )

    def _failed_result(
        self,
        status: str,
        message: str,
        elapsed: float,
        terminal_datacenters: Sequence[TerminalH60DataCenter],
        retried: bool,
    ) -> TerminalH60Result:
        return TerminalH60Result(
            status=status,
            message=message,
            solve_seconds=elapsed,
            objective_value=None,
            current_costs=self._zero_current_costs(),
            terminal_costs=self._zero_terminal_costs(),
            terminal_contribution=0.0,
            assignments=(),
            environment_actions=(),
            datacenter_allocations=(),
            terminal_projections=tuple(
                TerminalH60Projection(
                    dc_id=item.dc_id,
                    cpu_pressure=item.base_pressure[0],
                    gpu_pressure=item.base_pressure[1],
                    memory_pressure=item.base_pressure[2],
                    selected_active_task_count=0,
                )
                for item in terminal_datacenters
            ),
            presolve_retry=retried,
        )
