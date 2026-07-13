from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from datacenter_env.contracts import ExogenousInput
from datacenter_env.contracts.tasks import (
    EnvironmentalView,
    ResourceCapacity,
    RunningTaskView,
    TaskArrivalBatch,
    TaskEvent,
    TaskEventType,
    TaskOutcome,
    TaskSchedulingDecision,
    TaskSchedulingObservation,
    TaskSpec,
    TaskStatus,
    TaskStepResult,
    TaskView,
)
from datacenter_env.exceptions import InputValidationError, SchedulingError, UnschedulableTaskError
from datacenter_env.tasking.aggregation import TaskLoadAggregator
from datacenter_env.tasking.resources import ResourcePool


@dataclass
class TaskRuntimeState:
    spec: TaskSpec
    status: TaskStatus = TaskStatus.WAITING
    remaining_steps: int = 0
    first_start_time: datetime | None = None
    completion_time: datetime | None = None
    wait_steps: int = 0
    deferral_count: int = 0
    sla_violated: bool = False
    resource_blocked_count: int = 0

    def __post_init__(self) -> None:
        if self.remaining_steps == 0:
            self.remaining_steps = self.spec.duration_steps


class TaskRuntime:
    def __init__(self, config):
        capacity = ResourceCapacity(
            float(config.capacity["total_cpu_cores"]),
            float(config.capacity["total_gpu_units"]),
            float(config.capacity["total_memory_gb"]),
        )
        aggregation = config.load_aggregation
        self.step_minutes = config.step_minutes
        self.unschedulable_policy = str(config.tasks.get("unschedulable_policy", "record"))
        self.pool = ResourcePool(capacity)
        self.aggregator = TaskLoadAggregator(
            capacity,
            mode=str(aggregation.get("mode", "weighted_sum")),
            cpu_weight=float(aggregation.get("cpu_weight", 0.45)),
            gpu_weight=float(aggregation.get("gpu_weight", 0.45)),
            memory_weight=float(aggregation.get("memory_weight", 0.10)),
        )
        self.reset()

    def reset(self) -> None:
        self.pool.reset()
        self.states: dict[str, TaskRuntimeState] = {}
        self._prepared_timestamp: datetime | None = None
        self._decision_applied = False
        self._events: list[TaskEvent] = []
        self._arrived: list[TaskSpec] = []
        self._started: list[str] = []
        self._unschedulable: list[str] = []
        self._resource_blocked: list[str] = []
        self._forced_resource_blocked: list[str] = []
        self._interval_usage = self.pool.usage
        self._interval_workload = 0.0

    def begin_step(
        self, timestamp: datetime, arrivals: TaskArrivalBatch, current_input: ExogenousInput
    ) -> TaskSchedulingObservation:
        if self._prepared_timestamp is not None:
            if self._prepared_timestamp != timestamp:
                raise SchedulingError("previous task step has not been completed")
            return self.observation(timestamp, current_input)
        if arrivals.timestamp != timestamp:
            raise InputValidationError("task arrival batch must match the current step timestamp")
        self._prepared_timestamp = timestamp
        self._decision_applied = False
        self._events = []
        self._arrived = []
        self._started = []
        self._unschedulable = []
        self._resource_blocked = []
        self._forced_resource_blocked = []
        for task in arrivals.tasks:
            if task.task_id in self.states:
                raise InputValidationError(f"task id has already arrived: {task.task_id}")
            state = TaskRuntimeState(task)
            self.states[task.task_id] = state
            self._arrived.append(task)
            self._events.append(TaskEvent(timestamp, task.task_id, TaskEventType.ARRIVED))
            if not self.pool.task_fits_capacity(task):
                if self.unschedulable_policy == "raise":
                    raise UnschedulableTaskError(
                        f"task {task.task_id} exceeds data-center resource capacity"
                    )
                state.status = TaskStatus.UNSCHEDULABLE
                self._unschedulable.append(task.task_id)
                self._events.append(
                    TaskEvent(
                        timestamp,
                        task.task_id,
                        TaskEventType.UNSCHEDULABLE,
                        {"reason": "resource demand exceeds total capacity"},
                    )
                )
        return self.observation(timestamp, current_input)

    def observation(
        self, timestamp: datetime, current_input: ExogenousInput
    ) -> TaskSchedulingObservation:
        waiting = tuple(
            self._view(state, timestamp)
            for state in self.states.values()
            if state.status is TaskStatus.WAITING
        )
        running = tuple(
            self._running_view(state, timestamp)
            for state in self.states.values()
            if state.status is TaskStatus.RUNNING
        )
        return TaskSchedulingObservation(
            timestamp=timestamp,
            waiting_tasks=waiting,
            running_tasks=running,
            total_resources=self.pool.capacity,
            used_resources=self.pool.usage,
            available_resources=self.pool.availability,
            environment=EnvironmentalView(
                current_input.electricity_price_per_kwh,
                current_input.carbon_intensity_kg_per_kwh,
                current_input.outdoor_temperature_c,
                current_input.renewable_power_kw,
            ),
        )

    def apply_decision(self, timestamp: datetime, decision: TaskSchedulingDecision) -> float:
        if self._prepared_timestamp != timestamp:
            raise SchedulingError("task step must be prepared before applying a decision")
        if self._decision_applied:
            raise SchedulingError("task scheduling decision was already applied")
        waiting_ids = {
            task_id for task_id, state in self.states.items() if state.status is TaskStatus.WAITING
        }
        unknown = tuple(task_id for task_id in decision.start_task_ids if task_id not in waiting_ids)
        if unknown:
            raise SchedulingError(f"decision may only start waiting tasks: {unknown}")
        unknown_deferred = tuple(
            task_id for task_id in decision.deferred_task_ids if task_id not in waiting_ids
        )
        if unknown_deferred:
            raise SchedulingError(
                f"decision may only defer waiting tasks: {unknown_deferred}"
            )
        for task_id in decision.start_task_ids:
            state = self.states[task_id]
            if not self.pool.can_allocate(state.spec):
                raise SchedulingError(f"decision over-allocates resources at task {task_id}")
            self.pool.allocate(state.spec)
            state.status = TaskStatus.RUNNING
            state.first_start_time = timestamp
            self._started.append(task_id)
            self._events.append(TaskEvent(timestamp, task_id, TaskEventType.STARTED))
        explicitly_deferred = set(decision.deferred_task_ids)
        for task_id in sorted(waiting_ids - set(decision.start_task_ids)):
            state = self.states[task_id]
            state.wait_steps += 1
            resource_blocked = not self.pool.can_allocate(state.spec)
            if resource_blocked:
                state.resource_blocked_count += 1
                self._resource_blocked.append(task_id)
                event_type = (
                    TaskEventType.RESOURCE_BLOCKED
                    if state.spec.deferrable
                    and timestamp < self.latest_start_time(state.spec)
                    else TaskEventType.FORCED_RESOURCE_BLOCKED
                )
                if event_type is TaskEventType.FORCED_RESOURCE_BLOCKED:
                    self._forced_resource_blocked.append(task_id)
                self._events.append(TaskEvent(timestamp, task_id, event_type))
            elif state.spec.deferrable and (
                task_id in explicitly_deferred or not decision.classification_complete
            ):
                state.deferral_count += 1
                self._events.append(TaskEvent(timestamp, task_id, TaskEventType.DEFERRED))
        self._decision_applied = True
        self._interval_usage = self.pool.usage
        self._interval_workload = self.aggregator.aggregate(self._interval_usage)
        return self._interval_workload

    def finish_step(self, timestamp: datetime) -> TaskStepResult:
        if self._prepared_timestamp != timestamp or not self._decision_applied:
            raise SchedulingError("task decision must be applied before completing a step")
        completion_time = timestamp + timedelta(minutes=self.step_minutes)
        completed: list[str] = []
        violated: list[str] = []
        for task_id, state in self.states.items():
            if state.status is not TaskStatus.RUNNING:
                continue
            state.remaining_steps -= 1
            if state.remaining_steps > 0:
                continue
            state.completion_time = completion_time
            state.sla_violated = completion_time > state.spec.deadline_time
            state.status = (
                TaskStatus.SLA_VIOLATED if state.sla_violated else TaskStatus.COMPLETED
            )
            self.pool.release(state.spec)
            completed.append(task_id)
            self._events.append(
                TaskEvent(completion_time, task_id, TaskEventType.COMPLETED)
            )
            if state.sla_violated:
                violated.append(task_id)
                self._events.append(
                    TaskEvent(completion_time, task_id, TaskEventType.SLA_VIOLATED)
                )
        cpu, gpu, memory = self.aggregator.utilizations(self._interval_usage)
        result = TaskStepResult(
            arrived_task_ids=tuple(task.task_id for task in self._arrived),
            started_task_ids=tuple(self._started),
            completed_task_ids=tuple(completed),
            newly_sla_violated_task_ids=tuple(violated),
            unschedulable_task_ids=tuple(self._unschedulable),
            resource_blocked_task_ids=tuple(self._resource_blocked),
            forced_resource_blocked_task_ids=tuple(self._forced_resource_blocked),
            waiting_count=self.count(TaskStatus.WAITING),
            running_count=self.count(TaskStatus.RUNNING),
            completed_count=self.count(TaskStatus.COMPLETED) + self.count(TaskStatus.SLA_VIOLATED),
            at_risk_count=self.at_risk_count(completion_time),
            resource_usage=self._interval_usage,
            cpu_utilization=cpu,
            gpu_utilization=gpu,
            memory_utilization=memory,
            workload_fraction=self._interval_workload,
            events=tuple(self._events),
            arrived_tasks=tuple(self._arrived),
        )
        self._prepared_timestamp = None
        self._decision_applied = False
        return result

    @property
    def current_workload_fraction(self) -> float:
        return self._interval_workload

    @property
    def prepared_timestamp(self) -> datetime | None:
        return self._prepared_timestamp

    @property
    def decision_applied(self) -> bool:
        return self._decision_applied

    def count(self, status: TaskStatus) -> int:
        return sum(state.status is status for state in self.states.values())

    def at_risk_count(self, timestamp: datetime) -> int:
        return sum(
            state.status is TaskStatus.WAITING
            and timestamp > self.latest_start_time(state.spec)
            for state in self.states.values()
        )

    def outcomes(self) -> tuple[TaskOutcome, ...]:
        values: list[TaskOutcome] = []
        for state in self.states.values():
            final_status = state.status
            lateness = None
            if state.completion_time is not None:
                lateness = max(
                    0.0,
                    (state.completion_time - state.spec.deadline_time).total_seconds() / 60.0,
                )
            values.append(
                TaskOutcome(
                    state.spec.task_id,
                    final_status,
                    state.first_start_time,
                    state.completion_time,
                    state.wait_steps,
                    state.deferral_count,
                    state.sla_violated,
                    lateness,
                    state.resource_blocked_count,
                )
            )
        return tuple(values)

    def latest_start_time(self, spec: TaskSpec) -> datetime:
        return spec.deadline_time - timedelta(minutes=self.step_minutes * spec.duration_steps)

    def _view(self, state: TaskRuntimeState, timestamp: datetime) -> TaskView:
        del timestamp
        return TaskView(
            state.spec,
            state.status,
            state.remaining_steps,
            state.wait_steps,
            state.deferral_count,
            self.latest_start_time(state.spec),
        )

    def _running_view(self, state: TaskRuntimeState, timestamp: datetime) -> RunningTaskView:
        view = self._view(state, timestamp)
        if state.first_start_time is None:
            raise SchedulingError("running task is missing first_start_time")
        return RunningTaskView(
            view.spec,
            view.status,
            view.remaining_steps,
            view.wait_steps,
            view.deferral_count,
            view.latest_start_time,
            state.first_start_time,
        )
