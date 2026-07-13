# Single-Center Task Runtime Architecture

## Scope And Boundary

Version 0.2 adds a non-preemptive task runtime ahead of the existing
thermal-electric plant. `datacenter_env` owns queue state, resource accounting,
scheduler contracts, lifecycle and SLA evaluation. `external_workloads` owns
task generation, CSV loading, full timelines and arrival slicing. The package
never imports that external layer and never receives tasks whose arrival time is
after the current step.

## Step Order

Each `task_queue` step follows one deterministic sequence:

1. Validate the current timestamp and zero external aggregate workload.
2. Ingest exactly one `TaskArrivalBatch` for the timestamp.
3. Mark impossible resource requests `UNSCHEDULABLE`, or raise by policy.
4. Build an immutable scheduler observation.
5. Validate and apply the scheduler decision against one resource ledger.
6. Aggregate running CPU, GPU and memory utilization into `workload_fraction`.
7. Run the existing cooling controller and unchanged thermal-electric plant.
8. Execute every running task for one interval and complete duration-one tasks.
9. Release completed resources, evaluate SLA and emit task events.
10. Persist input, physical result and task events in one SQLite transaction.

Arrivals at time `t` may execute during `[t, t + dt)`. A duration-one task that
starts at `t` completes at `t + dt`. Running tasks cannot be preempted and retain
fixed resources until completion.

## Workload Modes

`legacy_aggregate` is the default and preserves the v0.1 path. It consumes
`ExogenousInput.workload_fraction`, creates no task runtime and returns
`StepResult.tasking=None`.

`task_queue` requires a `TaskArrivalBatch` every step. External
`workload_fraction` must be zero. The plant input is generated only from current
resource usage, which prevents callers from accidentally combining aggregate
and task-derived load.

## Resources And Aggregation

CPU cores, GPU units and memory GB share a central resource pool. Allocation is
validated against total and currently used capacity before any task starts.

The standard experiment uses:

```text
workload = clip(0.45 * cpu_util + 0.45 * gpu_util + 0.10 * memory_util, 0, 1)
```

These normalized weights are a reproducible proxy for driving the existing IT
power curve; they do not claim device-level physical power precision. A
`max_utilization` aggregation mode is also available.

## Schedulers

- `fifo_immediate`: arrival time then task id, packing every feasible task.
- `earliest_deadline_first`: deadline, descending priority, arrival, task id.
- `energy_aware_deferral`: starts non-deferrable and latest-start tasks, and may
  delay only deferrable work when a bounded environmental forecast has a lower
  price/carbon/renewable score.

Schedulers see immutable task views, resource totals/usage, current environment
and a bounded `ForecastWindow`. They do not see the plant, database, full task
timeline or future task arrivals.

Resource infeasibility and active deferral are distinct. Every task left waiting
at physical-step end gains one wait step. A deferrable task explicitly postponed
while feasible gains a deferral; a task that cannot fit current availability
instead gains a resource-blocked count/event. Non-deferrable and latest-start
tasks remain subject to the same physical capacity and are never over-allocated.

## SLA Semantics

The latest start is `deadline - duration_steps * dt`. A waiting task is at risk
after current time exceeds that boundary. Final SLA compliance is based only on
`completion_time <= deadline_time`. Tasks still waiting or running when a run
ends are stored with their real status and counted as unfinished; they are never
marked completed synthetically.

## Persistence

Schema v3 adds `tasks`, `run_task_events` and `run_task_outcomes`, plus task
columns on runs and simulation results. Existing databases migrate in place.
Each step stores the external input, thermal-electric result, task utilization
and task events atomically. Run outcomes are finalized only when the run is
completed. `NullRunStore` implements the same lifecycle without I/O.

## Gymnasium Boundary

Version 0.3 wraps this runtime through a sequential decision adapter. It does
not alter task execution or physical formulas: multiple binary decisions are
accumulated and submitted to this runtime as one scheduling decision before the
single physical step. Gym-specific observations and rewards are documented in
`gymnasium_environment.md`.
