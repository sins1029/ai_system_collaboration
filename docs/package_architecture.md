# datacenter_env Package Architecture

## Responsibility

`datacenter_env` owns the internal behavior of one data center: plant state,
cooling control, thermal safety, actuator dynamics, measurement, controller-side
prediction, energy accounting, diagnostics, run metrics, and optional SQLite
persistence. It does not generate or download external signals, select an
experiment timeline, coordinate multiple data centers, expose a web service, or
implement Gymnasium. It does own the internal single-center task queue,
resource accounting, task SLA evaluation and scheduler protocol.

## Package Layout

```text
src/datacenter_env/
|-- __init__.py
|-- api.py
|-- config.py
|-- exceptions.py
|-- version.py
|-- contracts/
|   |-- actions.py
|   |-- inputs.py
|   |-- observations.py
|   |-- protocols.py
|   `-- results.py
|   `-- tasks.py
|-- core/
|   |-- actuator.py
|   |-- environment.py
|   |-- measurement.py
|   |-- plant.py
|   |-- prediction.py
|   |-- primitives.py
|   |-- safety.py
|   `-- state.py
|-- control/
|   |-- controllers.py
|   `-- factory.py
|-- evaluation/
|   |-- metrics.py
|   `-- records.py
|-- tasking/
|   |-- aggregation.py
|   |-- resources.py
|   |-- runtime.py
|   `-- schedulers.py
|-- gym/
|   |-- action.py
|   |-- adapters.py
|   |-- config.py
|   |-- environment.py
|   |-- masks.py
|   |-- observation.py
|   |-- policies.py
|   |-- registration.py
|   `-- rewards.py
|-- storage/
|   |-- database.py
|   |-- schema.sql
|   `-- stores.py
`-- runtime/
    |-- runner.py
    `-- system.py
```

## Public API

The package root exposes `DataCenterSystem`, `DataCenterEnvironment`,
`DataCenterSystemConfig`, `ExogenousInput`, `ForecastWindow`,
`DataCenterAction`, `DataCenterObservation`, `StepResult`, `RunMetadata`,
`RunSummary`, `SQLiteRunStore`, `NullRunStore`, `run_single_center`, and
`TaskSpec`, `TaskArrivalBatch`, `TaskStatus`, `TaskSchedulingDecision`,
`TaskSchedulingObservation`, `TaskStepResult`, `TaskEvent`, `TaskScheduler`, and
`SingleCenterTaskSchedulingEnv`, its three configuration/reward objects,
`MaskedRandomPolicy`, `register_gym_environments`, and `__version__`. Internal
plant, actuator, controller, encoder and adapter implementation classes
are intentionally not re-exported.

## External Signal Boundary

`external_signals/` owns full timelines, CSV loading, synthetic generation,
alignment, workload preparation, and forecast-window slicing. The installed
package receives immutable `ExogenousInput` values and bounded
`ForecastWindow` tuples only. It does not import the external provider modules.

The first forecast step must match the current input timestamp. Timestamps in a
window must be strictly increasing, and the system rejects windows longer than
the active controller permits. Baseline receives one current step; heuristic
and finite-horizon controllers receive only their configured lookahead.

## Single-Step Chain

```text
DataCenterSystem
  -> controller.act(observation, bounded forecast)
  -> proposed cooling
  -> thermal safety constraint
  -> actuator-aware target
  -> applied cooling
  -> true plant state update
  -> temperature measurement
  -> energy/cost/carbon accounting
  -> balance and safety diagnostics
  -> optional RunStore append
```

`DataCenterEnvironment` implements the chain from external action onward and
has no controller or database dependency. This is the intended future
Gymnasium/RL integration point.

## Storage Chain

`DataCenterSystem` alone coordinates persistence. Plant, actuator, controller,
safety, accounting, and diagnostics never execute SQL.

`RunStore` defines initialize, create, append input, append step, finish, and
fail operations. `SQLiteRunStore` writes the legacy-compatible tables plus
`run_input_timeseries` and `schema_versions`. Version 2 migrations add run
status, completion time, package version, and error information without
invalidating existing SQLite files. `NullRunStore` provides the same lifecycle
without persistence.

A run starts as `running`, becomes `completed` only after metrics are stored,
and becomes `failed` if a runtime or storage operation raises. Failed appends
are never silently swallowed.

## Configuration

`DataCenterSystemConfig` groups physical configuration, optimization settings,
controller name, condition name, step duration, actuator model, plant
parameters, prediction parameters, measurement model, mismatch label, and an
optional database path. Nested mappings are deep-copied and frozen. Units remain
explicit in field names or in the existing configuration documentation.

External signal configuration such as `configs/signals.yaml` remains outside
the package because it determines how inputs are generated rather than how a
data center evolves.

## Controller Contract

Baseline, heuristic, and finite-horizon controllers implement the same
`reset()` and `act(observation, forecast_window)` protocol. Controllers see
measured temperature, previous applied cooling, and external inputs. They do
not see true temperature, plant objects, database connections, or unbounded
future arrays.

The finite-horizon controller retains the independent prediction model,
actuator prediction, rolling first-action execution, objective decomposition,
and fallback behavior.

## Reset And Isolation

Each environment owns its plant configuration, actuator state, delay FIFO,
measurement RNG, temperatures, previous action, step index, and timestamp.
Each system owns its controller, metric aggregator, run handle, and store.
`reset(seed)` rebuilds stateful components. No mutable runtime singleton is
used, so resetting one instance cannot affect another.

## Task Scheduling

`DataCenterAction` composes an optional task scheduling decision with the
cooling target. The `legacy_aggregate` mode leaves it empty. In `task_queue`
mode, arrivals are ingested before scheduling, running resource usage is
aggregated into the plant workload, and tasks advance after the physical step.
Generation and complete task timelines remain in `external_workloads/`. See
`task_runtime_architecture.md` for the detailed contract.

## Gymnasium Adapter

The Gym layer owns fixed numeric encoding, sequential binary interaction,
action masks, reward composition and episode boundaries. It accumulates binary
decisions and calls `DataCenterSystem.step_with_task_decision()` once per
physical interval. Provider factories remain external and are rebuilt on every
reset. See `gymnasium_environment.md` for the complete interface.

## Compatibility

Legacy project-level modules remain available during migration. The deprecated
`experiments.mvp.simulate()` is now a forwarding adapter around
`DataCenterSystem`; it exists for old tests and callers and contains no plant,
actuator, accounting, diagnostic, or SQL formula. New code should use the
top-level package API. Once downstream imports have migrated, the legacy model
and controller modules can be removed in a later, separately reviewed change.
