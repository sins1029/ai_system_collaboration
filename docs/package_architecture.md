# datacenter_env Package Architecture

## Responsibility

`datacenter_env` owns the internal behavior of one data center: plant state,
cooling control, thermal safety, actuator dynamics, measurement, controller-side
prediction, energy accounting, diagnostics, run metrics, and optional SQLite
persistence. It does not generate or download external signals, select an
experiment timeline, coordinate multiple data centers, expose a web service, or
implement Gymnasium.

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
`__version__`. Internal plant, actuator, and controller implementation classes
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

## Future Task Scheduling

`DataCenterAction` currently contains only `cooling_target_kw`. Future task
scheduling can add a separate scheduling contract for admitted/deferred tasks
and resource allocation, then compose it ahead of the current environment.
The plant and external-action interface do not need to be rewritten for that
extension.

## Compatibility

Legacy project-level modules remain available during migration. The deprecated
`experiments.mvp.simulate()` is now a forwarding adapter around
`DataCenterSystem`; it exists for old tests and callers and contains no plant,
actuator, accounting, diagnostic, or SQL formula. New code should use the
top-level package API. Once downstream imports have migrated, the legacy model
and controller modules can be removed in a later, separately reviewed change.
