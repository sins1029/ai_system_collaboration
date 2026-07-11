# Single-Center Thermal-Electric Database

## Model Chain

The current single-center model keeps the existing calculation chain:

```text
load -> IT power -> heat -> room temperature -> cooling power -> grid power
```

The core model equations were not rewritten. Each step now follows a closed
loop: strategy command, temperature feedback, feasible-action constraint,
thermal state update, and diagnostics.

## Temperature Semantics

- `min_temp_c` and `max_temp_c` are physical safety limits.
- `setpoint_temp_c` is the normal operating target.
- `deadband_c` suppresses unnecessary feedback changes close to the setpoint.
- `precool_floor_c` is the lowest control target allowed during price-aware
  pre-cooling and must be greater than or equal to `min_temp_c`.

The temperature state is not clipped. If no cooling action can satisfy the
physical limits, the run records `thermal_infeasible` and keeps the actual
model result visible.

## Scenario Difference

Scenario definitions live in `configs/experiment.yaml`.

- `baseline`
  - `dispatch_strategy`: `immediate`
  - `cooling_strategy`: `load_following`
  - Batch load is processed as it arrives.
  - Cooling combines heat-load feedforward with current-temperature feedback.

- `heuristic`
  - `dispatch_strategy`: `price_aware_temporal_shift`
  - `cooling_strategy`: `price_aware_precooling`
  - Batch load is shifted away from high-price periods when possible.
  - Cooling uses the same feedback controller and performs bounded pre-cooling
    only when a low-price step precedes a high-price window.

- `finite_horizon`
  - `dispatch_strategy`: `immediate`
  - `cooling_strategy`: `finite_horizon`
  - Cooling is re-optimized over a bounded forecast window at every step.

Both scenarios use the same simulation flow and differ only through explicit
strategy settings.

## Diagnostics

Step-level diagnostics include:

- `power_balance_error`
- `thermal_balance_error`
- `temperature_is_finite`
- `temperature_within_physical_range`
- `temperature_step_change`
- `below_min_temperature`
- `above_max_temperature`
- `temperature_violation`
- `thermal_infeasible`
- `cooling_constraint_intervention`
- `cooling_constraint_intervention_energy_kwh`
- `invalid_value_count`
- `negative_power_count`

Run-level diagnostic metrics include:

- `max_abs_power_balance_error`
- `mean_abs_power_balance_error`
- `max_abs_thermal_balance_error`
- `mean_abs_thermal_balance_error`
- `temperature_violation_count`
- `below_min_temperature_count`
- `above_max_temperature_count`
- `minimum_temperature_c`
- `mean_temperature_deviation_from_setpoint`
- `longest_below_min_streak`
- `longest_above_max_streak`
- `thermal_infeasibility_count`
- `cooling_constraint_intervention_count`
- `cooling_constraint_intervention_energy_kwh`
- `invalid_value_count`
- `negative_power_count`

The thermal balance residual is valid for this version because the model uses
an explicit one-step temperature update and stores the previous temperature,
heat, cooling, outdoor temperature, and step length needed to recompute it.

## Schema Migration

This is a development-stage SQLite database. `initialize_database()` creates
missing tables and also adds missing columns with `ALTER TABLE` for local
compatibility. Existing local rows remain readable, but older runs may have
NULL values for newly added diagnostic columns.

If a clean database is desired, delete `data/db/single_center.sqlite` and rerun:

```powershell
.\.venv311\Scripts\python.exe scripts\run_single_center.py
```

The database file is ignored by Git.
