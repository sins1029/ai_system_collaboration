# Single-Center Compute-Power Coordination

## Exogenous Signal Audit

The original `data/sample/mvp_24h.csv` was a fixed synthetic sequence with
naive timestamps. All five signal families existed and were stored in
`input_timeseries`, but carbon intensity was stored as gCO2/kWh and divided by
1000 in the metric layer, while renewable output was stored as a normalized
fraction and multiplied by 100 in the simulation layer.

The main experiment now uses `ExogenousSignals` and the generated
`data/timeseries/single_center_24h.csv` dataset:

| Signal | Canonical unit | Model use | Control use |
|---|---|---|---|
| online and batch workload | dimensionless capacity fraction | IT power and heat | dispatch |
| electricity price | currency/kWh | energy cost | heuristic and finite horizon |
| carbon intensity | kgCO2/kWh | carbon emissions | finite horizon |
| outdoor temperature | degC | thermal state and cooling COP | all cooling controllers |
| renewable power | kW | grid power balance | finite-horizon objective forecast |

All timestamps are timezone-aware, strictly increasing, unique, and exactly
15 minutes apart. The generator seed, parameters, units, range, and step count
are stored in dataset metadata.

## Energy Accounting

```text
P_dc = P_IT + P_cool + P_aux
P_renewable_used = min(P_renewable_available, P_dc)
P_grid = P_dc - P_renewable_used
P_renewable_curtailed = P_renewable_available - P_renewable_used
```

There is no export and no storage. Cost and carbon are both calculated from
grid energy, so renewable power is subtracted exactly once in the power
balance. A per-step residual verifies available = used + curtailed.

## Strategy Definitions

- `baseline`: immediate batch service and current-state heat feedforward plus
  temperature feedback. It receives one current signal row and no future data.
- `heuristic`: bounded-lookahead price-aware batch dispatch and constrained
  pre-cooling. Price thresholds and lookahead rules are explicit in config.
- `finite_horizon`: immediate batch service plus an eight-step receding-horizon
  cooling optimizer. It receives only the current bounded `SignalWindow`.

## Finite-Horizon Problem

State: room temperature and previous cooling command.

Control: cooling command in kW of thermal removal.

Forecast window: configurable `horizon_steps`, currently 8 steps or 2 hours.

The weighted objective is:

```text
J = objective_energy_cost
  + objective_carbon
  + objective_temperature
  + objective_control_movement
```

The components use currency, weighted kgCO2, weighted squared degC deviation,
and weighted squared normalized cooling movement. Every component and the
total horizon objective are stored for each solve.

Constraints enforce cooling capacity and predicted temperature between the
configured MPC control floor and physical maximum. The MPC floor is validated
inside `[T_min, T_setpoint]`.

The candidate/beam search optimizes the available horizon, executes only the
first action, advances the plant once, and solves again. Solver failure invokes
the baseline feedback controller and records failure and fallback counters.

## Plant and Prediction Boundary

`DatacenterModel.plant_step()` advances the actual experiment. The optimizer
uses `ThermalPredictionModel.predict_step()` and never calls future plant
steps. Prediction capacity, heat transfer, and cooling effectiveness scales
are independently configurable and default to 1.0.

## Database Compatibility

Canonical dataset metadata and signal/result columns are added through the
existing automatic `ALTER TABLE ADD COLUMN` migration. Legacy columns remain
populated for older local tools, while new code uses canonical units. Existing
runs retain NULL in newly introduced columns and remain readable.
