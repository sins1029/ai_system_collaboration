# Single-Center Closed-Loop Robustness

## Action Semantics

The control path now records three distinct thermal-removal powers in kW:

```text
proposed_cooling_kw
  -> constrained_cooling_kw
  -> applied_cooling_kw
  -> plant_step
```

The proposed action comes from the controller. The constrained action includes
capacity, temperature safety, and actuator-aware feasibility adjustments. The
applied action is the actuator output used by the plant. Legacy
`cooling_command_kw` and `cooling_kw` remain as proposed and applied aliases.

## Actuator Models

- `ideal`: applied equals constrained.
- `rate_limited`: asymmetric per-step ramp-up and ramp-down clipping.
- `first_order`: exact zero-order-hold discretization with
  `alpha = 1 - exp(-step_minutes / time_constant_minutes)`.
- `delayed`: explicit FIFO action queue with configurable integer delay steps.

Tracking error is `abs(constrained - applied)`. Ramp counts distinguish upward
and downward limitation. The safety layer compares the actuator reachable set
with the thermally feasible applied-action interval and records actuator-caused
infeasibility separately from optimizer failure.

## Plant, Prediction, and Measurement

Plant and prediction parameter dictionaries are deep-copied and persisted
separately. Plant scales are used only by `plant_step`; prediction scales are
used only by `ThermalPredictionModel`.

The controller receives measured temperature. The plant advances true
temperature, while diagnostics can access both. Measurement supports `none`
and seeded Gaussian temperature noise.

One-step error uses the signed definition:

```text
prediction_error = predicted_next_temperature - actual_next_temperature
```

Run metrics include mean absolute error, maximum absolute error, RMSE, and
signed bias.

## Robustness Matrix

Each controller is evaluated under five conditions:

| Condition | Actuator | Prediction model | Measurement |
|---|---|---|---|
| ideal | ideal | perfect | none |
| rate_limited | asymmetric ramp | perfect | none |
| mild_mismatch | ideal | capacity +5%, heat -5%, cooling +5% | none |
| rate_limited_mild_mismatch | asymmetric ramp | mild mismatch | none |
| combined_nonideal | first order, 30 min | mild mismatch | Gaussian, 0.15 degC |

The mismatch deliberately contains both overestimation and underestimation.
All condition, actuator, measurement, plant, and prediction snapshots are
stored as independent experiment metadata.

## Finite-Horizon Actuator Awareness

Candidate targets are propagated through `predict_actuator_step` before the
thermal prediction. Temperature constraints and objective energy use predicted
applied cooling, not an ideal target. The movement penalty is the sum of
squared applied-action changes normalized by maximum cooling capacity across
the prediction horizon.

Prediction infeasibility, actuator infeasibility, generic optimizer failure,
and fallback are recorded separately. Fallback uses the baseline temperature
controller but still passes through the configured safety and actuator chain.
