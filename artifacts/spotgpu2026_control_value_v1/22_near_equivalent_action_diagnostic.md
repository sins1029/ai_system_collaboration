
# Near-equivalent Action Diagnostic

- Public v3 shadow task disagreement rate: 26.244091%.
- Disagreeing task decisions evaluated: 122525.
- Frozen deterministic tie-break epsilon: 1e-9.
- Maximum destination-rank tie-break gap: 4e-9.
- Conservative tie-break-scale equivalent fraction: 0.000000000%.
- Median absolute fixed-state objective gap: 0.0167611223086.
- P95 absolute fixed-state objective gap: 0.749435157935.

The binary label is deliberately named TIE_BREAK_SCALE_EQUIVALENT. SciPy/HiGHS did
not expose a task-level solver optimality tolerance in the frozen configuration, so
this audit does not claim that every numerically small gap is solver-equivalent.
The continuous gap distribution in 21_fixed_state_action_cost_gap.csv and the raw
Parquet table is the primary evidence.
