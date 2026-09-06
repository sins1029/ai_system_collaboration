# Spot Dynamic State-Restore Audit

- Windows: LOW_LOAD, MEDIUM_LOAD, LONG_TASK_DENSE, HIGH_LOAD; 16 steps each, train split only.
- Window starts: 2309, 2949, 7054, 9144.
- Restored pre-window running tasks: 5011, 5307, 5616, 6389.
- Empty-state starts: 0.
- H1/H4 optimal steps: 64/64 of 64 each.
- Solver failure/timeout: 0/0.
- Maximum queue after dispatch: 0; ending queue: 0.
- Hard SLA violations: 0.
- Restore method: continuous deterministic prefix replay from train step 0. No window used an empty reset.
