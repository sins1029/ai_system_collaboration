
# SpotGPU2026 Control Value v1 Protocol

- Primary comparison: two independent closed loops, H1 versus repaired H4 Oracle.
- Primary timeline: steps 0 through 17669, continuous with no split reset.
- Tail diagnostic: no new arrivals; continue until pending, running, and in-transit are all zero, or 17670 common drain steps.
- H1 information: current deployable state only; Oracle future access is a hard failure.
- H4 information: current state plus exact +15, +30, +45, and +60 minute workload pressure and external energy signals.
- Workload, duration truth, train-only duration estimate, memory, bandwidth, origin, SLA, capacity, network, and initial state are identical.
- Stage cost is the frozen optimizer first-step cost. Reporting reward is exactly negative stage cost and is not independent evidence.
- Physical energy, electricity expenditure, and carbon are integrated from actual running occupancy in each 15-minute interval. They are separate from one-time assignment objective contributions.
- Daily and weekly blocks come from one continuous trajectory and are descriptive, not independent random samples.
- Fixed-state equivalence uses only the deterministic tie-break scale, 4 x 1e-9. A broader solver-tolerance equivalence class is not asserted.
- No composite score is constructed. No Transformer, BC, RL, reward tuning, MPC tuning, workload scaling, or silent fallback is used.
