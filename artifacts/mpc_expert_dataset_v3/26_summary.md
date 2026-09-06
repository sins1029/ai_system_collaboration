# MPC Expert Dataset v3 Current Summary

1. Alibaba2020 v3: 80 episodes, 7,680 states, 259,920 decisions; independently generated without Transformer.
2. v2/v3 student state, feasible mask, H1 action, and H4 Oracle action agreement: all 100%; mismatches: 0.
3. Spot source: 466,867 tasks unchanged; node pool: 4278 nodes, 632636 CPU, 10412 GPU.
4. Native 5DC: deterministic exclusive node partition; CPU/GPU strictly conserved; memory is MODELED_DC_MEMORY.
5. Native GPU pressure: offered 65.433559%, mandatory peak 78.502401%, immediate-start peak 81.768824%.
6. Arrival pressure P50/P90/P95/P99/max is recorded in `36_spot_native_arrival_pressure_summary.csv`.
7. Dynamic preflight: four post-warm-up train windows, 64 H1 and 64 H4 solves, zero failure/timeout and zero hard SLA violation.
8. Comparable-pressure capacity: NOT REQUIRED because native GPU pressure is material rather than too loose.
9. Spot formal 184-day expert generation: NOT RUN in this round.
10. Final status: SPOT NATIVE CAPACITY READY.
