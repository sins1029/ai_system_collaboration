# MPC Expert Dataset v3 Final Diagnosis

Final status: **MPC EXPERT DATASET v3 READY WITH DOCUMENTED LIMITATIONS**

1. Spot full continuous timeline completed: **YES**, 17670 steps with no daily or split reset.
2. States: **17670** (17663 nonempty, 7 empty).
3. Task decisions: **466867** over **466867** unique tasks.
4. H1/H4 final solver failures: **0 / 0**; verified same-model presolve retries: **0 / 5**; fallback labels: **0**.
5. H1/H4 task exact disagreement: **26.244091%**.
6. Nonempty state any-disagreement: **64.264281%**.
7. Spot train-only risk P95: **0.724635036**; high-risk task/state disagreement: **39.629803% / 96.956246%**.
8. HP versus Spot disagreement: **25.744684% / 30.302616%**.
9. H1 defer: **0.000000%**.
10. H4 defer: **0.000000%**.
11. Defer concentration is reported by priority in `50_spot_defer_analysis.csv`; no objective or SLA parameter was tuned to create defer.
12. `<1h` versus `>72h` disagreement: **26.559536% / 23.972456%**; descriptive only.
13. Low versus high duration-error disagreement: **25.860414% / 26.330774%**; descriptive only, estimator unchanged.
14. Alibaba2020 versus Spot exact disagreement: **3.679978% / 26.244091%**. This compares behavior inside each source-matched frozen capacity scenario, not workload difficulty under controlled capacity.
15. Expert Dataset v3 can be the single formal basis for later learning experiments: **YES**, with labels requested explicitly and with the documented scenario/license limitations retained.

## Completion gates

- Alibaba2020 Scenario A remains frozen and was not regenerated.
- Spot Scenario B contains the complete arrival timeline and all 466,867 source tasks.
- Checkpoint/restart semantic equality: PASS.
- Information-region isolation: PASS.
- Split-state exclusivity and continuous boundary crossing: PASS.
- Formal file hashes: `60_spot_integrity_manifest.csv`.
- Transformer, BC and RL training: not run.

## Documented limitations

- MODELED_DC_MEMORY is not measured Alibaba2026 host memory
- regional price/carbon are SustainCluster 2023 EXTERNAL_SCENARIO_SIGNAL, not Alibaba2026 measurements
- origin, memory, bandwidth, duration estimate and SLA are frozen modeled fields
- GPU models are descriptive metadata; compatibility constraints are disabled
- cross-dataset behavior is not a capacity-controlled causal comparison
- public redistribution remains blocked pending source/derived-data license confirmation
