
# SpotGPU2026 Forward-control Value v1 Summary

1. H4 total stage-cost improvement (H1-H4): -3342.26565078 objective units.
2. Relative stage-cost improvement: -0.077358212%.
3. SLA: H1=40.506396897%, H4=40.784206209%, delta=0.0027780931186.
4. Electricity: H4-H1=-1179.51118879 USD under the frozen SustainCluster 2023 external signal scenario.
5. Carbon: H4-H1=15752.2677958 kgCO2 under the same external signal scenario.
6. Transmission/migration: cost delta=1.68662103479 USD; migration-count delta=1094.
7. Backlog/completion: mean backlog delta=0.459026598755; primary completion-rate delta=0.
8. Pressure concentration: low/high mean stage-cost deltas per step are 0.168922029877 and 0.580688599989.
9. HP versus Spot: action-stage-cost improvements are HP=-0.070846707% and Spot=-0.441986160%; larger benefit=HP.
10. Of the 26.244091% shadow action disagreement, 0.000000000% is equivalent at the conservative 4e-9 tie-break scale. Broader solver-equivalence is not hard-classified.
11. Disagreement/gain association: Pearson=0.129657681; descriptive only, no causal claim.
12. Time stability: 21.081081081% of daily blocks have positive H4 stage-cost improvement.
13. Tail drain: H1=3 steps, H4=3 steps; conclusion changed=NO.
14. MPC mechanism: spatial placement; no objective tuning was used.
15. Next stage: 风险状态重点学习.

Final diagnosis: **CONTEXT_DEPENDENT_CONTROL_VALUE**
