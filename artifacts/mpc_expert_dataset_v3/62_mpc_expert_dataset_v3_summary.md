# MPC Expert Dataset v3 Summary

**MPC EXPERT DATASET v3 READY WITH DOCUMENTED LIMITATIONS**

- Scenario A: Alibaba2020 repaired v3, 7,680 states and 259,920 task decisions, frozen.
- Scenario B: SpotGPU2026 native-medium, 17670 continuous states and 466867 task decisions.
- Spot H1/H4 exact disagreement: 26.244091%.
- Spot nonempty-state disagreement: 64.264281%.
- H1/H4 final failures: 0 / 0; verified same-model presolve retries: 0 / 5.
- Checkpoint exact restart, information isolation, split grouping and data hashes: PASS.
- The dataset is ready as the formal expert-data foundation; future BC/Transformer/RL work is outside this run.
