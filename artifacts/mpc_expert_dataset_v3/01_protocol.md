# MPC Expert Dataset v3 Protocol

1. Scenario A was independently regenerated first with seeds 3001-3040, the original split, repaired H1, and repaired H4 Oracle.
2. No Transformer, RL, or policy training was run.
3. Scenario B keeps all 466,867 Spot tasks unchanged and replaces the mismatched legacy 5DC capacity with a deterministic partition of the complete node_info pool.
4. CPU/GPU are measured and strictly conserved. Memory is marked MODELED_DC_MEMORY using the frozen Alibaba2020/SustainCluster fleet Memory/CPU ratio.
5. HP/Spot max waits remain 1/8 steps. GPU model is metadata only.
6. Spot expert generation was not run; only train-window dynamic preflight with continuous prefix state restoration was allowed.
