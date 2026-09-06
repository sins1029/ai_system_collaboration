# SpotGPU2026 Capacity Calibration v1

Final status: **SPOT NATIVE CAPACITY READY**

## Source node pool

- Nodes: 4278.
- CPU: 632636 cores.
- GPU: 10412 units.
- GPU models: 6.
- Full node file was read and hashed; README aggregate numbers were not used as the measurement source.

## Capacity comparison

- Old GPU capacity: 2900; native GPU capacity: 10412; multiplier: 3.590344828x.
- Old mandatory GPU peak: 8173.67 / 2900 = 281.850690%.
- Native mandatory GPU peak: 8173.67 / 10412 = 78.502401%.
- Native GPU offered load: 65.433559%; immediate-start peak ratio: 81.768824%.
- Static single-task feasibility: 100.000000%.

## Dynamic preflight

- Selection: train-only low/medium/high GPU-arrival windows plus a >72h-task-dense window, 16 steps each.
- State initialization: continuous deterministic prefix replay from train step 0; no empty window reset.
- H1/H4 solver failures or timeouts: 0 / 0.
- Maximum post-dispatch queue: 0; hard SLA violations: 0.
- Price/carbon provenance for this solvability-only check: an actual SustainCluster five-DC signal snapshot, persistence for H1 and the existing four future signal nodes for H4. No policy-quality conclusion is drawn from this proxy.

No task was removed, sampled, retimed or resized. HP max wait remains 1 step and Spot max wait remains 8 steps. GPU models remain metadata only.
