
# Frozen Contract

- Scenario: B_SPOTGPU2026_NATIVE_MEDIUM
- Tasks: 466867; request scope: PER_JOB
- Stable order: submit_time, original_index, task_id
- Time step: 900 seconds; primary steps: 17670
- CPU: 632636 cores
- GPU: 10412; per DC: [1875, 2291, 2070, 2615, 1561]
- Memory: 542259.428571428522 GB, MODELED_DC_MEMORY
- Bandwidth: 0.020062580706 GB
- SLA: medium; HP max wait 1 step; Spot max wait 8 steps
- true_duration: simulator only
- estimated_duration: frozen train-only hierarchical conditional median
- Energy signals: SustainCluster 2023 EXTERNAL_SCENARIO_SIGNAL, not Alibaba2026 measured energy data
- Objective: unchanged configs/sustaincluster_mpc/h4_expert.yaml
- H1 horizon nodes: 1
- H4 horizon nodes: 5, offsets +15/+30/+45/+60 minutes
