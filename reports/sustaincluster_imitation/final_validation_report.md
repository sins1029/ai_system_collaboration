# SustainCluster imitation and SAC warm-start final validation

## Frozen versions

- Main project: `feature/single-center-task-runtime-v0.2@d49870b9d9fbee4b480100f4a64d74075b0e1a8b` (new work remains uncommitted).
- SustainCluster: `main@3f6ea95cb835b89ba50b0ef76d66d14b8037643e`, clean, remote `HewlettPackard/sustain-cluster`.
- Python: 3.10.11, CPU PyTorch 2.7.0.
- Workload SHA-256: `3BA8A8E0067288F8D2542752A292F7DA04CB79305689BA145CC6060E96CB7F2E`.

## Quality gates

- H=1/H=4 fair comparison: passed.
- H=4 infeasible, illegal action and resource overflow counts: zero.
- Forward-value scenarios: capacity release, normal trace and high-load trace.
- External SustainCluster source modifications: none.

## Expert datasets

Three physically separate Parquet datasets were written with SHA-256 manifests:

| Variant | Episodes | Steps | Task-actions |
|---|---:|---:|---:|
| deployable_no_future | 70 | 6,720 | 200,335 |
| deployable_baseline_forecast | 80 | 7,680 | 256,125 |
| oracle_upper_bound | 70 | 6,720 | 200,343 |

The primary deployable dataset uses complete, seed-exclusive episode splits. Oracle episode count in the primary split is zero. All semantic classes exceed 5% after adding new class-balance heterogeneity trajectories; no rows were duplicated.

## Behavior cloning

- Architecture: original shared SustainCluster `ActorNet`, 233 inputs, 6 semantic actions, hidden size 256 and LayerNorm.
- Three seeds: 11, 22 and 33.
- Test accuracy: 0.969809 +/- 0.001411.
- Top-2 accuracy: 0.996219 +/- 0.000661.
- Defer F1: 0.999914.
- Assign accuracy: 0.904918 +/- 0.004460.
- Migration F1: 0.947896 +/- 0.003163.
- BC-to-SAC bridge: exact 10-tensor key/shape match, 128,262 parameters.

## Real closed loop

Five fresh seeds (1201-1205) were run in real multi-action SustainCluster episodes.

- BC illegal actions: zero.
- BC resource overflows: zero.
- BC average inference: 4.076 ms; H=4 MPC: 20.513 ms.
- BC SLA violations: 1,552.8; H=4 MPC: 1,552.6; random actor: 1,597.6 per averaged episode.
- BC shared operating-cost gap versus H=4 MPC: -0.003758, using electricity + carbon + transmission.

The built-in RBC reward/SLA zeros are an interface limitation caused by internal routing and are not interpreted as zero-cost or violation-free operation.

## SAC warm start

Final fingerprint: `7EAA059E4EF6C764D089438CDD266617010D623CA0FEFDE469A9E2C32AF5CF25`.

All six runs use 10,000 environment steps, batch 8, update frequency 16, 618 updates, identical paired critic initialization, finite Q values and zero illegal actions.

- BC initial reward: -1,821.22; random initial reward: -3,641.44.
- BC initial SLA violations: 936; random: 966.
- BC initial expert agreement: 0.9256; random: 0.7796.
- BC reaches the reward threshold at step 0; random needs 666.7 steps on average.
- After SAC updates, BC reward falls to -2,432.21, SLA violations rise to 2,080 and expert agreement falls to 0.7690.

BC is a successful actor initializer, but the current unconstrained SAC fine-tuning recipe does not preserve cloned expert behavior. This is direct evidence of online distribution shift.

## Verification

- Focused imitation tests: 11 passed.
- Full project and MPC regression: 137 passed, 1 skipped.
- Dataset hash round-trip, seed leakage, oracle isolation and unseen burst checks: passed.

## Added files

- `configs/sustaincluster_mpc/h1_expert.yaml`, `h4_expert.yaml`.
- `configs/sustaincluster_imitation/dataset.yaml`, `bc_train.yaml`, `sac_warm_start.yaml`.
- `src/sustaincluster_imitation/`: schema, reader/writer, collectors, causal forecast, feature encoder, BC policy/trainer, SAC bridge, split and scenario modules.
- `reports/sustaincluster_imitation/`: reproducible experiment entry points plus JSON, Markdown and CSV results.
- `data/processed/sustaincluster_expert/`: three datasets and split manifest.
- `artifacts/sustaincluster_imitation/`: three BC checkpoints and training result JSON.
- `tests/test_sustaincluster_imitation.py`.

`sac_warm_start_rejected_mixed_config.json` is retained only as an audit record of a rejected mixed-configuration partial run. It is not referenced by any formal result.

## Readiness

The project is ready for controlled SAC-method development, but not for large-scale formal SAC training with the current loss alone. The next safe sequence is:

1. Add a BC/KL regularizer or expert replay mixing and freeze the actor during an initial critic warmup.
2. Keep H=4 MPC as an online correction/safety reference while validating SLA retention.
3. Consider RL outputs as high-level MPC reference parameters only after the regularized actor remains stable across unseen workload windows.

No formal long training, online MPC-RL controller, HVAC, storage, Transformer, GNN or single-center thermal integration was started.
