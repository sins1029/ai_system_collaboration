# Behavior cloning offline evaluation

The policy reuses SustainCluster's shared per-task `ActorNet`. Inputs use stable semantic DC ordering and deployable causal H=4 forecast features. Oracle rows are excluded.

- Feature dimension: 233
- Action dimension: 6
- Parameters: 128262
- Best checkpoint: `artifacts/sustaincluster_imitation/bc_actor_seed_11.pt`

## Three-seed results

| Seed | Test loss | Accuracy | Top-2 | Defer F1 | Assign accuracy | Migration F1 | Seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0.205103 | 0.971690 | 0.997091 | 0.999914 | 0.910864 | 0.952285 | 10.48 |
| 22 | 0.229155 | 0.968293 | 0.995490 | 0.999914 | 0.900123 | 0.946452 | 12.98 |
| 33 | 0.213015 | 0.969445 | 0.996076 | 0.999914 | 0.903765 | 0.944952 | 13.21 |

## Aggregate test metrics

- overall_accuracy: 0.969809 +/- 0.001411
- top2_accuracy: 0.996219 +/- 0.000661
- defer_f1: 0.999914 +/- 0.000000
- assign_accuracy: 0.904918 +/- 0.004460
- migration_f1: 0.947896 +/- 0.003163
- loss: 0.215758 +/- 0.010009

Confusion matrices and per-DC precision/recall/F1 are stored in the JSON result for every seed.
