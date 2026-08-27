# SustainCluster H=4 expert dataset

The three forecast variants are stored in separate directories. The deployable baseline forecast dataset is the only primary BC training source; oracle data is excluded from its split manifest.

## Primary dataset

- Episodes: 80
- Steps: 7680
- Task-action rows: 256125
- Defer ratio: 0.683760
- Local execution ratio: 0.262084
- Migration ratio: 0.737916
- GPU task ratio: 0.702840
- SLA-urgent ratio: 0.799668
- Solver failures: 0

## Labels

- dc_1: 17787 (0.069447)
- dc_2: 18431 (0.071961)
- dc_3: 15776 (0.061595)
- dc_4: 15526 (0.060619)
- dc_5: 13477 (0.052619)
- defer: 175128 (0.683760)

Rows are never duplicated for balancing. Labels below 5% are reported explicitly; class weighting and scenario-level sampling are used by training.

## Forecast boundary

`deployable_no_future` contains no future arrivals. `deployable_baseline_forecast` uses only observations at or before the current step. `oracle_upper_bound` reads synthetic/trace future arrivals and remains physically isolated from the deployable training split.

## Integrity

- Dataset schema: 1.0.0
- Storage: Parquet with Zstandard compression and SHA-256 file hashes.
- Split unit: complete episode, with seed-exclusive train/validation/test groups.
