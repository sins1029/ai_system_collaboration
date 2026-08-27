# Behavior cloning real closed-loop evaluation

Five fresh seeds (1201-1205) are disjoint from train/validation/test. Episodes alternate normal and high-load trace windows. `J = electricity + carbon + transmission`, so a positive gap means higher observed operating burden than H=4 MPC. SLA is reported separately because it has no shared physical unit.

| Policy | Reward | Completed | SLA violations | Wait avg | Defer | Migration | Electricity | Carbon | Transmission | Inference ms | Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h4_mpc | -1797.5083 | 3512.00 | 1552.60 | 0.0000 | 0.7496 | 0.5098 | 14786.3242 | 45277.9063 | 101.2325 | 20.5129 | 0.0000 |
| h1_mpc | -1796.6820 | 3512.00 | 1553.00 | 0.0000 | 0.7496 | 0.5145 | 14834.5842 | 45338.3843 | 100.6998 | 6.7645 | 0.0018 |
| bc_policy | -1866.3149 | 3512.20 | 1552.80 | 0.0000 | 0.7496 | 0.5385 | 14736.7824 | 45082.7059 | 119.8620 | 4.0763 | -0.0038 |
| random_untrained | -2439.2205 | 3506.60 | 1597.60 | 0.0000 | 0.7678 | 0.7783 | 14693.7968 | 45496.1290 | 340.5100 | 4.1969 | 0.0061 |
| original_rule_local_only | 0.0000 | 3551.60 | 0.00 | 0.0000 | 0.0000 | 0.0000 | 15275.2664 | 46706.7893 | 0.0000 | 0.0000 | 0.0302 |

All reported costs and rewards come from actions actually executed in SustainCluster. Internal optimizer objectives are not compared across policies.

The repository's RBC path routes tasks inside `DatacenterClusterManager`; `TaskSchedulingEnv` therefore passes an empty `current_tasks` list to `CompositeReward`. Its displayed reward and SLA violations are zeros from that interface and must not be interpreted as zero-cost or violation-free operation. The shared gap uses electricity, carbon and transmission metrics that are populated for every policy.
