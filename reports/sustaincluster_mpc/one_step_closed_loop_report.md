# SustainCluster one-step optimization closed-loop report

Date: 2026-07-17
Repository commit: `3f6ea95cb835b89ba50b0ef76d66d14b8037643e`

## Scope

This validates an external one-timestep assignment optimizer. It is not a multi-horizon MPC. No SustainCluster source, YAML, checkpoint, training entry, branch, commit, or Git history was changed.

The loop was:

`env.current_tasks -> frozen state adapter -> SciPy/HiGHS MILP -> semantic decisions -> dynamic action adapter -> env.step(actions)`

Runtime configuration was copied in memory:

- `single_action_mode=false`
- `disable_defer_action=false`
- `use_tensorboard=false`
- seed `123`
- 20 requested steps

The original YAML remained unchanged.

## Solver

- Interface: `scipy.optimize.milp`
- SciPy: 1.15.3
- Backend: bundled HiGHS
- Extra package installation: none
- Commercial solver: none
- External binary installation: none

The MILP uses one binary assignment variable for every task/DC pair and one binary defer variable for each task. It enforces exactly one assignment or defer decision and CPU, GPU, and memory capacity limits. Current DC queues and in-transit tasks are conservatively deducted as reservations before optimization.

Objective terms are independently weighted: electricity, carbon, transmission, defer, and SLA risk. All-zero weights remain deterministic through a configurable epsilon tie break that prefers assignment before defer when capacity exists.

## Action mapping

The reset-specific ordered DC list was `dc_id [2,4,5,1,3]`. With defer enabled:

| Semantic decision | Environment action |
|---|---:|
| defer | 0 |
| assign to dc_id 2 | 1 |
| assign to dc_id 4 | 2 |
| assign to dc_id 5 | 3 |
| assign to dc_id 1 | 4 |
| assign to dc_id 3 | 5 |

The mapping was derived after reset and checked again before every step. It was not inferred from `dc_id` arithmetic.

## Unit and integration tests

Command result: `15 passed in 8.52s`.

Covered behavior:

1. Empty task list returns `[]` and real `env.step([])` succeeds.
2. A single task receives a legal dynamically encoded destination.
3. Multiple feasible tasks all receive decisions.
4. GPU shortage redirects to another DC.
5. Memory limits are not exceeded.
6. Total shortage defers when defer is enabled.
7. Total shortage returns explicit `infeasible` when defer is disabled.
8. High transfer cost favors the origin DC.
9. Electricity cost favors a lower-price DC.
10. Carbon cost favors a lower-carbon DC.
11. An urgent task receives a higher defer cost than a loose task.
12. Task, assignment, and action order remain aligned.
13. Empty/nonempty transitions leave no adapter state residue.
14. Defer is rejected with task ID/index context when disabled.
15. Real state snapshots are frozen and state adaptation does not mutate the environment.

## Twenty-step results

Costs are weighted objective contributions. `E/C/X/D/S` means electricity/carbon/transmission/defer/SLA. Times are milliseconds.

| Step | Tasks | Assign/defer | DC1/2/3/4/5 | Solve ms | Objective | E/C/X/D/S | Reward | Scheduled | Overflow | Done |
|---:|---:|---:|---|---:|---:|---|---:|---:|---|---|
| 1 | 6 | 6/0 | 0/0/3/2/1 | 2.162 | 1.0609 | 0.2479/0.6235/0.1895/0/0 | -0.8221 | 6 | no | no |
| 2 | 5 | 5/0 | 0/0/2/2/1 | 1.581 | 1.0445 | 0.1756/0.7622/0.1068/0/0 | -0.4232 | 5 | no | no |
| 3 | 3 | 3/0 | 0/0/3/0/0 | 1.360 | 0.4127 | 0.0826/0.2084/0.1216/0/0 | -0.3041 | 3 | no | no |
| 4 | 79 | 79/0 | 0/0/74/4/1 | 2.854 | 117.2613 | 28.9467/85.1483/3.1663/0/0 | -343.6813 | 79 | no | no |
| 5 | 45 | 45/0 | 0/0/44/0/1 | 1.964 | 32.1141 | 7.0337/24.2782/0.8022/0/0 | -95.5720 | 45 | no | no |
| 6 | 29 | 29/0 | 0/0/28/0/1 | 4.136 | 14.7979 | 2.8158/11.8073/0.1748/0/0 | -47.3717 | 29 | no | no |
| 7 | 8 | 8/0 | 2/0/6/0/0 | 1.173 | 1.9455 | 0.3164/1.4494/0.1796/0/0 | -1.3852 | 8 | no | no |
| 8 | 38 | 38/0 | 0/0/37/1/0 | 1.276 | 17.4099 | 2.4715/13.9681/0.9704/0/0 | -34.1522 | 38 | no | no |
| 9 | 13 | 13/0 | 0/0/13/0/0 | 1.211 | 4.3730 | 0.6046/3.6521/0.1163/0/0 | -8.8934 | 13 | no | no |
| 10 | 51 | 51/0 | 1/0/43/7/0 | 1.454 | 39.0964 | 4.6901/29.5582/4.8481/0/0 | -60.6793 | 51 | no | no |
| 11 | 8 | 8/0 | 1/1/4/1/1 | 1.185 | 0.7701 | 0.0925/0.6218/0.0558/0/0 | -0.6597 | 8 | no | no |
| 12 | 21 | 21/0 | 1/0/18/0/2 | 1.380 | 8.0538 | 0.7906/6.2299/1.0333/0/0 | -5.6715 | 21 | no | no |
| 13 | 15 | 15/0 | 0/0/8/6/1 | 0.906 | 2.7074 | 0.3245/2.1355/0.2474/0/0 | -1.1316 | 15 | no | no |
| 14 | 18 | 18/0 | 0/0/16/2/0 | 1.341 | 7.6549 | 0.7615/5.2863/1.6072/0/0 | -2.9968 | 18 | no | no |
| 15 | 31 | 31/0 | 2/0/24/3/2 | 1.584 | 20.0808 | 2.0087/13.9553/4.1168/0/0 | -2.9549 | 31 | no | no |
| 16 | 47 | 47/0 | 0/0/45/2/0 | 1.909 | 6.1096 | 0.6121/4.2306/1.2669/0/0 | -2.7457 | 47 | no | no |
| 17 | 11 | 11/0 | 0/0/9/1/1 | 1.163 | 4.3978 | 0.5103/3.3386/0.5489/0/0 | -2.1881 | 11 | no | no |
| 18 | 9 | 9/0 | 0/1/5/3/0 | 0.838 | 1.6429 | 0.2014/1.1128/0.3288/0/0 | -0.3022 | 9 | no | no |
| 19 | 18 | 18/0 | 0/0/15/3/0 | 1.315 | 3.6472 | 0.3387/2.1168/1.1916/0/0 | -2.9738 | 18 | no | no |
| 20 | 76 | 76/0 | 3/6/55/10/2 | 2.158 | 14.2694 | 1.7562/9.8767/2.6364/0/0 | -8.6871 | 76 | no | no |

## Aggregate results

- Completed steps: 20/20
- Total task decisions: 531
- Assignment totals by real DC ID: DC1=10, DC2=8, DC3=452, DC4=47, DC5=14
- Average solve time: 0.0016474300 s (1.647 ms)
- Minimum solve time: 0.0008378000 s (0.838 ms)
- Maximum solve time: 0.0041355001 s (4.136 ms)
- Infeasible solves: 0
- Defer ratio: 0.0
- Action validation failures: 0
- Task-order mismatches: 0
- Resource-overflow steps: 0
- State-adapter environment mutations: 0
- Terminated/truncated steps: 0

The real trace did not require defer because schedulable capacity was sufficient and the configured defer cost was intentionally high. Defer behavior was still executed in unit tests for both feasible-defer and forbidden-defer/infeasible cases.

## Current support for future rolling MPC

Already usable:

- Stable task ID, original ordering, origin DC, CPU/GPU/memory, duration, deadline, arrival, transfer size, wait count, and deferred flag.
- Absolute and ratio resource states for all DCs.
- Running-task finish times plus queued and in-transit reservation demand.
- Current physical electricity price and carbon intensity.
- Current IT/cooling/total power and temperatures as step diagnostics.
- Current network cost matrix and task-specific transfer delay calculation.
- Existing short carbon and price forecast methods, although not used in this stage.

Still insufficient:

- No public future task-arrival forecast.
- No true global defer count/history per task.
- No authoritative immutable scheduler-state API.
- No link capacity, congestion, or reservation model.
- No unified physical-unit price/CI forecast object aligned to a control horizon.
- No definitive future capacity trajectory for queued and in-transit scheduling outcomes.
- No thermal state transition/forecast suitable for HVAC control.

## Minimum path to multi-horizon MPC

1. Add an official read-only environment getter matching the external snapshot schema.
2. Add horizon-aligned task-arrival, price, and carbon forecast inputs with timestamps and physical units.
3. Add a deterministic capacity transition/reservation model for running, queued, deferred, and in-transit tasks.
4. Add time-indexed decision variables and inter-step task/resource conservation constraints.
5. Solve the rolling horizon, apply only the first action vector, then rebuild state and repeat.

No RL action, RL training, HVAC control, battery control, future-task prediction, or multi-horizon optimization was implemented here.
