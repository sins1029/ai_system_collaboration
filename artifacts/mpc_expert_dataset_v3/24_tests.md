# Validation

| Check | Result |
|---|---|
| compileall | PASS (`src scripts tests`) |
| formal output verifier | PASS (timeline, grouping, contracts, boundary continuity, isolation, hashes) |
| Spot v3 dedicated | 7 passed, 14 warnings |
| related regression | 88 passed, 14 warnings |
| full pytest | 397 passed, 17 warnings |
| checkpoint/restart exact gate | PASS (100 uninterrupted vs 50 + restart + 50) |
| Alibaba2020 v3 H1 solves | 7,680 optimal, 0 failures |
| Alibaba2020 v3 H4 solves | 7,680 optimal, 0 failures |
| Spot full H1 solves | 17,663 optimal, 0 failures/timeouts, 0 presolve retries |
| Spot full H4 solves | 17,663 optimal, 0 failures/timeouts, 5 verified same-model presolve retries |
| fallback expert labels | 0 |
| new test failures | 0 |

The five H4 retries were triggered only after HiGHS presolve reported infeasible while an explicit all-terminal-backlog witness satisfied the same model. Re-solving the unchanged MILP with presolve disabled returned optimal every time; each event is recorded in `58_spot_solver_quality.csv`. Warnings are dependency deprecations and existing explicit Oracle-mode audit warnings.
