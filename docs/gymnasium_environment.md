# Gymnasium Task Scheduling Environment

## Purpose

`SingleCenterTaskSchedulingEnv` is a Gymnasium adapter over the existing
single-center task runtime and thermal-electric system. It does not reproduce
task lifecycle, resource, cooling or plant formulas. The agent controls task
scheduling only; the configured cooling controller remains responsible for
HVAC decisions.

## Sequential Task Decisions

The action space is `Discrete(2)`: `0` defers the current candidate and `1`
starts it. A fixed `max_tasks` action vector would either truncate the queue or
make action meaning depend on padding and slot assignment. Sequential decisions
keep a fixed action space while allowing any number of waiting tasks and more
than one start per physical interval.

A decision step handles one candidate. A simulation step advances the data
center by 15 minutes. The environment accumulates selected task IDs across all
decision steps in one interval, submits one `TaskSchedulingDecision`, and then
calls the existing system exactly once. Energy cost and carbon are therefore
computed once per physical interval.

Candidates default to EDF order: deadline, descending priority, arrival time,
then task ID. FIFO and priority orders are configurable. Rule benchmarks mirror
the native scheduler order, including non-deferrable-first Energy-aware order,
so sequential bookkeeping cannot rewrite the scheduler plan.

## Action Mask

The observation includes `[defer_legal, start_legal]`:

- `[1, 1]`: deferrable and resource-feasible.
- `[0, 1]`: non-deferrable or at latest start, and feasible.
- `[1, 0]`: deferrable but currently resource-infeasible.
- `[0, 0]`: never returned as a normal decision state. The environment records
  the forced resource block, skips only that candidate, and continues scanning
  later candidates before closing the simulation step.

Invalid actions support `raise`, `penalize_and_noop` and `project_to_legal`.
The default records and penalizes the invalid action without starting a task.

## Observation

The fixed `spaces.Dict` contains only finite numeric arrays:

| Key | Shape | Content |
| --- | --- | --- |
| `global` | 4 | step fraction, time sine/cosine, remaining fraction |
| `candidate_task` | 10 | resource demand, duration, slack, wait, priority and flags |
| `resources` | 6 | used and available CPU/GPU/memory fractions |
| `queue_summary` | 8 | queue/running/risk counts, wait/slack and mean demands |
| `environment` | 4 | current price, carbon, outdoor temperature and renewable power |
| `thermal` | 4 | measured temperature, applied cooling, workload and last grid power |
| `action_mask` | 2 | legal defer/start actions |

Normalization uses `ObservationConfig` constants or physical capacity. It does
not scan the future signal/task timeline. Values outside configured scales are
clipped and counted in `observation_clipping_count`. True temperature, future
arrivals, full forecasts, plant parameters and database state are excluded.

## Reward

The benchmark reward is modular and replaceable. Signed components are:

```text
completion
- normalized energy cost
- normalized carbon
- normalized waiting and queue length
- new SLA violations
- invalid actions
- temperature violations
- unfinished tasks at episode end
```

Decision reward contains only immediate action effects. Simulation reward is
computed once after a physical step. Terminal unfinished penalty is computed
once when the episode is truncated. `info["reward_components"]` contains every
component and their exact sum as `total`. The defaults are a first benchmark,
not a claim that reward tuning is complete.

## Episode And Reset

The standard episode has 96 simulation steps and ends with `truncated=True`.
`terminated` is reserved for unrecoverable domain termination. Waiting and
running tasks remain unfinished; completion is never fabricated.

`reset(seed=...)` rebuilds both external providers, the data-center system,
task/resource state, controller, measurement RNG, reward state, candidate state,
metrics and optional run handle. Equal providers, seed and actions are
reproducible. Empty input steps auto-advance until a decision state appears;
the count is reported in `auto_advanced_simulation_steps`.

## Providers And Persistence

The constructor receives signal and task provider factories. The package does
not generate tasks or read a hard-coded CSV. `NullRunStore` is the default.
SQLite schema v4 adds Gym run metadata, `run_agent_decisions` and
`run_gym_episode_summaries`, while reusing the v3 simulation, task event and
outcome tables.

## Rule And Random Policies

`TaskSchedulerGymAdapter` calls FIFO, EDF or Energy-aware once at the start of a
simulation step and maps membership in its planned start set to binary actions.
Blocked forced candidates are internal bookkeeping, not agent decisions, and
cannot prevent later planned tasks from being processed. Native and Gym traces
must have identical planned/executed sets and deterministic trajectory hashes.

`MaskedRandomPolicy` samples only actions enabled by the mask and is reproducible
with a fixed seed. It is a smoke-test policy, not a trained scheduling result.

The environment passes Gymnasium `check_env` and supports two-instance
`SyncVectorEnv` with independent stores and state.

## Authoritative Domain Semantics

The native scheduler observes the waiting/running set and resource ledger once
at physical-step start, returns one start set, and the runtime validates and
commits that set centrally. Sequential Gym decisions are one transaction: the
candidate set/order is frozen, starts consume only a virtual ledger, and task
state, wait counters, SLA and plant time do not move until one final commit.

Non-deferrable or latest-start means business deferral is forbidden; it never
permits resource over-allocation. If start is physically impossible, the task
stays waiting and receives a resource-blocked event. `wait_steps` increases once
at physical-step end. `deferral_count` increases only for an explicit legal
choice not to start a deferrable task; resource blocking instead increments
`resource_blocked_count`. Auto skips create no action, decision reward or
decision-step increment.

## Current Limits

There is no trained RL policy, systematic reward tuning, direct arbitrary task
ID action, preemption, multi-center migration, network cost or exogenous forecast
error model. Candidate order remains rule-defined.
