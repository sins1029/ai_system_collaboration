# SustainCluster temporal model audit

Date: 2026-07-17
Repository commit: `3f6ea95cb835b89ba50b0ef76d66d14b8037643e`

## Canonical time unit

All rolling-horizon model times use integer environment steps.

- One step is exactly 15 minutes: `TaskSchedulingEnv.__init__`, `envs/task_scheduling_env.py` line 37.
- Minutes to steps: `ceil(minutes / 15)` for duration or future release.
- Seconds to steps: `max(1, ceil(seconds / 900))` for a newly dispatched transfer. The minimum is one because arrivals are processed before new actions in each environment step.
- Deadline steps: conservative integer completion boundary derived from `floor(remaining_deadline_minutes / 15)`.
- No optimizer constraint mixes minutes, seconds, timestamps, and steps. Physical values are converted once by the horizon adapter.

## Task time fields

Source: `rl_components.task.Task`.

- `duration` is minutes (lines 17 and 48).
- `arrival_time` and `sla_deadline` are timestamps (lines 47 and 58-60).
- `sla_deadline = arrival_time + sla_multiplier * duration`, with default multiplier 1.5.
- `start_time` and `finish_time` are set when a DC schedules the task.
- `SustainDC.try_to_schedule_task`, lines 214-220, sets `finish_time = current_time + duration minutes`.

A running task releases CPU/GPU/memory at the first environment grid time whose timestamp is greater than or equal to `finish_time`. Relative to the horizon snapshot, its release step is `max(0, ceil((finish_time-current_time)/15min))`.

## Environment step event order

At the beginning of `TaskSchedulingEnv.step(actions)` for timestamp `t`:

1. Existing `in_transit_tasks` with `arrival_time <= t` move to the destination DC's `pending_tasks` deque (`task_scheduling_env.py` lines 255-265).
2. Current external tasks are processed in list order. An assignment creates a new `(arrival_time, task, destination)` transit tuple; defer appends to `deferred_tasks`.
3. `DatacenterClusterManager.step(t)` runs (`task_scheduling_env.py` line 359).
4. Each DC releases tasks with `finish_time <= t`, then attempts its local pending queue in FIFO order (`sustaindc_env.py` lines 393-404).
5. Reward and step diagnostics are calculated.
6. `env.current_time` advances by 15 minutes (`task_scheduling_env.py` line 372).
7. `_load_new_tasks()` forms the next external pending sequence as `deferred_tasks + newly_loaded_trace_tasks` (lines 389-397).

The term `pending` has two distinct locations:

- `env.current_tasks`: tasks awaiting the external global scheduling decision.
- `dc.pending_tasks`: tasks already routed to a DC but waiting for local capacity.

The MPC adapter keeps these separate.

## State transitions

### External pending to deferred

Action zero, when defer is enabled, stores the same task object in `env.deferred_tasks` and sets `temporarily_deferred=true`. After the clock advances, deferred tasks are prepended to new trace arrivals and become `env.current_tasks` for the next step. Thus a defer decision has exactly one-step receding-horizon persistence before replanning.

### External pending to transmitting

An assignment sets destination fields and computes a physical transfer arrival timestamp. The tuple enters `env.in_transit_tasks`. Because arrival processing already occurred earlier in the same call, even zero/sub-step delay is not admitted to a DC until at least the next environment step.

Discrete arrival offset:

`arrival_step = dispatch_step + max(1, ceil(delay_seconds / 900))`.

### Transmitting to DC pending

At the beginning of a later `step`, arrived tuples enter the destination `dc.pending_tasks` deque. During the same call, the DC may schedule them after releasing completed tasks.

### DC pending to running

The DC checks absolute available CPU, GPU, and memory. If all fit, it subtracts resources, sets start/finish timestamps, and appends the task to `running_tasks`. Otherwise the task returns to the DC queue and increments `wait_intervals`.

### Running to completed

At each DC step, tasks with `finish_time <= current_time_task` release resources before the local queue is attempted. The horizon adapter therefore adds each running task's resources back from its discrete release step onward.

## Resource timeline construction

For each DC and horizon step, the rolling model starts from current raw available capacity and applies immutable events:

1. Add resources released by currently running tasks at their release step.
2. Reserve resources for existing DC-queued tasks from the current step for their discrete duration.
3. Reserve resources for existing in-transit tasks from their arrival step for their discrete duration.
4. In oracle modes, reserve aggregated predicted arrivals at their origin DC from predicted arrival through predicted duration.
5. Current `env.current_tasks` are not pre-reserved; their occupancy is represented by MILP decision variables.

Each resource timeline is clipped to `[0,total_capacity]`. Raw available, known reservations, and optimizer-controlled occupancy remain distinct.

## Exogenous sequence indexing

### Electricity price

`ElectricityPrice_Manager` interpolates hourly data to four points per hour (`utils/managers.py` lines 541-545). `index` identifies the current 15-minute value; `get_current_price()` returns `prices[index]`, and `step()` increments the index with annual wrap (lines 547-567). Horizon step `h` reads `(index+h) mod len(prices)` in USD/MWh.

### Carbon intensity

`CI_Manager` also interpolates to four points per hour. Its `time_step` is the current 15-minute index. The physical series is `carbon_smooth` in gCO2e/kWh. The existing public forecast getter is normalized and only covers configured future steps, so the external adapter snapshots physical values by copied indexed scalars. This internal read is documented technical debt for a future public getter.

### Weather

Weather advances once in each `SustainDC.step`, but weather is not a control input or optimization coefficient in this stage. No weather forecast enters the MILP.

### Manager update nuance

Inside `SustainDC.step`, time/weather/CI/price managers advance before task release and local queue scheduling (`sustaindc_env.py` lines 380-404), while the cluster still passes timestamp `t`. The external optimizer always reads manager getters before calling `env.step`; after the step, `env.current_time` and manager indexes both represent the next receding-horizon state. Step-returned DC diagnostics may therefore contain manager values advanced during the call.

## Forecast modes

- `no_future_arrivals`: no trace arrivals are added; only current running, queued, and in-transit transitions are modeled.
- `oracle`: future trace rows are converted to temporary task values under saved/restored Python and NumPy RNG states, then aggregated by future step and origin DC. The environment, trace, RNG state, and generated task objects are not mutated or exposed.
- `noisy_oracle`: the same aggregate forecast receives seeded demand and arrival-step perturbations. Negative resource demand is prohibited and the same seed produces identical output.

Only current real external pending tasks receive per-task integer variables. Future arrivals are aggregate resource reservations, preventing unbounded variable growth.

## Rolling execution invariant

The optimizer may plan dispatches for steps `0..H-1`, but only step-zero decisions are encoded for SustainCluster:

- Planned dispatch at `h=0`: assign to the planned real `dc_id`.
- Planned dispatch at `h>0`: defer now.
- Not dispatched inside the horizon: defer now.

After exactly one environment step, all unexecuted plan entries are discarded and the horizon is rebuilt from real state.

For `H>1`, a dispatch variable is disabled when its transfer would place the
execution start at or beyond step `H`; such work is represented by the terminal
backlog variable instead of being compressed into the final capacity slot. For
`H=1` only, step-zero assignment uses the sole capacity/price value as an
explicit terminal estimate so the result remains comparable with the existing
one-step optimizer.
