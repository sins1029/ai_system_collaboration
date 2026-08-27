# SustainCluster external state interface audit

Date: 2026-07-17
Repository commit: `3f6ea95cb835b89ba50b0ef76d66d14b8037643e`

## Classification

1. Existing and directly accessible through a public attribute or method.
2. Existing but held in an internal or mutable implementation object; snapshot before exposing.
3. Derivable from existing fields without changing the environment.
4. Not represented by the current model.
5. A future rolling MPC implementation needs a getter, reservation, or forecast interface.

The scheduler's current pending task sequence is `env.current_tasks`, not `env.pending_tasks`. `TaskSchedulingEnv._load_new_tasks` assigns it at lines 389-397, and `step` consumes it in order at lines 323-354.

## Task fields

Primary source: `rl_components.task.Task`, lines 6-119. Runtime reset confirmed all fields listed as existing.

| Requested field | Class | Source / derivation | Notes |
|---|---:|---|---|
| Task ID | 1 | `Task.job_name`, lines 46 and 77-78 | Random suffix is added at construction; fixed within the episode. |
| Pending-list order | 3 | `enumerate(env.current_tasks)` | Adapter stores `original_index`; no separate field exists. |
| Origin DC | 1 | `origin_dc_id`, lines 66-68 | Stable configured `dc_id`, not an environment action. |
| CPU demand | 1 | `cores_req`, line 49 | Absolute CPU cores. |
| GPU demand | 1 | `gpu_req`, line 50 | Absolute GPU units. |
| Memory demand | 1 | `mem_req`, line 51 | GB. |
| Duration | 1 | `duration`, line 48 | Minutes. |
| Remaining duration | 3 | For unscheduled current tasks, full duration; for a scheduled task, `finish_time-current_time` | No dedicated field. |
| SLA/deadline | 1 | `sla_deadline`, lines 58-60 | Timestamp. |
| Remaining SLA | 3 | `sla_deadline-env.current_time` | Minutes; may be negative after violation. |
| Arrival time | 1 | `arrival_time`, line 47 | Timestamp. |
| Data/transfer size | 1 | `bandwidth_gb`, line 52 | GB. |
| Task type | 4 | None | No task class/category field. |
| Defer count | 4/5 | None | `wait_intervals` counts failed scheduling in a DC queue, not global defer actions. |
| Ever deferred | 1 | `temporarily_deferred`, line 75 | Set true by `TaskSchedulingEnv.step`; never reset to false in the observed code. |
| Queue wait count | 1 | `wait_intervals`, line 64 | Incremented by `SustainDC.try_to_schedule_task`, lines 237-243. |
| Start/finish time | 1 | `start_time`, `finish_time`, lines 55-56 | Normally unset for current unscheduled tasks. |
| Destination | 1 | `dest_dc_id`, lines 70-72 | Normally unset until routing. The mutable `dest_dc` object is not exposed. |
| SLA met | 1 | `sla_met`, line 61 | Meaningful after completion. |

`utils.workload_utils.extract_tasks_from_row`, lines 73-105, defines actual units and scaling: duration in minutes, CPU cores, GPU units, memory GB, bandwidth GB, and one-based configured origin IDs.

## Datacenter fields

Primary sources: `envs.sustaindc.sustaindc_env.SustainDC` and `envs.sustaindc.dc_gym.dc_gymenv`.

| Requested field | Class | Source / derivation | Notes |
|---|---:|---|---|
| DC ID | 1 | `dc.dc_id`, `SustainDC.__init__` line 45 | Stable configured ID. |
| CPU total/available | 1 | `total_cores`, `available_cores`, lines 57 and 80-83 | Absolute cores. |
| GPU total/available | 1 | `total_gpus`, `available_gpus`, lines 58 and 80-83 | Absolute GPU units. |
| Memory total/available | 1 | `total_mem_GB`, `available_mem`, lines 59 and 80-83 | GB. |
| Current running tasks | 2 | `running_tasks`, lines 103-105 | Mutable task list; adapter exposes only immutable IDs/counts/release times. |
| Current queued tasks | 2 | `pending_tasks`, lines 103-105 | Mutable deque local to each DC. |
| In-transit tasks | 2 | `env.in_transit_tasks`, `TaskSchedulingEnv` lines 43-45 and 255-265 | Adapter aggregates immutable reservations by destination. |
| Expected resource release | 3 | Running-task `finish_time` values | Exact per-task finish timestamps exist; no aggregate capacity forecast API. |
| Schedulable capacity | 3 | available minus queued and in-transit reservations | Conservative external reservation; distinct from raw available capacity. |
| Electricity price | 1 | `price_manager.get_current_price()` | USD/MWh; manager loads `Price (USD/MWh)` at `utils.managers` lines 513-526. Negative wholesale prices are present and valid. |
| Carbon intensity | 1 | `ci_manager.get_current_ci(norm=False)` | gCO2e/kWh; method at `utils.managers` lines 295-299. |
| Current total power | 1/2 | `dc.dc_info['dc_total_power_kW']` | Step diagnostic dictionary; snapshot it. Defined in `dc_gym.py` lines 229-244. |
| IT power | 1/2 | `dc.dc_info['dc_ITE_total_power_kW']` | kW. |
| Cooling power | 1/2 | `dc.dc_info['dc_HVAC_total_power_kW']` | kW; CT cooling plus compressor in current code. |
| Internal temperature | 1/2 | `dc.dc_info['dc_int_temperature']` | Degrees C. |
| Ambient temperature | 1/2 | `dc.dc_info['dc_exterior_ambient_temp']` | Degrees C. |
| CRAC setpoint | 1 | `current_crac_setpoint` or `dc_info['dc_crac_setpoint']` | Degrees C; not controlled in this stage. |

Power and temperature values are initialized to zero/default values at reset (`dc_gym.py` lines 137-153) and become modeled values after a datacenter step (`dc_gym.py` lines 229-244).

## Network fields

| Requested field | Class | Source / derivation | Notes |
|---|---:|---|---|
| Cost per GB | 1/2 | `cluster_manager.transmission_matrix` | Mutable pandas table loaded by `utils.transmission_cost_loader`, lines 6-23; adapter copies scalar values. |
| Task transfer cost | 3 | matrix cost per GB multiplied by `task.bandwidth_gb` | USD. Region mapping is in `utils.transmission_region_mapper`. |
| Task transfer delay | 3 | `data.network_cost.network_delay.get_transmission_delay` | Seconds; depends on provider, source, destination, and task GB. |
| Link capacity/congestion | 4/5 | None | Needed for a network-capacity-aware future MPC. |
| In-flight arrival time | 2 | `env.in_transit_tasks` | Existing tuples contain arrival time, mutable task, and destination name; adapter exposes copied reservations. |

## Exogenous and forecast signals

- Current time: `env.current_time` (public).
- Current price and carbon intensity: public manager getters.
- Current weather: public `Weather_Manager` getters where available.
- Carbon forecast: `CI_Manager.get_forecast_ci()` exists but is normalized and limited to the configured future steps.
- Future prices: `ElectricityPrice_Manager.get_future_prices(n)` exists.
- Future tasks: no forecast or deterministic arrival interface for an external controller; only the full internal workload table exists. Directly reading future rows would couple an MPC to simulator internals and is not used here.
- Future available resources: no consolidated forecast; it can be partly derived from running task finish times but not from pending/in-transit scheduling outcomes.

## Snapshot policy

The external adapter may read these objects but must never return task objects, datacenter objects, deques, lists, pandas DataFrames, or manager arrays. It returns frozen dataclasses containing scalar values and tuples. Task order is preserved with `original_index`; raw and conservatively schedulable capacities are separate fields.

## Gaps for rolling MPC

Minimum missing interfaces are: future task arrivals, explicit defer count/history, immutable task and DC getters, network capacity/congestion, consistent future price/CI values in physical units, and a reservation/release forecast that includes running, queued, and in-transit tasks. Thermal forecasts are also absent if HVAC later becomes a control variable.
