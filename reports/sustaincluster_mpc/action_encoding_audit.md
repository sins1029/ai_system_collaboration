# SustainCluster action encoding audit

Date: 2026-07-17
Repository commit: `3f6ea95cb835b89ba50b0ef76d66d14b8037643e`

## Conclusion

Environment actions are positional datacenter indexes, not stable datacenter IDs. The mapping must be rebuilt from the ordered values of `env.cluster_manager.datacenters` after every reset.

For the real reset used in this stage (`seed=123`), the ordered datacenters are:

| Position | Dictionary key | `dc_id` |
|---:|---|---:|
| 0 | DC2 | 2 |
| 1 | DC4 | 4 |
| 2 | DC5 | 5 |
| 3 | DC1 | 1 |
| 4 | DC3 | 3 |

Therefore, with defer disabled, actions `0,1,2,3,4` map to `dc_id 2,4,5,1,3` for this reset. They do not map to `dc_id 0,1,2,3,4` or `1,2,3,4,5`.

## Active YAML configuration

Source: `configs/env/sim_config.yaml`.

- Line 11: `shuffle_datacenters: true`
- Line 12: `strategy: manual_rl`
- Line 16: `disable_defer_action: true`
- Line 17: `single_action_mode: true`
- Line 20: `aggregation_method: average`

The external multi-action runner deep-copies this configuration in memory and sets `single_action_mode=false`. It may set `disable_defer_action=false` in memory for defer validation, but it never writes the YAML file.

## Action-space construction

Class: `envs.task_scheduling_env.TaskSchedulingEnv`

Function: `__init__`, lines 57-68 and 76-116.

| Mode | Defer setting | Gym action space | Meaning |
|---|---|---|---|
| single | disabled | `Discrete(num_dcs)` | One positional action is applied to every current task. |
| single | enabled | `Discrete(num_dcs + 1)` | `0=defer`; `1..N` select ordered DC positions `0..N-1`. |
| multi | disabled | `Discrete(num_dcs)` per task | One positional action per task. |
| multi | enabled | `Discrete(num_dcs + 1)` per task | `0=defer`; `1..N` select ordered DC positions `0..N-1`. |

The multi-action environment exposes a single `Discrete` space because each element of the externally supplied list is validated against that per-task space.

## Step-time mapping

Class: `TaskSchedulingEnv`

Function: `step`, lines 255-386.

1. Line 267 creates `dc_list_values = list(cluster_manager.datacenters.values())`.
2. Defer disabled: lines 280-282 and 325-327 add one to the agent action.
3. Defer enabled: action zero is preserved as defer at lines 283-285 and 328-330.
4. Assignment selects `dc_list_values[action - 1]` at lines 304 and 346.
5. The selected object's real `dc_id` is copied into the task at lines 305 and 347.

This creates the following general mapping:

- Defer disabled: `environment_action = ordered_dc_position`.
- Defer enabled: `environment_action = ordered_dc_position + 1`; `0=defer`.
- A semantic `dc_id` must first be located in the current ordered datacenter list.

Single and multi modes use the same mapping. Their only difference is whether one action is broadcast to all current tasks or one action is consumed for each task.

## Why the mapping changes

Class: `simulation.cluster_manager.DatacenterClusterManager`

Function: `reset`, lines 93-130.

When `shuffle_datacenter_order` is true, lines 116-119 shuffle dictionary items and replace `self.datacenters`. Since `TaskSchedulingEnv.step` later indexes the dictionary's ordered values, a reset can change the action-to-`dc_id` mapping.

Datacenter configuration IDs are stable and one-based (`1..5`) in `configs/env/datacenters.yaml`, lines 2, 17, 32, 47, and 62. Task origins use these real IDs: `utils.workload_utils.assign_task_origins`, lines 23-45, chooses from configured `dc_id` values and writes `task.origin_dc_id`.

## Defer behavior

- Defer is legal only when `disable_defer_action=false`.
- The defer action is always zero when enabled.
- Defer appends the task to `env.deferred_tasks` and sets `task.temporarily_deferred=true`: `TaskSchedulingEnv.step`, lines 297-301 and 340-344.
- `_load_new_tasks`, lines 389-397, prepends deferred tasks to newly loaded tasks on the next state, preserving deferred-task order before new arrivals.

## Action length and invalid values

- In multi-action plus `manual_rl`, lines 318-320 assert `len(actions) == len(env.current_tasks)`. A mismatch raises `AssertionError` before task routing.
- Empty tasks therefore require `[]`, which satisfies the assertion.
- The original environment does not call `action_space.contains` inside `step`.
- A negative or too-large assignment can index the Python list incorrectly or raise `IndexError`; action zero can also accidentally defer if a caller uses the wrong mode.
- The external action adapter must validate count, type, range, defer legality, and current mapping before calling `env.step`.

## Required adapter invariant

For every scheduler state, output action `i` must correspond to `env.current_tasks[i]`. The adapter must preserve `original_index`, verify task identity, translate semantic `dc_id` through the reset-specific positional mapping, and reject stale or impossible decisions.
