# 前后端接口合同

## 当前状态

```text
CURRENT STATUS: NO FROZEN WEB API YET
```

仓库当前没有正式 Flask/FastAPI 服务。下面定义的是 `PROPOSED DASHBOARD CONTRACT`，用于前后端并行开发和 fixture 对齐，不表示 REST API 已实现。

## Dashboard Data Contract v1

建议顶层结构：

```yaml
contract_version: dashboard-v1-proposed
mode: frozen_replay
system_summary: {}
datacenters: []
current_tasks: []
decisions: []
forecast: []
metrics: {}
timeline: []
```

### `system_summary`

- `timestamp`
- `scenario`
- `controller`
- `total_tasks`
- `running_tasks`
- `waiting_tasks`
- `completed_tasks`

### `datacenters[]`

- `id`
- `cpu_utilization`
- `gpu_utilization`
- `memory_utilization`
- `electricity_price`
- `carbon_intensity`
- `running_tasks`
- `reserved_tasks`

### `current_tasks[]`

- `task_id`
- `priority`
- `source_dc`
- `cpu_request`
- `gpu_request`
- `memory_request`
- `waiting_steps`
- `sla_remaining_steps`
- `status`

### `forecast[]`

- `horizon_minutes`
- `predicted_task_count`
- `predicted_cpu_pressure`
- `predicted_gpu_pressure`
- `predicted_memory_pressure`
- `information_class`

`information_class` 必须允许 UI 区分 deployable forecast、persistence 和离线 Oracle。生产或演示默认不得暴露 Oracle。

### `decisions[]`

- `task_id`
- `source_dc`
- `target_dc`
- `controller`
- `action`
- `reason`

`reason` 是拟议解释字段，当前数据库不保证已有自然语言解释。backend 可以从 objective component、capacity 和 action metadata 生成结构化解释，但不得伪造 optimizer 内部因果。

### `metrics`

- `total_cost`
- `electricity_cost`
- `carbon_cost`
- `transmission_cost`
- `waiting_cost`
- `sla_cost`
- `backlog_cost`
- `sla_violation_rate`

### `timeline[]`

- `step_index`
- `timestamp`
- `event_type`
- `task_id`
- `datacenter_id`
- `details`

## 数据来源映射

| 合同区域 | 首选来源 |
|---|---|
| system summary | `experiment_runs` + 当前 `simulation_results` |
| datacenters | simulator snapshot 或 replay adapter |
| tasks | `tasks` + `run_task_events` + `run_task_outcomes` |
| decisions | `run_agent_decisions` 或调度实验的小型 decision replay |
| forecast | forecast manifest/summary 或后端内存中的 deployable prediction |
| metrics | `run_metrics` + `simulation_results` |
| timeline | `run_task_events` + `simulation_results` |

## 版本与安全规则

1. 正式实现前把状态保持为 `proposed`。
2. 字段新增应向后兼容；删除或改义需要新 contract version。
3. 所有比例明确为 `0..1` 或 `0..100`，不可混用。
4. 时间统一为带时区 ISO 8601。
5. 前端不接收原始 checkpoint、完整 Parquet 或 privileged future。
6. replay 响应必须标记 `mode: frozen_replay`，不得伪装实时生产控制。
