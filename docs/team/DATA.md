# 数据说明

## 数据边界总览

| 数据 | 当前用途 | 是否随 Git 提供完整数据 |
|---|---|---|
| Alibaba2020 | 基线机制、预测、repaired MPC、Scenario A | 否 |
| SpotGPU2026 | 现代任务级 workload、Scenario B、Expert Dataset v3 | 否 |
| Alibaba GPU2026 Main | 小时级外部真实性与场景验证 | 否 |
| SustainCluster Energy Signal | 多区域电价与碳强度外部场景信号 | 由固定参考仓库提供 |
| 仓库样例数据 | 安装、单中心演示和测试 | 是，小规模 |

## Alibaba2020

Alibaba2020 用于早期机制开发、current-only baseline、Transformer workload forecast 和 repaired expert experiments。当前所谓 full-year workload 包含重复构造的 49-day cycle，不应描述为 365 天相互独立的真实 workload。预测实验使用经过审计的时间范围和 split，避免重复周期造成跨 split 泄漏。

原始 trace 和大规模派生数据不进入 Git。需要重建时，使用者必须自行确认数据来源、许可和本地路径。

## SpotGPU2026

SpotGPU2026 是当前现代任务级场景。冻结 v3 规模为：

- 17,670 continuous states
- 466,867 task decisions
- 4,278 nodes
- 10,412 GPUs

当前合同要点：

- CPU/GPU request scope 为 `PER_JOB`。
- `arrival_step = floor(submit_time / 900)`。
- `true_duration` 只进入 simulator truth。
- 主机内存不是源数据实测值，当前标记为 `MODELED_DC_MEMORY`。
- origin DC、bandwidth 和部分 SLA 字段是冻结建模字段。
- GPU model 只作为 metadata，不进入当前兼容约束。
- 五 DC 容量由完整 `node_info` 确定性守恒分区得到。

SpotGPU2026 v3 数据合同是稳定基础，但完整 workload、Expert Dataset v3、privileged future 和逐状态 value parquet 不公开提交。

## Alibaba GPU2026 Main

Alibaba GPU2026 Main trace 与 SpotGPU2026 task-level trace 不是同一数据。当前仅用于 hourly external realism、workload type 和 GPU-model realism 检查，不能被表述为精确的 15 分钟 task arrival 重建来源。

## SustainCluster Energy Signal

SpotGPU2026 不提供当前实验所需的完整多区域电价和碳强度。项目使用固定 SustainCluster commit 中的 2023 外部能源信号，标记为：

```text
SustainCluster 2023 EXTERNAL_SCENARIO_SIGNAL
```

这些信号不是 Alibaba2026 实测电价或实测碳强度。相关能源结论只适用于冻结的外部信号场景。

## 本地没有完整数据时

仍然可以：

- 安装项目并运行小型 smoke test。
- 阅读和修改源码、配置与测试。
- 开发 SQLite query、replay API、Dashboard 和可视化。
- 使用已提交的小型 summary、manifest 和样例数据。

无法：

- 全量复现 SpotGPU2026 v3 实验。
- 重训冻结 Compact Transformer。
- 重新生成完整 Expert Dataset v3。
- 运行依赖完整 checkpoint 或大规模 Parquet 的实验。

## 数据和许可责任

大型数据不会进入 Git。SpotGPU2026 派生数据的公开再分发许可仍需确认。开发者只能使用自己有权访问的数据，不得把原始 trace、完整派生数据或 privileged future 提交到协作分支。

SQLite 与 Parquet 的职责不同：SQLite 用于运行记录和 replay；Parquet 用于大规模离线训练与分析。不要为了 Dashboard 把完整 Parquet 导入 SQLite。
