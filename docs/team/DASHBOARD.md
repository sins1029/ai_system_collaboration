# 驾驶舱开发说明

## 产品定位

竞赛驾驶舱优先展示真实冻结实验 replay。它不是 live production cloud control，也不应暗示系统正在操作真实云资源。

推荐第一版围绕一条可复核的 15 分钟粒度轨迹设计：用户选择 scenario、controller 和 run，然后播放、暂停、拖动时间轴，并查看同一 step 的任务、数据中心、预测、决策与成本。

## 页面布局

### 顶部：系统 KPI

- 当前时间、scenario、controller、replay 状态。
- total/running/waiting/completed tasks。
- SLA violation、total cost、electricity、carbon。

### 左侧：任务流

- 新到达、等待、运行、完成任务。
- priority、资源请求、等待时间和 SLA 风险。
- 支持按状态和优先级筛选。

### 中部：DC1-DC5 状态

- CPU、GPU、memory utilization。
- running/reserved task count。
- electricity price 与 carbon intensity。
- 当前调度目标和容量告警。

桌面端优先使用紧凑表格、条形资源指示和可比较的小倍图。

### 右侧：未来 60 分钟预测

- `t+60` task、CPU、GPU、memory pressure。
- 模型和 information class。
- 清楚区分 deployable Transformer、Persistence 与离线 Oracle。

H60 value audit 不支持 H60 作为默认控制器。预测面板只能说明 forecast，不得自动宣称产生控制收益。

### 下方：调度决策时间线

- task、source DC、target DC、action、controller。
- 同一时间点的 arrival/start/defer/complete 事件。
- 允许展开结构化 reason 或 objective component。

### 底部：成本与约束

- total stage cost。
- electricity、carbon、transmission/migration。
- waiting、SLA、backlog。
- 支持当前 step、累计值和固定窗口对比。

## 推荐交互

- Run selector：选择 SQLite 中已完成的 run。
- Timeline scrubber：以 step 为单位回放。
- Play/pause、单步前进/后退、速度选择。
- Controller comparison：仅在两个 run 的 scenario、数据和时间对齐时启用。
- Empty/loading/error：缺数据时明确显示，不用零值伪装。
- Provenance drawer：显示 Git commit、dataset、seed、config 和 contract version。

## 数据来源

- SQLite：`experiment_runs`、`simulation_results`、`run_metrics`、任务事件/结果/决策。
- 小型 CSV：研究 summary、slice 和 controller comparison。
- JSON manifest：数据 hash、模型、split 和协议 provenance。
- 内存 adapter：五 DC 的结构化 snapshot。

前端不直接读取大型 Parquet、checkpoint、Oracle future 或个人文件路径。

## 验收建议

1. 同一个 replay step 的 KPI、任务、DC 和时间线一致。
2. 时间与比例单位明确，不混用百分数和小数。
3. 五 DC 在 1366x768 和常见宽屏下可比较，不发生文本重叠。
4. replay 标识始终可见。
5. 无完整数据时可用小型 fixture 开发并展示 empty state。
6. UI 不出现“H60 显著优于 H1”或“实时生产控制”等错误表述。
