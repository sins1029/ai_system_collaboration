# SustainCluster 确定性滚动时域混合整数 MPC 报告

日期：2026-07-17
SustainCluster commit：`3f6ea95cb835b89ba50b0ef76d66d14b8037643e`

## 范围与结论摘要

本阶段在外部包 `src/sustaincluster_mpc` 中完成了确定性滚动时域 MILP
骨架。SustainCluster 源码、YAML、checkpoint、Git 分支和 commit 均未修改。

闭环为：

`真实当前状态 -> H 步只读快照 -> SciPy/HiGHS MILP -> 仅执行 h=0 -> env.step -> 重建`

已支持 `H=1/2/4/8`、`no_future_arrivals`、`oracle`、
`noisy_oracle`。正常轨迹和压力实验共生成 98 条结果记录，所有闭环求解均无
infeasible 和资源越界。当前规模上，H=4 平均求解 2.715 ms，最大 8.331 ms，
具备在线运行余量。

## 实现模型

### 时间与状态

- 统一时间单位为 15 分钟离散 step。
- duration、SLA、running release、transit arrival 均在适配器边界转换为整数 step。
- running、DC pending、in-transit、external pending 分开建模。
- 价格和碳强度按各 DC 的物理序列构造 H 步时间线。
- future arrivals 仅按 `arrival step + origin DC` 聚合并预留容量，不创建未来逐任务整数变量。
- oracle 读取 trace 时保存并恢复 Python/NumPy RNG，测试确认不改变环境状态。

### 决策与约束

当前真实任务使用二元变量 `x[j,d,h]`，另为每个任务设置一个终端积压变量。
变量总数为：

`J * D * H + J`

约束包括：

1. 每个当前任务恰好选择一次时域内派发或终端积压；
2. CPU、GPU、memory 在任务完整 duration 内持续占用；
3. 任务到达目标中心后才开始执行；
4. completion 不晚于离散 deadline；
5. 已知队列、在途任务和预测任务预留后的逐时段容量不超限；
6. `H>1` 时禁止执行起点落在时域外的派发，防止反复把任务推到时域边缘；
7. `H=1` 使用唯一容量/价格格作为显式终端估计，以保持与 one-step 基线可比。

目标函数分解为 electricity、carbon、transmission、waiting/defer、SLA risk、
terminal backlog，全部权重可配置。优化器输出完整 H 步计划、首步语义决策、
真实环境 action、逐 DC 占用、目标分解、变量数和求解时间。

## 验证结果

测试命令结果：`33 passed, 1 skipped in 5.48s`。

跳过的是原 one-step 测试中依赖 `SUSTAINCLUSTER_REPO` 环境变量的重复集成项；
新增滚动测试中的真实仓库集成验证已执行。覆盖 H=1 等价、H=2/4/8、容量释放、
预测预留、低价等待、紧急 SLA、duration 占用、在途去重、终端积压、动态 action
映射、时域终端边界，以及真实三预测模式的只读性和可复现性。

- `pip check`：No broken requirements found
- SciPy：1.15.3
- NumPy：1.26.4
- 求解器：`scipy.optimize.milp` 内置 HiGHS，CPU
- 新增依赖：无

## 正常轨迹

seed 123，每组 8 步。滚动行是三种 forecast mode 的平均时间；三种模式在该片段的
首步决策、完成数、SLA 和成本完全相同。

| 方法 | H | 决策数 | defer | 完成 | SLA 违约 | 平均/最大求解 ms | 平均/最大整数变量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| one-step | 1 | 213 | 0.0% | 11 | 9 | 1.432 / 2.600 | 159.8 / 474 |
| rolling | 1 | 242 | 17.8% | 9 | 8 | 1.326 / 2.375 | 181.5 / 504 |
| rolling | 2 | 242 | 17.8% | 9 | 8 | 1.672 / 6.044 | 332.8 / 924 |
| rolling | 4 | 242 | 17.8% | 9 | 8 | 2.715 / 8.331 | 635.2 / 1764 |
| rolling | 8 | 242 | 17.8% | 9 | 8 | 4.747 / 12.143 | 1240.2 / 3444 |

rolling 比 one-step 多出的决策来自被 defer 后再次进入下一步的任务。该 8 步片段
没有形成预测容量瓶颈，因此 H>1 没有改变首步动作。这是“此片段没有收益”，不是
“MPC 总体没有收益”。

正常轨迹 H=4 的一步到达预测误差：

| forecast mode | task-count MAE | GPU-demand MAE | 首步决策是否变化 |
|---|---:|---:|---|
| no future arrivals | 27.50 | 7.65 | 否 |
| oracle | 0.00 | 0.00 | 否 |
| noisy oracle | 5.88 | 6.94 | 否 |

注意：one-step 与 rolling 的绝对目标值不能直接横比。rolling 增加了硬 completion
deadline、H 步 waiting/SLA 和 terminal backlog 项；正常轨迹中的目标值差异不是
实际电费差异。

## 压力场景

压力场景使用同一组确定性轨迹和统一压力权重。下表比较 one-step 与 H=4 oracle；
成本为闭环实际执行的 stage objective，适合在该表内部比较。

| 场景 | 方法 | 完成/积压 | defer | 平均等待 step | SLA 违约 | stage cost | electricity |
|---|---|---:|---:|---:|---:|---:|---:|
| A GPU burst | one-step | 6 / 0 | 50.0% | 1.00 | 2 | 2986.88 | 503.40 |
| A GPU burst | H4 oracle | 6 / 0 | 66.7% | 2.00 | 0 | 792.86 | 503.40 |
| B future low price | one-step | 3 / 0 | 0.0% | 0.00 | 0 | 399.39 | 399.38 |
| B future low price | H4 oracle | 3 / 0 | 50.0% | 1.00 | 0 | 71.10 | 7.99 |
| C capacity release | one-step | 1 / 0 | 66.7% | 2.00 | 0 | 715.93 | 100.43 |
| C capacity release | H4 oracle | 1 / 0 | 50.0% | 1.00 | 0 | 120.30 | 100.43 |
| D SLA conflict | one-step | 1 / 0 | 0.0% | 0.00 | 0 | 262.63 | 262.63 |
| D SLA conflict | H4 oracle | 1 / 0 | 0.0% | 0.00 | 0 | 312.63 | 262.63 |
| E center difference | one-step | 3 / 0 | 0.0% | 0.00 | 0 | 321.99 | 321.94 |
| E center difference | H4 oracle | 3 / 0 | 40.0% | 0.67 | 0 | 245.31 | 176.55 |

关键行为：

- A：oracle 为下一步紧急 GPU burst 主动留出容量，将 SLA 违约从 2 降到 0。
- B：MPC 等待一个决策步进入低价执行窗，electricity 项下降 98.0%。此收益来自已知价格时间线，三种 arrival forecast mode 相同。
- C：MPC 看到容量释放，比 one-step 少等待一个 step。
- D：低价窗可见，但硬 deadline 只允许立即派发，MPC 没有错误等待。
- E：H4 先使用大小中心的即时容量，再把第三个任务推迟到小型低价中心，electricity 项下降 45.2%。

A 场景 oracle 的 horizon 对比：

| H | 完成/积压 | SLA 违约 | 平均等待 step | stage cost |
|---:|---:|---:|---:|---:|
| 1 | 4 / 2 | 2 | 2.33 | 928.77 |
| 2 | 4 / 2 | 2 | 3.00 | 964.57 |
| 4 | 6 / 0 | 0 | 2.00 | 792.86 |
| 8 | 6 / 0 | 0 | 2.00 | 792.86 |

H=4 已覆盖该 burst 的关键到达与 duration；H=8 没有新增收益，只增加变量和时间。

## 预测误差敏感性

在 A、H=4 上，对 demand/duration/arrival 噪声统一使用 4 个强度乘数，每档 5 个固定
seed。`1.0` 对应 20% demand/duration 标准差与 0.75 step arrival 标准差。

| 噪声倍数 | 运行数 | 平均完成 | 平均 SLA 违约 | 违约 min/max | 平均 stage cost |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 5 | 6.0 | 0.0 | 0 / 0 | 792.86 |
| 0.5 | 5 | 6.0 | 0.0 | 0 / 0 | 792.86 |
| 1.0 | 5 | 6.0 | 0.0 | 0 / 0 | 792.86 |
| 2.0 | 5 | 5.8 | 0.2 | 0 / 1 | 808.24 |

中等误差未改变该场景的容量预留决策；误差加倍后出现性能退化。因此当前骨架对一定
误差有余量，但不是鲁棒 MPC，不能将 oracle 结果当作部署保证。

## 八项判断

1. **H>1 是否产生不同决策：**正常 8 步片段没有；A、B、C、E 压力场景有。D 场景因 deadline 正确保持立即执行。
2. **预测未来是否带来收益：**future-arrival oracle 在 A 将 SLA 违约 2 降为 0；B、C 的收益来自价格和已知容量时间线，不依赖 arrival forecast mode。
3. **何时主动 defer：**为未来紧急 GPU 任务留容量、等待低价、等待容量释放、等待小型低价中心；紧急 SLA 冲突时不 defer。
4. **误差是否抵消收益：**1 倍设定未抵消；2 倍时平均完成下降 0.2、平均 SLA 违约增加 0.2。
5. **规模增长：**变量严格按 `J*D*H+J` 线性增长；正常轨迹平均求解从 H1 的 1.326 ms 增至 H8 的 4.747 ms。
6. **H=4 是否适合在线：**当前最多 1764 个整数变量、最大 8.331 ms，远小于 15 分钟控制周期；H4 是当前场景的合理默认值。
7. **纯 rolling MPC 是否足够：**已经足够作为确定性、可解释、带硬约束的调度基座；尚不足以处理分布外预测误差、概率 SLA 和模型偏差。
8. **后续 RL 应学习什么：**学习预测残差、终端价值、风险裕度或目标权重/有限动作残差；CPU/GPU/memory、deadline 和 action 编码仍由 MPC 约束层保证，不应让 RL 无约束替换 MPC。

## RL 接口建议

后续若进入 RL+MPC，应保持当前模块边界：

1. `HorizonStateAdapter` 作为唯一只读环境状态入口；
2. forecast provider 返回带时间戳、单位和不确定度的聚合预测；
3. RL 仅输出有界的权重、风险 margin、terminal value 或候选残差；
4. `RollingHorizonOptimizer` 返回可行语义计划和诊断；
5. `SustainClusterActionAdapter` 仍负责 reset 后动态 `dc_id -> action` 映射；
6. 训练日志同时保存预测误差、MILP status、slack、变量数和首步/全计划目标。

本阶段按要求停止在纯 MPC，不实现 RL wrapper 或联合训练。

## 限制

- future arrivals 在 origin DC 做保守聚合预留，尚未优化其未来跨中心路由。
- DC pending 采用确定性容量预留，不是 SustainDC FIFO 的完整复制模型。
- 不建模网络带宽/拥塞、HVAC、储能、热状态或随机机会约束。
- H=1 使用终端容量估计；严格的传输后执行轨迹从 H>=2 开始完整表达。
- 正常轨迹仅运行 8 步，适合阶段验证，不等同于长期统计显著性实验。
- trace oracle 是实验上界，不是可部署预测器。

## 产物与版本状态

- 完整结果：`rolling_horizon_experiment_results.json`（98 条）
- 独立压力结果：`rolling_horizon_stress_results.json`（85 条）
- 时间审计：`temporal_model_audit.md`
- 外部仓库 HEAD 保持 `3f6ea95cb835b89ba50b0ef76d66d14b8037643e`
- 外部仓库 Git 工作区干净
- 未执行 pull、checkout、reset、依赖升级或正式训练
