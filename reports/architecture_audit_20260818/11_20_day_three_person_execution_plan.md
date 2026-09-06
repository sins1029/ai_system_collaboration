# 多数据中心调度算法 20 天三人执行计划

制定日期：2026-08-21  
计划口径：20 个有效工作日，3 人并行，共约 60 人日  
目标：初步完成可运行、可评估、可复现的多数据中心逐任务调度算法

## 1. 结论先行

20 天内可以初步完成调度算法，但“完成”必须限定为：

> 在现有 SustainCluster multi-action 环境、H1/H4 MPC、expert dataset 和 BC checkpoint 基础上，完成 Architecture A/B 的最小判别，并交付一个可连续运行 96-step episode、具有明确风险检查与 fallback、能统一输出 SLA/成本/迁移/调用率指标的初版调度器。

建议主交付采用 Architecture B 的最小实现：

```text
SustainCluster state
        ↓
frozen BC / regularized actor proposal
        ↓
transparent risk detector
   ├─ normal → direct execute
   └─ risky  → H4 MPC full-batch correction
        ↓
ActionAdapter → env.step(actions)
```

同时用小预算 regularized SAC 判断 Architecture A 是否仍有竞争力。Architecture C、thermal/HVAC、storage、renewable、真实中国电价/碳数据和电网互动不纳入这 20 天。

## 2. 与原技术路线的关系

原技术路线是：

```text
数据底座
→ 单中心热电模型
→ 数学优化
→ 多中心时空调度
→ 模仿学习 / PPO / SAC
→ 泛化与电网扩展
```

当前仓库并不是从零开始：

| 原路线阶段 | 当前状态 | 20 天安排 |
|---|---|---|
| 数据与环境 | SustainCluster、workload、15 min step、processed expert data 已有 | 只冻结版本和实验窗口，不重建数据底座。 |
| 数学优化 | H1/H4 rolling MPC 已实现并闭环 | 作为 teacher、benchmark 和风险纠正器。 |
| 多中心调度 | multi-action、defer/migration/transfer/SLA 已打通 | 继续作为主研究对象。 |
| 模仿学习 | BC offline 与真实闭环已完成 | frozen BC 作为交付基线。 |
| 强化学习 | BC-init SAC 起点改善，但 vanilla update 退化 | 只做正则化小预算判别，不扩大训练。 |
| 泛化/物理扩展 | 尚未系统完成 | 只做最小 stress check；正式扩展放到第 20 天以后。 |

因此本计划不是重走原路线，而是集中完成原路线“阶段 4 AI 调度”中的架构收口。

## 3. 20 天主目标与非目标

### 3.1 主目标

1. 冻结统一 state/action/reward/metrics/seed 接口。
2. 完成 regularized direct RL 的小预算实验，判断 A 是否可保留。
3. 完成 transparent event trigger + H4 correction，判断 B 是否成立。
4. 在第 12 天完成 A/B 架构门评审，只继续一条主线。
5. 第 16 天形成初版调度器 release candidate。
6. 第 20 天交付代码、配置、测试、结果 JSON、对比报告和运行说明。

### 3.2 明确不做

- 不重新训练大规模 expert/BC dataset；
- 不开展 Architecture C 完整 hierarchical RL；
- 不加入 thermal/HVAC/storage/grid control；
- 不做论文级大样本显著性实验；
- 不更改 SustainCluster 源码；
- 不同时开发 PPO、SAC、DQN 多条算法线；
- 不把“reward 提升”作为唯一验收，SLA 和安全指标拥有否决权。

## 4. 初版调度器定义

### Observation

- 当前 233-d per-task feature 与 feasible mask；
- task set 联合容量需求；
- SLA slack、pending/GPU burst、forecast reserve/release；
- actor entropy/margin；
- 可选的简单训练分布分位数 OOD score。

### Action

每个 current task 输出一个 semantic action：`defer` 或 `dc_1..dc_5`。

### Decision frequency

每 15 分钟处理一次动态 task set。

### Risk trigger MVP

任一条件满足则调用 H4 MPC 对整批动作纠正：

1. actor proposal 导致某 DC 的联合 CPU/GPU/memory demand 超过保守可用量；
2. 最小 SLA completion slack 不超过 1 step；
3. pending count 或 GPU demand 超过训练分布 P99；
4. actor top-1/top-2 margin 低于阈值；
5. horizon 内存在显著 capacity release / forecast reservation event。

首版使用透明规则，不在 20 天内训练复杂 risk detector。

### Fallback

- H4 optimal：执行 H4 full-batch correction；
- H4 timeout/error：降级 H1；
- H1 仍失败：执行 conservative defer/origin-safe rule，并记录原因；
- 所有路径必须经过动态 `dc_id→environment action` adapter。

## 5. 三人分工

### 成员 1 — 系统集成与实验负责人

主要职责：

- 冻结 Git/config/data/checkpoint/seed 清单；
- 统一实验入口、结果 JSON schema 和 metrics；
- 维护 normal/high-load/stress evaluation matrix；
- 集成 A/B 路径与 fallback；
- 负责回归、可复现性、最终运行说明和汇总报告。

验收产出：一条命令可运行全部基线和候选，输出统一结果。

### 成员 2 — MPC、风险检测与安全负责人

主要职责：

- 完善 H1/H4 solver diagnostics、timeout 和 fallback 状态；
- 实现 task-set joint-capacity/SLA/future-event risk features；
- 实现 event-trigger wrapper 与 full-batch MPC correction；
- 标注 risky events，计算 trigger precision/recall 与 call rate；
- 负责 illegal、overflow、solver failure、安全尾部指标。

验收产出：不改 SustainCluster 的可解释 trigger + correction 模块。

### 成员 3 — BC/RL 与策略质量负责人

主要职责：

- 冻结 BC checkpoint 与 direct policy baseline；
- 实现 BC/KL regularized SAC；
- 选择一种额外稳定化方法：critic warmup 或 expert replay，二选一；
- 修正或显式记录 reward/SLA/transmission 的目标接口；
- 输出 agreement、KL drift、confidence/margin 与学习曲线。

验收产出：小预算 regularized A 结果和可供 B 使用的 policy confidence 信号。

### 共同规则

- 成员 1 审核接口与结果可复现性；
- 成员 2 审核所有候选的约束与安全表述；
- 成员 3 审核学习实验是否公平；
- 每项核心代码至少一人交叉 review；
- 每天 15 分钟同步，报告 blocker、当日 JSON/测试产物和次日目标。

## 6. 工作包与人日预算

| 工作包 | 主责 | 复核 | 预计人日 | 输出 |
|---|---|---|---:|---|
| W1 版本与实验协议冻结 | 成员1 | 2、3 | 3 | manifest、metric spec、seed matrix |
| W2 baseline 一键复现 | 成员1 | 3 | 4 | H1/H4/BC/random 统一 runner |
| W3 regularized A | 成员3 | 1、2 | 9 | BC/KL SAC、小预算结果、漂移分析 |
| W4 trigger signals | 成员2 | 1、3 | 6 | capacity/SLA/burst/confidence/event detector |
| W5 MPC correction/fallback | 成员2 | 1 | 7 | B wrapper、H4→H1→safe fallback |
| W6 A/B 判别实验 | 成员1 | 2、3 | 6 | paired evaluation、decision memo |
| W7 主线集成与校准 | 三人 | 三人 | 9 | scheduler RC、配置、阈值 |
| W8 robustness smoke | 成员1、2 | 3 | 6 | unseen window/capacity/forecast stress |
| W9 测试、文档、交付 | 三人 | 三人 | 7 | tests、report、README、demo |
| W10 预留与问题修复 | 三人 | 三人 | 3 | blocker buffer |

总计约 60 人日。

## 7. D1-D20 时间线

| 时间 | 成员 1 | 成员 2 | 成员 3 | 阶段输出 / Gate |
|---|---|---|---|---|
| D1 | 冻结当前 Git、configs、datasets、checkpoint 清单 | 核对 H1/H4 diagnostics 与 failure path | 复核 BC/SAC baseline、reward 与 mask | Kickoff；范围、非目标和负责人签字确认 |
| D2 | 定义统一 JSON schema、seed/window matrix、验收阈值 | 定义 5 类 trigger 信号及计算口径 | 确定 BC/KL + critic warmup 或 expert replay 方案 | M0：实验协议冻结，禁止中途改指标 |
| D3-D4 | 统一 H1/H4/BC/random runner 与 report aggregation | 实现 joint proposal capacity、SLA slack、burst detector | 实现 regularized actor loss 与日志 | 所有方法能走同一评估入口 |
| D5 | 复跑 frozen baselines，核对当前报告数值 | 添加 H4 timeout/H1/safe fallback skeleton | 做短 smoke，确认 BC 权重和 critic 配对 | M1：baseline freeze；回归必须全绿 |
| D6-D7 | 运行/汇总 paired 10k 小预算实验 | 并行搭建 event-trigger wrapper | 主跑 regularized A 与一个稳定化消融 | Experiment 1 数据形成 |
| D8 | unseen workload/capacity-shift 小评估 | 完成整批 MPC correction 与调用日志 | 分析 reward/SLA/agreement/KL drift | 判断 A 是否满足最低门槛 |
| D9 | 形成 A decision memo | 复核 A 的联合容量和安全结论 | 固化可用于 B 的 frozen/regularized actor | M2：A go/no-go；不继续调参拖延 |
| D10-D11 | 运行 BC、full-H4、simple-trigger、oracle-trigger 对照 | 主导 trigger threshold 初始校准 | 提供 confidence margin/OOD 分位数 | Experiment 2 数据形成 |
| D12 | 汇总 quality/safety/call-rate 三类指标 | 分析 false negative、over-trigger、fallback | 比较 actor 选择对 B 的影响 | M3：A/B 架构门评审，确定唯一主线 |
| D13-D14 | 集成胜出主线、统一配置与 CLI | 若选 B：校准 trigger；若选 A：加强安全 checker/fallback | 若选 A：稳定 regularized actor；若选 B：冻结 actor 与 confidence | scheduler alpha |
| D15-D16 | 连续 96-step 与多 seed 回归 | 检查 solver、trigger、资源/SLA tail | 检查 policy drift 与 inference | M4：初版 scheduler release candidate |
| D17 | unseen workload + GPU burst smoke | capacity release、forecast error、solver failure | 复核 shift 下 confidence/agreement | 最小 robustness matrix |
| D18 | 汇总失败案例与局限 | 修复安全路径和过度触发 | 修复策略/日志问题，不扩大训练 | RC2；停止新增功能 |
| D19 | 一键复现、结果清单、README | 安全与 fallback 文档 | 模型卡、训练与 checkpoint 文档 | delivery candidate |
| D20 | 最终测试、演示、20天总结 | 最终安全签字 | 最终策略签字 | M5：初版算法交付与下一阶段 backlog |

## 8. 关键里程碑

### M0 — D2：协议冻结

- 当前代码/数据/checkpoint/seed 有 manifest；
- 统一 metrics、decision rules 和结果 schema；
- 明确 20 天不做 C 和物理能源扩展。

### M1 — D5：基线冻结

- H1、H4、frozen BC、random 可由统一入口复现；
- 当前 138 项回归不退化；
- baseline 数值与现有 JSON 在允许误差内一致。

### M2 — D9：A 判别

- regularized A 有完整 paired 小预算结果；
- 若 final reward 不差于 BC、SLA 恶化不超过 1%、agreement 不低于 0.90、illegal/overflow=0，则 A 保留；
- 未达到则停止 direct RL 扩张，B 使用 frozen BC。

### M3 — D12：B 判别与主线选择

- simple-trigger 与 full-H4 有统一对照；
- 目标：MPC call rate 不超过 20%，SLA/联合物理成本与 H4 差异不超过 1%，risky-event recall 不低于 95%；
- 若只能靠超过 50% 的调用率达到质量，B 不视为成功 selective architecture。

### M4 — D16：初版调度器 RC

- 可连续运行真实 multi-action episode；
- 动态 task 数、空任务、DC order reset、defer、migration 均可处理；
- fallback 与 failure diagnostics 完整；
- 可重复输出统一 metrics。

### M5 — D20：交付

- 初版算法实现；
- configs、checkpoint/manifest、测试、结果 JSON；
- A/B 选择依据；
- 已知局限与下一阶段 robustness backlog；
- 一条命令复现实验和一份演示结果。

## 9. 验收标准

### 功能验收

- 96-step multi-action episode 连续运行；
- action count/order/mapping 全合法；
- H4→H1→safe rule fallback 可被测试触发；
- 所有策略走同一 state、metrics 和 environment path。

### 安全验收

- illegal action=0；
- resource overflow=0；
- solver failure 不静默；
- SLA、queue tail 和 trigger false negative 单独报告；
- 不能用环境 queue 不出现负资源替代联合容量审计。

### 性能目标

- B 目标：call rate≤20%，SLA/cost 与 H4 差≤1%，risky recall≥95%；
- A 目标：final reward≥frozen BC、SLA≤BC+1%、agreement≥0.90；
- 两者均不满足时，不伪造“算法完成”：交付 frozen BC + deterministic safety/fallback baseline，并把架构状态标记为未决。

### 工程验收

- 当前 138 项回归全部通过，新增模块有针对性测试；
- 不修改 SustainCluster；
- configs 不硬编码在算法中；
- 结果记录 commit/worktree、config hash、seed、checkpoint 与数据版本；
- README 和运行命令可由非作者复现。

## 10. 主要风险与降级方案

| 风险 | 最晚发现日 | 降级方案 |
|---|---:|---|
| regularized SAC 仍漂移 | D8 | D9 停止 A，冻结 BC，集中 B。 |
| reward 与 SLA/transmission 错位导致结论混乱 | D2 | 不临时大改 reward；统一分项评估，并把 reward variant 作为后续消融。 |
| trigger 过度调用 MPC | D11 | 先删除弱信号，保留 capacity/SLA/future-event 三类强 trigger。 |
| trigger 漏掉风险状态 | D12 | 降低阈值或增加 deterministic joint-capacity checker；宁可提高 call rate，不隐瞒漏报。 |
| H4 timeout/solver failure | D5 | H1 fallback，再降级 safe rule。 |
| 评估耗时超预算 | D6 | 保留 96-step、少量 paired seeds；停止新增训练量，不削减 metrics。 |
| A/B 都未达到目标 | D12 | 交付 frozen BC + safety checker + MPC fallback 的工程 MVP，架构研究结论保持未决。 |

## 11. 日常管理方式

- 每日上午：15 分钟站会，只讨论 blocker、接口和当天交付；
- 每日下午：每人提交可审计产物，如测试、JSON、曲线或 decision memo；
- D2、D5、D9、D12、D16、D20 为正式 gate；
- gate 后禁止继续为已淘汰方案调参；
- 所有实验先写 config 和 decision rule，再运行；
- 报告区分 implementation fact、experiment fact 与 possible explanation。

## 12. 20 天后的预计状态

### 可以达到

- 一个真实可运行的多数据中心逐任务调度器 MVP；
- 一个清楚的 A/B 架构选择或带证据的未决结论；
- direct learned policy、risk trigger、MPC correction 与 fallback 的统一闭环；
- 小规模 normal/stress/OOD 证据；
- 可复现的工程与实验交付。

### 尚不能宣称

- 论文级泛化与统计显著性；
- RL 已普遍超过 MPC；
- event trigger 已达到生产级安全保证；
- thermal/HVAC/storage/renewable 已与 task scheduling 联合优化；
- 多种 DC 规模和真实区域数据已充分验证。

## 13. 第 20 天后的推荐顺序

1. 用 2-4 周完成正式 robustness/generalization；
2. 固化论文问题为 learned policy replacement + selective MPC correction；
3. 再决定是否需要 high-level RL + low-level MPC；
4. 最后接回原技术路线中的 thermal/HVAC/storage 和电网扩展。

