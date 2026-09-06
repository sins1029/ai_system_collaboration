# 项目研究逻辑重建

审计日期：2026-08-18

## 1. 审计口径

本报告讨论的是当前工作区中新形成的 SustainCluster 多数据中心调度研究线，而不是把根仓库中所有历史模块都合并成一个模糊课题。当前仓库存在两层事实：

1. Git `HEAD=d49870b` 与根 `README.md` 仍把已版本化主包定义为单数据中心热电环境和单中心任务队列；
2. 当前磁盘上的 `src/sustaincluster_mpc/`、`src/sustaincluster_imitation/`、对应 configs、reports、datasets 和 checkpoints 均未被 Git 跟踪，但构成了最新的多数据中心研究工作。

因此，“项目目前真正研究的问题”应按当前研究线定义，同时把尚未完成的仓库整合状态列为审计发现，不能假装根项目叙述已经同步。

## 2. 当前研究对象

当前研究对象是：

> 在多数据中心环境下，对动态到达的 AI/GPU 任务，在不同时间与不同数据中心之间进行逐任务调度，并同时考虑 CPU、GPU、memory、SLA、electricity、carbon、transmission、transfer delay、task defer 与 spatial migration。

这一定义与当前代码大体一致：

- SustainCluster `Task` 提供 CPU、GPU、memory、duration、bandwidth、arrival 与 SLA deadline；
- 多数据中心状态包含资源、价格、碳强度、队列、运行中和传输中任务；
- 环境执行 defer、跨中心传输、目的中心排队、资源占用、完成与 SLA 统计；
- MPC 目标显式包含 electricity、carbon、transmission、waiting/defer、SLA risk 与 terminal backlog；
- BC/RL 输出 `defer` 或语义化目的 `dc_id`，再由适配器转换为环境位置动作。

但必须补充两个实现边界：

- 当前 SustainCluster 环境 reward 配置实际只有 electricity price 与 carbon emissions 两项；SLA 与 transmission 没有进入当前 SAC reward，虽然它们被单独统计或进入 MPC 目标。
- 网络 transfer delay 和 cost 已建模，但链路容量与拥塞没有建模。

所以“同时考虑全部目标”准确地描述了研究目标和整体评估面，不代表每个现有算法都已用同一标量目标联合优化所有指标。

## 3. SustainCluster 的角色

SustainCluster 是实验环境，不是本项目的算法创新。它负责：

- workload 与 task arrival；
- external current tasks、deferred、DC pending、running 与 in-transit 生命周期；
- CPU/GPU/memory 容量与释放；
- 数据中心、价格、碳强度、传输成本和延迟；
- SLA 统计、reward 计算和 `env.step()` 状态推进。

本项目研究的是外部调度器：

```text
SustainCluster state snapshot
        ↓
external scheduler (MPC / BC / RL)
        ↓
one semantic decision per current task
        ↓
dynamic dc_id → environment-action adapter
        ↓
SustainCluster env.step(actions)
```

SustainCluster 原仓库保持 `main@3f6ea95c` 且工作区干净；当前研究代码通过只读状态适配器与动作适配器集成，没有修改外部仓库。

## 4. 起点：single-action 与 multi-action 的矛盾

### 4.1 single-action

`single_action_mode=true` 时，环境把动态任务集合聚合为固定维 observation，并接受一个离散动作。该动作被广播给全部当前任务。

优势是标准 PPO/APPO/SAC 接口容易使用，observation/action 维度固定。代价是同一步内 GPU 需求、SLA、duration、origin、bandwidth 与 transfer delay 不同的任务仍得到同一目的地或 defer 决策，无法表达真正的逐任务调度。

### 4.2 multi-action

`single_action_mode=false` 时，环境返回一个 per-task observation 列表，并要求动作列表长度严格等于 `len(env.current_tasks)`。每个元素代表一个任务的 defer 或目的中心。

这保留了逐任务表达能力，却把学习问题变成动态集合决策：

- pending task 数量变化；
- observation/action 列表长度变化；
- 联合动作组合随任务数指数增长；
- 策略同时要学习任务匹配、容量耦合、defer、SLA、migration、transfer 与长期收益；
- SustainCluster 的 Gym space 只描述单个任务的 `Discrete` 动作，不是完整可变长联合空间。

项目真正的起点因此是：

> 如何保留 multi-action 的逐任务调度能力，同时降低 RL 直接学习复杂联合动作空间及基础约束规律的难度？

## 5. 为什么引入 MPC

引入 MPC 的原因不是“算法更先进”，而是其结构正好覆盖 multi-action RL 最难从随机探索中重新发现的知识：

- CPU/GPU/memory 联合容量；
- 任务只能选择一次；
- duration 跨时段占用；
- SLA completion deadline；
- defer 与 terminal backlog；
- migration、transmission cost 与 transfer delay；
- running release、queued/in-transit reservation；
- future price/carbon/capacity/arrival forecast。

最初假设是：

> 能否利用显式模型和约束知识，避免 RL 从随机探索中重新学习所有基础调度规律，再把优化器知识迁移给神经策略？

由此产生了 `MPC → expert data → BC → RL warm start` 路线。

## 6. 因果研究链

```text
问题 1：multi-action RL 直接训练困难
        ↓
尝试：建立外部逐任务模型优化器
        ↓
实现：one-step 与 H=1/2/4/8 rolling MILP
        ↓
发现：MPC 能稳定产生合法的逐任务动作并执行真实闭环
        ↓
问题 2：H=4 比同口径 H=1 是否显著更好？
        ↓
实验：相同优化器、目标、约束、种子和 96-step episode 的公平比较
        ↓
发现：H=4 有可构造的前瞻能力，但在多数公平场景中与 H=1 相同或只微幅变化；
      normal/high-load 的 SLA 甚至略差
        ↓
推论：MPC 的价值不能只等同于“永久在线长时域求解”
        ↓
问题 3：模型知识能否迁移给神经网络？
        ↓
实现：可部署 causal H=4 expert dataset + Behavior Cloning
        ↓
发现：BC 的 overall accuracy 96.98%，assign accuracy 90.49%，migration F1 94.79%
        ↓
问题 4：BC 能否脱离在线 MPC 控制真实环境？
        ↓
实验：H=4/H=1/BC/random 的真实 multi-action 闭环
        ↓
发现：BC 已能独立闭环；完成数与 SLA 几乎追平 H=4，联合运行成本低 0.376%，
      但 reward 更差、transmission 与 migration 更高
        ↓
问题 5：RL 能否在专家初始化后继续改善？
        ↓
实验：random actor 与 BC actor 的配对 SAC warm start
        ↓
发现：BC 初始 reward、SLA 与 expert agreement 显著更好；
      vanilla SAC 更新后 BC reward、SLA 与 agreement 同时退化
        ↓
当前问题：MPC 应是 teacher、conditional corrector，还是 permanent executor？
```

## 7. 按研究问题划分的完成阶段

| 阶段 | 研究问题 | 当前答案 |
|---|---|---|
| 1 环境与 multi-action | 能否直接执行逐任务动作？ | 能。动态动作映射、任务顺序和空任务路径已验证。 |
| 2 模型优化器 | 显式优化器能否稳定逐任务调度？ | 能。one-step 与 rolling MPC 已真实闭环，公平比较中无 infeasible/illegal/overflow。 |
| 3 时域价值 | H=4 是否明显优于 H=1？ | 仅在特定 future-resource 场景有明确收益；普通场景多数相同或近似。 |
| 4 知识迁移 | AI 能否学习 MPC 路由？ | 能。不能只看 overall；assign accuracy 与 migration F1 也较高。 |
| 5 AI 在线执行 | BC 能否不调用 MPC 独立运行？ | 能。在 5 个新 seed 的真实 multi-action 闭环中已执行。 |
| 6 RL 微调 | BC 后加入 vanilla SAC 是否改善？ | 当前否。初始化成功，但在线更新发生策略漂移与 SLA 退化。 |
| 当前架构决策 | A/B/C 哪个应成为最终架构？ | 尚不能最终确定；A 有最多直接证据，B 是领先待验证候选，C 尚无直接实验。 |

## 8. 更准确的论文研究问题候选

1. **RQ1：** Can model-based expert knowledge improve the trainability of fine-grained multi-action RL scheduling?
2. **RQ2：** Can a learned task-level policy replace most online MPC decisions while retaining SLA and resource feasibility?
3. **RQ3：** Can event-triggered MPC correction achieve a better safety-computation tradeoff than pure learned scheduling or full-time MPC?
4. **RQ4：** Does high-level RL plus low-level MPC outperform fixed-objective MPC under forecast error and workload distribution shift?

当前实验基础最完整的是 RQ1，但它已经接近一个阶段性肯定答案；最适合作为下一篇完整主线的是 RQ2，并自然过渡到 RQ3。RQ4 目前缺少已定义且有决策权的高层动作、实现与数据，暂不应抢占主线。

