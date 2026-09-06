# Architecture C — High-Level RL + Low-Level MPC

## 1. 定义

```text
fixed-dimensional global state
            ↓
       high-level RL
            ↓
 bounded intent / constraint parameters
            ↓
      low-level MPC every step
            ↓
 task-level semantic decisions
            ↓
 SustainCluster
```

RL 不直接输出每个任务动作；MPC 每步把固定维高层意图转换为可行的 task-level actions。

## 2. 统一接口

| 接口项 | Architecture C |
|---|---|
| Observation | 固定维 global state：各 DC 资源/价格/碳/forecast、task-set 按 urgency/GPU/duration/origin 的 histogram、backlog、uncertainty 与过去结果。 |
| Action | 固定维、带物理含义且有界的 high-level intent。首选 reservation/quota/budget/risk margins；不建议只输出任意无归一化 objective weights。 |
| Decision frequency | 可每 15 分钟，也可每 30–60 分钟更新 intent；low-level MPC 仍每 15 分钟执行。 |
| MPC role | 永久在线 task-level executor 和 hard-constraint layer。 |
| RL role | 改变 feasible region、admission/risk/migration policy 或长期资源预留。 |
| Hard constraints | CPU/GPU/memory、task uniqueness、deadline、transfer 与 action encoding 由 MPC；RL action 本身也要有范围与可行 default。 |
| Training method | high-level off-policy/on-policy RL，环境 transition 内含 MPC；需要对 solver failure、intent clipping 和 stale intent 明确定义。 |
| Execution path | aggregate observation→RL intent→MPC config/constraints→solve→adapter→environment。 |
| Fallback | RL 缺失/OOD 时回到固定 default intent 的 H=4/H=1 MPC。 |

## 3. 高层动作候选与决策权分析

| 候选输出 | 是否推荐 | 理由与风险 |
|---|---|---|
| 每 DC resource reservation | 推荐 | RL 可为未来 burst 保留 GPU/CPU 比例，直接改变当前可行域；有长期决策权。需避免把容量保留到不可行。 |
| 每 DC admission quota | 推荐 | 控制每步各中心可接收的任务/资源量，固定维且能抑制拥塞。可能与 MPC 容量约束重复，需证明有分布变化价值。 |
| migration budget | 推荐 | 直接约束跨中心任务比例或 transmission budget，回应 BC migration/transmission 偏高。 |
| defer aggressiveness / backlog target | 条件推荐 | 可控制时移程度，但若只是给 waiting weight 乘系数，容易退化成调权重。用 quota/target constraint 更有决策权。 |
| risk margin | 推荐 | 在 forecast uncertainty 下收紧资源或 SLA slack，适合研究 model mismatch；需要 calibration。 |
| DC preference | 条件推荐 | 可作为 bounded additive bias；若直接指定唯一首选 DC，会压缩 MPC 价值。 |
| dynamic objective weights | 弱推荐 | 容易成为“RL 调权重”，可通过缩放或 reward hacking 产生不可解释动作；必须归一化并证明优于固定权重。 |
| terminal value / forecast residual | 研究性 | 理论上有长期价值，但接口、监督信号与稳定性复杂，不适合作为最小原型。 |

## 4. 建议的最小高层动作

为避免 C 过于模糊，最小原型可固定为 7 维：

```text
[gpu_reserve_dc1, ..., gpu_reserve_dc5,
 migration_budget, backlog_target]
```

- 每个 reserve 为 `[0, r_max]` 的占比，转成 low-level MPC 的可用 GPU 上界；
- migration budget 为当前步 assign 中允许跨中心的资源或任务比例；
- backlog target 为允许 defer/terminal backlog 的上界或软目标；
- CPU/memory 与 SLA/deadline 仍由 MPC 硬约束；
- 所有动作必须有 fixed default，且不能通过整体缩放取消目标。

若这个最小接口都不能优于最佳 fixed intent，则不应扩展到更复杂 high-level RL。

## 5. 当前支持证据

### 支持

- MPC 已能每步稳定生成带联合容量和 deadline 约束的 task-level 动作。
- H4 绝对求解开销远低于 15 分钟周期，因此“每步 MPC”在当前规模技术上可行。
- 构造场景说明 resource reservation、migration 和 defer 等高层概念可能影响长期结果。

### 反向或中性证据

- BC 直接闭环已经接近 H4，提出了“是否需要永久 MPC”的疑问。
- 最新公平比较没有证明 H4 普遍优于 H1。
- 当前没有 high-level observation、action、RL wrapper、训练或 closed-loop 结果。
- 没有证据说明动态 intent 优于 fixed MPC objective；RL 的必要决策权尚未被证明。

因此现有 MPC 成功只能证明 C 的 low-level executor 可实现，不能证明 high-level RL 有价值。

## 6. C 的研究意义陷阱

如果 RL 只输出 objective weights，而 MPC 仍决定所有实际 tradeoff，论文可能退化成超参数自适应；如果 weights 没有归一化，RL 甚至可通过统一缩放造成等价动作。要使 C 有研究意义，高层 action 必须：

- 固定维；
- 物理可解释；
- 对 feasible region 或 budget 有真实影响；
- 有固定策略基线；
- 在 distribution shift 下带来可测收益；
- 不破坏 MPC hard constraints。

## 7. 最低证据门槛

C 至少需要一个小型原型证明：

- high-level action 不会长期饱和到边界或退化为固定值；
- 相比 best fixed-intent MPC，在 workload/capacity/forecast shift 上有一致改善；
- 改善不是来自放宽 SLA 或隐藏更多 defer；
- solver feasibility 与 latency 保持；
- RL 的信息和动作确实提供了 MPC 固定目标无法表达的长期适应。

## 8. 当前判断

**直接证据强度：低。**C 是合理长期备选，不是当前领先方案。现有证据只验证了其低层 MPC 部分；高层 RL 的 observation、action 与增益均未验证。当前不应直接投入完整 hierarchical RL 训练。

