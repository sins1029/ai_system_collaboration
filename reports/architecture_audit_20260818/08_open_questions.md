# 已确定事实、未确定问题与角色审计

## 1. 已经确定

1. 当前新研究线以 SustainCluster 为多数据中心实验环境，不应把 SustainCluster 本身写成项目创新。
2. 研究对象是 multi-action 的逐任务 defer/目的中心决策。
3. single-action 容易接标准 RL，但不能表达同一步任务差异。
4. multi-action 在当前环境可运行，动作数量、顺序和动态 DC 映射已验证。
5. 外部 one-step/rolling MILP 可以稳定产生 task-level decisions。
6. 当前 MPC 显式约束 CPU/GPU/memory、duration、transfer、completion deadline 与 reservation。
7. H=4 具有真实前瞻能力，但最新公平比较没有证明其普遍优于 H=1。
8. 当前 H=4 绝对求解延迟不是 15 分钟周期瓶颈。
9. causal H=4 MPC expert dataset 已生成，oracle 与 primary split 隔离。
10. BC 高保真学习了 defer、具体 DC assignment 与 migration 规律。
11. BC 已可独立驱动真实 multi-action SustainCluster，不能再称为“仅离线分类器”。
12. MPC→BC 显著改善 SAC actor 的初始 reward、SLA 与 expert agreement。
13. 当前 vanilla SAC 更新不能稳定保留 BC expert behavior。
14. 当前 SAC 环境 reward 只含 energy/carbon；SLA/transmission 是评估量而非训练 reward。
15. Architecture B 与 C 尚未实现。
16. 本轮回归为 138 passed；没有训练或新实验。

## 2. 尚未确定

1. regularized SAC 是否能保留 BC 并超过 frozen BC/MPC。
2. direct policy 在 unseen workload、capacity change、forecast error、DC scale change 下是否保持 SLA。
3. per-task mask 缺少联合容量约束时，是否会在 burst/OOD 下造成拥塞与 SLA 退化。
4. H=4 在真实而非构造 future events 中的触发频率与收益分布。
5. MPC 是否需要在线，以及需要 H=1 还是 H=4。
6. event trigger 的最小充分特征、threshold 与 risky-event 定义。
7. B 能否在低 MPC call rate 下达到 full-H4 质量。
8. B 应纠正整批动作还是只纠正风险任务。
9. high-level RL 最合理的固定维 action 是 reservation、quota、migration budget、risk margin 还是组合。
10. dynamic objective weights 是否有超出“自动调参”的研究价值。
11. RL 能否在 stochastic/model mismatch 下真正超过 fixed MPC；当前没有证据。
12. SAC 退化中 reward mismatch、critic cold start、entropy、Q bias、replay、mask omission、forgetting 各自的因果贡献。
13. 根仓库的单中心 thermal 线与新 multi-center SustainCluster 线如何正式版本化整合。

## 3. 为什么还需要 MPC

| MPC 角色 | 状态 | 当前证据 | 限制 |
|---|---|---|---|
| Role 1 — Benchmark | 已验证 | H1/H4 公平比较与闭环指标完整；数学目标可解释。 | 目标权重与真实 reward 不完全一致，不能视为绝对最优真值。 |
| Role 2 — Expert Teacher | 已验证 | 256k causal samples；BC assign/migration 高保真；warm start 有明显初始增益。 | teacher 固定 objective 会把其偏好和模型误差传给学生。 |
| Role 3 — Safety/Correction Layer | 部分验证 | MPC 本身能做联合约束动作；特殊 future event 有价值。 | detector/correction/call-rate 未实现，尚无系统级 safety 证据。 |
| Role 4 — Permanent Executor | 部分验证 | H4 每步可运行、约束稳定、latency 足够。 | 未证明每一步都需要；普通场景与 H1/BC 近似。 |

所以当前保留 MPC 的最强理由是 benchmark 与 teacher；把它升级为 online corrector 有强动机但未验证；把它固定为 permanent executor 尚无必要性证据。

## 4. 为什么还需要 RL

### 当前已经证明的价值

- 神经 actor 可以压缩 MPC 的 state→action mapping；
- BC 可把 online optimization 转为直接 inference；
- expert initialization 显著提高 RL 起点和早期策略质量；
- learned policy 能在真实 multi-action 环境中执行。

严格来说，以上前三项中 BC 是监督学习贡献；当前 RL 在线更新本身尚未证明最终性能价值。

### 理论上可能的价值

- stochastic arrivals 与 forecast uncertainty 下学习长期 value；
- 适应模型误差和 distribution change；
- 动态 objectives/risk preference；
- 大规模时避免反复求解；
- 学习 MPC terminal value、risk margin 或触发策略。

### 当前尚未证明的价值

- RL 超过 H1/H4 MPC；
- RL 超过 frozen BC；
- RL 在 OOD 下更稳健；
- RL 的长期 value 能弥补当前 H=4 模型；
- RL 调权重/高层 intent 确实有独立决策价值。

因此不能以“理论上可适应”替代实验证据。下一步必须让 RL 在可观测的架构判别指标上证明必要性。

## 5. SAC 退化：事实与可能解释

### 已确定事实

- BC-init step-0 reward -1821.22，final -2432.21；
- SLA 936→2080；
- expert agreement .9256→.7690；
- 所有 Q 统计有限，illegal action=0；
- random-init reward 改善但 SLA 同样大幅恶化；
- environment reward 不含 SLA/transmission；
- execution 使用 feasible mask，SAC loss 不使用 per-action feasible mask。

### 只能作为 possible explanations

- critic cold start；
- fixed entropy/alpha；
- Q-value bias；
- reward 与安全指标错位；
- online distribution shift；
- 小 replay 与 replay distribution；
- catastrophic forgetting；
- per-action feasibility mask 未进入 update；
- 一个 task-set reward 被广播到多个 per-task critic targets 的 credit assignment。

结论：

> 现有结果说明 vanilla SAC 不能稳定保留 BC 学到的专家行为，但不能由此推出“RL 不能独立执行”。

