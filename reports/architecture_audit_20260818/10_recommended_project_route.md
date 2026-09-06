# 推荐项目路线

## 1. 当前架构决策状态

**Recommended current architecture：**保持 frozen BC direct executor 作为当前可运行 baseline；H4 MPC 作为 offline teacher 与在线 benchmark。此状态是决策期基线，不是最终架构。

**Leading candidate：**Architecture B，置信度 `medium`。

**Alternative：**Architecture A with regularized RL，置信度 `medium`。

**Not yet justified：**Architecture C 与 full-time H4 必要性，置信度 `low`。

为什么 B 领先：

- BC 已证明多数已测真实状态可以直接处理；
- H4 的增量价值集中在特定 future-resource/risk states；
- vanilla RL drift 又说明完全取消在线纠正可能过早；
- B 正好把这三点变成一个可测问题：低 call rate 能否换来 H4 级安全性。

为什么还不能最终选择：

- B 没有任何 trigger/correction 直接实验；
- A 的 regularized 版本也没有实验；
- C 的 high-level action 尚未证明有 signal；
- 当前评估分布窄、SAC reward 与 SLA/transmission 不一致；
- 当前研究线未纳入 Git，项目叙述也未统一。

## Phase 1 — Architecture Decision

目标：用三个最小判别实验确定 A/B/C，而不是扩大现有训练。

顺序：

1. regularized direct RL 小预算验证；
2. frozen BC + event-trigger H4 correction；
3. 仅在有 upper-bound signal 时做 minimal high-level intent prototype。

Phase 1 的退出条件：至少一种方案达到预注册 SLA/cost/safety 阈值，并且对其核心接口有直接闭环证据。若 A/B 都可行，优先比较系统复杂度、MPC call rate 和 OOD failure mode，而不是只比较平均 reward。

## Phase 2 — Algorithm Development

只有 Phase 1 决策后才进入对应开发：

- 选 A：开发 BC/KL regularized SAC、expert replay、critic warmup 与 direct-policy fallback；
- 选 B：开发 calibrated trigger、partial/full correction、训练时 correction credit；
- 选 C：开发固定维 high-level state/action、constraint-preserving intent adapter 与 hierarchical training。

不得同时把三套方案都扩展成完整系统。

## Phase 3 — Robustness / Generalization

对最终候选统一验证：

- unseen workload windows 与 burst intensity；
- CPU/GPU/memory capacity changes；
- forecast arrival/demand/duration errors；
- datacenter count/order/heterogeneity；
- price/carbon distribution changes；
- solver timeout、missing forecast 与 model mismatch。

报告均需分 normal/stress/OOD，并同时看平均和 SLA/queue tail。

## Phase 4 — Physical / Energy Extension

最后才考虑：

- thermal/HVAC；
- storage 与 renewable；
- 真实中国电价/碳数据；
- compute-power coordination。

这些扩展会改变 state、objective 和 timescale。在 A/B/C 的 task-scheduling ownership 尚未确定前加入，只会让架构归因更困难，因此不是当前重点。

## 2. 下一步第一优先级

> 先完成 Experiment 1 与 Experiment 2 的预注册和小预算闭环，尤其验证 B 的 MPC call-rate–SLA tradeoff；不要继续扩大 vanilla SAC 训练。

## 3. 项目叙述建议

当前主线宜重新表述为：

> Model-based expert distillation and selective optimization for fine-grained multi-action multi-datacenter scheduling.

这比宽泛的“RL + MPC 数据中心调度”更贴近已完成证据，也自然容纳 A 与 B 的最终选择。若 C 后续有独立证据，再扩展为 hierarchical adaptive optimization。

## 4. 当前停止点

本轮只完成逻辑审计、实现核验、证据矩阵与最小实验设计。没有修改算法、训练模型、运行大规模实验、改变分支、commit 或修改 SustainCluster。

