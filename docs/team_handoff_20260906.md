# 研究里程碑协作基线（2026-09-06）

## 当前研究基线

MPC Expert Dataset v3 已完成，当前状态为：

`MPC EXPERT DATASET v3 READY WITH DOCUMENTED LIMITATIONS`

### Alibaba2020

- 7,680 states
- 259,920 task decisions
- H1/H4 disagreement = 3.679978%

### SpotGPU2026

- 17,670 states
- 466,867 task decisions
- H1/H4 disagreement = 26.244091%
- high-risk task disagreement = 39.629803%

### Spot 闭环控制价值

- H4 相比 H1 总阶段成本没有改善
- 电力成本略降
- 碳排增加
- SLA 略差
- defer = 0
- 最终诊断：`CONTEXT_DEPENDENT_CONTROL_VALUE`

## 下一阶段研究问题

下一阶段不是继续机械模仿 H4，而是回答：

> 哪些 H4 前瞻动作真正具有正的控制价值？

## 后续可并行任务

1. H1/H4 分歧动作价值审计
2. 前瞻价值状态特征分析
3. SpotGPU2026 工作负载预测
4. Alibaba2020 / SpotGPU2026 跨数据集分析

建议后续分支名：

- `task/action-value-audit`
- `task/value-state-analysis`
- `task/spot-forecast`
- `task/cross-dataset-analysis`

后续组员应从本次 milestone commit/tag 创建独立分支，不要直接在冻结基线上开展新研究。

## 数据与许可边界

本次基线只公开代码、配置、测试、协议和汇总级研究结果。原始 Alibaba/SpotGPU2026 数据、完整专家数据集、逐任务或逐状态派生数据、模型权重与大型 checkpoint 不进入 Git。SpotGPU2026 派生数据的公开再分发许可仍为 `NOT_CONFIRMED`。
