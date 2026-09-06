# Architecture A 第一轮决策

## 1. Gate 判定

| Gate | M4 结果 | 通过 |
|---|---|---|
| Expert agreement 不出现同等级崩塌 | 0.9356 -> 0.7794，几乎等于 M1 的 0.7799 | NO |
| SLA 不明显恶化 | 1449.67 -> 3055.00，几乎等于 M1 的 3062.33 | NO |
| Reward 不持续下降 | -2193.32 -> -2596.35，7.5k 后保持退化平台 | NO |
| Illegal / overflow | 均为 0 | YES |

M3 相对 M1 有部分 reward/SLA 改善，但仍明显差于固定 BC，且 expert agreement 没有改善，因此同样不通过 Gate。

## 2. Outcome

```text
Outcome C
```

加入 Critic Warmup 和 Behavior Regularization 后仍出现快速、同等级的策略退化。不能继续把当前 SAC 配方直接扩大到 50k/100k，也不能把保存的 checkpoint 视为可继续长训练的初步模型。

## 3. Architecture A 判断

Architecture A 总体结论为 `UNCERTAIN`，不是直接否定。当前被否定的是“现有 reward + 当前 task-level Critic + vanilla discrete SAC，再附加轻量 warmup/KL”这条具体配方。BC、Adapter、Expert Dataset 和 task-level policy 部署路径仍然有效。

## 4. 最佳候选的定位

相对最佳方法为 M3 Regularized，保存 `artifacts/bc_reg_sac_v0_1/best_actor.pt`，seed 42，step 10000，SHA-256 为 `6D90971CA25E078A03240B97BC231066A8374B950B143C949CDBAE7FC1FBC693`。

该文件只用于人工审查、可复现诊断和后续对照。它没有超过 BC，也没有达到 `READY FOR MEDIUM-SCALE TRAINING`。

## 5. 下一步最小判别路线

1. 先把 SAC reward 与最终评价口径对齐，显式检查 SLA、defer/terminal backlog、transmission 项，并用固定轨迹逐 step 验证 reward 分解。
2. 修正或重新定义 task-level Critic transition：加入 task identity/terminal 对齐，避免把全局 reward 无差别复制给每个任务；为 Q target 建立专门单元测试。
3. 在上述两项完成后，复用同一 3-seed x 10k 矩阵，只比较 BC、修正后 vanilla 与一个受控 KL 候选，不立即做大 sweep。
4. 若 reward/Q 语义修正后仍退化，再把 Architecture B 的事件触发 MPC 作为下一轮候选，而不是继续盲目增加 SAC 训练步数。

## 6. 本轮停止条件

- 不执行中长规模训练；
- 不加入 Expert Replay；
- 不加入 MPC online correction；
- 不修改 SustainCluster；
- 不 commit，不 push，等待算法与代码人工审查。
