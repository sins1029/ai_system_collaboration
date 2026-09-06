# Algorithm v0.1 失败分析

## 1. 已排除的问题

本轮证据可以排除以下直接原因：

- BC 权重桥接失败：10/10 tensors 的 key、shape、dtype、数值与输出完全一致；
- Teacher 被训练：所有 run 的 Teacher 前后 SHA-256 完全一致且无梯度；
- Warmup 未执行：M2/M4 首次 Actor 更新均为 step 2512；
- Critic 初始化不成对：同一 seed 下 M1-M4 初始 Critic 哈希一致；
- 非法动作污染：Replay 与全部 SAC 目标已使用 per-task feasible mask，illegal 为 0；
- 明显数值爆炸：loss、Q、环境指标均 finite，NaN/Inf 为 0；
- 单次随机偶然：三个 paired seeds 都出现同方向的策略漂移。

## 2. BUG FIX 的实际结论

历史 Replay 丢失 feasible-action mask 是真实实现 Bug，本轮已经修复。修复后 M1 仍在 5k 左右退化为全 defer，因此该 Bug 不是退化的唯一原因，也不能把修复后的任何差异解释为算法创新。

## 3. Mechanism A 为什么不够

Critic Warmup 在 0-2500 steps 完全保持 BC，证明冻结机制有效；一旦 Actor 解冻，M2/M4 仍沿相似轨迹退化。Warmup 只能避免随机 Critic 在训练最初立即推动 Actor，不能修正后续持续存在的 reward、credit assignment 或 Q target 偏差。

## 4. Mechanism B 为什么不够

`lambda=0.01` 的 weighted KL 具有可测影响：M3/M4 平均约为 0.010/0.007，并提高了 entropy。但 M3 最终 agreement 为 0.7755，与 M1 的 0.7799 同等级，说明该强度没有保住专家行为。M4 也几乎收敛到全 defer。

本轮按 loss 数量级选择 10% 约束，没有进行大 sweep。更强 KL 可能延缓漂移，但在 reward/critic 语义未厘清前直接增加 lambda，最多可能把策略锁回 BC，不能证明 RL 有额外价值。

## 5. 首要结构性风险

### 5.1 Reward 与评价目标不一致

当前 SAC reward 只有 `0.9 * energy_price + 0.3 * carbon_emissions`，不包含 SLA、transmission、waiting/defer 和 terminal backlog。对即时已调度任务计成本，而对推迟行为缺少与最终 SLA 对齐的直接惩罚。这与最终评价指标和 H4 objective 明显不一致，是全 defer 固定点的首要解释候选。

这是强相关证据，不是已经完成的因果证明。下一轮需要通过最小 reward 对齐实验验证。

### 5.2 Task-level Critic 的 transition/credit assignment

一个环境 step 的全局 reward 被复制给所有 task Q target；current/next tasks 通过 padded 行位置 bootstrap，没有 task identity 对齐。任务完成、迁移或新任务进入时，下一行未必是同一任务。这可能让 Critic 学到错误的 task-action credit，并持续向 Actor 提供有偏梯度。

### 5.3 Replay 与熵设置

capacity 128、batch size 8 对高维、变长任务集合提供的覆盖很窄；alpha 固定为 0.01，没有 automatic entropy tuning。M1 最终 entropy 约 0.02，并转为确定性全 defer。这些是次级候选，需要在前两项语义问题修正后再做判别实验。

## 6. 归因边界

本轮能得出的结论是：在当前 reward 与 task-level Q 建模下，mask BUG FIX、2500-step Critic Warmup 和 `lambda=0.01` Teacher KL 都不足以阻止 SAC 退化。不能据此证明所有 Architecture A 方法无效，也不能证明更长训练会自行恢复。
