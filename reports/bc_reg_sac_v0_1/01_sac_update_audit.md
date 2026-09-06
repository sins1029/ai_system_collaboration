# 当前 SAC 更新审计

## 1. 真实训练流程

```text
SustainCluster env.step
        |
        v
(obs, semantic actions, normalized reward, next_obs, done)
        |
        v
FastReplayBuffer
        |
        v
twin critic update
        |
        v
categorical actor update
        |
        v
Polyak target critic update
```

实现入口为 `reports/sustaincluster_imitation/run_sac_warm_start.py`。Actor/Critic 来自只读第三方仓库 `rl_components.agent_net`，Replay 来自 `rl_components.replay_buffer`。

## 2. 组件审计

### Actor

- `ActorNet(233, 6, hidden_dim=256, LayerNorm=True)`。
- 每个任务独立输出 6 个 categorical logits。
- 执行路径会对 `EncodedTaskBatch.feasible_action_mask` 做 masked fill，然后 sample/argmax。
- BC checkpoint 的 10 个 tensor 与 SAC Actor key、shape、dtype 和数值完全一致。

### Critic / Target Critic

- 两个独立 Q 网络，每个任务输出 6 个动作的 Q 值。
- Critic 为随机初始化；同一 paired seed 下不同组的初始 critic hash 相同。
- Target Critic 从 Critic 精确复制，之后每次 update 都做 `tau=0.005` 的 Polyak 更新。

### Replay

- 历史 `FastReplayBuffer` 保存 obs、action、reward、next_obs、done 和 task-presence mask。
- capacity=128，max_tasks=2048，batch_size=8。
- 不保存 per-task feasible-action mask。
- Expert Dataset 虽有 step-level reward/done 和 task-level feature/action，但没有可直接消费的 `(task state, action, next task state)` transition 对齐。本轮不加入 Expert Replay。

### Entropy

- 固定 `alpha=0.01`。
- 没有 automatic entropy tuning。
- 没有 target entropy，也没有 alpha optimizer。

### Reward

SustainCluster 的 `CompositeReward` 当前为：

```text
0.9 * energy_price_reward + 0.3 * carbon_emissions_reward
```

两项均是对当前已调度任务的负成本，默认各除以 100。reward 不包含 SLA、transmission、waiting/defer 或 terminal backlog；与 H4 MPC objective 不一致。本轮只记录，不修改。

## 3. 历史更新时序

- environment steps：10,000
- 第 1-128 步：随机可行动作，不更新 critic/actor
- 第 128 步：首次 critic update，同时首次 actor update
- update frequency：每 16 environment steps 一次，共 618 次
- `policy_update_frequency=2` 作用在 global environment step；所有 update step 都是 16 的倍数，因此条件永远成立，实际 actor update 也是 618 次
- target update：每次 critic update 后执行，共 618 次

历史 `warmup_steps` 实际含义是“随机采样 + 延迟所有更新”，不是本轮要求的“BC Actor 保持执行、Critic-only update、Actor freeze”。Algorithm v0.1 将分离 `learning_starts` 与 `critic_warmup_steps`。

## 4. BUG FIX：训练目标没有可行动作 mask

历史执行动作时使用 feasible-action mask，但 replay 不保存该 mask，导致：

1. target value 对不可行动作也分配 policy probability；
2. actor SAC loss 对不可行动作也计算概率和 Q；
3. 执行时再 mask 只能避免非法 action 落地，不能修复已经被污染的训练梯度。

这是实现 Bug，不是 Algorithm v0.1 的算法贡献。新实现将在 replay 中保存 current/next feasible-action mask，并在 target、actor、entropy 和 teacher KL 中统一应用。

## 5. 其他限制

- 当前每个环境 transition 只有一个全局 reward，训练时复制给每个任务 Q target。
- current/next task 通过 padded 行位置参与 bootstrap，没有显式 task identity 对齐；任务完成或新任务进入时该假设并不严格。
- reward normalization 使用每个 run 独立的 RunningStats。

后两项属于状态/价值建模限制，本轮不扩大范围重构。若 mask 修复、warmup 和正则化后仍快速退化，应优先复查这些假设。

## 6. Algorithm v0.1 更新顺序

```text
env.step
   -> transition + current/next feasible masks
   -> semantic replay
   -> masked critic update
   -> 计算 masked SAC actor loss
   -> 计算 frozen BC teacher KL
   -> warmup 后执行 actor update
   -> target critic update
```

Critic Warmup 期间仍计算并记录 actor loss/teacher loss，但不执行 actor optimizer step，从而可以验证 Actor hash 完全不变。
