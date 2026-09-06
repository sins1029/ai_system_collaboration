# Architecture A Algorithm v0.1 设计

## 1. 验证目标

本轮只回答一个问题：在不改变 SustainCluster 环境、任务级语义动作、BC checkpoint 和 Architecture A 主路径的前提下，Critic Warmup 与 Behavior Regularization 能否阻止 BC-init SAC 的快速退化。

目标 checkpoint 仍是 task-level policy：输入当前 233 维 task/DC/forecast 特征，输出 `defer, DC1, DC2, DC3, DC4, DC5` 六个语义动作，经 ActionAdapter 后进入 `env.step()`。

## 2. 共同实现基础

所有 M1-M4 都使用同一份 BC Actor 初始化、随机 Twin Critic、Target Critic、固定熵系数和训练环境。唯一共同的行为修正是 SAC 审计发现的动作掩码 BUG FIX：

- Replay 保存 current/next per-task feasible-action mask；
- Critic target、Actor loss、entropy 与 teacher KL 只在可行动作集合上计算；
- 执行动作继续使用同一语义 mask；
- mask 修复属于实现正确性，不计作 Algorithm v0.1 收益。

当前仍保留两个已知价值建模限制：全局 step reward 被复制给各任务，current/next task 通过 padded 行位置 bootstrap 而没有 task identity 对齐。本轮没有扩大范围重构。

## 3. Mechanism A：Critic Warmup

Warmup 期间继续用 BC Actor 与环境交互，Replay 正常写入，只更新 Critic 和 Target Critic，不执行 Actor optimizer step。与历史的“随机动作 + 延迟全部更新”不同。

正式矩阵采用：

```text
learning_starts = 128 env steps
update_frequency = 16 env steps
critic_warmup_steps = 2500 env steps
```

因此 Critic 从 step 128 开始更新，step 128-2496 共 149 次 Critic-only update；Actor 首次更新为 step 2512。0、1000、2500 三个候选写入配置，第一轮全矩阵只使用 2500，避免扩大 sweep。

## 4. Mechanism B：Behavior Regularization

离散动作最自然的约束是冻结 BC Teacher 分布到当前 Actor 分布的 masked KL：

```text
L_actor = L_SAC + lambda_bc * KL(pi_BC || pi_actor)
```

KL 对每个真实任务分别计算，只在该任务当前可行动作集合内归一化。Teacher 由同一 BC checkpoint 深拷贝得到，固定为 `eval()`、`requires_grad=False`，不进入任何 optimizer。

## 5. 正则强度标定

先使用 M1 的 1,851 条更新诊断统计 loss 数量级：

| 项目 | 中位数 |
|---|---:|
| `abs(sac_actor_loss)` | 0.177305 |
| `expert_loss` | 1.745036 |

以加权 KL 约为 SAC Actor loss 的 10% 为目标：

| lambda | 估计比例 |
|---:|---:|
| 0.01 | 0.098420 |
| 0.1 | 0.984198 |

因此正式 M3/M4 使用 `lambda_bc=0.01`。`lambda=0` 通过单元测试证明与 vanilla actor update 数值一致。

## 6. 方法矩阵

| 方法 | 训练 | Warmup | Teacher KL |
|---|---|---:|---:|
| M0 BC Fixed | 否 | 0 | 0 |
| M1 BC-init SAC | 是 | 0 | 0 |
| M2 + Warmup | 是 | 2500 | 0 |
| M3 + Regularization | 是 | 0 | 0.01 |
| M4 + Warmup + Regularization | 是 | 2500 | 0.01 |

## 7. 受控实验条件

- training seeds：41、42、43；每个方法与种子共享 Critic 初始化哈希；
- evaluation seeds：2201、2202、2203，与训练种子分离，含 normal/high-load 窗口；
- 每个 run：10,000 environment steps；episode/evaluation 均为 96 steps；
- 评估点：0、1000、2500、5000、7500、10000；
- Replay capacity 128，batch size 8，Actor/Critic learning rate 均为 `3e-4`；
- `gamma=0.99`，`alpha=0.01` 固定，`tau=0.005`；
- 不使用 Expert Replay，不使用 MPC online correction，不修改 reward 和动作定义。

## 8. 可验证不变量

测试和正式运行共同检查：BC 到 SAC Actor 完全一致、Teacher 哈希不变、Warmup 期间 Actor 不变、Warmup 后 Actor 改变、采样动作始终合法、全部 loss/Q 统计 finite。正式运行的 paired Critic 哈希也按 seed 完全一致。
