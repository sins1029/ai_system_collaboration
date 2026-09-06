# Architecture A Algorithm v0.1 历史基线复核

## 1. 版本与证据边界

- 主仓库基线：`feature/single-center-task-runtime-v0.2@90eb972f78742603b522fabc3e1cf65391434418`
- SustainCluster：`3f6ea95cb835b89ba50b0ef76d66d14b8037643e`，工作区 clean
- BC checkpoint：`artifacts/sustaincluster_imitation/bc_actor_seed_11.pt`
- 历史机器结果：
  - `reports/sustaincluster_imitation/bc_offline_evaluation_results.json`
  - `reports/sustaincluster_imitation/bc_closed_loop_results.json`
  - `reports/sustaincluster_imitation/sac_warm_start_results.json`
- 本文只复核已保存结果，不把历史 Markdown 摘要当作原始证据。

## 2. BC 离线结果

三次训练 seed 为 11、22、33。下表为 test split 的 mean +/- std。

| 指标 | 结果 |
|---|---:|
| overall accuracy | 0.969809 +/- 0.001411 |
| assign accuracy | 0.904918 +/- 0.004460 |
| migration F1 | 0.947896 +/- 0.003163 |
| defer F1 | 0.999914 +/- 0.000000 |
| top-2 accuracy | 0.996219 +/- 0.000661 |

Actor 输入维度为 233，语义动作数为 6，参数量为 128,262。动作语义保持为 `defer, DC1, DC2, DC3, DC4, DC5`。

## 3. BC 真实闭环结果

评估使用 5 个未见 seed（1201-1205），交替覆盖 normal/high-load trace。数值为每个 96-step episode 的均值。

| Policy | Reward | Completed | SLA | Electricity | Carbon | Transmission | Migration | Inference ms | Illegal | Overflow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H4 MPC | -1797.5083 | 3512.0 | 1552.6 | 14786.3242 | 45277.9063 | 101.2325 | 0.5098 | 20.5129 | 0 | 0 |
| H1 MPC | -1796.6820 | 3512.0 | 1553.0 | 14834.5842 | 45338.3843 | 100.6998 | 0.5145 | 6.7645 | 0 | 0 |
| BC | -1866.3149 | 3512.2 | 1552.8 | 14736.7824 | 45082.7059 | 119.8620 | 0.5385 | 4.0763 | 0 | 0 |

BC 的 `electricity + carbon + transmission` 相对 H4 MPC gap 为 -0.003758。环境 reward 与该物理成本和 SLA 不是同一目标，不能只用 reward 判断策略优劣。

## 4. BC-init Vanilla SAC 历史结果

历史配置使用 3 个 paired seeds（41、42、43），每个 run 10,000 environment steps；evaluation seed 为 2201，每次 96 steps。

| 指标 | 初始 | 10k steps |
|---|---:|---:|
| reward | -1821.2219 | -2432.2136 |
| SLA violations | 936 | 2080 |
| expert agreement | 0.925637 | 0.768960 |

三个 BC-init run 的首尾评估值完全相同。训练共执行 618 次 critic update；历史代码实际也执行了 618 次 actor update。Q 值均 finite，illegal action 计数为 0。

## 5. 权重桥接复核

本轮重新加载 checkpoint 并检查得到：

- source/target keys：10 / 10
- missing keys：0
- unexpected keys：0
- shape mismatch：0
- dtype mismatch：0（全部 float32）
- 最大参数差：0
- 同一随机 observation 上最大 logits 差：0

因此 `BC checkpoint -> SAC Actor` 初始化完全成功；历史退化不是由权重未加载导致。

## 6. 口径说明

- BC closed-loop 使用 5 个 mixed-window evaluation seeds；历史 SAC 使用单个 normal-window evaluation seed 2201，二者 reward 不应直接作绝对差值比较。
- SustainCluster CompositeReward 当前只组合电费与碳排 reward，不包含 SLA、传输、等待和终端积压。
- H4 MPC objective 还包含 transmission、waiting/defer、SLA risk 与 terminal backlog。因此 SAC reward 与 MPC objective 不一致。
- Built-in RBC 的 reward/SLA 零值属于接口限制，不作为本轮基线。

## 7. 历史结论

BC 是有效的 Actor 初始化器，但历史无约束 SAC 更新显著破坏了初始策略。本轮需要先修正更新中的可行动作 mask，再分别验证 Critic Warmup 和固定 BC Teacher 正则化。
