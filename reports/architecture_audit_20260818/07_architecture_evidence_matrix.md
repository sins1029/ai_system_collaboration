# Architecture Evidence Matrix

## 1. 三种架构统一比较

| 项目 | A：Direct BC/RL | B：Event-Triggered MPC | C：High-Level RL + MPC |
|---|---|---|---|
| RL 输出 | variable-length task-level action | task-level proposal + risk signals | fixed-dimensional intent |
| MPC 在线 | 否 | 条件触发 | 每步 |
| MPC 作用 | teacher / benchmark | teacher + corrector | permanent executor |
| 动作空间 | 每任务 6 类，任务数动态 | 每任务 6 类，任务数动态 | 固定维 reservation/quota/budget |
| 决策频率 | 15 min | actor/detector 15 min，MPC on demand | RL 15–60 min，MPC 15 min |
| 当前约束保证 | adapter + per-task mask；无联合保证 | normal 同 A；trigger 后由 MPC 联合约束 | MPC 联合约束 |
| 在线求解成本 | 最低 | 中等且取决于 call rate | 最高；当前绝对值仍很小 |
| 当前已实现 | BC direct + vanilla SAC prototype | 否 | 否 |
| 当前最大风险 | OOD、联合容量、RL 漂移 | detector 漏报/过触发 | 高层动作无价值或仅调权重 |

## 2. 证据矩阵

支持等级：`strong / moderate / weak / neutral / weakens`。strength 描述证据本身可靠度，不等于对某架构的最终置信度。

| Evidence | supports A? | supports B? | supports C? | Strength | Limitation |
|---|---|---|---|---|---|
| multi-action 环境、动态 action mapping 与 task-order 已打通 | strong | strong | neutral | strong | 证明接口可用，不证明策略质量。 |
| one-step/rolling MPC 无 infeasible、illegal、overflow | moderate：可靠 teacher | strong：可作 corrector | strong：可作 executor | strong | 只在已测状态；模型不含网络拥塞/鲁棒约束。 |
| H4 构造压力场景可为 burst 留容量、等低价、等 release | neutral | strong | moderate | moderate | 较早比较非完全同口径，部分使用 oracle。 |
| 最新 capacity-release 公平比较 H4 wait 2→1、cost 1644.7→829.2 | neutral | strong | moderate | strong | 只是一个合成场景，完成和 SLA 不变。 |
| 最新 future-low-price 与 GPU-burst 中 H4=H1 | moderate | moderate | weakens permanent H4 | strong | 场景定义可能没有暴露可利用的前瞻差异。 |
| normal/high-load H4 与 H1 几乎相同且 SLA 略差 | moderate | strong | weakens permanent H4 | strong | 仅 5 seeds、固定 96-step windows。 |
| H4 最大 29.6 ms，远小于 15 min | neutral | moderate | strong | strong | 说明可运行，不说明必须运行。 |
| causal expert dataset 256,125 task-actions、solver failure 0 | strong | strong | weak/neutral | strong | 数据由固定 MPC objective 与固定 5-DC schema 产生。 |
| BC overall 96.98%、assign 90.49%、migration F1 94.79% | strong | strong | neutral | strong | defer 占 68.4%；仍有约 9.5% assign 路由误差。 |
| BC closed loop completed/SLA ≈ H4、cost gap -0.376% | strong | strong | raises question | strong | 仅 normal/high-load、5 fresh seeds；未覆盖 OOD。 |
| BC illegal/overflow=0 | moderate | moderate | neutral | moderate | mask 不做 task-set 联合容量；环境 queue 可掩盖拥塞。 |
| BC transmission +18.4%、migration +2.88pp、reward 更差 | weakens pure A | moderate：可定向纠正 | neutral | strong | 不说明 MPC correction 一定能以低 call rate修复。 |
| BC-init reward/SLA/agreement 明显优于 random-init | strong | strong | weak | strong | BC step-0 三 run 使用同 checkpoint/eval seed；不等同于多分布稳健性。 |
| vanilla SAC 后 agreement↓、SLA↑、reward↓ | weakens vanilla A only | motivates B | weak/neutral | strong | 未做 regularizer/reward/mask/critic 消融，不能支持“RL 不可 direct”。 |
| current SAC reward 不含 SLA/transmission | weakens current A recipe | motivates guard | motivates MPC constraints | strong implementation fact | 是否是退化主因未验证。 |
| event trigger 未实现、无 call-rate 实验 | neutral | weakens confidence | neutral | strong | B 当前只有间接证据。 |
| high-level action/interface 未实现 | neutral | neutral | strongly weakens confidence | strong | 现有 MPC 能力不能替代高层 RL 价值证据。 |

## 3. 按架构汇总

### Architecture A

- **最强支持：**BC 离线与真实 direct closed loop；MPC→BC warm start。
- **最强反例：**vanilla SAC 漂移；BC transmission/migration/reward 仍弱于 H4；联合容量/OOD 未验证。
- **结论：**frozen BC A 已成立为强基线；regularized RL A 仍待判别。当前证据为 `moderate-to-strong`，不能扩展到“pure RL 已安全”。

### Architecture B

- **最强支持：**BC 常态近似 MPC + MPC 在少数前瞻事件有价值 + learned update 可能漂移。
- **最强缺口：**没有 detector、correction 或 MPC call-rate 数据。
- **结论：**它最符合现有证据的结构性解释，但仍是待验证候选。当前直接证据 `low`，综合动机 `strong`。

### Architecture C

- **最强支持：**MPC 约束与 latency 足以作 permanent executor。
- **最强反问：**BC 已可直接执行；H4 未普遍胜过 H1；高层 RL 没有定义/实现/收益。
- **结论：**当前仅 low-level half 被验证，整体架构证据 `low`。

## 4. 架构决策状态

```text
Recommended current architecture:
  冻结 BC direct executor 作为当前可运行基线；保留 H4 MPC 作为 teacher/benchmark，
  不宣称任何 A/B/C 已最终胜出。

Leading candidate:
  Architecture B — Event-Triggered MPC（medium confidence）。

Alternative:
  Architecture A — BC-regularized direct RL（medium confidence）。

Not yet justified:
  Architecture C — permanent low-level MPC + high-level RL（low confidence）；
  以及“每一步必须 H=4”的主张（low confidence）。
```

这里“B 领先”指下一步最值得判别，不代表 B 的直接证据多于 A。按已实现证据排序，A 第一；按最能解释当前矛盾并形成新研究问题排序，B 第一。

