# Architecture B — BC / RL + Event-Triggered MPC

## 1. 定义

```text
state → BC / RL → task-level proposal
                    ↓
                risk detector
              ┌─────┴─────┐
          normal         risky
            ↓              ↓
      direct execute   H=1/H=4 MPC correction
              └─────┬─────┘
                    ↓
              ActionAdapter → env.step
```

MPC 不在每一步运行，而在检测到容量、SLA、置信度、OOD 或未来资源风险时纠正整批或局部任务动作。

## 2. 统一接口

| 接口项 | Architecture B |
|---|---|
| Observation | A 的动态 per-task features/masks，加 risk detector 所需的 policy entropy/margin、capacity slack、SLA slack、forecast uncertainty、burst/OOD score。 |
| Action | BC/RL 先提出每任务 `defer/dc_id`；触发后由 MPC 返回替换后的 task-level action vector。 |
| Decision frequency | 每 15 分钟；risk detector 每步运行，MPC 只在触发时运行。 |
| MPC role | 在线 corrector/safety fallback；也继续作为 offline expert 与 benchmark。 |
| RL/BC role | 正常状态直接执行，承担大多数 task-level 决策。 |
| Hard constraints | 未触发路径与 A 相同；触发路径由 MPC 提供 task-set CPU/GPU/memory/deadline 约束。要称为 safety architecture，必须证明 detector 对危险状态的 recall。 |
| Training method | BC 或 regularized RL；risk detector 可先用可解释规则，再用标注的 MPC disagreement/violation data 校准。 |
| Execution path | actor proposal→risk score→direct 或 MPC re-solve→adapter→environment。 |
| Fallback | detector error/solver error 时使用 conservative H=1、defer-safe rule 或上一安全计划；MPC infeasible/timeout 必须显式记录。 |

## 3. 合理的首版触发条件

首版应以可解释、无需先运行 full H=4 的信号为主：

1. **capacity pressure：**actor proposals 的联合 CPU/GPU/memory demand 超过目的 DC 可用或保守 forecast capacity；
2. **urgent SLA：**最小 completion slack 低于 1–2 steps，或只剩 defer 可行；
3. **workload burst：**pending count/GPU demand 超出训练分位数；
4. **low confidence：**动作概率 margin 低或 entropy 高；
5. **OOD：**feature distance/ensemble disagreement 超出 calibration threshold；
6. **future resource event：**已知 release、价格低谷、forecast reserve 或 uncertainty 在 H=4 内显著变化；
7. **cheap disagreement：**可选用 H=1/heuristic 与 actor 比较。若必须先运行 H=4 才知道 disagreement，就失去了减少 H=4 调用的意义。

触发后的 correction 可先替换整批 action vector，避免局部纠正破坏 task-set 联合容量。后续才研究只纠正风险任务。

## 4. 当前支持证据

### 直接支持

当前没有 event-trigger controller 的实现或实验，因此 B 还没有直接系统级支持。

### 强动机证据

- BC 已在普通真实闭环中接近 H4，说明多数状态可能不需要在线 MPC。
- H4 与 H1 在 future-low-price、GPU-burst 公平场景完全相同，在 normal/high-load 上差异极小，说明 full H4 的边际价值是状态依赖的。
- capacity-release 公平场景 H4 的 wait 和 stage cost 明显改善；早期构造场景也证明 future events 确实可能改变正确动作。
- vanilla SAC 漂移说明纯 learned execution 仍可能需要在线保护。
- H4 最大约 29.6 ms，触发时调用的绝对成本很低，B 的价值主要是减少对模型正确性的持续依赖和研究 safety-computation tradeoff，而不是解决当前实时性瓶颈。

## 5. B 的核心科学问题

> 能否用低 MPC 调用率，获得接近或优于 full-H4 的 SLA、资源可行性与运行成本？

这个问题只有同时报告以下三组量才有意义：

- quality：reward、SLA、completed、electricity、carbon、transmission；
- safety：illegal、overflow、joint-capacity pressure、solver failure；
- intervention：MPC call rate、trigger precision/recall、corrected task ratio、额外 latency。

只报告“调用率低”或“没有溢出”都不足以支持 B。

## 6. B 的主要风险

1. detector 漏报会让危险动作直接执行；
2. detector 过度触发会退化为 full-time MPC；
3. 以 MPC disagreement 作为标签可能只复制模型偏好，不等于真实风险；
4. correction 改变行为分布，RL 训练时若看不到被纠正动作会产生 credit assignment 问题；
5. 当前环境本地队列会掩盖联合容量冲突，风险标签不能只看 resource negative。

## 7. 最低证据门槛

建议预注册首轮判据：

- MPC call rate ≤20%；
- SLA violations 与 full H4 差异 ≤1%；
- `electricity+carbon+transmission` 与 full H4 差异 ≤1%；
- illegal=0、overflow=0；
- 对预定义 risky events 的 detector recall ≥95%；
- 相比 frozen BC，在 OOD/stress 上出现可重复的 SLA 或成本改善。

这些只是阶段门槛，不是论文最终统计标准。

## 8. 当前判断

**直接证据强度：低；动机强度：高；综合置信度：中。**B 是当前领先候选，因为它最直接地解释了两个同时存在的事实：BC 在常态下已经够好，而 MPC 在少数前瞻/风险状态仍有独特价值。但在触发器、调用率和纠正收益被实验验证前，不能把 B 写成已确定架构。

