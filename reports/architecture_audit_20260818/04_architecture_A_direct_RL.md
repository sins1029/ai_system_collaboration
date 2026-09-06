# Architecture A — BC / Regularized RL Direct Execution

## 1. 定义

```text
H=4 MPC ──offline──> expert state-action data
                           ↓
                    BC / regularized RL
                           ↓
          one defer/DC action per current task
                           ↓
                    ActionAdapter
                           ↓
                  SustainCluster
```

MPC 只作为离线 teacher、数学 benchmark 与必要时的数据再生成器；在线执行不调用 MPC。BC/RL 直接拥有 task routing 决策权。

## 2. 统一接口

| 接口项 | Architecture A |
|---|---|
| Observation | 动态 task set；当前实现为每任务 233-d feature、6-d feasible mask，并共享 DC/forecast 全局特征。 |
| Action | 每任务一个 semantic action：`defer` 或 `dc_1..dc_5`。 |
| Decision frequency | 每 15 分钟一次，对当步全部 `env.current_tasks` 联合生成动作列表。 |
| MPC role | 离线 expert、benchmark、dataset refresh；在线不调用。 |
| RL/BC role | 直接决定 task defer 与目的 DC。 |
| Hard constraints | 当前只有合法 action、task-order、per-task transfer/deadline/capacity mask；没有联合 task-set 容量保证。环境本地队列防止资源变成负值，但可能把任务排队至 SLA 违约。 |
| Training method | 已实现 BC 与 vanilla SAC warm start；目标方案需 BC/KL regularization、expert replay、actor freeze/critic warmup 等。 |
| Execution path | state→forecast→encoder/mask→actor→semantic decisions→dynamic adapter→`env.step(actions)`。 |
| Fallback | 当前未实现。候选为 defer-safe rule、H=1/H=4 MPC 或 last-known-safe policy。若在线不允许 MPC，至少需 deterministic safe rule。 |

## 3. 当前实现状态

已实现并有证据的是：

- BC direct execution；
- 真实 multi-action closed loop；
- BC→SAC actor 权重桥；
- vanilla SAC prototype；
- 执行阶段 action mask 和 dynamic action mapping。

未实现的是 Architecture A 名称中的关键限定词 `regularized`：当前没有 BC/KL anchor、expert replay mixing、frozen actor critic warmup、policy trust region 或 OOD fallback。

## 4. 支持证据

### 强证据

- BC overall accuracy 96.98%、assign accuracy 90.49%、migration F1 94.79%。
- BC 在 5 个 fresh seed 的真实环境中独立闭环，illegal action=0、resource overflow=0。
- BC completed 3512.2 vs H4 3512.0，SLA 1552.8 vs 1552.6。
- BC 物理联合成本比 H4 低 0.376%，说明直接策略至少在已测分布上不依赖在线求解也可复现总体质量。
- MPC→BC 将 SAC 初始 reward 从 -3641.44 提升至 -1821.22，expert agreement 从 .7796 提升至 .9256。

### 削弱证据

- vanilla SAC 将 BC reward 降至 -2432.21、SLA 提高至 2080、agreement 降至 .7690。
- 当前 action mask 是逐任务而非联合约束，零 overflow 不等于所有 task-set 都安全。
- BC transmission 比 H4 高约 18.4%，migration 高 2.88 个百分点，reward 也更差。
- 尚无未见 workload window、DC 容量变化、forecast error 或 DC scale 变化下的直接策略证据。

## 5. A 不能被当前 SAC 失败否定

当前负结果只针对一个具体 recipe：随机 critic、128 replay、固定 entropy、环境 reward 仅含 energy/carbon、SAC loss 不使用 per-action feasibility mask、没有专家锚定。它证明 vanilla fine-tuning 不稳定，不证明：

- frozen BC 不能直接执行；事实上已经能执行；
- regularized SAC 不能保持或超过 BC；尚未实验；
- direct RL 必然产生非法动作；执行 mask 当前已避免编码非法动作；
- 所有 OOD 下 A 都不安全；这一点也尚未验证。

## 6. A 的主要研究风险

1. **联合容量耦合：**独立 per-task logits 可能把很多任务同时送往同一中心。
2. **SLA 与 reward 不一致：**在线 RL 可能提高当前 reward 而恶化 SLA。
3. **分布外状态：**BC 只覆盖当前数据生成器与固定 5 DC semantic space。
4. **灾难性遗忘：**online updates 会快速抹去 expert routing。
5. **fallback 空白：**当前 direct path 没有部署级异常处理。

## 7. 使 A 成为最终架构的最低证据门槛

A 至少需要证明：

- regularized RL 在同样 10k 小规模交互中保持或提高 frozen BC reward；
- SLA 相对 frozen BC 不恶化超过预设容差；
- expert agreement 不发生当前量级的坍塌；
- 未见 workload/capacity/forecast error 下 illegal=0 且无联合拥塞导致的显著 SLA 恶化；
- 即使不调用 MPC，也有可说明的 deterministic fallback。

## 8. 当前判断

**直接证据强度：中高。**A 是唯一已有完整 direct closed-loop 证据的架构族，但已经验证的是 frozen BC 和失败的 vanilla SAC，不是最终所需的 regularized A。它应作为主要对照和可保留的替代方案，而不是因一次 SAC 失败被排除。

