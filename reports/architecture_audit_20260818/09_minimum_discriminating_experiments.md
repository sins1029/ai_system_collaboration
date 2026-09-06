# 最小架构判别实验

原则：不扩大训练规模，先用现有 96-step windows、已有 checkpoint、统一 metrics 和少量 paired seeds 判断 A/B/C 是否值得继续。

## Experiment 1 — Regularized RL Direct Execution

### Research question

vanilla SAC 的失败是否足以否定 Architecture A，还是专家正则能稳定保留并改善 BC？

### Compared methods

1. frozen BC；
2. 当前 vanilla BC-init SAC（复现实验对照，不需扩大）；
3. BC-init SAC + actor BC/KL regularizer；
4. 可选最小消融：3 + frozen-actor critic warmup 或 expert replay（二选一，不同时堆叠所有技巧）。

### Controlled variables

- 相同 ActorNet/CriticNet、features、masks、reward、environment steps=10k；
- 相同 paired critic init、replay、evaluation seeds/windows；
- 相同 update 数和超参数，仅改变正则/critic warmup；
- H4 expert 只用于 agreement 或 expert replay，不作为在线 executor。

### Metrics

- reward、completed、SLA、electricity、carbon、transmission；
- expert agreement 与 KL drift；
- illegal/overflow、destination concentration、queue/SLA tail；
- initial/final/area-under-learning-curve；
- 至少一个 unseen workload 或 capacity-shift evaluation window。

### Expected evidence

- 若正则策略保留 BC 并在 reward/成本上改善，则 A 仍是强候选；
- 若所有合理正则仍发生同方向 drift，A 的 direct online RL 可被明显削弱。

### Decision rule

A 进入下一阶段需同时满足：final reward 不差于 frozen BC、SLA 相对 frozen BC 不恶化超过 1%、agreement ≥0.90、illegal/overflow=0，并在至少一个 shift window 不劣于 BC。否则停止扩大 direct SAC。

## Experiment 2 — Event-Triggered MPC

### Research question

B 能否在低 MPC 调用率下保持 full-H4 的安全性和运行质量？

### Compared methods

1. frozen BC；
2. full H4 MPC；
3. frozen BC + simple transparent trigger；
4. frozen BC + oracle trigger upper bound（只作分析上界，可事后用 full-H4 disagreement/violation 定义，不作为部署方案）。

simple trigger 建议先由以下任一条件触发整批 H4 correction：联合 proposal 超容量、最小 SLA slack ≤1 step、GPU/pending burst 超训练 P99、policy margin 低、future reserve/release 变化超过阈值。

### Controlled variables

- 同一 BC checkpoint、forecast、seeds/windows；
- H4 config 完全相同；
- 所有方法执行相同环境步；
- 不在线训练 actor，隔离 trigger/correction 价值。

### Metrics

- full quality/safety metrics；
- MPC call rate、corrected task ratio、trigger precision/recall；
- false-negative risky events、solver failures/timeouts；
- total latency 与 per-trigger latency；
- normal 与 stress/OOD 分开报告。

### Expected evidence

- 若常态 BC 足够且风险状态 correction 有效，B 应接近 H4 且调用率显著低于 100%；
- oracle-trigger upper bound 可判断 detector 改进是否仍有空间。

### Decision rule

B 成为领先实现需满足：call rate ≤20%、SLA 与 full H4 差异 ≤1%、联合物理成本差异 ≤1%、risky-event recall ≥95%、illegal/overflow=0，并在 stress 上优于 frozen BC。若达到质量只能靠 >50% 调用，则 B 的结构性优势不足。

## Experiment 3 — Minimal High-Level RL + MPC Prototype

### Research question

C 中 fixed-dimensional high-level intent 是否真的有价值，还是 fixed-objective MPC 已经足够？

### Compared methods

1. H4 MPC + fixed default intent；
2. H4 MPC + best small grid fixed intent；
3. H4 MPC + scripted oracle intent upper bound；
4. small high-level RL，输出 7-d `[5×GPU reserve, migration budget, backlog target]`。

### Controlled variables

- low-level optimizer、hard constraints、forecast、steps、seeds 完全相同；
- high-level action bounds 固定；
- objective weights 不由 RL 任意缩放；
- 先用很小训练预算，只验证 action 是否有 signal，不追求最终性能。

### Metrics

- reward/SLA/physical costs/migration/backlog；
- intent action variance、boundary saturation、与 fixed intent 的实际 decision disagreement；
- MPC feasibility、variable count、solve latency；
- distribution shift 前后分别统计。

### Expected evidence

- scripted upper bound 若都不能超过 best fixed intent，则没有必要训练 C；
- RL 只有在 shift 下稳定优于 best fixed intent 且 action 非退化时才证明高层接口有价值。

### Decision rule

C 只有在 scripted upper bound 先显示 ≥1% 可利用收益，且小型 RL 能捕获其中方向、SLA 不恶化、solver 始终可行时才进入正式开发；否则保留为长期备选。

## 4. 实验顺序

1. Experiment 1：最低改动，直接判断 A 是否仍然可行；
2. Experiment 2：当前最关键，直接验证领先候选 B；
3. Experiment 3：仅在前两者不能明确满足目标，或 scripted upper bound 先显示价值时进行。

三个实验都应预注册 metrics、threshold 和结果 JSON schema，避免看到结果后改变判据。

