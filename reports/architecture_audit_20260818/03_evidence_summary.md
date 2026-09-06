# 当前证据汇总

## 1. 证据优先级

本审计采用以下优先级：

1. 当前 source/config 与原始 JSON；
2. 由当前 JSON 生成的最新公平比较/闭环报告；
3. 较早阶段报告，用于说明能力与历史因果链；
4. 设计建议，不作为实验事实。

较早 `rolling_horizon_report.md` 的 one-step vs rolling 压力比较使用了不同模型/目标口径；它证明 H>1 可以构造出前瞻行为，但不能覆盖更新的同口径 H=1/H=4 公平比较。

## 2. H=1 与 H=4 最新公平比较

两者使用相同 `RollingHorizonOptimizer`、目标、约束、forecast mode、种子和 96-step episode，仅 horizon 不同，每场景 5 seeds。

| 场景 | H1→H4 主要变化 | 客观判断 |
|---|---|---|
| capacity_release | wait 2→1；defer 66.7%→50%；stage cost 1644.66→829.24；完成/SLA不变 | 明确前瞻收益。 |
| future_low_price | 所有行为/质量/成本指标相同 | H=4 无新增收益。 |
| gpu_burst | 完成 32、SLA 16、wait 17.67、defer 96.4%、成本完全相同 | H=4 无新增收益。 |
| high_load_trace | 完成相同；SLA 2477.4→2480.8（差 3.4）；electricity 低 14.06；stage cost 低 2.78 | 运行成本差极小，SLA 略差。 |
| normal_trace | 完成相同；SLA 936.0→936.2（差 0.2）；electricity 低 12.39；stage cost高 32.73 | 基本相同，且不是所有指标都更好。 |

所有 H=4 运行：infeasible=0、illegal action=0、resource overflow=0、action-order error=0。

现实 trace 的计算开销：

- normal：平均 1.733→4.007 ms，即 2.31 倍（绝对增加 2.274 ms）；
- high-load：平均 2.813→7.888 ms，即 2.80 倍（绝对增加 5.075 ms）；
- 公平比较最大 H=4 solve time：29.603 ms；
- 另一个 BC 闭环 harness 中 H1/H4 为 6.764/20.513 ms，即 3.03 倍。

15 分钟调度周期为 900,000 ms；29.603 ms 约占 0.0033%。所以 H=4 的相对开销明显，但绝对开销不是当前实时瓶颈。

结论：

> 当前证据仅支持 H=4 是有效的预测优化器和专家基线，并不能证明每一个调度时刻都必须使用 H=4。

## 3. 较早构造压力证据

较早、非同口径实验说明 H=4 的机制能力：

- oracle GPU burst：SLA violation 2→0；
- future low price：可等待进入低价窗；
- capacity release：少等待一步；
- SLA conflict：deadline 阻止错误等待；
- center heterogeneity：可利用未来低价小中心；
- oracle burst 中 H=4 与 H=8 相同，H=8 只增加开销；
- 预测噪声加倍后性能才出现退化，说明该单一构造场景有余量，但不是 robust MPC 证明。

这些是“前瞻能力存在”的证据，不是“部署中 H=4 总体优于 H=1”的证据。

## 4. Expert dataset

主训练集 `deployable_baseline_forecast`：

- 80 episodes，7,680 steps，256,125 task-actions；
- defer 68.376%；
- local execution 26.208%（在已 assign 内）；
- migration 73.792%（在已 assign 内）；
- GPU task 70.284%；
- SLA-urgent 79.967%；
- solver failure 0；
- 各语义 DC 类均超过 5%；
- 完整 episode、seed-exclusive 60/20/20 split；
- oracle 不进入 primary split；
- test 包含 unseen burst intensity。

此外有物理隔离的 `deployable_no_future` 和 `oracle_upper_bound` 数据集，但 BC 主训练只用 causal baseline 数据。

## 5. BC 离线证据

三 seed test aggregate：

| 指标 | 均值 ± std | 含义 |
|---|---:|---|
| overall accuracy | 96.981% ± 0.141% | 总分类命中。 |
| top-2 accuracy | 99.622% ± 0.066% | 专家动作大多在前两名。 |
| defer F1 | 99.991% | defer/assign 边界几乎完全复现。 |
| assign accuracy | 90.492% ± 0.446% | 只在专家选择 assign 的样本上判断具体 DC；这是路由学习的关键证据。 |
| migration F1 | 94.790% ± 0.316% | 是否跨中心迁移的判断高度一致。 |

overall accuracy 的一部分来自 defer 占主训练集 68.376%；永远 defer 的朴素基线本身就会得到约 68.4% accuracy。因此 overall 不能单独证明路由知识。assign accuracy 和 migration F1 排除了这一主要混淆，表明网络确实学到了目的中心与迁移模式，而不仅是学会 defer。

## 6. BC 真实闭环

5 个未用于 train/validation/test 的 fresh seeds，normal/high-load trace 交替，96 steps：

| Policy | Reward | Completed | SLA | Electricity | Carbon | Transmission | Migration | Inference ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H4 MPC | -1797.51 | 3512.0 | 1552.6 | 14786.32 | 45277.91 | 101.23 | 50.98% | 20.513 |
| H1 MPC | -1796.68 | 3512.0 | 1553.0 | 14834.58 | 45338.38 | 100.70 | 51.45% | 6.764 |
| BC | -1866.31 | 3512.2 | 1552.8 | 14736.78 | 45082.71 | 119.86 | 53.85% | 4.076 |
| random | -2439.22 | 3506.6 | 1597.6 | 14693.80 | 45496.13 | 340.51 | 77.83% | 4.197 |

所有四个 manual policies 的 illegal action 与 resource overflow 均为 0。

BC 相对 H4：

- completed +0.2，SLA violation +0.2，差距极小；
- electricity 低 49.54，carbon 低 195.20；
- transmission 高 18.63（约 +18.4%）；
- migration 高 2.88 个百分点；
- `electricity+carbon+transmission` 低 0.376%；
- reward 更差 68.81；
- inference 快约 80.1%，但 H4 的 20.5 ms 本身不构成 15 分钟周期瓶颈。

因此 BC 已经不只是离线分类器，而是能独立控制真实 multi-action 环境。它在完成、SLA 和联合物理成本上近似 H4，但不能写成全面优于 H4：MPC 的 reward、transmission 与 migration 更好，BC 的 electricity、carbon、联合成本和速度更好。

原始 `local_only` rule 的 reward/SLA 是接口零值，不能解释为无 SLA violation；只可比较其由 DC info 填充的 electricity/carbon/completed 等指标。

## 7. SAC warm-start

3 个 paired seeds、每组 10,000 环境交互：

| 指标 | random-init | BC-init |
|---|---:|---:|
| initial reward | -3641.44 ± 765.59 | -1821.22 ± 0.00 |
| initial SLA | 966 | 936 |
| initial expert agreement | 0.7796 | 0.9256 |
| steps to threshold | 666.7 | 0 |
| final reward | -2359.93 | -2432.21 |
| final SLA | 1977 | 2080 |
| final agreement | 0.7668 | 0.7690 |

BC 的 0 reward std 来自三次 run 在 step 0 使用同一 BC checkpoint 与同一 evaluation seed，不等同于跨环境稳健性零方差。

已确定：MPC→BC 显著改善 actor 初始化和初始策略质量。BC-init 更新后 reward 下降 610.99，SLA 增加 1,144，agreement 下降 0.1567；vanilla SAC 没有稳定保留专家行为。

尚不能确定退化原因。critic cold start、entropy/Q bias、reward mismatch、replay distribution、online distribution shift、catastrophic forgetting，以及 SAC loss 未使用 per-action feasible mask 都是可能解释，需要消融；当前结果不能推出“RL 不能直接执行”。

## 8. 当前最强结论

1. multi-action 逐任务闭环已经打通。
2. MPC 是可靠的约束调度器、benchmark 与 expert teacher。
3. H=4 具备前瞻能力，但没有被证明需要永久在线。
4. BC 学到的不只是 defer，还学到了具体路由和 migration 规律。
5. BC 可以独立控制真实 SustainCluster，并在已测分布上接近 H4。
6. BC warm start 明显改善 RL 起点。
7. 当前 vanilla SAC 更新会破坏专家行为；这削弱的是“vanilla A”，不是所有 direct RL。

