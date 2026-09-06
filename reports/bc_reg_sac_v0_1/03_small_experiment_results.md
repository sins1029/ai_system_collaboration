# Architecture A 第一轮小规模实验结果

## 1. 实验身份

- Git 基线：`90eb972f78742603b522fabc3e1cf65391434418`
- 开发分支：`feature/bc-regularized-sac-v0.1`
- 配置指纹：`0D0931664A8737F7A77C728A50964EDC708B21BC85702025070CBB8B2D092A29`
- 正式规模：3 paired training seeds x 10,000 steps，3 个 fresh evaluation seeds x 96 steps
- 本机历史复现：6/6 runs 与保存结果一致，legacy 配置指纹完全一致

## 2. 10k 最终聚合

M0 为同一固定 BC 在三个 evaluation windows 上的一次确定性评估；M1-M4 为 3 个训练种子的最终评估均值，每个训练种子都使用相同三个 evaluation windows。

| Method | Reward | SLA | Agreement | Completed | Electricity | Carbon | Transmission | Defer | Illegal | Overflow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 BC Fixed | -2193.32 | 1449.67 | 0.9356 | 3314.33 | 15892.19 | 43850.74 | 103.15 | 0.7473 | 0 | 0 |
| M1 Vanilla | -2598.22 | 3062.33 | 0.7799 | 3062.33 | 15900.88 | 44215.69 | 0.00 | 1.0000 | 0 | 0 |
| M2 Warmup | -2598.22 | 3062.33 | 0.7799 | 3062.33 | 15900.88 | 44215.69 | 0.00 | 1.0000 | 0 | 0 |
| M3 Regularized | -2567.07 | 2862.56 | 0.7755 | 3079.22 | 15865.82 | 44117.88 | 26.60 | 0.9928 | 0 | 0 |
| M4 Combined | -2596.35 | 3055.00 | 0.7794 | 3062.89 | 15894.39 | 44198.65 | 0.60 | 0.9997 | 0 | 0 |

所有方法的 `average_wait_steps=0`、`nan_inf=0`。这不是“无等待问题”的充分证据，因为当前环境/统计口径没有给出非零 wait 信号。

## 3. 退化曲线

下表为三个训练种子在固定评估点的均值。

| Method | Step | Reward | SLA | Agreement | Defer ratio |
|---|---:|---:|---:|---:|---:|
| M1 | 0 | -2193.32 | 1449.67 | 0.9356 | 0.7473 |
| M1 | 2500 | -2541.44 | 2526.89 | 0.7926 | 0.9165 |
| M1 | 5000 | -2598.22 | 3062.33 | 0.7799 | 1.0000 |
| M2 | 2500 | -2193.32 | 1449.67 | 0.9356 | 0.7473 |
| M2 | 5000 | -2539.23 | 2537.78 | 0.7929 | 0.9216 |
| M2 | 7500 | -2598.22 | 3062.33 | 0.7799 | 1.0000 |
| M3 | 2500 | -2463.30 | 2013.89 | 0.8132 | 0.8433 |
| M3 | 5000 | -2591.00 | 2942.44 | 0.7776 | 0.9951 |
| M3 | 10000 | -2567.07 | 2862.56 | 0.7755 | 0.9928 |
| M4 | 2500 | -2193.32 | 1449.67 | 0.9356 | 0.7473 |
| M4 | 5000 | -2530.76 | 2538.11 | 0.7965 | 0.9216 |
| M4 | 10000 | -2596.35 | 3055.00 | 0.7794 | 0.9997 |

Warmup 在冻结期按设计完全保持 BC，但解除冻结后只把坍缩时间推迟约 2,500 steps。M1 在 5k 已进入全 defer；M2 在 7.5k 进入同一固定点。M3 保留了少量非 defer 行为，但没有保住 expert agreement 或 SLA。

## 4. 更新与数值诊断

| Method | Critic updates | Actor updates | First actor step | Critic loss | SAC actor loss | KL | Weighted KL | Q mean | Q std | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | 618 | 618 | 128 | 0.64 | 0.07 | 1.73 | 0.000 | -0.07 | 0.28 | 0.02 |
| M2 | 618 | 469 | 2512 | 0.90 | 0.13 | 1.44 | 0.000 | -0.13 | 0.27 | 0.04 |
| M3 | 618 | 618 | 128 | 0.62 | 0.06 | 0.96 | 0.010 | -0.06 | 0.23 | 0.07 |
| M4 | 618 | 469 | 2512 | 0.82 | 0.11 | 0.74 | 0.007 | -0.12 | 0.24 | 0.07 |

全部 Q/loss 有限，Teacher 训练前后哈希完全一致。M2/M4 的 Actor 更新时序证明 warmup 代码真实执行。M1 与 M2 最终指标完全相同，是三个种子都收敛到全 defer 行为，而不是汇总重复或 Actor 未更新。

## 5. 相对比较

M3 是本轮相对最佳的训练方法：相较 M1，reward 提高 31.15，SLA 减少 199.78，completed 增加 16.89，electricity 减少 35.06，carbon 减少 97.82；但 expert agreement 反而再下降 0.00434，transmission 增加 26.60。

相较固定 BC，M3 的 reward 下降 373.75，SLA 增加 1412.89，completed 减少 235.11，expert agreement 下降 0.16004。物理成本只有 electricity 和 transmission 改善，carbon 变差，不能抵消调度质量退化。

H1/H4 历史闭环使用 seeds 1201-1205，本轮使用 2201-2203，因此不能做严格 paired 差值。历史 H1/H4 reward 约为 `-1797`，且 SLA、completed 与 BC 接近；本轮没有证据表明任何 RL 方法优于 H1/H4。

## 6. Checkpoint 与原始数据

相对选择规则得到 M3，并保存最接近 M3 聚合表现的 seed 42：

```text
artifacts/bc_reg_sac_v0_1/best_actor.pt
SHA256: 6D90971CA25E078A03240B97BC231066A8374B950B143C949CDBAE7FC1FBC693
step: 10000
```

该 checkpoint 只用于算法审查和失败模式复核，没有通过继续训练 Gate。原始数据包括 `experiment_results.json`、`training_curve.csv`、`evaluation_summary.csv`、`seed_summary.csv`、`update_diagnostics.csv` 与 `loss_calibration.json`。
