# Duration Estimator Contract

`true_duration` 直接来自 Spot duration，只供 simulator 完成、资源释放与离线评估，禁止进入 deployable observation。

`estimated_duration` 只用时间前缀 70% train 拟合层级条件中位数：GPU model+GPU request+worker+priority -> GPU model+GPU request+priority -> GPU request+priority -> priority -> 全局中位数。support 20/50/100 仅按 train 内部交错子样本稳定性选择，冻结为 100；validation/test 未参与选择。

Validation：MAE=137366.314s，Median AE=1560.000s，P90 AE=63153.700s。重尾误差标记 `DURATION_TAIL_RISK`；这是无泄漏基线，不是高精度 predictor。
