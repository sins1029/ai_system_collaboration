# BC 检查点说明

`bc_actor_seed_11.pt` 是当前 Architecture A 最小 BC 演示的默认检查点，仅用于推理和回归验证；演示入口不会继续训练或修改该文件。

## 生成谱系

- 生成代码：`reports/sustaincluster_imitation/train_bc.py`、`src/sustaincluster_imitation/bc_trainer.py` 和 `src/sustaincluster_imitation/bc_policy.py`。
- 训练配置：`configs/sustaincluster_imitation/bc_train.yaml`。
- 专家数据：本项目 H=4 MPC 生成的 `deployable_baseline_forecast` 主数据变体；`oracle_upper_bound` 不进入主训练 split。
- 初始化：本项目 BC actor 随机初始化训练，不包含第三方预训练模型权重。
- checkpoint 元数据：seed 11、feature dimension 233、action DC IDs 1-5、validation loss `0.18717402264055813`。
- 文件大小：`517589 bytes`。
- SHA256：`124a86cce281c834b45e3a734015ec5fa79520babc7c19d01fd167993a5f64ab`。

训练数据统计和离线指标分别见 `reports/sustaincluster_imitation/expert_dataset_report.md` 与 `reports/sustaincluster_imitation/bc_offline_evaluation_report.md`。其他随机种子检查点、训练日志和批量实验输出属于可再生成产物，不进入 Git 候选清单。

## 发布边界

该 checkpoint 是本项目训练产物，但其训练与评价依赖 SustainCluster 及 Alibaba workload。团队私有协作可以保留冻结产物；公开发布前仍需由项目负责人结合上游代码和数据许可确认。本文不声明可自由商用，也不替代法律审核。
