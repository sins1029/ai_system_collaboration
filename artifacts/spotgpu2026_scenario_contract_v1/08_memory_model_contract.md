# Memory Model Contract

Spot 无 task host memory，故 `memory_request=MODELED`。v1 只用 Alibaba2020 时间前缀 70% train：CPU/GPU 固定 bin 条件中位数，support<100 回退 CPU bin，再回退全局 train 中位数；无随机采样。

该值仅补齐 SustainCluster 三资源场景，不能称为真实 Spot 内存；单位沿用当前 Alibaba2020->SustainCluster 链。分布及 memory/CPU、memory/GPU 比率见验证 CSV。
