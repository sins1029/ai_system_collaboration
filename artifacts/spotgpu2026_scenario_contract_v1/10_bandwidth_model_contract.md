# Bandwidth Model Contract

Spot 无 task bandwidth，故 `bandwidth=MODELED`。比较全局/GPU-bin/CPU-GPU-bin 中位数后，v1 按简单稳定原则冻结 Alibaba2020 train-only 全局中位数 0.020062580706 GB。低/基准/高为 0.5x/1x/2x。

Bandwidth 不在静态 CPU/GPU/memory 容量约束中；本轮未跑 MPC，migration decision 标记 NOT_MEASURED，不能把 transfer-cost 线性代理当成调度结论。
