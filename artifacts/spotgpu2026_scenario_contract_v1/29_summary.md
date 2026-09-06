# SpotGPU2026 调度场景合同冻结 v1

1. Request scope：PER_JOB，不乘 worker_num。
2. Arrival：floor(submit_time/900)，同 step 稳定排序。
3. Duration：true 仅 simulator；estimated 为 Spot train-only 层级中位数。
4. Memory：Alibaba2020 train-only 条件中位数，MODELED。
5. Bandwidth：Alibaba2020 train-only 全局中位数，MODELED。
6. Origin：SHA256 + population/activity 权重确定性映射，MODELED_ORIGIN。
7. SLA/defer：三套场景，v1 选 medium，HP/Spot 均 bounded。
8. GPU 型号：metadata-only，资源按 GPU-equivalent。
9. DIRECT：submit/raw CPU/GPU/worker/duration/model/priority；DERIVED：step/task_id；MODELED：memory/bandwidth/estimated/origin/SLA；SIMULATOR_ONLY：true duration。
10. 静态可调度率：100.000000%。
11. 严重 capacity mismatch：NO。
12. 可重建 v3：YES；仅技术/私有生成放行，公开许可未确认。
