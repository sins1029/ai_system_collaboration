# Static Feasibility Report

只检查单任务是否至少有一个冻结 DC 同时满足 CPU/GPU/memory 总容量；不运行 MPC，不计并发占用、GPU 型号、bandwidth、price 或 carbon。

Total=5,000；Feasible=5,000（100.000000%）；Infeasible=0；Main reason=NONE；Diagnosis=NO SERIOUS CAPACITY MISMATCH。

## 按 HP/Spot

| priority | count | sum | mean |
| --- | --- | --- | --- |
| HP | 4487 | 4487 | 1.0 |
| Spot | 513 | 513 | 1.0 |

## 按 GPU model

| gpu_model | count | sum | mean |
| --- | --- | --- | --- |
| A10 | 2058 | 2058 | 1.0 |
| A100-SXM4-80GB | 1774 | 1774 | 1.0 |
| A800-SXM4-80GB | 117 | 117 | 1.0 |
| GPU-series-1 | 98 | 98 | 1.0 |
| GPU-series-2 | 804 | 804 | 1.0 |
| H800 | 149 | 149 | 1.0 |

## 按 GPU request

| gpu_request | count | sum | mean |
| --- | --- | --- | --- |
| 0.3 | 367 | 367 | 1.0 |
| 0.5 | 496 | 496 | 1.0 |
| 1.0 | 2669 | 2669 | 1.0 |
| 2.0 | 185 | 185 | 1.0 |
| 4.0 | 223 | 223 | 1.0 |
| 8.0 | 1060 | 1060 | 1.0 |
