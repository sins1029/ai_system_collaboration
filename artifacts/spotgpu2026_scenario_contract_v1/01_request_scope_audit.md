# Request Scope Audit

`REQUEST_SCOPE = PER_JOB`

官方 README 明确写明 CPU cores 与 GPU 数是“requested by the job”，并将 `worker_num` 单独定义为 job 请求的 instances 数。[固定 commit README](https://github.com/alibaba/clusterdata/blob/0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71/cluster-trace-v2026-spot-gpu/README.md)

冻结公式：`CPU_job=cpu_request`，`GPU_job=gpu_request`，不乘 worker_num。双口径统计仅作诊断：误乘后最大 GPU 从 8 变为 1,456，并出现 65 个超过所有当前 DC GPU 容量的任务。官方目录无示例适配代码，公开 issue 无补充，Discussions 未启用；现有官方证据无冲突。
