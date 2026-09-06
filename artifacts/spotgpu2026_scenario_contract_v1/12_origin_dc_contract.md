# Origin DC Contract

Spot 无多 DC 来源，故 `origin_dc=MODELED_ORIGIN`。v1 用 `namespace+task_id+submit_time` 的 SHA256 值映射到 population weight x local-time activity 累积分布；场景锚点为 2026-01-01T00:00:00Z，本地 08:00-19:59 权重 1.0，其余 0.3。

无运行时 RNG；同任务和配置必得同一 origin。全量总体与 UTC 六小时分段分布见 CSV。
