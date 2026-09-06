# SLA / Defer Contract

Alibaba 只给 HP/Spot，不给 deadline、SLA minutes、waiting penalty 或 maximum defer。数值全部标记 `MODELED_SCENARIO_PARAMETER`。

公式：`sla_steps=ceil(estimated_duration/900)+max_wait_steps`。v1 选择 medium：HP 等待上限 1 step，Spot 8 steps。两类都允许 defer，但都有硬上限；保守/弹性配置保留作敏感性，不修改 reward。
