# Resource Quality Report

冻结样本 5,000 条，取 validation 起始段并保持稳定顺序。CPU zero=0，CPU negative=0，GPU nonpositive=0，duration nonpositive=0，worker nonpositive=0。

不自动删除 outlier。Spot 全量源的 2 条 CPU=0 保留并标记，不静默修正。完整 mean/P50/P90/P99/max 和容量越界计数见 CSV。
