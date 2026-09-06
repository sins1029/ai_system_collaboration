# GPU Type Contract

`GPU_HETEROGENEITY_MODE=METADATA_ONLY`。六种 gpu_model 原样保留；v1 只按 per-job gpu_request 的 GPU-equivalent count 调度，不声称验证型号兼容、性能或功耗。异构约束另立版本。
