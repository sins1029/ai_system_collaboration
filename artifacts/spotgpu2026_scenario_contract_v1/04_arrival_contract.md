# Arrival Contract

保留原始相对秒 `submit_time`，首任务 t=0；`arrival_step=floor(submit_time/900)`。不扰动、不合成、不 shuffle。同 step 使用稳定顺序 `submit_time -> original_index -> task_id`。边界：899->0，900->1，1799->1，1800->2。
