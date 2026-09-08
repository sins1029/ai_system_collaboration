# 使用手册

本文面向只想运行系统、查看结果，不修改算法的团队成员。首次安装的完整命令见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 1. 环境安装

```powershell
git switch competition/team-dev-ready
powershell -ExecutionPolicy Bypass -File .\scripts\clone_reference_repos.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -PythonPath python
.\.venv-sustain-cluster\Scripts\python.exe -m pip check
```

SustainCluster 应固定为 `3f6ea95cb835b89ba50b0ef76d66d14b8037643e`。

## 2. 数据准备

仓库提供小型样例时序和任务数据，可支持基础运行。Alibaba/Spot 原始 workload、完整 Expert Dataset v3 和 checkpoint 不随 Git 分发。

如果具备合法的 Alibaba workload 本地来源，可查看：

```powershell
.\.venv-sustain-cluster\Scripts\python.exe .\scripts\prepare_workload.py --help
```

不要把准备后的大文件提交 Git。SpotGPU2026 全量实验需要团队内部的数据目录和 manifest，缺少它们时不要尝试伪造输入。

## 3. 运行 smoke test

```powershell
.\.venv-sustain-cluster\Scripts\python.exe .\scripts\run_sustaincluster_demo.py --policy mpc --steps 5
```

该命令验证 SustainCluster 导入、环境初始化、adapter、MPC 调度和短闭环。它不是完整竞赛评估。当前 demo 会打印 Oracle information mode 的 upper-bound / non-deployable 警告，使用者必须保留这一解释，不能把结果称为在线可部署性能。

## 4. 运行基础实验

仓库的基础入口包括：

```powershell
.\.venv-sustain-cluster\Scripts\python.exe .\scripts\run_mvp.py
.\.venv-sustain-cluster\Scripts\python.exe .\scripts\run_single_center.py
.\.venv-sustain-cluster\Scripts\python.exe .\scripts\run_task_scheduling.py
```

运行前先阅读对应 `configs/`，确认输出位置、输入数据和数据库不会覆盖需要保留的本地结果。

`scripts/competition/` 下的 H60 脚本是研究入口，依赖本地冻结数据和 checkpoint。普通演示不应启动它们，也不得把 H60 描述为默认 controller。

## 5. 查看结果

- `results/`：基础实验导出的 CSV，默认不进入 Git。
- `artifacts/`：阶段 summary、metrics、manifest、报告和本地大文件。
- `data/db/*.sqlite`：运行数据库。
- 终端输出：smoke test 和 runner 的即时状态。

优先阅读小型 `summary.md`、`diagnosis.md`、CSV summary 和 JSON manifest。不要直接用单个大 Parquet 作为演示接口。

## 6. 查看 SQLite

```powershell
.\.venv-sustain-cluster\Scripts\python.exe .\scripts\inspect_database.py .\data\db\single_center.sqlite
```

常用数据：

- `experiment_runs`：运行配置和状态。
- `simulation_results`：逐 step 资源、温度、功率和 objective。
- `run_metrics`：run 级指标。
- `run_task_events`、`run_task_outcomes`：任务生命周期与 SLA。
- `run_agent_decisions`：action、mask 和 reward。

## 7. 查看 artifacts

Git 中只保留明确选择的小型研究摘要。完整 H60 value table、forecast checkpoint 和 Expert Dataset v3 Parquet 只在授权本地环境存在。若文档链接的本地大产物不存在，应向数据负责人申请，不要从不明来源下载。

## 8. Dashboard / replay 数据来源

当前推荐使用：

1. SQLite 的冻结 run 轨迹。
2. 小型 CSV summary。
3. JSON manifest 中的模型、数据和协议 provenance。

当前没有冻结 Web API。前端应先按 [INTERFACES.md](INTERFACES.md) 的拟议合同开发本地 adapter，并清楚标注 replay 模式。

## 9. 常见报错

| 报错 | 处理 |
|---|---|
| 找不到 SustainCluster | 运行 `clone_reference_repos.ps1` 并核对 commit |
| 缺少 Python 包 | 重新运行 `setup_env.ps1`，再执行 `pip check` |
| 缺少 workload/checkpoint | 确认任务是否需要受限本地数据；不要提交或伪造 |
| 数据库不存在 | 先运行会启用 SQLite store 的基础实验 |
| frozen provenance test 失败 | 记录期望和当前 Git HEAD，确认是否仅为历史精确 commit guard |
| H60 实验入口报 manifest 缺失 | 当前机器没有完整冻结产物，停止该实验并联系算法负责人 |
