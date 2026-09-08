# 算能协同——多数据中心 AI 任务低碳智能调度系统

基于负载预测与优化决策的跨数据中心算力协同平台

> 团队统一开发分支：`competition/team-dev-ready`

## 快速入口

- [部署](docs/team/DEPLOYMENT.md)
- [架构](docs/team/ARCHITECTURE.md)
- [数据](docs/team/DATA.md)
- [数据库](docs/team/DATABASE.md)
- [使用](docs/team/USER_GUIDE.md)
- [开发](docs/team/DEVELOPMENT.md)
- [接口](docs/team/INTERFACES.md)
- [Dashboard](docs/team/DASHBOARD.md)
- [Git](docs/team/GIT_WORKFLOW.md)
- [产物边界](docs/team/ARTIFACT_POLICY.md)

## 1. 项目是做什么的

本项目面向多数据中心 AI 任务调度。系统根据任务 CPU、GPU、内存、优先级和 SLA，以及五个数据中心的容量、电价和碳强度，决定任务应进入 `DC1` 至 `DC5`，或在合同允许时延迟。当前仿真粒度为 15 分钟。

项目同时保留单数据中心热电耦合环境，用于计算负载、制冷、温度、电力和碳排放的联合建模。多数据中心调度主线通过 SustainCluster 适配器和本地 simulator 验证跨中心放置决策。

## 2. 当前竞赛目标

竞赛版本优先交付一个可解释、可回放、可分工开发的系统演示：

1. 展示任务到达、排队、运行和完成过程。
2. 展示五个数据中心的 CPU、GPU、内存及能源状态。
3. 展示未来 60 分钟负载预测。
4. 展示 H1 调度决策及成本、SLA、迁移和 backlog 指标。
5. 使用真实冻结实验轨迹驱动 Dashboard replay，而不是宣称连接生产云平台。

## 3. 系统核心模块

| 模块 | 作用 | 主要目录 |
|---|---|---|
| 热电环境 | 单中心计算、制冷、温度和能耗仿真 | `src/datacenter_env/` |
| 数据与合同 | workload、任务、时间线和信息边界 | `src/sustaincluster_contract/`、`scripts/audit/` |
| 预测 | 24 小时历史窗口与 Compact Transformer | `src/forecasting/` |
| 优化调度 | state/action adapter、H1、H4、MPC-H60 | `src/sustaincluster_mpc/` |
| 历史学习实验 | expert dataset、BC 和状态表示研究 | `src/sustaincluster_imitation/` |
| 存储 | SQLite 运行记录、任务事件和指标 | `src/datacenter_env/storage/` |
| 实验入口 | 数据准备、审计、训练和评估脚本 | `scripts/` |

## 4. 当前稳定模块

- SpotGPU2026 v3 数据合同和确定性五数据中心容量映射。
- H1 current-only optimizer，作为当前稳定在线优化基线。
- Compact Transformer 的 `t+60` workload forecast 基础设施。
- MPC-H60 的实验、标签、counterfactual value audit 基础设施。
- SQLite schema v4、运行存储和查询脚本。
- 单中心 simulator、多中心适配器及测试基础设施。

稳定表示接口、合同和执行路径已验证，不表示所有研究控制器都已证明优于基线。

## 5. 当前实验模块

### MPC-H60

MPC-H60 是研究和诊断分支，不是当前默认控制器。最新 TRAIN+VALIDATION common-continuation value audit 的结论为：

```text
H60_VALUE_NOT_SUPPORTED
```

Oracle H60 与 Learned H60 均未在验证集形成稳定正净收益。因此不得描述为“H60 显著优于 H1”，也不进入 Gate 训练。

### Forecast-Aware H1 v1

当前后续方向是：

```text
H1 current optimization
+ Transformer t+60 forecast
+ task-lifecycle-aware future-risk correction
```

状态：`EXPERIMENTAL / UNDER VALIDATION`。它尚未成为正式成功方案或默认控制器。

## 6. 目录说明

| 目录 | 内容 | Git 定位 |
|---|---|---|
| `src/` | 可复用源码 | 提交 |
| `scripts/` | setup、dataset、experiment、competition 入口 | 提交 |
| `configs/` | 冻结和实验配置 | 提交 |
| `tests/` | 单元、契约、回归和阶段测试 | 提交 |
| `docs/` | 架构、部署和协作文档 | 提交 |
| `artifacts/` | 摘要、metrics、manifest 和本地大产物 | 仅明确选择的小文件可提交 |
| `data/` | 样例、本地数据库和受限 workload | 只提交明确的小样例 |
| `references/external_repos/` | 只读第三方参考仓库 | 不提交第三方仓库本体 |

## 7. 五分钟快速部署

在 Windows PowerShell 中运行：

```powershell
git clone https://github.com/sins1029/ai_system_collaboration.git
cd ai_system_collaboration
git fetch --all --tags
git switch competition/team-dev-ready
powershell -ExecutionPolicy Bypass -File .\scripts\clone_reference_repos.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -PythonPath python
.\.venv-sustain-cluster\Scripts\python.exe -m pip check
```

要求 Git 和 Python 3.10 或更高版本。完整说明见 [DEPLOYMENT.md](docs/team/DEPLOYMENT.md)。

## 8. 最小运行验证

```powershell
.\.venv-sustain-cluster\Scripts\python.exe .\scripts\run_sustaincluster_demo.py --policy mpc --steps 5
```

这只是 5-step 环境与调度链路 smoke test，不代表竞赛最终 controller，也不用于证明算法收益。当前 demo 会明确警告其 Oracle information mode 属于上界、不可部署模式。

## 9. 数据说明

当前冻结 SpotGPU2026 v3 规模为 17,670 states 和 466,867 task decisions。大型 workload、完整 Expert Dataset v3 Parquet、privileged future 和新 checkpoint 不进入 Git。

没有完整本地数据时，团队成员仍可安装项目、阅读源码、开发 SQLite 查询、前后端和 replay 界面；但无法全量复现实验、重训 forecast 或重建完整 expert dataset。详见 [DATA.md](docs/team/DATA.md)。

## 10. 数据库说明

运行数据使用 SQLite schema v4，保存实验元数据、仿真时序、任务事件、决策和汇总指标。大型训练数据继续使用 Parquet，不导入 SQLite。详见 [DATABASE.md](docs/team/DATABASE.md)。

## 11. 前后端开发入口

- 前端读取 SQLite、已提交的小型 CSV summary 和 JSON manifest。
- 后端负责 loader、controller adapter、experiment runner、SQLite query 和 replay data service。
- 当前没有冻结 REST API。`docs/team/INTERFACES.md` 中的 Dashboard Data Contract v1 是拟议合同，不是已实现服务。
- 竞赛演示优先使用 frozen trajectory replay，不宣称 live production cloud control。

## 12. Git 协作规则

所有成员从 `competition/team-dev-ready` 创建自己的 `feature/xxx` 分支，通过 PR 合回团队分支。暂不直接修改或合并 `main`。禁止 force push、`git reset --hard`、`git clean`，也禁止提交大型数据、虚拟环境、数据库、checkpoint 和第三方仓库本体。

## 13. 常见问题

**为什么 clone 后没有完整实验数据？** 这些文件体积大且存在许可边界，只保留在授权的本地环境。

**H60 是当前最优控制器吗？** 不是。当前 value audit 不支持 H60 作为默认控制器。

**没有数据能做什么？** 可以开发 Dashboard、SQLite 查询、接口适配、文档和使用样例，也可以运行仓库内小型 smoke test。

**完整 pytest 中 frozen provenance test 为什么可能失败？** 某些历史测试要求精确 Git HEAD。新增文档或正常 commit 会改变 HEAD，这类失败应单独记录为历史 provenance guard，不等同于算法 runtime failure。

**从哪里开始开发？** 先完成 [DEPLOYMENT.md](docs/team/DEPLOYMENT.md)，再按角色阅读 [DEVELOPMENT.md](docs/team/DEVELOPMENT.md) 和 [GIT_WORKFLOW.md](docs/team/GIT_WORKFLOW.md)。
