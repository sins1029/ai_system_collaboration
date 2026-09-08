# 开发手册

## 开发基线

所有新工作从 `competition/team-dev-ready` 创建独立 feature 分支。不要直接在 `main`、团队基线或冻结研究 tag 上开发。

## A. Dashboard / Frontend

### 可以依赖

- SQLite schema v4 和只读查询结果。
- 已提交的小型 CSV summary、JSON manifest 和 Markdown 说明。
- [INTERFACES.md](INTERFACES.md) 中标记为 proposed 的 Dashboard Data Contract v1。
- 冻结 trajectory replay。

### 不得依赖

- 完整训练 Parquet。
- Oracle future 或 simulator-only `true_duration`。
- 开发者机器上的个人绝对路径。
- 尚未冻结的 REST 字段或 H60 成功假设。

### Ownership

- 页面、组件、交互和可视化。
- replay 时间轴和 loading/error/empty 状态。
- 数据合同 consumer tests。

## B. Backend / Integration

### 负责

- 数据加载和 schema validation。
- current-state/controller/action adapter。
- experiment runner 与 replay data service。
- SQLite read model 和查询性能。
- proposed dashboard contract 的版本化转换。

### 约束

- 当前没有冻结 Web API，不得让临时 endpoint 成为隐式正式合同。
- 服务层只能暴露 deployable 信息；Oracle 数据必须隔离。
- 大型 Parquet 不应直接由浏览器加载，也不应整体导入 SQLite。

## C. Algorithm

### 负责

- workload forecast 和评估。
- H1、实验性 Forecast-Aware H1 与调度 optimizer。
- information leakage、counterfactual value 和闭环 evaluation。
- 算法 config、tests、small summary 和 protocol provenance。

### 当前边界

- H1 是稳定 current-only baseline。
- Compact Transformer `t+60` forecast 已冻结。
- MPC-H60 是诊断分支，value audit 结论为 `H60_VALUE_NOT_SUPPORTED`。
- Forecast-Aware H1 v1 是 `EXPERIMENTAL / UNDER VALIDATION`。
- 不启动 Gate，不把 test 用于路线选择，不修改冻结 v3。

## 模块 ownership

| 路径 | 主要 owner | 变更要求 |
|---|---|---|
| `src/datacenter_env/` | Backend + Algorithm | 运行与存储回归测试 |
| `src/forecasting/` | Algorithm | split、leakage、metrics tests |
| `src/sustaincluster_mpc/` | Algorithm + Backend | adapter、feasibility、solver tests |
| `src/sustaincluster_imitation/` | Algorithm | 数据 schema 与历史策略回归 |
| `scripts/` | 对应模块 owner | CLI/help 和最小运行验证 |
| `docs/team/` | 全体 | 保持状态和命令真实 |
| Dashboard 新目录 | Frontend | contract fixtures 与视觉验收 |

## 开发前检查

```powershell
git fetch origin
git switch competition/team-dev-ready
git pull --ff-only
git switch -c feature/your-task
```

先阅读：

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [DATA.md](DATA.md)
- [ARTIFACT_POLICY.md](ARTIFACT_POLICY.md)
- 对应模块源码和 tests

## 提交前检查

```powershell
git status --short
git diff --check
.\.venv-sustain-cluster\Scripts\python.exe -m pytest -q path\to\relevant_tests.py
```

共享接口、schema 或 controller contract 变更还应运行完整 pytest。提交 PR 时说明数据范围、信息边界、测试结果和未提交本地产物。
