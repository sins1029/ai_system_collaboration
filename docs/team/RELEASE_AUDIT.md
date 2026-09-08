# TEAM-DEV-READY 发布审计

审计日期：2026-09-08

## 发布前保护记录

| 项目 | 结果 |
|---|---|
| 初始分支 | `competition/mpc-h60-rush` |
| 初始 HEAD | `3de39f9cf4d0808f1f63f54ad35e5c2d8100e00c` |
| Git remote | `git@github.com:sins1029/ai_system_collaboration.git` |
| 工作区 | 存在未跟踪 H60 源码和本地报告，无 tracked 修改 |
| 破坏性 Git 操作 | 未使用 |

初始最近历史：

```text
3de39f9 docs: refine counterfactual action-value audit protocol
ec0fab9 docs: refresh research README and team task plan
c83515b research: freeze MPC expert v3 and Spot control-value baseline
90eb972 fix: 修复协作环境初始化与数据准备流程
7ea8d00 feat: 冻结首版调度架构并整理三人协作版本
```

## 算法源码冻结

- 审计并提交 12 个 H60 source/config/test 文件。
- Commit：`e01cae65d0c7171669115d7e064445b4f75dc5b4`
- Message：`competition: add H60 forecast and control experiments`
- Source branch：`competition/mpc-h60-rush`
- Source push：PASS
- 远程分支：`origin/competition/mpc-h60-rush`

未提交：`artifacts/mpc_h60_v1/` 下 Parquet、checkpoint 和生成结果；历史 `reports/pre_git_release/` 与 `reports/team_onboarding/` 本地材料。

## 团队分支

- Branch：`competition/team-dev-ready`
- 创建基线：`e01cae65d0c7171669115d7e064445b4f75dc5b4`
- `main`：未修改
- force push / merge / rebase：未执行

## 当前研究状态

- SpotGPU2026 v3：17,670 states / 466,867 task decisions。
- H1：稳定 current-only baseline。
- Compact Transformer `t+60` forecast：冻结实验基础设施。
- MPC-H60：研究与诊断分支。
- H60 common-continuation value audit：`H60_VALUE_NOT_SUPPORTED`。
- Forecast-Aware H1 v1：`EXPERIMENTAL / UNDER VALIDATION`。

## 环境

| 检查 | 结果 |
|---|---|
| Python | `3.10.11` |
| SustainCluster commit | `3f6ea95cb835b89ba50b0ef76d66d14b8037643e` |
| SustainCluster status | clean |
| Mermaid CLI | 未安装；保留 Mermaid 源 |
| compileall | PASS，`python -m compileall -q src scripts tests` |
| relevant tests | PASS，`16 passed` |
| full pytest | `422 passed, 1 failed`；唯一失败为已知历史 HEAD provenance guard |
| smoke test | PASS，5 steps / 157 actions / 0 solver failure / 0 invalid action / 0 resource violation |

## 已知测试边界

唯一完整测试失败：

```text
tests/test_structured_current_state_representation_v1.py::test_frozen_contract_and_split_identity
RuntimeError: current project HEAD differs from frozen baseline
```

该测试要求 `expected_git_head = 90eb972f78742603b522fabc3e1cf65391434418`。当前 HEAD 已包含后续研究和团队交接提交，因此精确 HEAD 不同。相关 dataset hash、manifest 和运行测试未报告失败；此项记录为历史 provenance guard，不是算法 runtime failure，测试原样保留。

5-step smoke test 同时打印 Oracle information mode 为 upper-bound / non-deployable 的警告。该命令只验收环境与调度链路，不作为正式 controller 效果证据。

## Git 边界

| 类别 | 本轮团队文档提交 |
|---|---|
| 大型 Parquet | 排除 |
| 新 checkpoint | 排除 |
| SQLite / DB | 排除 |
| virtualenv / cache | 排除 |
| external repository | 排除 |
| 本地 PPT / 图片报告 | 排除 |
| README 和 `docs/team/` | 纳入 |
