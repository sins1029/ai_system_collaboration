# Windows 本地部署

## 支持范围

- Windows 10 / 11
- Windows PowerShell 5.1 或 PowerShell 7
- Git
- Python 3.10 或更高版本

## 1. 克隆团队分支

```powershell
git clone https://github.com/sins1029/ai_system_collaboration.git
cd ai_system_collaboration
git fetch --all --tags
git switch competition/team-dev-ready
```

## 2. 准备固定 SustainCluster

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\clone_reference_repos.ps1
git -C .\references\external_repos\sustain-cluster rev-parse HEAD
```

期望输出：

```text
3f6ea95cb835b89ba50b0ef76d66d14b8037643e
```

脚本会拒绝覆盖存在本地修改或 commit 不匹配的 SustainCluster 工作区。其他四个参考仓库不是运行主线依赖；仅研究需要时使用 `-AllReferences`。

## 3. 创建项目环境

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -PythonPath python
```

环境路径：

```text
.\.venv-sustain-cluster\Scripts\python.exe
```

脚本安装固定 SustainCluster requirements，并以 `research,test` extras 安装本项目。

## 4. 验证环境

```powershell
.\.venv-sustain-cluster\Scripts\python.exe -m pip check
.\.venv-sustain-cluster\Scripts\python.exe .\scripts\verify_env.py
```

## 5. 编译与测试

```powershell
.\.venv-sustain-cluster\Scripts\python.exe -m compileall -q src scripts tests
.\.venv-sustain-cluster\Scripts\python.exe -m pytest -q
```

某些历史 frozen tests 会校验精确 Git HEAD。团队文档 commit 使 HEAD 改变后，此类测试可能报告 provenance mismatch。应在测试记录中单列它，不应误报为算法 runtime failure，也不要直接删除历史测试。

## 6. 最小 smoke test

```powershell
.\.venv-sustain-cluster\Scripts\python.exe .\scripts\run_sustaincluster_demo.py --policy mpc --steps 5
```

该脚本真实支持 `--policy {mpc,bc}`、`--steps 5..20`、`--seed`、`--start-time` 和 `--sustaincluster-root`。默认 MPC 运行只验证 5-step 环境、适配器和调度链路，不代表竞赛最终 controller。当前脚本会警告 Oracle information mode 为 upper-bound / non-deployable；这条 smoke 命令不得被当作正式部署证据。

## 7. 常见部署问题

### 找不到 Python

确认 `python --version` 至少为 3.10，或向 setup 脚本传入明确路径：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -PythonPath C:\Path\To\python.exe
```

### SustainCluster 目录已存在但检查失败

不要删除或重置该目录。运行：

```powershell
git -C .\references\external_repos\sustain-cluster status --short
git -C .\references\external_repos\sustain-cluster rev-parse HEAD
```

把结果交给仓库维护者处理。

### 缺少完整数据或 checkpoint

这是正常的 Git 边界。基础开发和 smoke test 不依赖完整训练数据；全量实验需要团队授权的本地数据和模型产物。

### pytest 只有 provenance guard 失败

记录失败测试名、期望 commit 和当前 commit。确认其余单元、接口和 runtime tests 通过后，将该项标记为历史来源保护问题，不修改 frozen test 来掩盖差异。
