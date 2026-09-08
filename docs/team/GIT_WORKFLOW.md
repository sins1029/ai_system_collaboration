# Git 协作规范

## 分支模型

```text
competition/team-dev-ready
        |
        +-- feature/dashboard-xxx
        +-- feature/backend-xxx
        +-- feature/algorithm-xxx
        +-- feature/docs-xxx
        |
        +-- Pull Request
        v
competition/team-dev-ready
```

暂时不要直接向 `main` 合并，也不要把个人开发直接提交到团队分支。

## 开始开发

```powershell
git fetch origin
git switch competition/team-dev-ready
git pull --ff-only
git switch -c feature/your-task
```

建议分支名清楚表达 owner 和范围，例如 `feature/dashboard-replay-v1`、`feature/backend-sqlite-query-v1`。

## 开发过程中

- 小步提交，commit message 说明行为而非操作过程。
- 不改与任务无关的冻结实验、schema 或 controller。
- 不覆盖他人的未提交修改。
- 数据和模型只保存在受控本地目录。
- 需要共享接口时，先在 PR 中更新合同和 fixture。

## 提交前

```powershell
git status --short
git diff --check
.\.venv-sustain-cluster\Scripts\python.exe -m pytest -q path\to\relevant_tests.py
```

只使用明确路径 `git add <files>`。不要使用 `git add .` 把本地产物一起带入。

## Pull Request

PR 至少写明：

- 目标和非目标。
- 修改的模块与接口。
- 数据和 information class。
- 测试命令及结果。
- 未提交的本地 artifacts。
- 截图或 replay 证据，若涉及前端。

## 禁止操作

- force push。
- `git reset --hard`。
- `git clean`。
- 直接提交大型数据、Parquet、SQLite、checkpoint 或虚拟环境。
- 提交 `references/external_repos/` 中的第三方源码。
- 未经审查改写冻结实验历史。

遇到历史冲突或大文件误暂存时先停止并请求 review，不要用破坏性命令自行清理。
