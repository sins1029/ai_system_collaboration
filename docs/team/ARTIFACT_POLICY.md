# 数据与模型 Git 边界

## 可以提交

- `src/`、`scripts/` 中的源码。
- 小型 YAML/JSON config 和 schema。
- tests、Markdown 文档。
- 经过审计的小型 summary CSV/JSON。
- 不含受限逐任务数据的 manifest、hash 和 protocol。

## 默认不要提交

- `*.pt`、`*.pth`、`*.ckpt`、`*.onnx`、`*.safetensors`。
- `*.parquet`、`*.npz`、`*.pkl`、完整 result dataset。
- `*.sqlite`、`*.db` 及 WAL/journal 文件。
- Alibaba/Spot 原始 workload 和受限派生数据。
- virtualenv、cache、临时日志和渲染中间文件。
- `references/external_repos/` 中的第三方仓库本体。

## 当前例外

Git 历史已经合法跟踪 `artifacts/sustaincluster_imitation/bc_actor_seed_11.pt`。本轮不删除或改写该历史文件。这个例外不意味着可以提交新的 checkpoint。

## `.gitignore` 状态

当前 `.gitignore` 已覆盖：

- `.venv*/`、`__pycache__/`、`.cache/`。
- 新 checkpoint 与模型格式。
- Parquet 和常见离线数组。
- SQLite/DB 文件。
- 默认 generated artifacts。
- raw workload 和 external reference repositories。

因此本轮不追加粗暴的 `artifacts/**` 规则，也不改动现有合法跟踪的小型 artifacts。

## 提交前审计

```powershell
git status --short
git diff --cached --stat
git diff --cached --name-status
git diff --cached --check
```

逐项确认：

1. 没有大型二进制或逐任务数据。
2. 没有 checkpoint、SQLite、虚拟环境或 cache。
3. 没有第三方仓库源码。
4. 没有 API key、token、密码或个人绝对路径。
5. summary 不包含受限数据的可逆明细。

发现误暂存时只对明确路径执行 `git restore --staged <path>`，然后重新审计。不要使用 `git reset --hard` 或 `git clean`。
