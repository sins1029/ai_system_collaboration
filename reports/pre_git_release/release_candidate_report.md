# Architecture A 最小可用发布候选报告

日期：2026-08-27
分支：`feature/single-center-task-runtime-v0.2`
基线提交：`d49870b9d9fbee4b480100f4a64d74075b0e1a8b`

## 1. 第一版架构

- Architecture A：当前第一版主架构。H=4 MPC 作为离线专家/优化基线，BC 作为在线策略，未来带正则 RL 复用同一策略接口。
- Architecture B：保留 `Policy → optional correction → ActionAdapter` 接口位置，尚未实现风险检测和事件触发 MPC。
- Architecture C：研究候选，不进入第一版代码主路径。

现有 H=4、专家数据、BC 和 SAC 报告没有出现足以推翻 Architecture A 的新证据。

## 2. 当前系统执行路径

```text
SustainCluster
  ↓
StateAdapter / HorizonAdapter
  ↓
H=4 RollingHorizonOptimizer
  ↓
Expert Dataset（seed 互斥、oracle 隔离）
  ↓
FeatureEncoder / BCPolicy
  ↓
AssignmentDecision
  ↓
ActionAdapter
  ↓
SustainCluster env.step
```

最小 Demo 直接选择 MPC 或 BC，不训练模型、不写专家数据。BC checkpoint 可通过 `sac_weight_bridge.py` 精确初始化同结构 actor，但 vanilla SAC 不属于当前默认执行路径。

## 3. 核心代码

- `src/sustaincluster_mpc/`：状态、时域和动作适配；H=1/H=4 优化。
- `src/sustaincluster_imitation/`：路径、环境工厂、数据 schema/读写/划分、因果预测、特征、BC 和 SAC bridge。
- `scripts/run_sustaincluster_demo.py`：唯一 Architecture A 演示入口。
- `scripts/run_tests.py`：统一 pytest 入口。
- `configs/sustaincluster_mpc/`、`configs/sustaincluster_imitation/`：冻结配置。
- `tests/test_sustaincluster_imitation.py` 和 `reports/sustaincluster_mpc/test_*.py`：核心与端到端回归。

`src/datacenter_env/` 等既有单中心模块继续参与测试，但不是多中心默认入口。

## 4. 外部依赖

- Python 3.10+；当前验证 3.10.11。
- NumPy 1.26.4、Pandas 2.2.3、SciPy 1.15.3、PyTorch 2.7.0+cpu、PyArrow 19.0.1、PyYAML 6.0.2、Pytest 8.3.5、Gymnasium 1.0.0。
- `pip check`：`No broken requirements found.`
- SustainCluster：只读第三方仓库，commit `3f6ea95cb835b89ba50b0ef76d66d14b8037643e`，`main` 工作树干净，LICENSE 存在。
- workload：由 SustainCluster 的 `sim_config.yaml` 指向，不复制进主项目 Git。
- BC checkpoint：`artifacts/sustaincluster_imitation/bc_actor_seed_11.pt`，517589 bytes。

缺少仓库、workload 或 checkpoint 时，项目路径层提供中文错误和准备说明。

## 5. Demo

最终 5 步 MPC：157 个任务动作，平均决策 10.689 ms，求解失败 0、非法动作 0、资源越界 0。

最终 5 步 BC：157 个任务动作，平均决策 5.139 ms，求解失败 0、非法动作 0、资源越界 0。

从系统临时目录启动新 Python 进程，通过 `SUSTAINCLUSTER_ROOT` 运行 5 步 BC：通过；平均决策 4.027 ms，失败/非法/越界均为 0。

时延受机器和冷启动影响，只作为本机烟测记录。

## 6. 测试

- 5 个主线 YAML 解析通过。
- `compileall` 通过。
- 全量受控 pytest：`140 passed in 60.00s`，0 failed，0 skipped。
- one-step MPC 文件清理后单独复核：15 passed。
- 两个 PowerShell 部署脚本静态语法通过。
- 依赖导入与 `pip check` 通过。

测试收集只覆盖 `tests/` 和 `reports/sustaincluster_mpc/`，不递归进入上游、数据、PPT 或 artifacts。

## 7. 文档

- 根 `README.md`：中文项目定位、Mermaid、真实 clone/安装、Demo、测试、目录、三人分工和结论。
- `docs/使用说明.md`：环境、SustainCluster、workload、checkpoint、expert dataset、Demo、测试和 Windows 注意事项。
- `docs/架构说明.md`：Architecture A/B/C、六层输入输出边界和信息泄漏规则。
- `docs/开发说明.md`：中文规范、模块边界、三人职责、feature branch + PR。
- `reports/pre_git_release/`：架构冻结决策、发布候选总报告与本次提交白名单。

## 8. Git 建议提交文件

本轮白名单包含 14 个已跟踪修改和 59 个新增文件，共 73 个文件。按组包括：

- `.gitattributes`、`.gitignore`、README、pyproject 和必要部署/测试脚本。
- MPC/模仿学习源码、5 个冻结 YAML、3 个中文文档和 1 个集成测试。
- seed 11 BC checkpoint 及说明。
- 23 份长期架构/MPC/BC 证据与复现脚本。
- 本目录 3 份冻结记录：架构决策、发布候选总报告与提交白名单。

人工必须逐组确认，尤其是 checkpoint 发布授权和 23 份长期证据范围；不使用无差别暂存。

## 9. 不建议提交文件

- 五个 `references/external_repos/*` 只读仓库。
- 原始/处理后 workload 和 expert Parquet。
- 虚拟环境、`.runtime/`、缓存、SQLite、PPT 工作目录。
- seed 22/33 等中间 checkpoint、训练 JSON、批量 CSV/JSON/Parquet。
- 报告 PDF/TeX/辅助文件和旧机器绑定部署快照。
- 19 份重复或时效性强的架构/发布审计子报告，以及临时 MPC 依赖说明。

工作区非虚拟环境文件中 >5 MiB 共 18 个，均已排除在候选集外；Git 候选 >5 MiB 为 0。

## 10. 已知问题

- 5 步 Demo 不替代既有 96/10000 步实验。
- 当前 BC 检查点许可与发布授权需要人工确认。
- 上游未来改字段/相对路径时仍需重新做契约测试。
- 历史单中心和 vanilla SAC 复现代码为可复现性保留，尚未物理归档。
- 项目自身主线异常、核心 CLI、关键 docstring 和新文档已中文化；内部字段/状态码保持英文，第三方原始异常不翻译。

## 11. 三人后续开发边界

- 开发者 A，环境与优化：adapter、MPC、constraint、forecast。
- 开发者 B，学习算法：expert dataset、BC、regularized RL、policy。
- 开发者 C，实验与评价：runner、metrics、evaluation、plots、regression。

三人都不能长期直接修改上游 SustainCluster。Architecture B 只能在语义策略和 ActionAdapter 之间接入，并必须继续通过合法性与资源边界测试。

## 12. 上传 Git 前人工确认

- 逐文件审阅 14 个已跟踪修改和 59 个新增白名单文件。
- 确认 README 与版本 `0.3.0` 口径。
- 确认 SustainCluster license/引用方式与固定 commit。
- 确认 seed 11 checkpoint 来源、许可和是否随仓库发布。
- 确认 23 份长期研究证据/复现脚本需要进入第一版。
- 确认原始数据、处理后数据、旧部署快照和 PPT 均未进入候选。
- 确认 secrets、个人路径、大文件和最终 diff 审计结果。
- 人工决定并执行暂存、提交、PR、标签和远程操作。

## Git 安全声明

候选审计结果：大于 5 MiB 文件 0，高置信敏感信息 0，邮箱 0，审计机器绝对路径 0；外部 SustainCluster 工作树干净。

本轮 Git 提交阶段仅按白名单显式暂存；未运行 `git commit`、`git push`、`git tag`、`git merge` 或 `git rebase`，没有修改 remote，也没有修改 SustainCluster 上游工作树。

**READY FOR HUMAN REVIEW**
