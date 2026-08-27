from __future__ import annotations

import os
import sys
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """查找同时包含 pyproject.toml 和 src 目录的项目根目录。"""
    origin = Path(start or Path.cwd()).resolve()
    candidates = (origin, *origin.parents, *Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError(
        "无法定位 ai_system_collaboration 项目根目录；请从项目目录运行命令。"
    )


def resolve_sustaincluster_root(explicit: Path | None = None) -> Path:
    """按显式参数、环境变量、项目内参考仓库的顺序定位 SustainCluster。"""
    configured = explicit or os.environ.get("SUSTAINCLUSTER_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        root = (
            find_project_root()
            / "references"
            / "external_repos"
            / "sustain-cluster"
        ).resolve()
    required = (
        root / "configs" / "env" / "sim_config.yaml",
        root / "envs" / "task_scheduling_env.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "未找到可用的 SustainCluster 参考仓库。请设置环境变量 "
            f"SUSTAINCLUSTER_ROOT；当前检查目录：{root}；缺少：{', '.join(missing)}"
        )
    return root


def prepare_sustaincluster_imports(root: Path | None = None) -> Path:
    """把只读参考仓库加入当前进程的模块搜索路径。"""
    resolved = resolve_sustaincluster_root(root)
    value = str(resolved)
    if value not in sys.path:
        sys.path.insert(0, value)
    return resolved


def resolve_bc_checkpoint(explicit: Path | None = None) -> Path:
    """定位最小 BC 演示使用的已验证检查点。"""
    path = Path(explicit).expanduser().resolve() if explicit else (
        find_project_root()
        / "artifacts"
        / "sustaincluster_imitation"
        / "bc_actor_seed_11.pt"
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"未找到 BC 检查点：{path}")
    return path
