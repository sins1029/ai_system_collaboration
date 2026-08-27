"""运行项目受控 pytest 测试集。"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "pytest", "-q"]
    env = os.environ.copy()
    python_paths = [str(root), str(root / "src")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    raise SystemExit(subprocess.call(command, cwd=root, env=env))


if __name__ == "__main__":
    main()
