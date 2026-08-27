"""准备并校验 Architecture A 使用的 SustainCluster workload。"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path


WORKLOAD_RELATIVE_PATH = Path(
    "data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl"
)
WORKLOAD_SIZE = 142_336_499
WORKLOAD_SHA256 = "3ba8a8e0067288f8d2542752a292f7da04cb79305689ba145cc6060e96cb7f2e"
ARCHIVE_SIZE = 75_625_706
ARCHIVE_SHA256 = "907090a42863c46887cdf9c66f99ac77b2a3b351544362d8c6c34b9ace6754a9"


def sha256(path: Path) -> str:
    """流式计算文件 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    """校验文件大小和 SHA256，不接受静默的数据版本漂移。"""
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"文件大小不匹配：{path}\n期望：{expected_size} bytes\n实际：{actual_size} bytes"
        )
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA256 不匹配：{path}\n期望：{expected_sha256}\n实际：{actual_sha256}"
        )


def resolve_sustaincluster_root(explicit: Path | None) -> Path:
    """按参数、环境变量和项目默认目录定位 SustainCluster。"""
    project_root = Path(__file__).resolve().parents[1]
    configured = explicit or os.environ.get("SUSTAINCLUSTER_ROOT")
    root = (
        Path(configured).expanduser().resolve()
        if configured
        else project_root / "references" / "external_repos" / "sustain-cluster"
    )
    if not (root / "configs" / "env" / "sim_config.yaml").is_file():
        raise FileNotFoundError(
            "未找到 SustainCluster。\n"
            f"当前检查目录：{root}\n"
            "请先运行 scripts/clone_reference_repos.ps1，或设置 SUSTAINCLUSTER_ROOT。"
        )
    return root.resolve()


def extract_archive(archive: Path, destination: Path) -> None:
    """只提取归档中的目标 workload，避免路径穿越和额外文件写入。"""
    validate_file(archive, ARCHIVE_SIZE, ARCHIVE_SHA256)
    with zipfile.ZipFile(archive) as bundle:
        members = [item for item in bundle.infolist() if not item.is_dir()]
        if len(members) != 1 or Path(members[0].filename).name != destination.name:
            raise ValueError(f"workload 归档内容不符合固定版本：{archive}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with bundle.open(members[0]) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            validate_file(temporary, WORKLOAD_SIZE, WORKLOAD_SHA256)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def prepare_workload(root: Path, source: Path | None, check_only: bool) -> Path:
    """准备目标文件；默认使用固定 commit 自带的受校验 ZIP。"""
    destination = root / WORKLOAD_RELATIVE_PATH
    if destination.is_file():
        validate_file(destination, WORKLOAD_SIZE, WORKLOAD_SHA256)
        return destination
    if check_only:
        raise FileNotFoundError(
            "未找到 Alibaba 2020 workload。\n\n"
            f"期望位置：\n{destination}\n\n"
            "请按照 docs/数据准备说明.md 完成数据准备。"
        )

    configured_source = source or os.environ.get("SUSTAINCLUSTER_WORKLOAD")
    candidate = (
        Path(configured_source).expanduser().resolve()
        if configured_source
        else destination.with_suffix(".zip")
    )
    if not candidate.is_file():
        raise FileNotFoundError(
            "未找到可用的 workload 来源。\n"
            f"已检查：{candidate}\n"
            "请确认 SustainCluster 位于固定 commit，或通过 --source / "
            "SUSTAINCLUSTER_WORKLOAD 指定团队提供的 .zip 或 .pkl 文件。"
        )
    if candidate.suffix.lower() == ".zip":
        extract_archive(candidate, destination)
    elif candidate.suffix.lower() == ".pkl":
        validate_file(candidate, WORKLOAD_SIZE, WORKLOAD_SHA256)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if candidate != destination:
            shutil.copy2(candidate, destination)
    else:
        raise ValueError("workload 来源只支持 .zip 或 .pkl 文件。")
    validate_file(destination, WORKLOAD_SIZE, WORKLOAD_SHA256)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备并校验 SustainCluster workload")
    parser.add_argument("--sustaincluster-root", type=Path, help="SustainCluster 仓库根目录")
    parser.add_argument("--source", type=Path, help="团队提供的受校验 .zip 或 .pkl")
    parser.add_argument("--check", action="store_true", help="只校验，不写入文件")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        root = resolve_sustaincluster_root(args.sustaincluster_root)
        workload = prepare_workload(root, args.source, args.check)
    except (FileNotFoundError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"workload 已通过校验：{workload}")
    print(f"文件大小：{WORKLOAD_SIZE} bytes")
    print(f"SHA256：{WORKLOAD_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
