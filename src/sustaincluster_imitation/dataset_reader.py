from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from sustaincluster_imitation.dataset_schema import DATASET_SCHEMA_VERSION


class ExpertDatasetReader:
    def __init__(self, dataset_dir: Path, verify_hashes: bool = True) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.manifest = json.loads(
            (self.dataset_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest["schema_version"] != DATASET_SCHEMA_VERSION:
            raise ValueError("不支持的专家数据集模式版本")
        if verify_hashes:
            self._verify_hashes()

    def read_episodes(self) -> list[dict[str, Any]]:
        return self._read("episodes.parquet")

    def read_steps(self) -> list[dict[str, Any]]:
        return self._read("steps.parquet")

    def read_tasks(self) -> list[dict[str, Any]]:
        return self._read("tasks.parquet")

    def _read(self, filename: str) -> list[dict[str, Any]]:
        return pq.read_table(self.dataset_dir / filename).to_pylist()

    def _verify_hashes(self) -> None:
        for filename, expected in self.manifest["files"].items():
            path = self.dataset_dir / filename
            digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
            if digest != expected["sha256"]:
                raise ValueError(f"数据集校验和不匹配： {filename}")
