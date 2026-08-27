from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from sustaincluster_imitation.dataset_schema import DATASET_SCHEMA_VERSION


class ExpertDatasetWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def write(
        self,
        episodes: Iterable[Any],
        steps: Iterable[Any],
        tasks: Iterable[Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tables = {
            "episodes": self._rows(episodes),
            "steps": self._rows(steps),
            "tasks": self._rows(tasks),
        }
        files: dict[str, dict[str, Any]] = {}
        for name, rows in tables.items():
            path = self.output_dir / f"{name}.parquet"
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, path, compression="zstd")
            files[path.name] = {
                "rows": table.num_rows,
                "bytes": path.stat().st_size,
                "sha256": self._sha256(path),
            }
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
            "files": files,
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return manifest

    @staticmethod
    def _rows(values: Iterable[Any]) -> list[dict[str, Any]]:
        rows = []
        for value in values:
            if is_dataclass(value):
                rows.append(asdict(value))
            elif isinstance(value, dict):
                rows.append(dict(value))
            else:
                raise TypeError(f"不支持的数据集行类型 {type(value)!r}")
        return rows

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest().upper()
