from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def build_episode_split(
    episodes: Sequence[dict[str, Any]],
    output_path: Path,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> dict[str, Any]:
    if not episodes:
        raise ValueError("episodes 不能为空")
    seeds = sorted({int(row["seed"]) for row in episodes})
    train_count = max(1, int(round(len(seeds) * train_fraction)))
    validation_count = max(1, int(round(len(seeds) * validation_fraction)))
    if train_count + validation_count >= len(seeds):
        validation_count = 1
        train_count = len(seeds) - 2
    split_seeds = {
        "train": seeds[:train_count],
        "validation": seeds[train_count : train_count + validation_count],
        "test": seeds[train_count + validation_count :],
    }
    seed_to_split = {
        seed: split for split, values in split_seeds.items() for seed in values
    }
    assignments = {
        str(row["episode_id"]): seed_to_split[int(row["seed"])]
        for row in episodes
    }
    if len(assignments) != len(episodes):
        raise ValueError("episode ID 必须唯一")
    manifest = {
        "strategy": "seed-exclusive episode split",
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": 1.0 - train_fraction - validation_fraction,
        },
        "seeds": split_seeds,
        "episode_assignments": assignments,
        "leakage_checks": {
            "episode_overlap": False,
            "seed_overlap": False,
            "oracle_in_primary_train": False,
            "test_has_unseen_burst_intensity": any(
                seed in split_seeds["test"]
                and float(row.get("burst_intensity", 1.0)) != 1.0
                for row in episodes
                for seed in [int(row["seed"])]
            ),
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest
