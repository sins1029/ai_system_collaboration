from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from generate_expert_dataset import write_report

from sustaincluster_imitation.dataset_reader import ExpertDatasetReader
from sustaincluster_imitation.dataset_writer import ExpertDatasetWriter
from sustaincluster_imitation.expert_collector import ExpertCollector, ExpertRuntimeConfig
from sustaincluster_imitation.split_dataset import build_episode_split
from sustaincluster_imitation.synthetic_scenarios import build_synthetic_scenarios


VARIANT = "deployable_baseline_forecast"


def value(row: Any, key: str) -> Any:
    return row[key] if isinstance(row, dict) else getattr(row, key)


def primary_statistics(
    episodes: list[Any], steps: list[Any], tasks: list[Any], previous: dict[str, Any]
) -> dict[str, Any]:
    labels = Counter(
        "defer"
        if value(task, "semantic_decision") == "defer"
        else f"dc_{value(task, 'destination_dc_id')}"
        for task in tasks
    )
    assigned = [task for task in tasks if value(task, "semantic_decision") == "assign"]
    local = sum(
        value(task, "destination_dc_id") == value(task, "origin_dc_id")
        for task in assigned
    )
    scenario_samples = Counter(value(task, "scenario_name") for task in tasks)
    total = len(tasks)
    return {
        **previous,
        "episode_count": len(episodes),
        "step_count": len(steps),
        "task_action_count": total,
        "variant_samples": {VARIANT: total},
        "scenario_samples": dict(sorted(scenario_samples.items())),
        "defer_count": labels["defer"],
        "defer_ratio": labels["defer"] / max(1, total),
        "assign_by_dc": {
            key.removeprefix("dc_"): count
            for key, count in sorted(labels.items())
            if key.startswith("dc_")
        },
        "label_distribution": {
            key: {"count": count, "ratio": count / max(1, total)}
            for key, count in sorted(labels.items())
        },
        "local_execution_ratio": local / max(1, len(assigned)),
        "migration_ratio": (len(assigned) - local) / max(1, len(assigned)),
        "sla_urgent_ratio": sum(
            value(task, "remaining_sla_minutes") <= 60.0 for task in tasks
        )
        / max(1, total),
        "gpu_task_ratio": sum(value(task, "gpu_units") > 0.0 for task in tasks)
        / max(1, total),
        "solver_failure_count": 0,
        "labels_below_five_percent": [
            key for key, count in sorted(labels.items()) if count / max(1, total) < 0.05
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expert-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--sustaincluster-commit", required=True)
    args = parser.parse_args()
    dataset_dir = args.dataset_root / VARIANT
    reader = ExpertDatasetReader(dataset_dir)
    episodes: list[Any] = reader.read_episodes()
    steps: list[Any] = reader.read_steps()
    tasks: list[Any] = reader.read_tasks()
    collector = ExpertCollector(
        args.repo,
        args.project_commit,
        args.sustaincluster_commit,
        ExpertRuntimeConfig.from_yaml(args.expert_config),
    )
    seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
    for seed in seeds:
        intensity = 1.5 if seed in (909, 1010) else 1.0
        scenario = next(
            item
            for item in build_synthetic_scenarios(96, intensity)
            if item.name == "class_balance_heterogeneity"
        )
        split = "train" if seed <= 606 else "validation" if seed <= 808 else "test"
        episode_id = f"{VARIANT}__{scenario.name}__seed_{seed}"
        result = collector.collect_synthetic_episode(
            episode_id=episode_id,
            scenario=scenario,
            seed=seed,
            dataset_variant=VARIANT,
            split_hint=split,
        )
        episodes.append(result.episode)
        steps.extend(result.steps)
        tasks.extend(result.tasks)
    metadata = {
        **reader.manifest["metadata"],
        "scenario_level_augmentation": "class_balance_heterogeneity",
        "augmentation_policy": "new expert trajectories; no row duplication",
    }
    manifest = ExpertDatasetWriter(dataset_dir).write(
        episodes, steps, tasks, metadata
    )
    build_episode_split(
        [row if isinstance(row, dict) else row.to_dict() for row in episodes],
        args.dataset_root / "split_manifest.json",
        0.6,
        0.2,
    )
    stats_path = args.report_dir / "expert_dataset_statistics.json"
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    payload["datasets"][VARIANT] = primary_statistics(
        episodes, steps, tasks, payload["datasets"][VARIANT]
    )
    payload["manifests"][VARIANT] = manifest
    stats_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(
        args.report_dir / "expert_dataset_report.md",
        payload["datasets"],
        payload["manifests"],
    )
    print(json.dumps(payload["datasets"][VARIANT], sort_keys=True))


if __name__ == "__main__":
    main()
