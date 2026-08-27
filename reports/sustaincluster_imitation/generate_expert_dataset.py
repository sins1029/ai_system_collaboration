from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from sustaincluster_imitation.dataset_writer import ExpertDatasetWriter
from sustaincluster_imitation.expert_collector import ExpertCollector, ExpertRuntimeConfig
from sustaincluster_imitation.split_dataset import build_episode_split
from sustaincluster_imitation.synthetic_scenarios import build_synthetic_scenarios


def split_hint(seed: int, seeds: list[int]) -> str:
    position = sorted(seeds).index(seed)
    if position < int(len(seeds) * 0.6):
        return "train"
    if position < int(len(seeds) * 0.8):
        return "validation"
    return "test"


def collect_variant(
    *,
    variant: str,
    settings: dict[str, Any],
    repo: Path,
    expert_config: Path,
    project_commit: str,
    sustaincluster_commit: str,
) -> tuple[list[Any], list[Any], list[Any], list[dict[str, Any]]]:
    seeds = [int(value) for value in settings["seeds"]]
    episode_steps = int(settings["episode_steps"])
    collector = ExpertCollector(
        repo,
        project_commit,
        sustaincluster_commit,
        ExpertRuntimeConfig.from_yaml(expert_config),
        baseline_history_window=int(settings["baseline_history_window"]),
    )
    episodes: list[Any] = []
    steps: list[Any] = []
    tasks: list[Any] = []
    summaries: list[dict[str, Any]] = []
    for scenario_name, start in settings["real_windows"].items():
        for seed in seeds:
            episode_id = f"{variant}__{scenario_name}__seed_{seed}"
            collected = collector.collect_real_episode(
                episode_id=episode_id,
                scenario_name=scenario_name,
                seed=seed,
                start_time=pd.Timestamp(start),
                episode_steps=episode_steps,
                dataset_variant=variant,
                split_hint=split_hint(seed, seeds),
            )
            episodes.append(collected.episode)
            steps.extend(collected.steps)
            tasks.extend(collected.tasks)
            summaries.append(
                {"episode_id": episode_id, "variant": variant, **collected.summary}
            )
    for seed in seeds:
        intensity = (
            float(settings["unseen_test_burst_multiplier"])
            if split_hint(seed, seeds) == "test"
            else 1.0
        )
        for scenario in build_synthetic_scenarios(episode_steps, intensity):
            episode_id = f"{variant}__{scenario.name}__seed_{seed}"
            collected = collector.collect_synthetic_episode(
                episode_id=episode_id,
                scenario=scenario,
                seed=seed,
                dataset_variant=variant,
                split_hint=split_hint(seed, seeds),
            )
            episodes.append(collected.episode)
            steps.extend(collected.steps)
            tasks.extend(collected.tasks)
            summaries.append(
                {"episode_id": episode_id, "variant": variant, **collected.summary}
            )
    return episodes, steps, tasks, summaries


def statistics(
    episodes: list[Any], steps: list[Any], tasks: list[Any], summaries: list[dict]
) -> dict[str, Any]:
    decisions = Counter(task.semantic_decision for task in tasks)
    assigned = [task for task in tasks if task.semantic_decision == "assign"]
    assignment_counts = Counter(str(task.destination_dc_id) for task in assigned)
    scenario_samples = Counter(task.scenario_name for task in tasks)
    variant_samples = Counter(task.dataset_variant for task in tasks)
    failures = sum(int(item.get("infeasible_count", 0)) for item in summaries)
    urgent = sum(task.remaining_sla_minutes <= 60.0 for task in tasks)
    gpu = sum(task.gpu_units > 0.0 for task in tasks)
    local = sum(task.destination_dc_id == task.origin_dc_id for task in assigned)
    forecast_fields = (
        "forecast_tasks_mae",
        "forecast_cpu_mae",
        "forecast_gpu_mae",
        "forecast_memory_mae",
    )
    forecast_mae = {
        field: sum(float(row.get(field, 0.0)) for row in summaries)
        / max(1, len(summaries))
        for field in forecast_fields
    }
    labels = Counter(
        "defer" if task.semantic_decision == "defer" else f"dc_{task.destination_dc_id}"
        for task in tasks
    )
    return {
        "episode_count": len(episodes),
        "step_count": len(steps),
        "task_action_count": len(tasks),
        "variant_samples": dict(sorted(variant_samples.items())),
        "scenario_samples": dict(sorted(scenario_samples.items())),
        "defer_count": decisions["defer"],
        "defer_ratio": decisions["defer"] / max(1, len(tasks)),
        "assign_by_dc": dict(sorted(assignment_counts.items())),
        "label_distribution": {
            key: {"count": value, "ratio": value / max(1, len(tasks))}
            for key, value in sorted(labels.items())
        },
        "local_execution_ratio": local / max(1, len(assigned)),
        "migration_ratio": (len(assigned) - local) / max(1, len(assigned)),
        "sla_urgent_ratio": urgent / max(1, len(tasks)),
        "gpu_task_ratio": gpu / max(1, len(tasks)),
        "solver_failure_count": failures,
        "forecast_mae": forecast_mae,
        "labels_below_five_percent": [
            key for key, value in sorted(labels.items()) if value / max(1, len(tasks)) < 0.05
        ],
    }


def write_report(path: Path, stats: dict[str, Any], manifests: dict[str, Any]) -> None:
    primary = stats["deployable_baseline_forecast"]
    lines = [
        "# SustainCluster H=4 expert dataset",
        "",
        "The three forecast variants are stored in separate directories. The deployable baseline forecast dataset is the only primary BC training source; oracle data is excluded from its split manifest.",
        "",
        "## Primary dataset",
        "",
        f"- Episodes: {primary['episode_count']}",
        f"- Steps: {primary['step_count']}",
        f"- Task-action rows: {primary['task_action_count']}",
        f"- Defer ratio: {primary['defer_ratio']:.6f}",
        f"- Local execution ratio: {primary['local_execution_ratio']:.6f}",
        f"- Migration ratio: {primary['migration_ratio']:.6f}",
        f"- GPU task ratio: {primary['gpu_task_ratio']:.6f}",
        f"- SLA-urgent ratio: {primary['sla_urgent_ratio']:.6f}",
        f"- Solver failures: {primary['solver_failure_count']}",
        "",
        "## Labels",
        "",
    ]
    for label, values in primary["label_distribution"].items():
        lines.append(f"- {label}: {values['count']} ({values['ratio']:.6f})")
    lines.extend(
        [
            "",
            "Rows are never duplicated for balancing. Labels below 5% are reported explicitly; class weighting and scenario-level sampling are used by training.",
            "",
            "## Forecast boundary",
            "",
            "`deployable_no_future` contains no future arrivals. `deployable_baseline_forecast` uses only observations at or before the current step. `oracle_upper_bound` reads synthetic/trace future arrivals and remains physically isolated from the deployable training split.",
            "",
            "## Integrity",
            "",
            f"- Dataset schema: {next(iter(manifests.values()))['schema_version']}",
            "- Storage: Parquet with Zstandard compression and SHA-256 file hashes.",
            "- Split unit: complete episode, with seed-exclusive train/validation/test groups.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--expert-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--sustaincluster-commit", required=True)
    args = parser.parse_args()
    with args.dataset_config.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)["dataset"]
    all_stats: dict[str, Any] = {}
    manifests: dict[str, Any] = {}
    primary_episodes: list[Any] | None = None
    for variant in settings["variants"]:
        episodes, steps, tasks, summaries = collect_variant(
            variant=variant,
            settings=settings,
            repo=args.repo.resolve(),
            expert_config=args.expert_config.resolve(),
            project_commit=args.project_commit,
            sustaincluster_commit=args.sustaincluster_commit,
        )
        metadata = {
            "variant": variant,
            "horizon": int(settings["horizon"]),
            "episode_steps": int(settings["episode_steps"]),
            "seeds": settings["seeds"],
            "project_commit": args.project_commit,
            "sustaincluster_commit": args.sustaincluster_commit,
            "dataset_config": str(args.dataset_config.resolve()),
            "expert_config": str(args.expert_config.resolve()),
        }
        manifests[variant] = ExpertDatasetWriter(
            args.output_root.resolve() / variant
        ).write(episodes, steps, tasks, metadata)
        all_stats[variant] = statistics(episodes, steps, tasks, summaries)
        if variant == settings["primary_variant"]:
            primary_episodes = episodes
    if primary_episodes is None:
        raise RuntimeError("primary dataset variant was not collected")
    split = settings["split"]
    build_episode_split(
        [episode.to_dict() for episode in primary_episodes],
        args.output_root.resolve() / "split_manifest.json",
        float(split["train_fraction"]),
        float(split["validation_fraction"]),
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    payload = {"datasets": all_stats, "manifests": manifests}
    (args.report_dir / "expert_dataset_statistics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(
        args.report_dir / "expert_dataset_report.md", all_stats, manifests
    )
    print(json.dumps(all_stats[settings["primary_variant"]], sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()
