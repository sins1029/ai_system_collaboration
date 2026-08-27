from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sustaincluster_imitation.bc_trainer import BehaviorCloningTrainer


def report_lines(result: dict[str, Any]) -> list[str]:
    lines = [
        "# Behavior cloning offline evaluation",
        "",
        "The policy reuses SustainCluster's shared per-task `ActorNet`. Inputs use stable semantic DC ordering and deployable causal H=4 forecast features. Oracle rows are excluded.",
        "",
        f"- Feature dimension: {result['feature_dim']}",
        f"- Action dimension: {result['action_dim']}",
        f"- Parameters: {result['model_parameter_count']}",
        f"- Best checkpoint: `{result['best_checkpoint']}`",
        "",
        "## Three-seed results",
        "",
        "| Seed | Test loss | Accuracy | Top-2 | Defer F1 | Assign accuracy | Migration F1 | Seconds |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in result["runs"]:
        test = run["test"]
        lines.append(
            f"| {run['seed']} | {test['loss']:.6f} | {test['overall_accuracy']:.6f} | "
            f"{test['top2_accuracy']:.6f} | {test['defer_f1']:.6f} | "
            f"{test['assign_accuracy']:.6f} | {test['migration_f1']:.6f} | "
            f"{run['training_seconds']:.2f} |"
        )
    lines.extend(["", "## Aggregate test metrics", ""])
    for name, values in result["aggregate"]["test"].items():
        lines.append(
            f"- {name}: {values['mean']:.6f} +/- {values['std']:.6f}"
        )
    lines.extend(
        [
            "",
            "Confusion matrices and per-DC precision/recall/F1 are stored in the JSON result for every seed.",
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    trainer = BehaviorCloningTrainer(
        args.dataset, args.split, args.config, args.artifacts
    )
    result = trainer.train_all()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "bc_offline_evaluation_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.report_dir / "bc_offline_evaluation_report.md").write_text(
        "\n".join(report_lines(result)) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["aggregate"]["test"], sort_keys=True))


if __name__ == "__main__":
    main()
