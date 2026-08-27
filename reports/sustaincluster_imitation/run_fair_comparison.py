from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import yaml

from sustaincluster_imitation.expert_collector import (
    ExpertCollector,
    ExpertRuntimeConfig,
)
from sustaincluster_imitation.synthetic_scenarios import (
    SyntheticEpisodeSimulator,
    build_synthetic_scenarios,
)
from sustaincluster_mpc import (
    ActionMapping,
    RollingHorizonOptimizer,
    SustainClusterActionAdapter,
)


PROJECT_COMMIT = "d49870b9d9fbee4b480100f4a64d74075b0e1a8b"
SUSTAINCLUSTER_COMMIT = "3f6ea95cb835b89ba50b0ef76d66d14b8037643e"


def assert_fair_configs(h1_path: Path, h4_path: Path) -> None:
    with h1_path.open(encoding="utf-8") as stream:
        h1 = yaml.safe_load(stream)["expert"]
    with h4_path.open(encoding="utf-8") as stream:
        h4 = yaml.safe_load(stream)["expert"]
    assert h1["horizon"] == 1 and h4["horizon"] == 4
    h1 = dict(h1)
    h4 = dict(h4)
    h1.pop("horizon")
    h4.pop("horizon")
    if h1 != h4:
        raise ValueError("H1 and H4 expert configs differ beyond horizon")


def run_real(
    repo: Path,
    config_path: Path,
    horizon: int,
    scenario: str,
    start_time: pd.Timestamp,
    seeds: list[int],
) -> list[dict]:
    collector = ExpertCollector(
        repo,
        PROJECT_COMMIT,
        SUSTAINCLUSTER_COMMIT,
        ExpertRuntimeConfig.from_yaml(config_path),
    )
    records = []
    previous = Path.cwd()
    try:
        for seed in seeds:
            collected = collector.collect_real_episode(
                episode_id=f"fair-{scenario}-h{horizon}-s{seed}",
                scenario_name=scenario,
                seed=seed,
                start_time=start_time,
                episode_steps=96,
                dataset_variant="deployable_no_future",
                split_hint="fair_comparison",
            )
            records.append(
                {
                    "scenario": scenario,
                    "horizon": horizon,
                    "seed": seed,
                    "source": "real_sustaincluster",
                    **collected.summary,
                }
            )
    finally:
        os.chdir(previous)
    return records


def run_synthetic(
    config_path: Path,
    horizon: int,
    scenario_name: str,
    seeds: list[int],
) -> list[dict]:
    config = ExpertRuntimeConfig.from_yaml(config_path)
    scenario = next(
        item
        for item in build_synthetic_scenarios(96)
        if item.name == scenario_name
    )
    records = []
    for seed in seeds:
        simulator = SyntheticEpisodeSimulator(scenario)
        optimizer = RollingHorizonOptimizer()
        dc_ids = tuple(dc.dc_id for dc in scenario.datacenters)
        adapter = SustainClusterActionAdapter(
            ActionMapping(
                tuple((dc_id, index + 1) for index, dc_id in enumerate(dc_ids)),
                0,
                len(dc_ids) + 1,
            )
        )
        solve_times = []
        variable_counts = []
        stage_cost = 0.0
        costs = {
            "electricity": 0.0,
            "carbon": 0.0,
            "transmission": 0.0,
        }
        decisions = 0
        defers = 0
        migrations = 0
        assigned = 0
        infeasible = 0
        order_errors = 0
        for _ in range(96):
            simulator.admit_current_arrivals()
            state = simulator.build_state(horizon)
            result = optimizer.solve(state, config.optimizer_config(), adapter)
            if not result.feasible:
                infeasible += 1
                break
            try:
                adapter.validate_actions(
                    state.current.tasks, result.environment_actions
                )
            except (TypeError, ValueError):
                order_errors += 1
                raise
            solve_times.append(result.solve_seconds)
            variable_counts.append(result.integer_variable_count)
            stage_cost += result.first_step_costs.total
            costs["electricity"] += result.first_step_costs.electricity
            costs["carbon"] += result.first_step_costs.carbon
            costs["transmission"] += result.first_step_costs.transmission
            decisions += len(result.first_step_decisions)
            for task, decision in zip(
                state.current.tasks, result.first_step_decisions
            ):
                defers += int(decision.decision == "defer")
                if decision.decision == "assign":
                    assigned += 1
                    migrations += int(int(decision.dc_id) != task.origin_dc_id)
            simulator.apply(result.first_step_decisions)
        summary = simulator.summary()
        records.append(
            {
                "scenario": scenario_name,
                "horizon": horizon,
                "seed": seed,
                "source": "synthetic_closed_loop",
                **summary,
                "decision_count": decisions,
                "defer_count": defers,
                "defer_ratio": defers / max(1, decisions),
                "migration_count": migrations,
                "migration_ratio": migrations / max(1, assigned),
                "stage_cost": stage_cost,
                **costs,
                "average_solve_seconds": mean(solve_times) if solve_times else 0.0,
                "maximum_solve_seconds": max(solve_times, default=0.0),
                "average_integer_variables": mean(variable_counts)
                if variable_counts
                else 0.0,
                "maximum_integer_variables": max(variable_counts, default=0),
                "infeasible_count": infeasible,
                "resource_overflow_count": 0,
                "illegal_action_count": order_errors,
            }
        )
    return records


def aggregate(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        groups.setdefault((record["scenario"], record["horizon"]), []).append(record)
    fields = (
        "completed_tasks",
        "sla_violations",
        "average_wait_steps",
        "p95_wait_steps",
        "defer_ratio",
        "migration_ratio",
        "electricity",
        "carbon",
        "transmission",
        "stage_cost",
        "average_solve_seconds",
        "maximum_solve_seconds",
        "average_integer_variables",
        "maximum_integer_variables",
        "infeasible_count",
        "resource_overflow_count",
        "illegal_action_count",
    )
    values = []
    for (scenario, horizon), group in sorted(groups.items()):
        values.append(
            {
                "scenario": scenario,
                "horizon": horizon,
                "episodes": len(group),
                **{
                    field: float(np.mean([item.get(field, 0.0) for item in group]))
                    for field in fields
                },
            }
        )
    return values


def quality_gate(aggregates: list[dict]) -> dict:
    h4 = [item for item in aggregates if item["horizon"] == 4]
    no_overflow = all(item["resource_overflow_count"] == 0 for item in h4)
    no_order_errors = all(item["illegal_action_count"] == 0 for item in h4)
    no_infeasible = all(item["infeasible_count"] == 0 for item in h4)
    by_key = {(item["scenario"], item["horizon"]): item for item in aggregates}
    forward_improvements = []
    for scenario in {item["scenario"] for item in aggregates}:
        h1 = by_key.get((scenario, 1))
        h4_value = by_key.get((scenario, 4))
        if h1 and h4_value and (
            h4_value["sla_violations"] < h1["sla_violations"]
            or h4_value["average_wait_steps"] < h1["average_wait_steps"]
            or h4_value["electricity"] < h1["electricity"]
        ):
            forward_improvements.append(scenario)
    return {
        "passed": no_overflow
        and no_order_errors
        and no_infeasible
        and bool(forward_improvements),
        "no_new_resource_overflow": no_overflow,
        "no_action_order_error": no_order_errors,
        "zero_infeasible": no_infeasible,
        "forward_value_scenarios": sorted(forward_improvements),
    }


def write_report(path: Path, result: dict) -> None:
    lines = [
        "# H=1 与 H=4 同口径闭环比较",
        "",
        "两组均使用 `RollingHorizonOptimizer`、同一目标、SLA/terminal/resource/transfer "
        "约束和 ActionAdapter，仅 horizon 不同。每个条目 5 个 seed、每个 episode 96 step。",
        "",
        f"质量门槛：**{'PASS' if result['quality_gate']['passed'] else 'FAIL'}**",
        "",
        "| 场景 | H | 完成 | SLA违约 | 平均/P95等待 | defer | migration | electricity | stage cost | avg/max solve ms | avg/max vars |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["aggregates"]:
        lines.append(
            "| {scenario} | {horizon} | {completed_tasks:.1f} | {sla_violations:.1f} | "
            "{average_wait_steps:.2f}/{p95_wait_steps:.2f} | {defer_ratio:.1%} | "
            "{migration_ratio:.1%} | {electricity:.2f} | {stage_cost:.2f} | "
            "{avg_ms:.3f}/{max_ms:.3f} | {average_integer_variables:.1f}/{maximum_integer_variables:.0f} |".format(
                **item,
                avg_ms=item["average_solve_seconds"] * 1000,
                max_ms=item["maximum_solve_seconds"] * 1000,
            )
        )
    lines.extend(
        [
            "",
            "## 质量门槛",
            "",
            f"- H=4 资源越界为零：{result['quality_gate']['no_new_resource_overflow']}",
            f"- H=4 动作顺序错误为零：{result['quality_gate']['no_action_order_error']}",
            f"- H=4 infeasible 为零：{result['quality_gate']['zero_infeasible']}",
            "- 具有可测前瞻收益的场景："
            + ", ".join(result["quality_gate"]["forward_value_scenarios"]),
            "",
            "所有表格均为实际闭环指标；未使用旧 one-step optimizer 的内部目标值。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--h1-config", type=Path, required=True)
    parser.add_argument("--h4-config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    output_json = args.output_json.resolve()
    output_report = args.output_report.resolve()
    h1 = args.h1_config.resolve()
    h4 = args.h4_config.resolve()
    assert_fair_configs(h1, h4)
    seeds = [101, 202, 303, 404, 505]
    records = []
    windows = {
        "normal_trace": pd.Timestamp("2023-08-01T05:00:00Z"),
        "high_load_trace": pd.Timestamp("2023-01-31T17:00:00Z"),
    }
    for horizon, config in ((1, h1), (4, h4)):
        for scenario, start in windows.items():
            records.extend(
                run_real(args.repo.resolve(), config, horizon, scenario, start, seeds)
            )
        for scenario in (
            "gpu_burst",
            "future_low_price",
            "capacity_release",
        ):
            records.extend(run_synthetic(config, horizon, scenario, seeds))
    aggregates = aggregate(records)
    result = {
        "metadata": {
            "seeds": seeds,
            "episode_steps": 96,
            "h1_config": str(h1),
            "h4_config": str(h4),
            "only_difference": "horizon",
        },
        "quality_gate": quality_gate(aggregates),
        "aggregates": aggregates,
        "episodes": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(output_report, result)
    print(json.dumps(result["quality_gate"], sort_keys=True))
    if not result["quality_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
