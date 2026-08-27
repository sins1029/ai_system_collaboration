from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
import torch

from sustaincluster_imitation.bc_policy import BCPolicy, BCPolicyConfig
from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_imitation.expert_collector import ExpertRuntimeConfig
from sustaincluster_imitation.feature_encoder import (
    SemanticActionSpace,
    SustainClusterFeatureEncoder,
)
from sustaincluster_imitation.forecast_baseline import HistoricalArrivalForecaster
from sustaincluster_mpc import (
    HorizonStateAdapter,
    RollingHorizonOptimizer,
    SustainClusterActionAdapter,
)


EVALUATION_EPISODES = (
    (1201, "normal_trace", "2023-08-01T05:00:00Z"),
    (1202, "high_load_trace", "2023-01-31T17:00:00Z"),
    (1203, "normal_trace", "2023-08-01T05:00:00Z"),
    (1204, "high_load_trace", "2023-01-31T17:00:00Z"),
    (1205, "normal_trace", "2023-08-01T05:00:00Z"),
)


def empty_metrics() -> dict[str, Any]:
    return {
        "total_reward": 0.0,
        "completed_tasks": 0,
        "sla_violations": 0,
        "wait_steps": [],
        "decision_count": 0,
        "defer_count": 0,
        "assign_count": 0,
        "migration_count": 0,
        "electricity": 0.0,
        "carbon": 0.0,
        "transmission": 0.0,
        "resource_overflow_count": 0,
        "illegal_action_count": 0,
        "inference_seconds": [],
    }


def resource_bounds(env: Any) -> bool:
    tolerance = 1e-7
    return all(
        -tolerance <= value <= total + tolerance
        for dc in env.cluster_manager.datacenters.values()
        for value, total in (
            (dc.available_cores, dc.total_cores),
            (dc.available_gpus, dc.total_gpus),
            (dc.available_mem, dc.total_mem_GB),
        )
    )


def update_actual_metrics(metrics: dict[str, Any], info: dict, reward: float) -> None:
    metrics["total_reward"] += reward
    metrics["transmission"] += float(info.get("transmission_cost_total_usd", 0.0))
    for dc_info in info["datacenter_infos"].values():
        common = dc_info["__common__"]
        metrics["electricity"] += float(common["energy_cost_USD"])
        metrics["carbon"] += float(common["carbon_emissions_kg"])
        metrics["completed_tasks"] += int(common["finished_tasks_count"])
        metrics["sla_violations"] += int(common["__sla__"]["violated"])


def finish(metrics: dict[str, Any]) -> dict[str, Any]:
    waits = metrics.pop("wait_steps")
    inference = metrics.pop("inference_seconds")
    metrics.update(
        {
            "average_wait_steps": mean(waits) if waits else 0.0,
            "p95_wait_steps": float(np.percentile(waits, 95)) if waits else 0.0,
            "defer_ratio": metrics["defer_count"]
            / max(1, metrics["decision_count"]),
            "migration_ratio": metrics["migration_count"]
            / max(1, metrics["assign_count"]),
            "average_inference_ms": 1000.0 * mean(inference)
            if inference
            else 0.0,
            "p95_inference_ms": 1000.0 * float(np.percentile(inference, 95))
            if inference
            else 0.0,
            "closed_loop_cost": metrics["electricity"]
            + metrics["carbon"]
            + metrics["transmission"],
        }
    )
    return metrics


def run_manual_episode(
    *,
    repo: Path,
    policy_name: str,
    seed: int,
    scenario: str,
    start: str,
    h1_config: ExpertRuntimeConfig,
    h4_config: ExpertRuntimeConfig,
    bc_policy: BCPolicy,
    random_policy: BCPolicy,
    steps: int = 96,
) -> dict[str, Any]:
    env = build_sustaincluster_env(
        repo, pd.Timestamp(start), steps, allow_defer=True, initial_seed=seed
    )
    env.reset(seed=seed)
    action_adapter = SustainClusterActionAdapter.from_env(env)
    horizon_adapter = HorizonStateAdapter()
    optimizer = RollingHorizonOptimizer()
    actions = SemanticActionSpace((1, 2, 3, 4, 5), True)
    encoder = SustainClusterFeatureEncoder(actions, 4)
    forecaster = HistoricalArrivalForecaster(4, 16, seed)
    metrics = empty_metrics()
    for _ in range(steps):
        action_adapter.assert_matches_env(env)
        horizon = 1 if policy_name == "h1_mpc" else 4
        state = horizon_adapter.build_horizon_state(
            env, horizon, "no_future_arrivals"
        )
        forecast = None
        if horizon == 4:
            forecaster.observe(state.current.tasks)
            forecast = forecaster.predict(state.timestep_minutes)
            state = forecaster.apply(state, forecast)
        started = time.perf_counter()
        if policy_name in ("h1_mpc", "h4_mpc"):
            config = h1_config if policy_name == "h1_mpc" else h4_config
            result = optimizer.solve(state, config.optimizer_config(), action_adapter)
            if not result.feasible:
                raise RuntimeError(f"{policy_name} became infeasible")
            decisions = result.first_step_decisions
            environment_actions = list(result.environment_actions)
        else:
            encoded = encoder.encode(
                state, forecast.uncertainties if forecast is not None else ()
            )
            policy = bc_policy if policy_name == "bc_policy" else random_policy
            decisions = policy.predict(encoded)
            environment_actions = action_adapter.encode_assignments(
                state.current.tasks, decisions
            )
        metrics["inference_seconds"].append(time.perf_counter() - started)
        if len(environment_actions) != len(state.current.tasks):
            metrics["illegal_action_count"] += 1
            raise RuntimeError("policy action count does not match pending tasks")
        action_adapter.validate_actions(state.current.tasks, environment_actions)
        for task, decision in zip(state.current.tasks, decisions):
            metrics["decision_count"] += 1
            metrics["wait_steps"].append(float(task.wait_intervals))
            if decision.decision == "defer":
                metrics["defer_count"] += 1
            else:
                metrics["assign_count"] += 1
                metrics["migration_count"] += int(
                    int(decision.dc_id) != task.origin_dc_id
                )
        before = resource_bounds(env)
        _, reward, terminated, truncated, info = env.step(environment_actions)
        after = resource_bounds(env)
        metrics["resource_overflow_count"] += int(not before or not after)
        update_actual_metrics(metrics, info, float(reward))
        if terminated or truncated:
            break
    return {
        "policy": policy_name,
        "seed": seed,
        "scenario": scenario,
        **finish(metrics),
    }


def run_rule_episode(
    repo: Path, seed: int, scenario: str, start: str, steps: int = 96
) -> dict[str, Any]:
    env = build_sustaincluster_env(
        repo,
        pd.Timestamp(start),
        steps,
        allow_defer=False,
        strategy="local_only",
        initial_seed=seed,
    )
    env.reset(seed=seed)
    metrics = empty_metrics()
    for _ in range(steps):
        before = resource_bounds(env)
        _, reward, terminated, truncated, info = env.step([])
        after = resource_bounds(env)
        metrics["resource_overflow_count"] += int(not before or not after)
        update_actual_metrics(metrics, info, float(reward))
        if terminated or truncated:
            break
    return {
        "policy": "original_rule_local_only",
        "seed": seed,
        "scenario": scenario,
        **finish(metrics),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    numeric = (
        "total_reward",
        "closed_loop_cost",
        "completed_tasks",
        "sla_violations",
        "average_wait_steps",
        "p95_wait_steps",
        "defer_ratio",
        "migration_ratio",
        "electricity",
        "carbon",
        "transmission",
        "resource_overflow_count",
        "illegal_action_count",
        "average_inference_ms",
        "p95_inference_ms",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["policy"]].append(row)
    result = {
        policy: {field: float(np.mean([row[field] for row in rows])) for field in numeric}
        for policy, rows in grouped.items()
    }
    mpc_cost = result["h4_mpc"]["closed_loop_cost"]
    for values in result.values():
        values["gap_vs_h4_mpc"] = (
            values["closed_loop_cost"] - mpc_cost
        ) / max(abs(mpc_cost), 1e-12)
    return result


def report(aggregates: dict[str, dict[str, float]]) -> str:
    lines = [
        "# Behavior cloning real closed-loop evaluation",
        "",
        "Five fresh seeds (1201-1205) are disjoint from train/validation/test. Episodes alternate normal and high-load trace windows. `J = electricity + carbon + transmission`, so a positive gap means higher observed operating burden than H=4 MPC. SLA is reported separately because it has no shared physical unit.",
        "",
        "| Policy | Reward | Completed | SLA violations | Wait avg | Defer | Migration | Electricity | Carbon | Transmission | Inference ms | Gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, values in aggregates.items():
        lines.append(
            f"| {policy} | {values['total_reward']:.4f} | {values['completed_tasks']:.2f} | "
            f"{values['sla_violations']:.2f} | {values['average_wait_steps']:.4f} | "
            f"{values['defer_ratio']:.4f} | {values['migration_ratio']:.4f} | "
            f"{values['electricity']:.4f} | {values['carbon']:.4f} | "
            f"{values['transmission']:.4f} | {values['average_inference_ms']:.4f} | "
            f"{values['gap_vs_h4_mpc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "All reported costs and rewards come from actions actually executed in SustainCluster. Internal optimizer objectives are not compared across policies.",
            "",
            "The repository's RBC path routes tasks inside `DatacenterClusterManager`; `TaskSchedulingEnv` therefore passes an empty `current_tasks` list to `CompositeReward`. Its displayed reward and SLA violations are zeros from that interface and must not be interpreted as zero-cost or violation-free operation. The shared gap uses electricity, carbon and transmission metrics that are populated for every policy.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--h1-config", type=Path, required=True)
    parser.add_argument("--h4-config", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    bc = BCPolicy.load_checkpoint(args.bc_checkpoint)
    torch.manual_seed(90210)
    random_policy = BCPolicy(BCPolicyConfig(233, (1, 2, 3, 4, 5), 256, True))
    h1 = ExpertRuntimeConfig.from_yaml(args.h1_config)
    h4 = ExpertRuntimeConfig.from_yaml(args.h4_config)
    records = []
    for seed, scenario, start in EVALUATION_EPISODES:
        for policy_name in ("h4_mpc", "h1_mpc", "bc_policy", "random_untrained"):
            records.append(
                run_manual_episode(
                    repo=args.repo.resolve(),
                    policy_name=policy_name,
                    seed=seed,
                    scenario=scenario,
                    start=start,
                    h1_config=h1,
                    h4_config=h4,
                    bc_policy=bc,
                    random_policy=random_policy,
                )
            )
        records.append(run_rule_episode(args.repo.resolve(), seed, scenario, start))
    aggregates = aggregate(records)
    payload = {
        "cost_definition": "J=electricity+carbon+transmission; lower is better",
        "evaluation_seeds": [item[0] for item in EVALUATION_EPISODES],
        "records": records,
        "aggregates": aggregates,
    }
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output_report.write_text(report(aggregates), encoding="utf-8")
    print(json.dumps(aggregates, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()
