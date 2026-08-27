from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import pandas as pd
import yaml

from sustaincluster_mpc import (
    ObjectiveWeights,
    OneStepOptimizer,
    OptimizationConfig,
    SustainClusterActionAdapter,
    SustainClusterStateAdapter,
)


def build_env(repo: Path, allow_defer: bool):
    from envs.task_scheduling_env import TaskSchedulingEnv
    from rewards.predefined.composite_reward import CompositeReward
    from simulation.cluster_manager import DatacenterClusterManager

    os.chdir(repo)
    with (repo / "configs/env/sim_config.yaml").open(encoding="utf-8") as stream:
        sim_config = copy.deepcopy(yaml.safe_load(stream)["simulation"])
    with (repo / "configs/env/datacenters.yaml").open(encoding="utf-8") as stream:
        dc_config = yaml.safe_load(stream)["datacenters"]
    with (repo / "configs/env/reward_config.yaml").open(encoding="utf-8") as stream:
        reward_config = yaml.safe_load(stream)["reward"]

    sim_config["single_action_mode"] = False
    sim_config["disable_defer_action"] = not allow_defer
    sim_config["use_tensorboard"] = False
    start = pd.Timestamp(
        year=sim_config["year"],
        month=sim_config["month"],
        day=sim_config["init_day"],
        hour=sim_config["init_hour"],
        tz="UTC",
    )
    cluster = DatacenterClusterManager(
        config_list=copy.deepcopy(dc_config),
        simulation_year=sim_config["year"],
        init_day=int(sim_config["month"] * 30.5),
        init_hour=sim_config["init_hour"],
        strategy="manual_rl",
        tasks_file_path=sim_config["workload_path"],
        shuffle_datacenter_order=sim_config["shuffle_datacenters"],
        cloud_provider=sim_config["cloud_provider"],
        logger=None,
    )
    reward = CompositeReward(
        components=reward_config["components"],
        normalize=reward_config.get("normalize", False),
        freeze_stats_after_steps=reward_config.get("freeze_stats_after_steps"),
    )
    env = TaskSchedulingEnv(
        cluster_manager=cluster,
        start_time=start,
        end_time=start + pd.Timedelta(days=sim_config["duration_days"]),
        reward_fn=reward,
        writer=None,
        sim_config=sim_config,
        initial_seed_for_resets=123,
    )
    return env


def env_fingerprint(env) -> tuple:
    tasks = tuple(
        (
            task.job_name,
            task.origin_dc_id,
            task.dest_dc_id,
            task.temporarily_deferred,
            task.wait_intervals,
        )
        for task in env.current_tasks
    )
    datacenters = tuple(
        (
            name,
            dc.dc_id,
            dc.available_cores,
            dc.available_gpus,
            dc.available_mem,
            len(dc.running_tasks),
            len(dc.pending_tasks),
        )
        for name, dc in env.cluster_manager.datacenters.items()
    )
    transit = tuple(
        (str(arrival), task.job_name, destination)
        for arrival, task, destination in env.in_transit_tasks
    )
    return tasks, datacenters, transit


def has_resource_overflow(env) -> bool:
    tolerance = 1e-7
    for dc in env.cluster_manager.datacenters.values():
        values = (
            (dc.available_cores, dc.total_cores),
            (dc.available_gpus, dc.total_gpus),
            (dc.available_mem, dc.total_mem_GB),
        )
        if any(value < -tolerance or value > total + tolerance for value, total in values):
            return True
    return False


def run(repo: Path, steps: int, seed: int, allow_defer: bool) -> dict:
    env = build_env(repo, allow_defer)
    env.reset(seed=seed)
    action_adapter = SustainClusterActionAdapter.from_env(env)
    optimizer = OneStepOptimizer(
        OptimizationConfig(
            allow_defer=allow_defer,
            weights=ObjectiveWeights(
                electricity=1.0,
                carbon=1.0,
                transmission=1.0,
                defer=100.0,
                sla_risk=10.0,
            ),
        )
    )

    records: list[dict] = []
    action_validation_failures = 0
    order_mismatches = 0
    adapter_mutations = 0
    infeasible_count = 0
    total_assignments = 0
    total_defers = 0
    assignment_by_dc = {
        int(dc.dc_id): 0 for dc in env.cluster_manager.datacenters.values()
    }

    for step_index in range(1, steps + 1):
        action_adapter.assert_matches_env(env)
        before_adapter = env_fingerprint(env)
        state = SustainClusterStateAdapter(env).build_scheduler_state()
        after_adapter = env_fingerprint(env)
        adapter_mutated = before_adapter != after_adapter
        adapter_mutations += int(adapter_mutated)

        result = optimizer.solve(state, action_adapter)
        if not result.feasible:
            infeasible_count += 1
            records.append(
                {
                    "step": step_index,
                    "pending_tasks": len(state.tasks),
                    "status": result.status,
                    "solve_seconds": result.solve_seconds,
                    "message": result.message,
                }
            )
            break

        actions = list(result.environment_actions)
        try:
            action_adapter.validate_actions(state.tasks, actions)
        except (TypeError, ValueError):
            action_validation_failures += 1
            raise
        order_mismatch = any(
            assignment.original_index != task.original_index
            or assignment.task_id != task.task_id
            for task, assignment in zip(state.tasks, result.assignments)
        )
        order_mismatches += int(order_mismatch)

        assigned = [item for item in result.assignments if item.decision == "assign"]
        deferred = [item for item in result.assignments if item.decision == "defer"]
        per_dc = {dc_id: 0 for dc_id in assignment_by_dc}
        for item in assigned:
            per_dc[int(item.dc_id)] += 1
            assignment_by_dc[int(item.dc_id)] += 1
        total_assignments += len(assigned)
        total_defers += len(deferred)

        _, reward, terminated, truncated, info = env.step(actions)
        overflow = has_resource_overflow(env)
        records.append(
            {
                "step": step_index,
                "pending_tasks": len(state.tasks),
                "action_count": len(actions),
                "assign_count": len(assigned),
                "defer_count": len(deferred),
                "assignments_by_dc": per_dc,
                "status": result.status,
                "solve_seconds": result.solve_seconds,
                "objective_value": result.objective_value,
                "costs": asdict(result.costs),
                "reward": float(reward),
                "scheduled_tasks": int(info["scheduled_tasks_this_step"]),
                "resource_overflow": overflow,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "order_mismatch": order_mismatch,
                "adapter_mutated_env": adapter_mutated,
            }
        )
        if terminated or truncated:
            break

    solve_times = [record["solve_seconds"] for record in records]
    decision_count = total_assignments + total_defers
    summary = {
        "requested_steps": steps,
        "completed_steps": len(records),
        "allow_defer": allow_defer,
        "action_mapping": action_adapter.mapping.dc_id_to_action_items,
        "defer_action": action_adapter.mapping.defer_action,
        "average_solve_seconds": mean(solve_times) if solve_times else 0.0,
        "minimum_solve_seconds": min(solve_times) if solve_times else 0.0,
        "maximum_solve_seconds": max(solve_times) if solve_times else 0.0,
        "infeasible_count": infeasible_count,
        "defer_ratio": total_defers / decision_count if decision_count else 0.0,
        "action_validation_failures": action_validation_failures,
        "assignment_by_dc": assignment_by_dc,
        "order_mismatches": order_mismatches,
        "adapter_mutations": adapter_mutations,
        "resource_overflow_steps": sum(
            int(record.get("resource_overflow", False)) for record in records
        ),
    }
    return {"summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--allow-defer", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    result = run(args.repo.resolve(), args.steps, args.seed, args.allow_defer)
    for record in result["records"]:
        print(json.dumps({"event": "closed_loop_step", **record}, sort_keys=True))
    print(
        json.dumps(
            {"event": "closed_loop_summary", **result["summary"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
