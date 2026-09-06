from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from sustaincluster_imitation.bc_regularized_sac import (
    SACUpdateSettings,
    SemanticReplayBuffer,
    initialize_bc_actor_and_teacher,
    load_batch_to_device,
    module_state_hash,
    sac_update,
    select_masked_actions,
)
from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_imitation.expert_collector import ExpertRuntimeConfig
from sustaincluster_imitation.feature_encoder import (
    EncodedTaskBatch,
    SemanticActionSpace,
    SustainClusterFeatureEncoder,
)
from sustaincluster_imitation.forecast_baseline import HistoricalArrivalForecaster
from sustaincluster_imitation.paths import prepare_sustaincluster_imports
from sustaincluster_mpc import (
    AssignmentDecision,
    HorizonStateAdapter,
    RollingHorizonOptimizer,
    SustainClusterActionAdapter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
prepare_sustaincluster_imports()
from rl_components.agent_net import CriticNet  # noqa: E402
from utils.running_stats import RunningStats  # noqa: E402


def encode_current(env, adapter, forecaster, encoder):
    """使用与 BC/Expert 一致的因果特征路径编码当前任务。"""
    state = adapter.build_horizon_state(env, 4, "no_future_arrivals")
    forecaster.observe(state.current.tasks)
    forecast = forecaster.predict(state.timestep_minutes)
    state = forecaster.apply(state, forecast)
    return state, encoder.encode(state, forecast.uncertainties)


def indices_to_decisions(encoded, indices, semantic_actions):
    if len(indices) != len(encoded.task_ids):
        raise ValueError("semantic action count mismatch")
    decisions = []
    for task_id, original_index, index in zip(
        encoded.task_ids, encoded.original_indices, indices
    ):
        dc_id = semantic_actions.dc_for_index(index)
        decisions.append(
            AssignmentDecision(
                task_id,
                original_index,
                "defer" if dc_id is None else "assign",
                dc_id,
            )
        )
    return tuple(decisions)


def choose_actions(actor, encoded, semantic_actions, deterministic, device):
    """严格在当前可行语义动作集合中选择动作。"""
    if not len(encoded.task_ids):
        return [], ()
    features = torch.as_tensor(encoded.features, dtype=torch.float32, device=device)
    mask = torch.as_tensor(
        encoded.feasible_action_mask, dtype=torch.bool, device=device
    )
    indices = select_masked_actions(
        actor, features, mask, deterministic=deterministic
    ).cpu().tolist()
    values = [int(index) for index in indices]
    return values, indices_to_decisions(encoded, values, semantic_actions)


def resource_bounds(env) -> bool:
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


def empty_metrics():
    return {
        "reward": 0.0,
        "completed": 0,
        "sla": 0,
        "electricity": 0.0,
        "carbon": 0.0,
        "transmission": 0.0,
        "decision_count": 0,
        "defer_count": 0,
        "assign_count": 0,
        "migration_count": 0,
        "wait_steps": [],
        "inference_seconds": [],
        "illegal": 0,
        "overflow": 0,
        "nan_inf": 0,
        "expert_equal": 0,
        "expert_compared": 0,
    }


def update_physical_metrics(metrics, info, reward):
    metrics["reward"] += reward
    metrics["transmission"] += float(
        info.get("transmission_cost_total_usd", 0.0)
    )
    for dc_info in info["datacenter_infos"].values():
        common = dc_info["__common__"]
        metrics["electricity"] += float(common["energy_cost_USD"])
        metrics["carbon"] += float(common["carbon_emissions_kg"])
        metrics["completed"] += int(common["finished_tasks_count"])
        metrics["sla"] += int(common["__sla__"]["violated"])


def finish_metrics(metrics):
    waits = metrics.pop("wait_steps")
    inference = metrics.pop("inference_seconds")
    compared = int(metrics.pop("expert_compared"))
    equal = int(metrics.pop("expert_equal"))
    result = {key: float(value) for key, value in metrics.items()}
    result.update(
        {
            "average_wait_steps": mean(waits) if waits else 0.0,
            "defer_ratio": result["defer_count"]
            / max(1.0, result["decision_count"]),
            "migration_ratio": result["migration_count"]
            / max(1.0, result["assign_count"]),
            "inference_ms": 1000.0 * mean(inference) if inference else 0.0,
            "expert_agreement": equal / max(1, compared),
        }
    )
    return result


def evaluate_episode(
    actor,
    episode,
    config,
    expert_config,
    encoder,
    semantic_actions,
    device,
):
    seed = int(episode["seed"])
    env = build_sustaincluster_env(
        None,
        pd.Timestamp(episode["start"]),
        int(config["evaluation_steps"]),
        allow_defer=True,
        initial_seed=seed,
    )
    previous_mode = actor.training
    actor.eval()
    try:
        env.reset(seed=seed)
        action_adapter = SustainClusterActionAdapter.from_env(env)
        state_adapter = HorizonStateAdapter()
        forecaster = HistoricalArrivalForecaster(4, 16, seed)
        optimizer = RollingHorizonOptimizer()
        metrics = empty_metrics()
        steps = 0
        for steps in range(1, int(config["evaluation_steps"]) + 1):
            before = resource_bounds(env)
            state, encoded = encode_current(
                env, state_adapter, forecaster, encoder
            )
            started = time.perf_counter()
            indices, decisions = choose_actions(
                actor, encoded, semantic_actions, True, device
            )
            metrics["inference_seconds"].append(time.perf_counter() - started)
            expert = optimizer.solve(
                state, expert_config.optimizer_config(), action_adapter
            )
            if not expert.feasible:
                raise RuntimeError("H=4 expert infeasible during evaluation")
            expert_indices = [
                semantic_actions.encode_decision(decision)
                for decision in expert.first_step_decisions
            ]
            if len(expert_indices) != len(indices):
                raise RuntimeError("expert/policy task count mismatch")
            metrics["expert_equal"] += sum(
                left == right for left, right in zip(indices, expert_indices)
            )
            metrics["expert_compared"] += len(indices)
            actions = action_adapter.encode_assignments(
                state.current.tasks, decisions
            )
            metrics["illegal"] += int(len(actions) != len(state.current.tasks))
            for task, decision in zip(state.current.tasks, decisions):
                metrics["decision_count"] += 1
                metrics["wait_steps"].append(float(task.wait_intervals))
                if decision.decision == "defer":
                    metrics["defer_count"] += 1
                else:
                    metrics["assign_count"] += 1
                    metrics["migration_count"] += int(
                        decision.dc_id != task.origin_dc_id
                    )
            _, reward, terminated, truncated, info = env.step(actions)
            metrics["overflow"] += int(not before or not resource_bounds(env))
            update_physical_metrics(metrics, info, float(reward))
            metrics["nan_inf"] += int(
                not all(
                    np.isfinite(value)
                    for value in (
                        reward,
                        metrics["electricity"],
                        metrics["carbon"],
                        metrics["transmission"],
                    )
                )
            )
            if terminated or truncated:
                break
        return {
            "evaluation_seed": seed,
            "scenario": str(episode["scenario"]),
            "start": str(episode["start"]),
            "steps": steps,
            **finish_metrics(metrics),
        }
    finally:
        actor.train(previous_mode)
        env.close()


def aggregate_records(records):
    averaged = (
        "reward",
        "completed",
        "sla",
        "electricity",
        "carbon",
        "transmission",
        "average_wait_steps",
        "defer_ratio",
        "migration_ratio",
        "inference_ms",
        "expert_agreement",
    )
    result = {
        key: float(mean(float(row[key]) for row in records)) for key in averaged
    }
    result.update(
        {
            key: float(sum(float(row[key]) for row in records))
            for key in ("illegal", "overflow", "nan_inf")
        }
    )
    return result


def evaluate_actor(actor, config, expert_config, encoder, semantic_actions, device):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        records = [
            evaluate_episode(
                actor,
                episode,
                config,
                expert_config,
                encoder,
                semantic_actions,
                device,
            )
            for episode in config["evaluation_episodes"]
        ]
        return aggregate_records(records), records
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


DIAGNOSTIC_FIELDS = (
    "critic_loss",
    "sac_actor_loss",
    "expert_loss",
    "weighted_expert_loss",
    "total_actor_loss",
    "q_mean",
    "q_std",
    "entropy",
    "alpha",
)


def diagnostics_mean(rows):
    if not rows:
        return {name: 0.0 for name in DIAGNOSTIC_FIELDS}
    return {
        name: float(mean(float(row[name]) for row in rows))
        for name in DIAGNOSTIC_FIELDS
    }


def train_run(
    method,
    method_config,
    seed,
    config,
    checkpoint,
    expert_config,
    behavior_reg_weight,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(str(config["device"]))
    actor, teacher, bc_config, bridge = initialize_bc_actor_and_teacher(
        checkpoint, device
    )
    if (
        bc_config.feature_dim != int(config["feature_dim"])
        or len(bc_config.action_dc_ids) + 1 != int(config["action_dim"])
        or not bridge.exact_key_match
    ):
        raise RuntimeError("BC/SAC Actor schema mismatch")
    teacher_initial_hash = module_state_hash(teacher)
    initial_actor_hash = module_state_hash(actor)
    torch.manual_seed(seed + 100_000)
    critic = CriticNet(
        int(config["feature_dim"]),
        int(config["action_dim"]),
        int(config["hidden_dim"]),
        bool(config["use_layer_norm"]),
    ).to(device)
    target_critic = CriticNet(
        int(config["feature_dim"]),
        int(config["action_dim"]),
        int(config["hidden_dim"]),
        bool(config["use_layer_norm"]),
    ).to(device)
    target_critic.load_state_dict(critic.state_dict())
    critic_initial_hash = module_state_hash(critic)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=float(config["actor_learning_rate"])
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=float(config["critic_learning_rate"])
    )
    replay = SemanticReplayBuffer(
        int(config["replay_capacity"]),
        int(config["max_tasks"]),
        int(config["feature_dim"]),
        int(config["action_dim"]),
        seed + 200_000,
    )
    semantic_actions = SemanticActionSpace(
        tuple(int(value) for value in config["canonical_dc_ids"]), True
    )
    encoder = SustainClusterFeatureEncoder(semantic_actions, 4)
    state_adapter = HorizonStateAdapter()
    env = build_sustaincluster_env(
        None,
        pd.Timestamp(config["training_start"]),
        int(config["training_steps"]) + int(config["episode_steps"]),
        allow_defer=True,
        initial_seed=seed,
    )

    def reset_episode(index):
        episode_seed = seed + index
        env.reset(seed=episode_seed)
        action_adapter = SustainClusterActionAdapter.from_env(env)
        forecaster = HistoricalArrivalForecaster(4, 16, episode_seed)
        state, encoded = encode_current(env, state_adapter, forecaster, encoder)
        return action_adapter, forecaster, state, encoded

    action_adapter, forecaster, state, encoded = reset_episode(0)
    reward_stats = RunningStats()
    evaluation_points = {int(value) for value in config["evaluation_points"]}
    initial_evaluation, initial_records = evaluate_actor(
        actor, config, expert_config, encoder, semantic_actions, device
    )
    curve_rows = [{"step": 0, **initial_evaluation, **diagnostics_mean([])}]
    evaluation_rows = [{"step": 0, **row} for row in initial_records]
    update_rows = []
    interval_updates = []
    training_metrics = empty_metrics()
    update_index = 0
    actor_update_count = 0
    first_actor_update_step = None
    episode_index = 0
    settings = SACUpdateSettings(
        gamma=float(config["gamma"]),
        alpha=float(config["alpha"]),
        tau=float(config["tau"]),
        behavior_reg_weight=float(behavior_reg_weight),
        behavior_reg_type=str(config["behavior_reg_type"]),
    )
    try:
        for step in range(1, int(config["training_steps"]) + 1):
            if len(encoded.task_ids) > int(config["max_tasks"]):
                raise RuntimeError("pending task count exceeds max_tasks")
            before = resource_bounds(env)
            indices, decisions = choose_actions(
                actor, encoded, semantic_actions, False, device
            )
            actions = action_adapter.encode_assignments(
                state.current.tasks, decisions
            )
            training_metrics["illegal"] += int(
                len(actions) != len(state.current.tasks)
            )
            for task, decision in zip(state.current.tasks, decisions):
                training_metrics["decision_count"] += 1
                training_metrics["wait_steps"].append(float(task.wait_intervals))
                if decision.decision == "defer":
                    training_metrics["defer_count"] += 1
                else:
                    training_metrics["assign_count"] += 1
                    training_metrics["migration_count"] += int(
                        decision.dc_id != task.origin_dc_id
                    )
            _, reward, terminated, truncated, info = env.step(actions)
            training_metrics["overflow"] += int(
                not before or not resource_bounds(env)
            )
            update_physical_metrics(training_metrics, info, float(reward))
            training_metrics["nan_inf"] += int(not np.isfinite(reward))
            reward_stats.update(float(reward))
            normalized_reward = float(reward_stats.normalize(float(reward)))
            episode_boundary = step % int(config["episode_steps"]) == 0
            transition_done = bool(terminated or truncated or episode_boundary)
            if transition_done:
                next_encoded = EncodedTaskBatch(
                    np.empty((0, int(config["feature_dim"])), dtype=np.float32),
                    np.empty((0, int(config["action_dim"])), dtype=bool),
                    (),
                    (),
                )
                next_state = None
            else:
                next_state, next_encoded = encode_current(
                    env, state_adapter, forecaster, encoder
                )
            if len(encoded.task_ids):
                replay.add(
                    encoded.features,
                    indices,
                    normalized_reward,
                    next_encoded.features,
                    transition_done,
                    encoded.feasible_action_mask,
                    next_encoded.feasible_action_mask,
                )
            if transition_done:
                episode_index += 1
                if step < int(config["training_steps"]):
                    action_adapter, forecaster, state, encoded = reset_episode(
                        episode_index
                    )
            else:
                state, encoded = next_state, next_encoded
            if (
                step >= int(config["learning_starts"])
                and step % int(config["update_frequency"]) == 0
                and len(replay) >= int(config["batch_size"])
            ):
                update_index += 1
                update_actor = (
                    step > int(method_config["critic_warmup_steps"])
                    and update_index % int(config["actor_update_frequency"]) == 0
                )
                update_target = (
                    update_index % int(config["target_update_frequency"]) == 0
                )
                diagnostics = sac_update(
                    actor=actor,
                    critic=critic,
                    target_critic=target_critic,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    batch=load_batch_to_device(
                        replay.sample(int(config["batch_size"])), device
                    ),
                    settings=settings,
                    teacher=teacher,
                    update_actor=update_actor,
                    update_target=update_target,
                )
                if update_actor:
                    actor_update_count += 1
                    first_actor_update_step = first_actor_update_step or step
                row = {
                    "method": method,
                    "seed": seed,
                    "step": step,
                    "update_index": update_index,
                    **diagnostics,
                }
                update_rows.append(row)
                interval_updates.append(row)
            if step in evaluation_points and step != 0:
                aggregate, records = evaluate_actor(
                    actor, config, expert_config, encoder, semantic_actions, device
                )
                curve_rows.append(
                    {
                        "step": step,
                        **aggregate,
                        **diagnostics_mean(interval_updates),
                    }
                )
                evaluation_rows.extend(
                    {"step": step, **row} for row in records
                )
                interval_updates = []
    finally:
        env.close()
    teacher_final_hash = module_state_hash(teacher)
    if teacher_final_hash != teacher_initial_hash:
        raise RuntimeError("Frozen BC teacher parameters changed")
    if not all(bool(row["all_finite"]) for row in update_rows):
        raise RuntimeError("Non-finite SAC diagnostics")
    final_evaluation = {
        key: float(value)
        for key, value in curve_rows[-1].items()
        if key not in {"step", *DIAGNOSTIC_FIELDS}
    }
    summary = {
        "method": method,
        "seed": seed,
        "training_steps": int(config["training_steps"]),
        "critic_warmup_steps": int(method_config["critic_warmup_steps"]),
        "behavior_reg_weight": float(behavior_reg_weight),
        "initial_actor_hash": initial_actor_hash,
        "final_actor_hash": module_state_hash(actor),
        "teacher_initial_hash": teacher_initial_hash,
        "teacher_final_hash": teacher_final_hash,
        "critic_initial_hash": critic_initial_hash,
        "critic_updates": update_index,
        "actor_updates": actor_update_count,
        "first_actor_update_step": first_actor_update_step,
        "replay_final_size": len(replay),
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "training": finish_metrics(training_metrics),
        "diagnostics": diagnostics_mean(update_rows),
    }
    return {
        "summary": summary,
        "curve_rows": curve_rows,
        "evaluation_rows": evaluation_rows,
        "update_rows": update_rows,
        "actor_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in actor.state_dict().items()
        },
    }


def fixed_bc_run(config, checkpoint, expert_config):
    device = torch.device(str(config["device"]))
    actor, teacher, _, _ = initialize_bc_actor_and_teacher(checkpoint, device)
    actions = SemanticActionSpace(
        tuple(int(value) for value in config["canonical_dc_ids"]), True
    )
    encoder = SustainClusterFeatureEncoder(actions, 4)
    aggregate, records = evaluate_actor(
        actor, config, expert_config, encoder, actions, device
    )
    actor_hash = module_state_hash(actor)
    teacher_hash = module_state_hash(teacher)
    summary = {
        "method": "M0_BC_FIXED",
        "seed": None,
        "training_steps": 0,
        "critic_warmup_steps": 0,
        "behavior_reg_weight": 0.0,
        "initial_actor_hash": actor_hash,
        "final_actor_hash": actor_hash,
        "teacher_initial_hash": teacher_hash,
        "teacher_final_hash": teacher_hash,
        "critic_initial_hash": None,
        "critic_updates": 0,
        "actor_updates": 0,
        "first_actor_update_step": None,
        "replay_final_size": 0,
        "initial_evaluation": aggregate,
        "final_evaluation": aggregate,
        "training": {},
        "diagnostics": diagnostics_mean([]),
    }
    return {
        "summary": summary,
        "curve_rows": [{"step": 0, **aggregate, **diagnostics_mean([])}],
        "evaluation_rows": [{"step": 0, **row} for row in records],
        "update_rows": [],
        "actor_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in actor.state_dict().items()
        },
    }


def calibrate_regularization(update_rows, candidates):
    rows = [
        row
        for row in update_rows
        if row["method"] == "M1_VANILLA"
        and bool(row["actor_updated"])
        and float(row["expert_loss"]) > 1e-12
    ]
    if not rows:
        raise RuntimeError("M1 未产生可用于 lambda 校准的 expert KL")
    sac_magnitude = float(
        np.median([abs(float(row["sac_actor_loss"])) for row in rows])
    )
    expert_magnitude = float(
        np.median([float(row["expert_loss"]) for row in rows])
    )
    positive = [float(value) for value in candidates if float(value) > 0]
    ratios = {
        str(value): value * expert_magnitude / max(sac_magnitude, 1e-12)
        for value in positive
    }
    selected = min(positive, key=lambda value: abs(ratios[str(value)] - 0.10))
    return {
        "sac_actor_loss_median_abs": sac_magnitude,
        "expert_loss_median": expert_magnitude,
        "target_weighted_ratio": 0.10,
        "candidate_weighted_ratios": ratios,
        "selected_weight": selected,
        "samples": len(rows),
    }


def method_aggregates(summaries):
    grouped = defaultdict(list)
    for summary in summaries:
        grouped[summary["method"]].append(summary)
    fields = (
        "reward",
        "completed",
        "sla",
        "electricity",
        "carbon",
        "transmission",
        "average_wait_steps",
        "defer_ratio",
        "migration_ratio",
        "expert_agreement",
        "illegal",
        "overflow",
        "nan_inf",
    )
    return {
        method: {
            **{
                field: float(
                    mean(row["final_evaluation"][field] for row in rows)
                )
                for field in fields
            },
            "runs": len(rows),
        }
        for method, rows in grouped.items()
    }


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select_best_candidate(summaries, aggregates):
    baseline = aggregates["M1_VANILLA"]
    candidates = {
        method: values
        for method, values in aggregates.items()
        if method != "M0_BC_FIXED"
        and values["illegal"] == 0
        and values["overflow"] == 0
        and values["nan_inf"] == 0
    }

    def score(values):
        return (
            (values["reward"] - baseline["reward"])
            / max(abs(baseline["reward"]), 1e-12)
            + (baseline["sla"] - values["sla"])
            / max(abs(baseline["sla"]), 1e-12)
            + values["expert_agreement"]
            - baseline["expert_agreement"]
        )

    best_method = max(candidates, key=lambda name: score(candidates[name]))
    rows = [row for row in summaries if row["method"] == best_method]
    aggregate = candidates[best_method]

    def distance(row):
        values = row["final_evaluation"]
        return sum(
            abs(values[key] - aggregate[key]) / max(abs(aggregate[key]), 1e-12)
            for key in ("reward", "sla", "expert_agreement")
        )

    return best_method, min(rows, key=distance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/rl/bc_reg_sac_v0_1.yaml",
    )
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["bc_regularized_sac"]
    torch.set_num_threads(int(config["torch_threads"]))
    checkpoint = (PROJECT_ROOT / config["bc_checkpoint"]).resolve()
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    checkpoint_dir = (PROJECT_ROOT / config["checkpoint_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    expert_config = ExpertRuntimeConfig.from_yaml(
        (PROJECT_ROOT / config["expert_config"]).resolve()
    )
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest().upper()
    runs = [fixed_bc_run(config, checkpoint, expert_config)]
    methods = config["methods"]
    for seed in config["training_seeds"]:
        print(f"starting M1_VANILLA seed={seed}", flush=True)
        runs.append(
            train_run(
                "M1_VANILLA",
                methods["M1_VANILLA"],
                int(seed),
                config,
                checkpoint,
                expert_config,
                0.0,
            )
        )
        print(f"finished M1_VANILLA seed={seed}", flush=True)
    calibration = calibrate_regularization(
        [row for run in runs for row in run["update_rows"]],
        config["behavior_reg_weight_candidates"],
    )
    selected_weight = float(calibration["selected_weight"])
    (output_dir / "loss_calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"selected behavior_reg_weight={selected_weight}", flush=True)
    for method in ("M2_WARMUP", "M3_REGULARIZED", "M4_COMBINED"):
        configured = methods[method]["behavior_reg_weight"]
        weight = selected_weight if configured == "auto" else float(configured)
        for seed in config["training_seeds"]:
            print(f"starting {method} seed={seed}", flush=True)
            runs.append(
                train_run(
                    method,
                    methods[method],
                    int(seed),
                    config,
                    checkpoint,
                    expert_config,
                    weight,
                )
            )
            print(f"finished {method} seed={seed}", flush=True)
    summaries = [run["summary"] for run in runs]
    curves = [
        {"method": run["summary"]["method"], "seed": run["summary"]["seed"], **row}
        for run in runs
        for row in run["curve_rows"]
    ]
    evaluations = [
        {"method": run["summary"]["method"], "seed": run["summary"]["seed"], **row}
        for run in runs
        for row in run["evaluation_rows"]
    ]
    updates = [row for run in runs for row in run["update_rows"]]
    aggregates = method_aggregates(summaries)
    paired_critics = {
        str(seed): {
            summary["method"]: summary["critic_initial_hash"]
            for summary in summaries
            if summary["seed"] == seed
        }
        for seed in config["training_seeds"]
    }
    if not all(len(set(values.values())) == 1 for values in paired_critics.values()):
        raise RuntimeError("paired methods did not use identical critic initialization")
    best_method, best_summary = select_best_candidate(summaries, aggregates)
    best_run = next(
        run
        for run in runs
        if run["summary"]["method"] == best_method
        and run["summary"]["seed"] == best_summary["seed"]
    )
    checkpoint_path = checkpoint_dir / "best_actor.pt"
    torch.save(
        {
            "format_version": 1,
            "algorithm": "bc_regularized_sac_v0_1",
            "method": best_method,
            "seed": best_summary["seed"],
            "step": int(config["training_steps"]),
            "actor_state_dict": best_run["actor_state_dict"],
            "source_bc_checkpoint": str(Path(config["bc_checkpoint"])),
            "config": config,
            "evaluation": best_summary["final_evaluation"],
            "teacher_hash": best_summary["teacher_final_hash"],
        },
        checkpoint_path,
    )
    manifest = {
        "path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest().upper(),
        "method": best_method,
        "seed": best_summary["seed"],
        "step": int(config["training_steps"]),
        "evaluation": best_summary["final_evaluation"],
        "config_fingerprint": fingerprint,
    }
    (checkpoint_dir / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    result = {
        "git_baseline": "90eb972f78742603b522fabc3e1cf65391434418",
        "config": config,
        "config_fingerprint": fingerprint,
        "loss_calibration": calibration,
        "paired_critic_hashes": paired_critics,
        "summaries": summaries,
        "aggregates": aggregates,
        "best_checkpoint": manifest,
    }
    (output_dir / "experiment_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(output_dir / "training_curve.csv", curves)
    write_csv(output_dir / "evaluation_summary.csv", evaluations)
    write_csv(output_dir / "update_diagnostics.csv", updates)
    seed_rows = [
        {
            "method": summary["method"],
            "seed": summary["seed"],
            "critic_warmup_steps": summary["critic_warmup_steps"],
            "behavior_reg_weight": summary["behavior_reg_weight"],
            "critic_updates": summary["critic_updates"],
            "actor_updates": summary["actor_updates"],
            "first_actor_update_step": summary["first_actor_update_step"],
            **summary["final_evaluation"],
        }
        for summary in summaries
    ]
    write_csv(output_dir / "seed_summary.csv", seed_rows)
    print(json.dumps({"aggregates": aggregates, "best": manifest}, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()
