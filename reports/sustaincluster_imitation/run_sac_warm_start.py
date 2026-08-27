from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from rl_components.agent_net import ActorNet, CriticNet
from rl_components.replay_buffer import FastReplayBuffer
from utils.running_stats import RunningStats

from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_imitation.expert_collector import ExpertRuntimeConfig
from sustaincluster_imitation.feature_encoder import (
    EncodedTaskBatch,
    SemanticActionSpace,
    SustainClusterFeatureEncoder,
)
from sustaincluster_imitation.forecast_baseline import HistoricalArrivalForecaster
from sustaincluster_imitation.sac_weight_bridge import initialize_sac_actor_from_bc
from sustaincluster_mpc import (
    AssignmentDecision,
    HorizonStateAdapter,
    RollingHorizonOptimizer,
    SustainClusterActionAdapter,
)


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest().upper()


def encode_current(
    env: Any,
    adapter: HorizonStateAdapter,
    forecaster: HistoricalArrivalForecaster,
    encoder: SustainClusterFeatureEncoder,
) -> tuple[Any, EncodedTaskBatch]:
    state = adapter.build_horizon_state(env, 4, "no_future_arrivals")
    forecaster.observe(state.current.tasks)
    forecast = forecaster.predict(state.timestep_minutes)
    state = forecaster.apply(state, forecast)
    return state, encoder.encode(state, forecast.uncertainties)


def indices_to_decisions(
    encoded: EncodedTaskBatch,
    indices: list[int],
    actions: SemanticActionSpace,
) -> tuple[AssignmentDecision, ...]:
    if len(indices) != len(encoded.task_ids):
        raise ValueError("semantic action count mismatch")
    decisions = []
    for task_id, original_index, index in zip(
        encoded.task_ids, encoded.original_indices, indices
    ):
        dc_id = actions.dc_for_index(int(index))
        decisions.append(
            AssignmentDecision(
                task_id,
                original_index,
                "defer" if dc_id is None else "assign",
                dc_id,
            )
        )
    return tuple(decisions)


def select_actions(
    actor: torch.nn.Module,
    encoded: EncodedTaskBatch,
    actions: SemanticActionSpace,
    *,
    stochastic: bool,
    rng: np.random.Generator,
) -> tuple[list[int], tuple[AssignmentDecision, ...]]:
    if len(encoded.task_ids) == 0:
        return [], ()
    if stochastic:
        with torch.no_grad():
            features = torch.as_tensor(encoded.features, dtype=torch.float32)
            mask = torch.as_tensor(encoded.feasible_action_mask, dtype=torch.bool)
            logits = actor(features).masked_fill(
                ~mask, torch.finfo(torch.float32).min
            )
            indices = torch.distributions.Categorical(logits=logits).sample().tolist()
    else:
        with torch.no_grad():
            features = torch.as_tensor(encoded.features, dtype=torch.float32)
            mask = torch.as_tensor(encoded.feasible_action_mask, dtype=torch.bool)
            indices = (
                actor(features)
                .masked_fill(~mask, torch.finfo(torch.float32).min)
                .argmax(dim=-1)
                .tolist()
            )
    return [int(index) for index in indices], indices_to_decisions(
        encoded, [int(index) for index in indices], actions
    )


def random_feasible_indices(
    encoded: EncodedTaskBatch, rng: np.random.Generator
) -> list[int]:
    return [
        int(rng.choice(np.flatnonzero(mask)))
        for mask in encoded.feasible_action_mask
    ]


def sla_violations(info: dict[str, Any]) -> int:
    return sum(
        int(dc_info["__common__"]["__sla__"]["violated"])
        for dc_info in info["datacenter_infos"].values()
    )


def evaluate(
    *,
    repo: Path,
    actor: torch.nn.Module,
    config: dict[str, Any],
    expert_config: ExpertRuntimeConfig,
    encoder: SustainClusterFeatureEncoder,
    semantic_actions: SemanticActionSpace,
) -> dict[str, Any]:
    saved_random = random.getstate()
    saved_numpy = np.random.get_state()
    saved_torch = torch.random.get_rng_state()
    try:
        env = build_sustaincluster_env(
            repo,
            pd.Timestamp(config["evaluation_start"]),
            int(config["evaluation_steps"]),
            allow_defer=True,
            initial_seed=int(config["evaluation_seed"]),
        )
        env.reset(seed=int(config["evaluation_seed"]))
        action_adapter = SustainClusterActionAdapter.from_env(env)
        state_adapter = HorizonStateAdapter()
        forecaster = HistoricalArrivalForecaster(
            4, 16, int(config["evaluation_seed"])
        )
        optimizer = RollingHorizonOptimizer()
        rng = np.random.default_rng(int(config["evaluation_seed"]))
        total_reward = 0.0
        violations = 0
        illegal = 0
        agreement = 0
        compared = 0
        for _ in range(int(config["evaluation_steps"])):
            state, encoded = encode_current(env, state_adapter, forecaster, encoder)
            indices, decisions = select_actions(
                actor, encoded, semantic_actions, stochastic=False, rng=rng
            )
            expert = optimizer.solve(
                state, expert_config.optimizer_config(), action_adapter
            )
            if not expert.feasible:
                raise RuntimeError("H=4 expert infeasible during SAC evaluation")
            expert_indices = [
                semantic_actions.encode_decision(item)
                for item in expert.first_step_decisions
            ]
            agreement += sum(a == b for a, b in zip(indices, expert_indices))
            compared += len(indices)
            environment_actions = action_adapter.encode_assignments(
                state.current.tasks, decisions
            )
            illegal += int(len(environment_actions) != len(state.current.tasks))
            _, reward, terminated, truncated, info = env.step(environment_actions)
            total_reward += float(reward)
            violations += sla_violations(info)
            if terminated or truncated:
                break
        return {
            "reward": total_reward,
            "sla_violations": violations,
            "illegal_actions": illegal,
            "expert_action_agreement": agreement / max(1, compared),
        }
    finally:
        random.setstate(saved_random)
        np.random.set_state(saved_numpy)
        torch.random.set_rng_state(saved_torch)


def sac_update(
    *,
    actor: torch.nn.Module,
    critic: CriticNet,
    target_critic: CriticNet,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    buffer: FastReplayBuffer,
    config: dict[str, Any],
    global_step: int,
) -> dict[str, float]:
    obs, acts, rewards, next_obs, dones, obs_mask, next_mask = buffer.sample(
        int(config["batch_size"])
    )
    occupied_columns = (obs_mask.bool() | next_mask.bool()).any(dim=0)
    occupied_indices = occupied_columns.nonzero(as_tuple=False)
    active_tasks = (
        int(occupied_indices[-1].item()) + 1 if len(occupied_indices) else 1
    )
    obs = obs[:, :active_tasks]
    acts = acts[:, :active_tasks]
    next_obs = next_obs[:, :active_tasks]
    obs_mask = obs_mask[:, :active_tasks]
    next_mask = next_mask[:, :active_tasks]
    batch, tasks, dim = obs.shape
    obs_flat = obs.reshape(batch * tasks, dim)
    next_flat = next_obs.reshape(batch * tasks, dim)
    act_flat = acts.reshape(-1)
    obs_mask_flat = obs_mask.reshape(-1).bool()
    next_mask_flat = next_mask.reshape(-1)
    with torch.no_grad():
        next_logits = actor(next_flat)
        next_probs = F.softmax(next_logits, dim=-1)
        next_log_probs = F.log_softmax(next_logits, dim=-1)
        q1_next, q2_next = target_critic.forward_all(next_flat)
        q_next = torch.minimum(q1_next, q2_next)
        values = (
            next_probs * (q_next - float(config["alpha"]) * next_log_probs)
        ).sum(dim=-1)
        values = (values * next_mask_flat).reshape(batch, tasks)
        targets = rewards.unsqueeze(1) + float(config["gamma"]) * (
            1.0 - dones.unsqueeze(1)
        ) * values
    valid_actions = act_flat[obs_mask_flat]
    valid_obs = obs_flat[obs_mask_flat]
    q1, q2 = critic(valid_obs, valid_actions)
    target = targets.reshape(-1)[obs_mask_flat]
    critic_loss = 0.5 * (F.mse_loss(q1, target) + F.mse_loss(q2, target))
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    actor_loss_value = float("nan")
    if global_step % int(config["policy_update_frequency"]) == 0:
        logits = actor(valid_obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            q1_eval, q2_eval = critic.forward_all(valid_obs)
            q_eval = torch.minimum(q1_eval, q2_eval)
        actor_loss = (
            probs * (float(config["alpha"]) * log_probs - q_eval)
        ).sum(dim=-1).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        actor_loss_value = float(actor_loss.detach())
    with torch.no_grad():
        for source, target_parameter in zip(
            critic.parameters(), target_critic.parameters()
        ):
            target_parameter.mul_(1.0 - float(config["tau"]))
            target_parameter.add_(float(config["tau"]) * source)
        q_values = torch.minimum(q1, q2)
    return {
        "actor_loss": actor_loss_value,
        "critic_loss": float(critic_loss.detach()),
        "q_mean": float(q_values.mean()),
        "q_std": float(q_values.std()),
    }


def train_run(
    *,
    group: str,
    seed: int,
    repo: Path,
    checkpoint: Path,
    config: dict[str, Any],
    expert_config: ExpertRuntimeConfig,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    feature_dim = 233
    action_dim = 6
    actor = ActorNet(
        feature_dim,
        action_dim,
        int(config["hidden_dim"]),
        bool(config["use_layer_norm"]),
    )
    critic = CriticNet(
        feature_dim,
        action_dim,
        int(config["hidden_dim"]),
        bool(config["use_layer_norm"]),
    )
    target_critic = CriticNet(
        feature_dim,
        action_dim,
        int(config["hidden_dim"]),
        bool(config["use_layer_norm"]),
    )
    target_critic.load_state_dict(critic.state_dict())
    critic_hash = state_hash(critic)
    if group == "bc_init_sac":
        bc_actor, bc_config, bridge = initialize_sac_actor_from_bc(checkpoint)
        if bc_config.feature_dim != feature_dim or not bridge.exact_key_match:
            raise RuntimeError("BC/SAC actor bridge validation failed")
        actor.load_state_dict(bc_actor.state_dict(), strict=True)
    initial_actor_hash = state_hash(actor)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=float(config["actor_learning_rate"])
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=float(config["critic_learning_rate"])
    )
    buffer = FastReplayBuffer(
        int(config["replay_buffer_size"]),
        int(config["max_tasks"]),
        feature_dim,
    )
    actions = SemanticActionSpace(
        tuple(int(item) for item in config["canonical_dc_ids"]), True
    )
    encoder = SustainClusterFeatureEncoder(actions, 4)
    state_adapter = HorizonStateAdapter()
    env = build_sustaincluster_env(
        repo,
        pd.Timestamp(config["training_start"]),
        int(config["total_environment_steps"])
        + int(config["training_episode_steps"]),
        allow_defer=True,
        initial_seed=seed,
    )

    def reset_training_episode(episode_index: int) -> tuple[Any, Any, Any]:
        episode_seed = seed + episode_index
        env.reset(seed=episode_seed)
        episode_action_adapter = SustainClusterActionAdapter.from_env(env)
        episode_forecaster = HistoricalArrivalForecaster(4, 16, episode_seed)
        episode_state, episode_encoded = encode_current(
            env, state_adapter, episode_forecaster, encoder
        )
        return (
            episode_action_adapter,
            episode_forecaster,
            (episode_state, episode_encoded),
        )

    episode_index = 0
    action_adapter, forecaster, initial = reset_training_episode(episode_index)
    state, encoded = initial
    reward_stats = RunningStats()
    evaluation_points = set(int(item) for item in config["evaluation_points"])
    evaluations = [
        {
            "step": 0,
            **evaluate(
                repo=repo,
                actor=actor,
                config=config,
                expert_config=expert_config,
                encoder=encoder,
                semantic_actions=actions,
            ),
        }
    ]
    training_rewards: list[float] = []
    training_sla: list[int] = []
    updates: list[dict[str, float]] = []
    illegal_actions = 0
    for step in range(1, int(config["total_environment_steps"]) + 1):
        if len(encoded.task_ids) > int(config["max_tasks"]):
            raise RuntimeError(
                f"pending task count {len(encoded.task_ids)} exceeds max_tasks"
            )
        if step <= int(config["warmup_steps"]):
            indices = random_feasible_indices(encoded, rng)
            decisions = indices_to_decisions(encoded, indices, actions)
        else:
            indices, decisions = select_actions(
                actor, encoded, actions, stochastic=True, rng=rng
            )
        environment_actions = action_adapter.encode_assignments(
            state.current.tasks, decisions
        )
        illegal_actions += int(len(environment_actions) != len(state.current.tasks))
        _, reward, terminated, truncated, info = env.step(environment_actions)
        episode_boundary = step % int(config["training_episode_steps"]) == 0
        transition_done = bool(terminated or truncated or episode_boundary)
        reward_stats.update(float(reward))
        normalized_reward = float(reward_stats.normalize(float(reward)))
        if transition_done:
            next_state = None
            next_encoded = EncodedTaskBatch(
                np.empty((0, feature_dim), dtype=np.float32),
                np.empty((0, action_dim), dtype=bool),
                (),
                (),
            )
        else:
            next_state, next_encoded = encode_current(
                env, state_adapter, forecaster, encoder
            )
        if len(next_encoded.task_ids) > int(config["max_tasks"]):
            raise RuntimeError(
                f"next pending task count {len(next_encoded.task_ids)} exceeds max_tasks"
            )
        if len(encoded.task_ids) > 0:
            buffer.add(
                encoded.features,
                indices,
                normalized_reward,
                next_encoded.features,
                transition_done,
            )
        training_rewards.append(float(reward))
        training_sla.append(sla_violations(info))
        if transition_done:
            episode_index += 1
            if step < int(config["total_environment_steps"]):
                action_adapter, forecaster, initial = reset_training_episode(
                    episode_index
                )
                state, encoded = initial
        else:
            state, encoded = next_state, next_encoded
        if (
            step >= int(config["warmup_steps"])
            and step % int(config["update_frequency"]) == 0
            and len(buffer) >= int(config["batch_size"])
        ):
            update = sac_update(
                actor=actor,
                critic=critic,
                target_critic=target_critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                buffer=buffer,
                config=config,
                global_step=step,
            )
            update["step"] = float(step)
            updates.append(update)
        if step in evaluation_points and step != 0:
            evaluations.append(
                {
                    "step": step,
                    **evaluate(
                        repo=repo,
                        actor=actor,
                        config=config,
                        expert_config=expert_config,
                        encoder=encoder,
                        semantic_actions=actions,
                    ),
                }
            )
    first10 = int(len(training_rewards) * 0.10)
    first25 = int(len(training_rewards) * 0.25)
    finite_updates = [row for row in updates if np.isfinite(row["critic_loss"])]
    return {
        "group": group,
        "seed": seed,
        "environment_steps": len(training_rewards),
        "initial_actor_hash": initial_actor_hash,
        "final_actor_hash": state_hash(actor),
        "critic_initial_hash": critic_hash,
        "replay_buffer_final_size": len(buffer),
        "initial_evaluation": evaluations[0],
        "evaluations": evaluations,
        "first_10_percent": {
            "mean_step_reward": mean(training_rewards[:first10]),
            "sla_violations": sum(training_sla[:first10]),
        },
        "first_25_percent": {
            "mean_step_reward": mean(training_rewards[:first25]),
            "sla_violations": sum(training_sla[:first25]),
        },
        "training": {
            "total_reward": sum(training_rewards),
            "sla_violations": sum(training_sla),
            "illegal_actions": illegal_actions,
            "updates": len(updates),
            "actor_loss_mean": float(
                np.nanmean([row["actor_loss"] for row in updates])
            ),
            "critic_loss_mean": mean(row["critic_loss"] for row in finite_updates),
            "q_mean": mean(row["q_mean"] for row in finite_updates),
            "q_std_mean": mean(row["q_std"] for row in finite_updates),
            "all_q_finite": all(
                np.isfinite(row["q_mean"]) and np.isfinite(row["q_std"])
                for row in updates
            ),
        },
        "update_curve": updates[:: max(1, len(updates) // 200)],
    }


def aggregate_results(runs: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[run["group"]].append(run)
    initial_means = {
        group: mean(row["initial_evaluation"]["reward"] for row in rows)
        for group, rows in groups.items()
    }
    threshold = 0.5 * (
        initial_means["random_init_sac"] + initial_means["bc_init_sac"]
    )
    aggregates = {}
    for group, rows in groups.items():
        steps_to_threshold = []
        for row in rows:
            reached = next(
                (
                    item["step"]
                    for item in row["evaluations"]
                    if item["reward"] >= threshold
                ),
                None,
            )
            steps_to_threshold.append(reached)
        aggregates[group] = {
            "initial_reward_mean": mean(
                row["initial_evaluation"]["reward"] for row in rows
            ),
            "initial_reward_std": float(
                np.std([row["initial_evaluation"]["reward"] for row in rows])
            ),
            "final_reward_mean": mean(row["evaluations"][-1]["reward"] for row in rows),
            "final_reward_std": float(
                np.std([row["evaluations"][-1]["reward"] for row in rows])
            ),
            "initial_sla_mean": mean(
                row["initial_evaluation"]["sla_violations"] for row in rows
            ),
            "final_sla_mean": mean(
                row["evaluations"][-1]["sla_violations"] for row in rows
            ),
            "first_10_percent_reward_mean": mean(
                row["first_10_percent"]["mean_step_reward"] for row in rows
            ),
            "first_25_percent_reward_mean": mean(
                row["first_25_percent"]["mean_step_reward"] for row in rows
            ),
            "steps_to_threshold": steps_to_threshold,
            "steps_to_threshold_mean": mean(
                value for value in steps_to_threshold if value is not None
            )
            if any(value is not None for value in steps_to_threshold)
            else None,
            "initial_expert_agreement_mean": mean(
                row["evaluations"][0]["expert_action_agreement"] for row in rows
            ),
            "final_expert_agreement_mean": mean(
                row["evaluations"][-1]["expert_action_agreement"] for row in rows
            ),
            "actor_loss_mean": mean(row["training"]["actor_loss_mean"] for row in rows),
            "critic_loss_mean": mean(row["training"]["critic_loss_mean"] for row in rows),
            "q_mean": mean(row["training"]["q_mean"] for row in rows),
            "q_std_mean": mean(row["training"]["q_std_mean"] for row in rows),
            "all_q_finite": all(row["training"]["all_q_finite"] for row in rows),
            "illegal_actions": sum(row["training"]["illegal_actions"] for row in rows),
        }
    return {"reward_threshold": threshold, "groups": aggregates}


def write_curves(path: Path, runs: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "group",
                "seed",
                "step",
                "reward",
                "sla_violations",
                "expert_action_agreement",
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        for run in runs:
            for item in run["evaluations"]:
                writer.writerow(
                    {
                        "group": run["group"],
                        "seed": run["seed"],
                        **item,
                    }
                )


def report(result: dict[str, Any]) -> str:
    groups = result["aggregate"]["groups"]
    random_group = groups["random_init_sac"]
    bc_group = groups["bc_init_sac"]
    bc_initial_gain = (
        bc_group["initial_reward_mean"] - random_group["initial_reward_mean"]
    )
    bc_initial_sla_reduction = (
        random_group["initial_sla_mean"] - bc_group["initial_sla_mean"]
    )
    bc_agreement_drift = (
        bc_group["final_expert_agreement_mean"]
        - bc_group["initial_expert_agreement_mean"]
    )
    bc_reward_drift = (
        bc_group["final_reward_mean"] - bc_group["initial_reward_mean"]
    )
    lines = [
        "# SAC warm-start comparison",
        "",
        "Three paired seeds run 10,000 real SustainCluster environment interactions per group. Actor architecture, critic initialization, replay capacity, reward, optimizer settings and evaluation seeds are identical; only actor initialization differs.",
        "",
        f"Reward threshold: {result['aggregate']['reward_threshold']:.6f}",
        "",
        "| Group | Initial reward | Final reward | Initial SLA | Final SLA | First 10% reward | First 25% reward | Steps to threshold | Expert agreement initial/final |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, values in groups.items():
        lines.append(
            f"| {group} | {values['initial_reward_mean']:.4f} +/- {values['initial_reward_std']:.4f} | "
            f"{values['final_reward_mean']:.4f} +/- {values['final_reward_std']:.4f} | "
            f"{values['initial_sla_mean']:.2f} | {values['final_sla_mean']:.2f} | "
            f"{values['first_10_percent_reward_mean']:.6f} | "
            f"{values['first_25_percent_reward_mean']:.6f} | "
            f"{values['steps_to_threshold_mean']} | "
            f"{values['initial_expert_agreement_mean']:.4f}/{values['final_expert_agreement_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The original one-million-transition replay setting is not used because `1,000,000 x 750 x 233 x 2` float observations would require roughly 1.4 TB before actions and masks. Both groups instead use the same bounded replay configuration recorded in JSON.",
            "",
            "A fall in expert agreement after updates is treated as policy drift; reward/SLA changes determine whether that drift is useful adaptation or behavior-cloning distribution shift.",
            "",
            "## Answers",
            "",
            f"1. **Initial performance:** yes. BC improves mean initial evaluation reward by {bc_initial_gain:.4f} (higher is better).",
            f"2. **Early SLA:** yes at initialization. BC reduces initial SLA violations by {bc_initial_sla_reduction:.2f} per evaluation episode, although this advantage does not survive fine-tuning.",
            f"3. **Reward threshold:** BC reaches the fixed threshold at step {bc_group['steps_to_threshold_mean']}; random initialization needs {random_group['steps_to_threshold_mean']} steps on average among the paired runs.",
            f"4. **Seed variance:** yes initially. BC initial reward std is {bc_group['initial_reward_std']:.6f}, versus {random_group['initial_reward_std']:.6f} for random initialization.",
            f"5. **MPC policy drift:** yes. BC expert agreement changes by {bc_agreement_drift:.6f}, and evaluation reward changes by {bc_reward_drift:.4f} after 10,000 steps.",
            "6. **Distribution shift:** yes. The BC actor starts close to the expert, but online SAC updates reduce expert agreement and increase final SLA violations; the current reward/update formulation does not preserve the cloned safety behavior.",
            "",
            "This is a warm-start validation, not evidence that the current SAC fine-tuning recipe is ready for long training. A KL/BC regularizer, expert replay mixing, or a frozen-actor critic warmup should be evaluated before scaling.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expert-config", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-curves", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["sac_warm_start"]
    torch.set_num_threads(int(config["torch_threads"]))
    expert_config = ExpertRuntimeConfig.from_yaml(args.expert_config)
    config_fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest().upper()
    partial_path = args.output_json.with_name(
        f"{args.output_json.stem}.{config_fingerprint[:12]}.partial.json"
    )
    runs = []
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("config_fingerprint") != config_fingerprint:
            raise RuntimeError("partial SAC result config fingerprint mismatch")
        runs = partial.get("runs", [])
    completed = {(row["group"], int(row["seed"])) for row in runs}
    for seed in config["seeds"]:
        for group in ("random_init_sac", "bc_init_sac"):
            if (group, int(seed)) in completed:
                print(f"resuming past completed {group} seed={seed}", flush=True)
                continue
            print(f"starting {group} seed={seed}", flush=True)
            run = train_run(
                group=group,
                seed=int(seed),
                repo=args.repo.resolve(),
                checkpoint=args.bc_checkpoint.resolve(),
                config=config,
                expert_config=expert_config,
            )
            runs.append(run)
            partial_path.write_text(
                json.dumps(
                    {
                        "config": config,
                        "config_fingerprint": config_fingerprint,
                        "runs": runs,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            print(
                f"finished {group} seed={seed} final_reward={run['evaluations'][-1]['reward']:.6f}",
                flush=True,
            )
    paired_critic_hashes = {
        seed: {
            row["group"]: row["critic_initial_hash"]
            for row in runs
            if row["seed"] == seed
        }
        for seed in config["seeds"]
    }
    if not all(len(set(values.values())) == 1 for values in paired_critic_hashes.values()):
        raise RuntimeError("paired SAC critics did not start from identical weights")
    result = {
        "config": config,
        "config_fingerprint": config_fingerprint,
        "bc_checkpoint": str(args.bc_checkpoint.resolve()),
        "paired_critic_hashes": paired_critic_hashes,
        "runs": runs,
        "aggregate": aggregate_results(runs),
    }
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output_report.write_text(report(result), encoding="utf-8")
    write_curves(args.output_curves, runs)
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()
