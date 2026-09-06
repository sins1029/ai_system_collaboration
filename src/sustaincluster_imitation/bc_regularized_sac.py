from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sustaincluster_imitation.bc_policy import BCPolicyConfig
from sustaincluster_imitation.sac_weight_bridge import (
    WeightBridgeResult,
    initialize_sac_actor_from_bc,
)


def module_state_hash(module: nn.Module) -> str:
    """计算包含参数名、dtype、shape 与数值的稳定 SHA-256。"""
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest().upper()


def initialize_bc_actor_and_teacher(
    checkpoint: Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, nn.Module, BCPolicyConfig, WeightBridgeResult]:
    """从同一 BC checkpoint 初始化可训练 Actor 和冻结 Teacher。"""
    actor, config, bridge = initialize_sac_actor_from_bc(checkpoint, device)
    teacher = copy.deepcopy(actor).to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    return actor, teacher, config, bridge


def masked_policy_terms(
    logits: torch.Tensor,
    feasible_action_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回只在可行动作上归一化的概率、对数概率与熵。"""
    mask = feasible_action_mask.bool()
    if logits.shape != mask.shape:
        raise ValueError("动作掩码形状必须与 logits 一致")
    if logits.ndim != 2:
        raise ValueError("logits 必须为 [N,A]")
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("每个真实任务至少需要一个可行动作")
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probabilities = F.softmax(masked_logits, dim=-1)
    log_probabilities = F.log_softmax(masked_logits, dim=-1)
    entropy_terms = torch.where(
        mask,
        probabilities * log_probabilities,
        torch.zeros_like(probabilities),
    )
    entropy = -entropy_terms.sum(dim=-1)
    return probabilities, log_probabilities, entropy


def masked_teacher_kl(
    actor_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    feasible_action_mask: torch.Tensor,
) -> torch.Tensor:
    """计算固定 BC Teacher 到当前 Actor 的 masked KL。"""
    actor_probs, actor_log_probs, _ = masked_policy_terms(
        actor_logits, feasible_action_mask
    )
    del actor_probs
    teacher_probs, teacher_log_probs, _ = masked_policy_terms(
        teacher_logits, feasible_action_mask
    )
    mask = feasible_action_mask.bool()
    terms = torch.where(
        mask,
        teacher_probs * (teacher_log_probs - actor_log_probs),
        torch.zeros_like(teacher_probs),
    )
    return terms.sum(dim=-1).mean()


@torch.no_grad()
def select_masked_actions(
    actor: nn.Module,
    features: torch.Tensor,
    feasible_action_mask: torch.Tensor,
    *,
    deterministic: bool,
) -> torch.Tensor:
    """从当前可行语义动作中选择动作。"""
    probabilities, _, _ = masked_policy_terms(
        actor(features), feasible_action_mask
    )
    if deterministic:
        return probabilities.argmax(dim=-1)
    return torch.distributions.Categorical(probs=probabilities).sample()


@dataclass(frozen=True)
class ReplayBatch:
    """包含 task mask 与 feasible-action mask 的 SAC batch。"""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    task_mask: torch.Tensor
    next_task_mask: torch.Tensor
    feasible_action_mask: torch.Tensor
    next_feasible_action_mask: torch.Tensor


@dataclass(frozen=True)
class _Transition:
    observations: np.ndarray
    actions: np.ndarray
    reward: float
    next_observations: np.ndarray
    done: bool
    feasible_action_mask: np.ndarray
    next_feasible_action_mask: np.ndarray


class SemanticReplayBuffer:
    """按实际任务数紧凑保存语义动作 transition，避免丢失可行性掩码。"""

    def __init__(
        self,
        capacity: int,
        max_tasks: int,
        observation_dim: int,
        action_dim: int,
        seed: int,
    ) -> None:
        if min(capacity, max_tasks, observation_dim, action_dim) <= 0:
            raise ValueError("Replay 维度必须为正数")
        self.capacity = int(capacity)
        self.max_tasks = int(max_tasks)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(seed)
        self._storage: list[_Transition] = []
        self._position = 0

    def add(
        self,
        observations: np.ndarray,
        actions: list[int] | np.ndarray,
        reward: float,
        next_observations: np.ndarray,
        done: bool,
        feasible_action_mask: np.ndarray,
        next_feasible_action_mask: np.ndarray,
    ) -> None:
        """加入一条环境 transition，并校验真实动作均可行。"""
        observations = np.asarray(observations, dtype=np.float32)
        next_observations = np.asarray(next_observations, dtype=np.float32)
        actions_array = np.asarray(actions, dtype=np.int64)
        feasible = np.asarray(feasible_action_mask, dtype=bool)
        next_feasible = np.asarray(next_feasible_action_mask, dtype=bool)
        self._validate_step(observations, feasible, "current")
        self._validate_step(next_observations, next_feasible, "next")
        if len(observations) > self.max_tasks or len(next_observations) > self.max_tasks:
            raise ValueError("transition 中任务数超过 max_tasks")
        if actions_array.shape != (len(observations),):
            raise ValueError("动作数量必须与 current observations 一致")
        if len(actions_array):
            in_range = (actions_array >= 0) & (actions_array < self.action_dim)
            if not bool(in_range.all()):
                raise ValueError("Replay 动作索引越界")
            chosen_feasible = feasible[
                np.arange(len(actions_array)), actions_array
            ]
            if not bool(chosen_feasible.all()):
                raise ValueError("Replay 拒绝保存不可行语义动作")
        transition = _Transition(
            observations.copy(),
            actions_array.copy(),
            float(reward),
            next_observations.copy(),
            bool(done),
            feasible.copy(),
            next_feasible.copy(),
        )
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._position] = transition
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> ReplayBatch:
        """有放回采样并只 pad 到当前 batch 的最大任务数。"""
        if not self._storage:
            raise ValueError("Replay 为空")
        indices = self._rng.integers(0, len(self._storage), size=int(batch_size))
        rows = [self._storage[int(index)] for index in indices]
        tasks = max(
            1,
            max(max(len(row.observations), len(row.next_observations)) for row in rows),
        )
        batch = len(rows)
        obs = np.zeros((batch, tasks, self.observation_dim), dtype=np.float32)
        next_obs = np.zeros_like(obs)
        actions = np.full((batch, tasks), -1, dtype=np.int64)
        task_mask = np.zeros((batch, tasks), dtype=bool)
        next_task_mask = np.zeros_like(task_mask)
        feasible = np.zeros((batch, tasks, self.action_dim), dtype=bool)
        next_feasible = np.zeros_like(feasible)
        rewards = np.empty(batch, dtype=np.float32)
        dones = np.empty(batch, dtype=np.float32)
        for index, row in enumerate(rows):
            current_tasks = len(row.observations)
            following_tasks = len(row.next_observations)
            if current_tasks:
                obs[index, :current_tasks] = row.observations
                actions[index, :current_tasks] = row.actions
                task_mask[index, :current_tasks] = True
                feasible[index, :current_tasks] = row.feasible_action_mask
            if following_tasks:
                next_obs[index, :following_tasks] = row.next_observations
                next_task_mask[index, :following_tasks] = True
                next_feasible[index, :following_tasks] = (
                    row.next_feasible_action_mask
                )
            rewards[index] = row.reward
            dones[index] = float(row.done)
        return ReplayBatch(
            observations=torch.from_numpy(obs),
            actions=torch.from_numpy(actions),
            rewards=torch.from_numpy(rewards),
            next_observations=torch.from_numpy(next_obs),
            dones=torch.from_numpy(dones),
            task_mask=torch.from_numpy(task_mask),
            next_task_mask=torch.from_numpy(next_task_mask),
            feasible_action_mask=torch.from_numpy(feasible),
            next_feasible_action_mask=torch.from_numpy(next_feasible),
        )

    def _validate_step(
        self,
        observations: np.ndarray,
        feasible_action_mask: np.ndarray,
        label: str,
    ) -> None:
        if observations.shape != (len(observations), self.observation_dim):
            raise ValueError(f"{label} observations 形状错误")
        expected = (len(observations), self.action_dim)
        if feasible_action_mask.shape != expected:
            raise ValueError(f"{label} feasible-action mask 形状错误")
        if len(observations) and not bool(feasible_action_mask.any(axis=1).all()):
            raise ValueError(f"{label} 每个任务至少需要一个可行动作")

    def __len__(self) -> int:
        return len(self._storage)


@dataclass(frozen=True)
class SACUpdateSettings:
    """单次离散 SAC 更新所需的固定超参数。"""

    gamma: float
    alpha: float
    tau: float
    behavior_reg_weight: float = 0.0
    behavior_reg_type: str = "teacher_kl"


def sac_update(
    *,
    actor: nn.Module,
    critic: nn.Module,
    target_critic: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch: ReplayBatch,
    settings: SACUpdateSettings,
    teacher: nn.Module | None,
    update_actor: bool,
    update_target: bool = True,
) -> dict[str, float | bool]:
    """执行一次 masked discrete SAC 更新，并可在 warmup 期间冻结 Actor。"""
    task_mask = batch.task_mask.reshape(-1).bool()
    next_task_mask = batch.next_task_mask.reshape(-1).bool()
    if not task_mask.any():
        raise ValueError("batch 中没有真实 current task")
    observation_dim = batch.observations.shape[-1]
    action_dim = batch.feasible_action_mask.shape[-1]
    observations = batch.observations.reshape(-1, observation_dim)
    next_observations = batch.next_observations.reshape(-1, observation_dim)
    feasible = batch.feasible_action_mask.reshape(-1, action_dim)
    next_feasible = batch.next_feasible_action_mask.reshape(-1, action_dim)
    actions = batch.actions.reshape(-1)

    next_values = torch.zeros(
        len(next_observations), dtype=observations.dtype, device=observations.device
    )
    with torch.no_grad():
        if next_task_mask.any():
            next_logits = actor(next_observations[next_task_mask])
            next_probs, next_log_probs, _ = masked_policy_terms(
                next_logits, next_feasible[next_task_mask]
            )
            q1_next, q2_next = target_critic.forward_all(
                next_observations[next_task_mask]
            )
            q_next = torch.minimum(q1_next, q2_next)
            next_terms = torch.where(
                next_feasible[next_task_mask],
                next_probs * (q_next - settings.alpha * next_log_probs),
                torch.zeros_like(next_probs),
            )
            next_values[next_task_mask] = next_terms.sum(dim=-1)
        next_values = next_values.reshape(batch.task_mask.shape)
        targets = batch.rewards.unsqueeze(1) + settings.gamma * (
            1.0 - batch.dones.unsqueeze(1)
        ) * next_values

    valid_observations = observations[task_mask]
    valid_actions = actions[task_mask]
    valid_feasible = feasible[task_mask]
    chosen_feasible = valid_feasible.gather(
        1, valid_actions.long().unsqueeze(1)
    ).squeeze(1)
    if not bool(chosen_feasible.all()):
        raise ValueError("batch 中包含不可行 semantic action")
    q1, q2 = critic(valid_observations, valid_actions)
    target = targets.reshape(-1)[task_mask]
    critic_loss = 0.5 * (
        F.mse_loss(q1, target) + F.mse_loss(q2, target)
    )
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_optimizer.step()

    logits = actor(valid_observations)
    probabilities, log_probabilities, entropy = masked_policy_terms(
        logits, valid_feasible
    )
    with torch.no_grad():
        q1_eval, q2_eval = critic.forward_all(valid_observations)
        q_eval = torch.minimum(q1_eval, q2_eval)
    actor_terms = torch.where(
        valid_feasible,
        probabilities * (settings.alpha * log_probabilities - q_eval),
        torch.zeros_like(probabilities),
    )
    sac_actor_loss = actor_terms.sum(dim=-1).mean()
    if teacher is None:
        expert_loss = sac_actor_loss.new_zeros(())
    else:
        if settings.behavior_reg_type != "teacher_kl":
            raise ValueError("Algorithm v0.1 仅支持 teacher_kl")
        with torch.no_grad():
            teacher_logits = teacher(valid_observations)
        expert_loss = masked_teacher_kl(
            logits, teacher_logits, valid_feasible
        )
    weighted_expert_loss = settings.behavior_reg_weight * expert_loss
    total_actor_loss = sac_actor_loss + weighted_expert_loss
    if update_actor:
        actor_optimizer.zero_grad(set_to_none=True)
        total_actor_loss.backward()
        actor_optimizer.step()

    if update_target:
        with torch.no_grad():
            for source, target_parameter in zip(
                critic.parameters(), target_critic.parameters()
            ):
                target_parameter.mul_(1.0 - settings.tau)
                target_parameter.add_(settings.tau * source)

    q_values = torch.minimum(q1.detach(), q2.detach())
    finite_tensors = (
        critic_loss.detach(),
        sac_actor_loss.detach(),
        expert_loss.detach(),
        weighted_expert_loss.detach(),
        total_actor_loss.detach(),
        q_values,
        entropy.detach(),
    )
    if not all(bool(torch.isfinite(value).all()) for value in finite_tensors):
        raise FloatingPointError("SAC update 出现 NaN 或 Inf")
    return {
        "critic_loss": float(critic_loss.detach()),
        "sac_actor_loss": float(sac_actor_loss.detach()),
        "expert_loss": float(expert_loss.detach()),
        "weighted_expert_loss": float(weighted_expert_loss.detach()),
        "total_actor_loss": float(total_actor_loss.detach()),
        "q_mean": float(q_values.mean()),
        "q_std": float(q_values.std(unbiased=False)),
        "entropy": float(entropy.mean()),
        "alpha": float(settings.alpha),
        "actor_updated": bool(update_actor),
        "all_finite": True,
    }


def load_batch_to_device(batch: ReplayBatch, device: torch.device | str) -> ReplayBatch:
    """把 ReplayBatch 的全部张量移动到同一设备。"""
    values: dict[str, Any] = {}
    for name in batch.__dataclass_fields__:
        values[name] = getattr(batch, name).to(device)
    return ReplayBatch(**values)
