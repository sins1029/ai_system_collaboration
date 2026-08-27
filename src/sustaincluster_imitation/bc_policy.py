from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sustaincluster_imitation.paths import prepare_sustaincluster_imports
from sustaincluster_imitation.feature_encoder import (
    EncodedTaskBatch,
    SemanticActionSpace,
)
from sustaincluster_mpc import AssignmentDecision


def build_sustaincluster_actor(
    feature_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    use_layer_norm: bool = True,
) -> nn.Module:
    prepare_sustaincluster_imports()
    module = importlib.import_module("rl_components.agent_net")
    return module.ActorNet(        feature_dim,
        action_dim,
        hidden_dim=hidden_dim,
        use_layer_norm=use_layer_norm,
    )


@dataclass(frozen=True)
class BCPolicyConfig:
    feature_dim: int
    action_dc_ids: tuple[int, ...]
    hidden_dim: int = 256
    use_layer_norm: bool = True


class BCPolicy(nn.Module):
    """加载或执行行为克隆策略，并输出稳定语义任务决策。"""
    def __init__(self, config: BCPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.semantic_actions = SemanticActionSpace(config.action_dc_ids, True)
        self.actor = build_sustaincluster_actor(
            config.feature_dim,
            self.semantic_actions.size,
            config.hidden_dim,
            config.use_layer_norm,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        shape = features.shape
        if features.ndim == 2:
            return self.actor(features)
        if features.ndim != 3:
            raise ValueError("特征张量必须为 [T,F] 或 [B,T,F]")
        flat = features.reshape(-1, shape[-1])
        return self.actor(flat).reshape(shape[0], shape[1], -1)

    def masked_logits(
        self, features: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        logits = self(features)
        if logits.shape != action_mask.shape:
            raise ValueError("动作掩码形状与 logits 不一致")
        if not torch.all(action_mask.any(dim=-1)):
            raise ValueError("每个真实任务至少需要一个可行动作")
        return logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)

    @torch.no_grad()
    def predict(
        self,
        encoded: EncodedTaskBatch,
        device: torch.device | str = "cpu",
        deterministic: bool = True,
    ) -> tuple[AssignmentDecision, ...]:
        if encoded.features.shape[0] == 0:
            return ()
        self.eval()
        features = torch.as_tensor(encoded.features, dtype=torch.float32, device=device)
        mask = torch.as_tensor(encoded.feasible_action_mask, dtype=torch.bool, device=device)
        logits = self.masked_logits(features, mask)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = torch.distributions.Categorical(logits=logits).sample()
        decisions = []
        for task_id, original_index, action in zip(
            encoded.task_ids,
            encoded.original_indices,
            actions.cpu().tolist(),
        ):
            dc_id = self.semantic_actions.dc_for_index(int(action))
            decisions.append(
                AssignmentDecision(
                    task_id=task_id,
                    original_index=original_index,
                    decision="defer" if dc_id is None else "assign",
                    dc_id=dc_id,
                )
            )
        return tuple(decisions)

    def save_checkpoint(
        self, path: Path, extra: dict[str, Any] | None = None
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "config": {
                    "feature_dim": self.config.feature_dim,
                    "action_dc_ids": list(self.config.action_dc_ids),
                    "hidden_dim": self.config.hidden_dim,
                    "use_layer_norm": self.config.use_layer_norm,
                },
                "actor_state_dict": self.actor.state_dict(),
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls, path: Path, device: torch.device | str = "cpu"
    ) -> "BCPolicy":
        payload = torch.load(path, map_location=device, weights_only=False)
        config = payload["config"]
        policy = cls(
            BCPolicyConfig(
                feature_dim=int(config["feature_dim"]),
                action_dc_ids=tuple(int(x) for x in config["action_dc_ids"]),
                hidden_dim=int(config["hidden_dim"]),
                use_layer_norm=bool(config["use_layer_norm"]),
            )
        )
        policy.actor.load_state_dict(payload["actor_state_dict"], strict=True)
        policy.to(device)
        return policy


def weighted_masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_mask: torch.Tensor,
    feasible_action_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape != feasible_action_mask.shape:
        raise ValueError("可行性掩码形状必须与 logits 一致")
    if labels.shape != task_mask.shape or labels.shape != logits.shape[:-1]:
        raise ValueError("标签与任务掩码形状不一致")
    flat_real = task_mask.reshape(-1).bool()
    if not flat_real.any():
        return logits.sum() * 0.0
    flat_logits = logits.reshape(-1, logits.shape[-1])[flat_real]
    flat_feasible = feasible_action_mask.reshape(-1, logits.shape[-1])[flat_real]
    flat_labels = labels.reshape(-1)[flat_real].long()
    chosen_feasible = flat_feasible.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
    if not chosen_feasible.all():
        raise ValueError("专家标签被可行性掩码判定为不可行")
    masked_logits = flat_logits.masked_fill(
        ~flat_feasible.bool(), torch.finfo(flat_logits.dtype).min
    )
    return F.cross_entropy(masked_logits, flat_labels, weight=class_weights)
