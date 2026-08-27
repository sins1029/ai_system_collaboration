from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from sustaincluster_imitation.bc_policy import (
    BCPolicy,
    BCPolicyConfig,
    build_sustaincluster_actor,
)


@dataclass(frozen=True)
class WeightBridgeResult:
    source_checkpoint: str
    parameter_tensors: int
    parameter_count: int
    exact_key_match: bool


def initialize_sac_actor_from_bc(
    checkpoint: Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, BCPolicyConfig, WeightBridgeResult]:
    policy = BCPolicy.load_checkpoint(checkpoint, device)
    config = policy.config
    actor = build_sustaincluster_actor(
        config.feature_dim,
        len(config.action_dc_ids) + 1,
        config.hidden_dim,
        config.use_layer_norm,
    ).to(device)
    source = policy.actor.state_dict()
    target = actor.state_dict()
    exact = tuple(source) == tuple(target) and all(
        source[key].shape == target[key].shape for key in source
    )
    if not exact:
        raise ValueError("BC actor 与 SAC actor 的参数模式不同")
    actor.load_state_dict(source, strict=True)
    return actor, config, WeightBridgeResult(
        str(Path(checkpoint).resolve()),
        len(source),
        sum(parameter.numel() for parameter in actor.parameters()),
        exact,
    )
