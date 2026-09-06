from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from sustaincluster_imitation.bc_regularized_sac import (
    SACUpdateSettings,
    SemanticReplayBuffer,
    initialize_bc_actor_and_teacher,
    module_state_hash,
    sac_update,
    select_masked_actions,
)
from sustaincluster_imitation.paths import prepare_sustaincluster_imports


WORKSPACE = Path(__file__).resolve().parents[1]
prepare_sustaincluster_imports()
from rl_components.agent_net import ActorNet, CriticNet  # noqa: E402


def _components(seed: int = 7):
    torch.manual_seed(seed)
    actor = ActorNet(4, 3, hidden_dim=16, use_layer_norm=True)
    teacher = copy.deepcopy(actor).eval().requires_grad_(False)
    critic = CriticNet(4, 3, hidden_dim=16, use_layer_norm=True)
    target = copy.deepcopy(critic)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-3)
    return actor, teacher, critic, target, actor_optimizer, critic_optimizer


def _batch(seed: int = 19):
    rng = np.random.default_rng(seed)
    replay = SemanticReplayBuffer(8, 4, 4, 3, seed)
    for index in range(4):
        observations = rng.normal(size=(2, 4)).astype(np.float32)
        next_observations = rng.normal(size=(2, 4)).astype(np.float32)
        feasible = np.array([[True, True, False], [True, False, True]])
        next_feasible = np.array([[True, False, True], [True, True, False]])
        replay.add(
            observations,
            [index % 2, 2 if index % 2 == 0 else 0],
            float(index - 2),
            next_observations,
            False,
            feasible,
            next_feasible,
        )
    return replay.sample(4)


def _update(components, *, weight: float, update_actor: bool):
    actor, teacher, critic, target, actor_optimizer, critic_optimizer = components
    return sac_update(
        actor=actor,
        critic=critic,
        target_critic=target,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        batch=_batch(),
        settings=SACUpdateSettings(0.99, 0.01, 0.005, weight),
        teacher=teacher,
        update_actor=update_actor,
    )


def test_bc_actor_and_sac_actor_are_exactly_equal() -> None:
    checkpoint = (
        WORKSPACE / "artifacts/sustaincluster_imitation/bc_actor_seed_11.pt"
    )
    actor, teacher, config, bridge = initialize_bc_actor_and_teacher(checkpoint)
    torch.manual_seed(101)
    observations = torch.randn(31, config.feature_dim)

    assert bridge.exact_key_match is True
    assert tuple(actor.state_dict()) == tuple(teacher.state_dict())
    assert torch.equal(actor(observations), teacher(observations))


def test_frozen_teacher_parameters_do_not_change() -> None:
    components = _components()
    teacher = components[1]
    before = module_state_hash(teacher)

    _update(components, weight=0.1, update_actor=True)

    assert module_state_hash(teacher) == before
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_actor_does_not_change_during_critic_warmup() -> None:
    components = _components()
    actor = components[0]
    before = module_state_hash(actor)

    result = _update(components, weight=0.1, update_actor=False)

    assert result["actor_updated"] is False
    assert module_state_hash(actor) == before


def test_actor_changes_after_critic_warmup() -> None:
    components = _components()
    actor = components[0]
    before = [parameter.detach().clone() for parameter in actor.parameters()]

    result = _update(components, weight=0.1, update_actor=True)

    assert result["actor_updated"] is True
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, actor.parameters())
    )


def test_zero_regularization_matches_vanilla_actor_update() -> None:
    first = _components(seed=29)
    second = _components(seed=29)

    first_result = _update(first, weight=0.0, update_actor=True)
    second_result = sac_update(
        actor=second[0],
        critic=second[2],
        target_critic=second[3],
        actor_optimizer=second[4],
        critic_optimizer=second[5],
        batch=_batch(),
        settings=SACUpdateSettings(0.99, 0.01, 0.005, 0.0),
        teacher=None,
        update_actor=True,
    )

    assert first_result["total_actor_loss"] == first_result["sac_actor_loss"]
    assert second_result["total_actor_loss"] == second_result["sac_actor_loss"]
    for left, right in zip(first[0].parameters(), second[0].parameters()):
        assert torch.equal(left, right)


def test_masked_action_selection_never_returns_illegal_action() -> None:
    actor = ActorNet(4, 3, hidden_dim=16, use_layer_norm=True)
    features = torch.randn(256, 4)
    mask = torch.zeros(256, 3, dtype=torch.bool)
    mask[:128, 1] = True
    mask[128:, 2] = True

    sampled = select_masked_actions(
        actor, features, mask, deterministic=False
    )

    assert bool(mask.gather(1, sampled.unsqueeze(1)).all())


def test_all_sac_losses_are_finite() -> None:
    result = _update(_components(), weight=0.1, update_actor=True)

    assert result["all_finite"] is True
    for name in (
        "critic_loss",
        "sac_actor_loss",
        "expert_loss",
        "weighted_expert_loss",
        "total_actor_loss",
        "q_mean",
        "q_std",
        "entropy",
        "alpha",
    ):
        assert np.isfinite(result[name])
