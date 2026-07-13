from __future__ import annotations

import gymnasium as gym


ENVIRONMENT_ID = "DataCenterTaskScheduling-v0"


def register_gym_environments() -> None:
    if ENVIRONMENT_ID not in gym.registry:
        gym.register(
            id=ENVIRONMENT_ID,
            entry_point="datacenter_env.gym.environment:SingleCenterTaskSchedulingEnv",
        )
