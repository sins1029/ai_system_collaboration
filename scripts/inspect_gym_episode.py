from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datacenter_env import MaskedRandomPolicy  # noqa: E402
from experiments.gymnasium_env import build_standard_gym_environment  # noqa: E402


def main() -> None:
    env = build_standard_gym_environment(ROOT)
    policy = MaskedRandomPolicy(seed=7)
    observation, info = env.reset(seed=7)
    for _ in range(12):
        action = policy.action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print(
            f"decision={info['decision_step_index']} sim={info['simulation_step_index']} "
            f"task={info['candidate_task_id']} mask={info['action_mask']} action={action} "
            f"reward={reward:.4f} physical={info['physical_step_advanced']} "
            f"started={info['started_task_ids']} completed={info['completed_task_ids']}"
        )
        if info["auto_skipped_candidate_ids"]:
            print(
                f"  auto_skipped={info['auto_skipped_candidate_ids']} "
                f"forced_blocked={info['forced_resource_blocked_task_ids']}"
            )
        print(f"  reward_components={info['reward_components']}")
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()
