from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datacenter_env import MaskedRandomPolicy  # noqa: E402
from experiments.gymnasium_env import build_standard_gym_environment  # noqa: E402


def main() -> None:
    env = build_standard_gym_environment(ROOT)
    policy = MaskedRandomPolicy(seed=42)
    observation, info = env.reset(seed=42)
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    metrics = info["final_metrics"]
    summary = info["episode_summary"]
    print(f"episode_reward: {summary['episode_reward']:.6f}")
    print(f"decision_steps: {summary['decision_steps']}")
    print(f"simulation_steps: {summary['simulation_steps']}")
    print(f"completed_tasks: {metrics['tasks_completed']:.0f}")
    print(f"sla_violations: {metrics['sla_violation_count']:.0f}")
    print(f"invalid_actions: {summary['invalid_actions']}")
    print(f"energy_cost: {metrics['energy_cost']:.6f}")
    print(f"carbon_kg: {metrics['carbon_kg']:.6f}")
    print(f"temperature_violations: {metrics['temperature_violation_count']:.0f}")
    env.close()


if __name__ == "__main__":
    main()
