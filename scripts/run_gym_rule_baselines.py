from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datacenter_env.gym.adapters import RuleBasedGymPolicy  # noqa: E402
from datacenter_env.tasking import build_task_scheduler  # noqa: E402
from experiments.gymnasium_env import build_standard_gym_environment  # noqa: E402


SCHEDULERS = (
    "fifo_immediate",
    "earliest_deadline_first",
    "energy_aware_deferral",
)


def main() -> None:
    for scheduler_name in SCHEDULERS:
        env = build_standard_gym_environment(ROOT, scheduler_name=scheduler_name)
        policy = RuleBasedGymPolicy(
            env, build_task_scheduler(env.config.system_config)
        )
        observation, info = env.reset(seed=2026)
        policy.reset(2026)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy.action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        metrics = info["final_metrics"]
        plan_mismatches = sum(
            bool(trace["scheduler_planned_start_ids"])
            and set(trace["scheduler_planned_start_ids"])
            != set(trace["actual_started_task_ids"])
            for trace in env.simulation_traces
        )
        print(f"[{scheduler_name}]")
        print(f"  episode_reward: {info['episode_summary']['episode_reward']:.6f}")
        print(f"  planned_executed_mismatches: {plan_mismatches}")
        for key in (
            "tasks_completed",
            "tasks_unfinished",
            "sla_violation_count",
            "energy_cost",
            "carbon_kg",
            "total_grid_energy_kwh",
            "temperature_violation_count",
        ):
            print(f"  {key}: {metrics[key]:.6f}")
        env.close()


if __name__ == "__main__":
    main()
