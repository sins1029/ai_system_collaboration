from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import tempfile
import unittest

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np

from datacenter_env import (
    DataCenterSystemConfig,
    ExogenousInput,
    GymEnvironmentConfig,
    MaskedRandomPolicy,
    NullRunStore,
    ObservationConfig,
    RewardConfig,
    SQLiteRunStore,
    SingleCenterTaskSchedulingEnv,
    TaskArrivalBatch,
    TaskSpec,
    register_gym_environments,
)
from datacenter_env.contracts import ResourceAvailability, TaskStatus, TaskView
from datacenter_env.exceptions import InputValidationError
from datacenter_env.gym.adapters import RuleBasedGymPolicy
from datacenter_env.gym.masks import task_action_mask
from datacenter_env.tasking import build_task_scheduler
from experiments.trace_equivalence import compare_scheduler_traces
from models.config_loader import load_simple_yaml


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 1, 1)


class TaskProvider:
    def __init__(self, tasks: tuple[TaskSpec, ...]):
        self.tasks = tasks

    def arrivals_at(self, timestamp: datetime) -> TaskArrivalBatch:
        return TaskArrivalBatch(
            timestamp,
            tuple(task for task in self.tasks if task.arrival_time == timestamp),
        )


def task(
    task_id: str,
    *,
    arrival_step: int = 0,
    duration: int = 1,
    cpu: float = 20,
    deadline_steps: int = 4,
    deferrable: bool = True,
    priority: int = 0,
) -> TaskSpec:
    arrival = NOW + timedelta(minutes=15 * arrival_step)
    return TaskSpec(
        task_id,
        arrival,
        duration,
        cpu,
        0,
        10,
        arrival + timedelta(minutes=15 * deadline_steps),
        priority,
        deferrable,
    )


def signals(count: int = 3) -> tuple[ExogenousInput, ...]:
    return tuple(
        ExogenousInput(
            NOW + timedelta(minutes=15 * index),
            0,
            0.5 + 0.05 * index,
            0.4,
            25,
            10,
        )
        for index in range(count)
    )


def setup(
    tasks: tuple[TaskSpec, ...],
    *,
    max_steps: int = 3,
    policy: str = "penalize_and_noop",
    scheduler: str = "fifo_immediate",
    store_factory=None,
    reward_overrides: dict | None = None,
    provider_counter: dict[str, int] | None = None,
):
    system_config = DataCenterSystemConfig.from_dict(
        load_simple_yaml(ROOT / "configs" / "datacenter.yaml"),
        load_simple_yaml(ROOT / "configs" / "optimization.yaml"),
        workload={"mode": "task_queue"},
        capacity={
            "total_cpu_cores": 100,
            "total_gpu_units": 10,
            "total_memory_gb": 200,
        },
        tasks={"unschedulable_policy": "record"},
        scheduler={"name": scheduler, "forecast_steps": 3},
    )
    reward = RewardConfig(**(reward_overrides or {}))
    config = GymEnvironmentConfig(
        system_config,
        invalid_action_policy=policy,
        max_simulation_steps=max_steps,
        observation=ObservationConfig(max_queue_length=20),
        reward=reward,
    )
    episode_signals = signals(max_steps)

    def signal_factory():
        if provider_counter is not None:
            provider_counter["signals"] = provider_counter.get("signals", 0) + 1
        return tuple(episode_signals)

    def task_factory():
        if provider_counter is not None:
            provider_counter["tasks"] = provider_counter.get("tasks", 0) + 1
        return TaskProvider(tuple(tasks))

    env = SingleCenterTaskSchedulingEnv(
        config,
        signal_factory,
        task_factory,
        store_factory=store_factory or NullRunStore,
    )
    return env, config, signal_factory, task_factory


def finish_with_legal_actions(env, seed: int = 1):
    observation, info = env.reset(seed=seed)
    terminated = truncated = False
    rewards = []
    while not (terminated or truncated):
        legal = np.flatnonzero(observation["action_mask"])
        action = int(legal[-1])
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
    return observation, info, rewards


class GymnasiumEnvironmentTest(unittest.TestCase):
    def test_reset_step_spaces_and_api_shapes(self) -> None:
        env, *_ = setup((task("a"),), max_steps=1)
        reset_result = env.reset(seed=3)
        self.assertEqual(len(reset_result), 2)
        observation, _ = reset_result
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(env.action_space, gym.spaces.Discrete(2))
        step_result = env.step(1)
        self.assertEqual(len(step_result), 5)
        self.assertTrue(env.observation_space.contains(step_result[0]))
        self.assertFalse(step_result[2])
        self.assertTrue(step_result[3])
        env.close()

    def test_action_masks_cover_flexible_forced_latest_and_infeasible(self) -> None:
        flexible = task("flexible")
        forced = task("forced", deferrable=False)
        latest = task("latest", deadline_steps=1)
        for spec, expected, available in (
            (flexible, (1, 1), ResourceAvailability(100, 10, 200)),
            (forced, (0, 1), ResourceAvailability(100, 10, 200)),
            (latest, (0, 1), ResourceAvailability(100, 10, 200)),
            (flexible, (1, 0), ResourceAvailability(0, 10, 200)),
        ):
            view = TaskView(
                spec,
                TaskStatus.WAITING,
                spec.duration_steps,
                0,
                0,
                spec.deadline_time - timedelta(minutes=15 * spec.duration_steps),
            )
            self.assertEqual(
                tuple(task_action_mask(view, NOW, available)), expected
            )

    def test_multiple_decisions_advance_one_physical_step_and_charge_once(self) -> None:
        env, *_ = setup(
            (task("a", deadline_steps=2), task("b", deadline_steps=3)),
            max_steps=1,
        )
        observation, _ = env.reset(seed=1)
        observation, first_reward, _, first_truncated, first_info = env.step(1)
        self.assertFalse(first_info["physical_step_advanced"])
        self.assertFalse(first_truncated)
        self.assertEqual(first_info["reward_components"]["energy_cost"], 0.0)
        _, _, _, truncated, second_info = env.step(1)
        self.assertTrue(truncated)
        self.assertTrue(second_info["physical_step_advanced"])
        self.assertEqual(second_info["episode_summary"]["simulation_steps"], 1)
        self.assertEqual(len(env._system.aggregator._steps), 1)
        self.assertEqual(first_reward, 0.0)
        env.close()

    def test_empty_steps_auto_advance_to_next_candidate(self) -> None:
        env, *_ = setup((task("later", arrival_step=1),), max_steps=2)
        observation, info = env.reset(seed=1)
        self.assertEqual(info["auto_advanced_simulation_steps"], 1)
        self.assertEqual(env.simulation_step_index, 1)
        self.assertEqual(info["candidate_task_id"], "later")
        self.assertTrue(env.observation_space.contains(observation))
        env.close()

    def test_invalid_action_penalize_noop_raise_and_projection(self) -> None:
        forced = (task("forced", deferrable=False),)
        penalized, *_ = setup(forced, max_steps=1)
        _, _ = penalized.reset(seed=1)
        _, reward, _, _, info = penalized.step(0)
        self.assertTrue(info["invalid_action"])
        self.assertLess(info["reward_components"]["invalid_action"], 0)
        self.assertNotIn("forced", info["started_task_ids"])
        self.assertLess(reward, 0)
        penalized.close()

        raising, *_ = setup(forced, max_steps=1, policy="raise")
        raising.reset(seed=1)
        with self.assertRaises(InputValidationError):
            raising.step(0)
        raising.close()

        projected, *_ = setup(forced, max_steps=1, policy="project_to_legal")
        projected.reset(seed=1)
        _, _, _, _, projected_info = projected.step(0)
        self.assertTrue(projected_info["invalid_action"])
        self.assertIn("forced", projected_info["started_task_ids"])
        projected.close()

    def test_reward_components_sum_and_terminal_penalty_occurs_once(self) -> None:
        env, *_ = setup((task("long", duration=4),), max_steps=1)
        _, info, rewards = finish_with_legal_actions(env)
        components = info["reward_components"]
        subtotal = sum(value for key, value in components.items() if key != "total")
        self.assertAlmostEqual(subtotal, components["total"])
        self.assertEqual(components["terminal_unfinished"], -2.0)
        self.assertAlmostEqual(sum(rewards), info["episode_summary"]["episode_reward"])
        with self.assertRaises(RuntimeError):
            env.step(1)
        env.close()

    def test_reward_sensitivity_switches_expected_components(self) -> None:
        late = TaskSpec("late", NOW, 1, 20, 0, 10, NOW + timedelta(minutes=10))
        low, *_ = setup(
            (late,), max_steps=1, reward_overrides={"sla_violation_weight": 1.0}
        )
        _, low_info, low_rewards = finish_with_legal_actions(low)
        high, *_ = setup(
            (late,), max_steps=1, reward_overrides={"sla_violation_weight": 10.0}
        )
        _, high_info, high_rewards = finish_with_legal_actions(high)
        self.assertLess(sum(high_rewards), sum(low_rewards))
        self.assertLess(
            high_info["reward_components"]["sla_violation"],
            low_info["reward_components"]["sla_violation"],
        )
        no_cost, *_ = setup(
            (task("a"),),
            max_steps=1,
            reward_overrides={"energy_cost_weight": 0.0, "carbon_weight": 0.0},
        )
        _, no_cost_info, _ = finish_with_legal_actions(no_cost)
        self.assertEqual(no_cost_info["reward_components"]["energy_cost"], 0.0)
        self.assertEqual(no_cost_info["reward_components"]["carbon"], 0.0)
        low.close()
        high.close()
        no_cost.close()

    def test_reset_rebuilds_providers_and_is_reproducible(self) -> None:
        counter: dict[str, int] = {}
        env, *_ = setup((task("a"),), max_steps=1, provider_counter=counter)
        first, _ = env.reset(seed=9)
        env.step(1)
        second, _ = env.reset(seed=9)
        self.assertEqual(counter, {"signals": 2, "tasks": 2})
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        env.close()

    def test_observation_is_fixed_numeric_and_has_no_hidden_domain_fields(self) -> None:
        env, *_ = setup((task("a"),), max_steps=1)
        observation, _ = env.reset(seed=1)
        self.assertEqual(
            {key: value.shape for key, value in observation.items()},
            {
                "global": (4,),
                "candidate_task": (10,),
                "resources": (6,),
                "queue_summary": (8,),
                "environment": (4,),
                "thermal": (4,),
                "action_mask": (2,),
            },
        )
        serialized = repr(observation).lower()
        self.assertNotIn("true_temperature", serialized)
        self.assertNotIn("taskspec", serialized)
        env.close()

    def test_masked_random_policy_is_legal_and_seed_reproducible(self) -> None:
        observation = {"action_mask": np.asarray([1, 1], dtype=np.int8)}
        first = MaskedRandomPolicy(11)
        second = MaskedRandomPolicy(11)
        left = [first.action(observation) for _ in range(20)]
        right = [second.action(observation) for _ in range(20)]
        self.assertEqual(left, right)
        self.assertTrue(all(action in {0, 1} for action in left))
        forced = {"action_mask": np.asarray([0, 1], dtype=np.int8)}
        self.assertEqual(first.action(forced), 1)

    def test_rule_adapters_complete_for_all_schedulers(self) -> None:
        for scheduler_name in (
            "fifo_immediate",
            "earliest_deadline_first",
            "energy_aware_deferral",
        ):
            env, *_ = setup(
                (task("a"), task("b", deadline_steps=2)),
                max_steps=2,
                scheduler=scheduler_name,
            )
            policy = RuleBasedGymPolicy(
                env, build_task_scheduler(env.config.system_config)
            )
            observation, info = env.reset(seed=1)
            policy.reset(1)
            terminated = truncated = False
            while not (terminated or truncated):
                action = policy.action(observation, info)
                observation, _, terminated, truncated, info = env.step(action)
            self.assertEqual(info["final_metrics"]["tasks_completed"], 2.0)
            env.close()

    def test_gym_checker_and_registration(self) -> None:
        env, config, signal_factory, task_factory = setup(
            (task("a"), task("b", arrival_step=1)), max_steps=2
        )
        check_env(env, skip_render_check=True)
        env.close()
        register_gym_environments()
        register_gym_environments()
        made = gym.make(
            "DataCenterTaskScheduling-v0",
            config=config,
            signal_provider_factory=signal_factory,
            task_provider_factory=task_factory,
        )
        observation, _ = made.reset(seed=1)
        self.assertTrue(made.observation_space.contains(observation))
        made.close()

    def test_sync_vector_env_instances_are_isolated(self) -> None:
        vector = gym.vector.SyncVectorEnv(
            [
                lambda: setup((task("a"),), max_steps=2)[0],
                lambda: setup((task("a"),), max_steps=2)[0],
            ]
        )
        observation, _ = vector.reset(seed=[1, 2])
        self.assertEqual(observation["global"].shape, (2, 4))
        actions = np.argmax(observation["action_mask"], axis=1)
        next_observation, rewards, *_ = vector.step(actions)
        self.assertEqual(next_observation["candidate_task"].shape, (2, 10))
        self.assertEqual(rewards.shape, (2,))
        vector.close()

    def test_sqlite_round_trip_records_gym_metadata_and_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gym.sqlite"
            env, *_ = setup(
                (task("a"),),
                max_steps=1,
                store_factory=lambda: SQLiteRunStore(path, commit_each_step=True),
            )
            _, info, _ = finish_with_legal_actions(env)
            env.close()
            connection = sqlite3.connect(path)
            run = connection.execute(
                """
                SELECT interface_type, environment_id, invalid_action_policy,
                       candidate_order, episode_seed, status
                FROM experiment_runs
                """
            ).fetchone()
            decision = connection.execute(
                """
                SELECT decision_step_index, simulation_step_index, candidate_task_id,
                       action, action_legal, action_mask_json, total_reward
                FROM run_agent_decisions
                """
            ).fetchone()
            episode = connection.execute(
                """
                SELECT episode_reward, decision_steps, simulation_steps,
                       invalid_actions, terminated, truncated
                FROM run_gym_episode_summaries
                """
            ).fetchone()
            version = connection.execute(
                "SELECT MAX(version) FROM schema_versions"
            ).fetchone()[0]
            connection.close()
            self.assertEqual(
                run,
                (
                    "gymnasium",
                    "DataCenterTaskScheduling-v0",
                    "penalize_and_noop",
                    "edf",
                    1,
                    "completed",
                ),
            )
            self.assertEqual(decision[:5], (0, 0, "a", 1, 1))
            self.assertEqual(decision[5], "[1, 1]")
            self.assertAlmostEqual(decision[6], info["reward_components"]["total"])
            self.assertAlmostEqual(episode[0], info["episode_summary"]["episode_reward"])
            self.assertEqual(episode[1:], (1, 1, 0, 0, 1))
            self.assertEqual(version, 4)

    def test_forced_resource_blocked_task_does_not_block_later_small_task(self) -> None:
        tasks = (
            task("running", duration=2, cpu=70, deadline_steps=4),
            task(
                "blocked-forced",
                arrival_step=1,
                cpu=50,
                deadline_steps=1,
                deferrable=False,
            ),
            task("small", arrival_step=1, cpu=20, deadline_steps=3),
        )
        env, *_ = setup(tasks, max_steps=2)
        observation, _ = env.reset(seed=1)
        self.assertEqual(env.current_candidate.spec.task_id, "running")
        observation, _, _, _, info = env.step(1)
        self.assertEqual(env.current_candidate.spec.task_id, "small")
        self.assertEqual(info["auto_skipped_candidate_ids"], ("blocked-forced",))
        self.assertEqual(
            info["forced_resource_blocked_task_ids"], ("blocked-forced",)
        )
        self.assertNotEqual(tuple(observation["action_mask"]), (0, 0))
        _, _, _, truncated, final_info = env.step(1)
        self.assertTrue(truncated)
        self.assertIn("small", final_info["started_task_ids"])
        outcomes = {
            item.task_id: item for item in env._system.environment.task_outcomes()
        }
        blocked = outcomes["blocked-forced"]
        self.assertEqual(blocked.wait_steps, 1)
        self.assertEqual(blocked.deferral_count, 0)
        self.assertEqual(blocked.resource_blocked_count, 1)
        self.assertEqual(final_info["episode_summary"]["decision_steps"], 2)
        env.close()

    def test_reward_configuration_does_not_change_domain_trajectory(self) -> None:
        tasks = (task("a", duration=2), task("b", deadline_steps=2))
        normal, *_ = setup(tasks, max_steps=2)
        _, normal_info, normal_rewards = finish_with_legal_actions(normal, seed=5)
        changed, *_ = setup(
            tasks,
            max_steps=2,
            reward_overrides={
                "completion_weight": 8.0,
                "energy_cost_weight": 0.0,
                "carbon_weight": 0.0,
                "sla_violation_weight": 20.0,
            },
        )
        _, changed_info, changed_rewards = finish_with_legal_actions(changed, seed=5)
        self.assertEqual(normal.simulation_traces, changed.simulation_traces)
        self.assertEqual(normal_info["final_metrics"], changed_info["final_metrics"])
        self.assertNotEqual(sum(normal_rewards), sum(changed_rewards))
        normal.close()
        changed.close()

    def test_sqlite_and_null_store_do_not_change_domain_trajectory(self) -> None:
        tasks = (task("a", duration=2), task("b", deadline_steps=2))
        null_env, *_ = setup(tasks, max_steps=2)
        _, null_info, null_rewards = finish_with_legal_actions(null_env, seed=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "isolation.sqlite"
            sqlite_env, *_ = setup(
                tasks,
                max_steps=2,
                store_factory=lambda: SQLiteRunStore(path, commit_each_step=True),
            )
            _, sqlite_info, sqlite_rewards = finish_with_legal_actions(
                sqlite_env, seed=8
            )
            self.assertEqual(null_env.simulation_traces, sqlite_env.simulation_traces)
            self.assertEqual(null_info["final_metrics"], sqlite_info["final_metrics"])
            self.assertEqual(null_rewards, sqlite_rewards)
            sqlite_env.close()
        null_env.close()

    def test_native_and_gym_rule_trajectories_are_identical(self) -> None:
        for scheduler in ("fifo", "edf", "energy_aware"):
            comparison = compare_scheduler_traces(ROOT, scheduler)
            self.assertEqual(comparison["mismatch_count"], 0)
            self.assertEqual(comparison["event_mismatch_count"], 0)
            self.assertEqual(comparison["planned_executed_mismatch_count"], 0)
            self.assertEqual(comparison["native_hash"], comparison["gym_hash"])
