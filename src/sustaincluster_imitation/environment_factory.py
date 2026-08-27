from __future__ import annotations

import copy
import os
from pathlib import Path

import pandas as pd
import yaml

from sustaincluster_imitation.paths import prepare_sustaincluster_imports


def build_sustaincluster_env(
    repo: Path | None,
    start_time: pd.Timestamp,
    episode_steps: int,
    *,
    allow_defer: bool = True,
    strategy: str = "manual_rl",
    initial_seed: int = 123,
):
    """构建 SustainCluster 环境，并在返回前恢复调用方工作目录。"""
    repo = prepare_sustaincluster_imports(repo)
    from envs.task_scheduling_env import TaskSchedulingEnv
    from rewards.predefined.composite_reward import CompositeReward
    from simulation.cluster_manager import DatacenterClusterManager

    previous_cwd = Path.cwd()
    try:
        os.chdir(repo)
        with (repo / "configs/env/sim_config.yaml").open(encoding="utf-8") as stream:
            sim_config = copy.deepcopy(yaml.safe_load(stream)["simulation"])
        with (repo / "configs/env/datacenters.yaml").open(encoding="utf-8") as stream:
            dc_config = copy.deepcopy(yaml.safe_load(stream)["datacenters"])
        with (repo / "configs/env/reward_config.yaml").open(encoding="utf-8") as stream:
            reward_config = copy.deepcopy(yaml.safe_load(stream)["reward"])
        workload_path = Path(sim_config["workload_path"])
        if not workload_path.is_absolute():
            workload_path = (repo / workload_path).resolve()
        if not workload_path.is_file():
            raise FileNotFoundError(f"未找到 SustainCluster 工作负载文件：{workload_path}")
        sim_config["workload_path"] = str(workload_path)
        start_time = pd.Timestamp(start_time)
        if start_time.tzinfo is None:
            start_time = start_time.tz_localize("UTC")
        else:
            start_time = start_time.tz_convert("UTC")
        sim_config.update(
            {
                "year": int(start_time.year),
                "month": int(start_time.month),
                "init_day": int(start_time.day),
                "init_hour": int(start_time.hour),
                "duration_days": episode_steps / 96.0,
                "single_action_mode": False,
                "disable_defer_action": not allow_defer,
                "use_tensorboard": False,
                "strategy": strategy,
            }
        )
        cluster = DatacenterClusterManager(
            config_list=dc_config,
            simulation_year=int(start_time.year),
            init_day=int(start_time.dayofyear - 1),
            init_hour=int(start_time.hour),
            strategy=strategy,
            tasks_file_path=sim_config["workload_path"],
            shuffle_datacenter_order=bool(sim_config["shuffle_datacenters"]),
            cloud_provider=sim_config["cloud_provider"],
            logger=None,
        )
        reward = CompositeReward(
            components=reward_config["components"],
            normalize=reward_config.get("normalize", False),
            freeze_stats_after_steps=reward_config.get("freeze_stats_after_steps"),
        )
        return TaskSchedulingEnv(
            cluster_manager=cluster,
            start_time=start_time,
            end_time=start_time + pd.Timedelta(minutes=episode_steps * 15),
            reward_fn=reward,
            writer=None,
            sim_config=sim_config,
            initial_seed_for_resets=initial_seed,
        )
    finally:
        os.chdir(previous_cwd)
