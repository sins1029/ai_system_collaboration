from __future__ import annotations

import argparse
import contextlib
import io
import logging
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

import pandas as pd

from sustaincluster_imitation.bc_policy import BCPolicy
from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_imitation.expert_collector import ExpertRuntimeConfig
from sustaincluster_imitation.feature_encoder import (
    SemanticActionSpace,
    SustainClusterFeatureEncoder,
)
from sustaincluster_imitation.forecast_baseline import HistoricalArrivalForecaster
from sustaincluster_imitation.paths import (
    resolve_bc_checkpoint,
    resolve_sustaincluster_root,
)
from sustaincluster_mpc import (
    HorizonStateAdapter,
    RollingHorizonOptimizer,
    SustainClusterActionAdapter,
)


LOGGER = logging.getLogger("sustaincluster_demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 SustainCluster 最小闭环演示",
        add_help=False,
    )
    parser._optionals.title = "选项"
    parser.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    parser.add_argument(
        "--policy", choices=("mpc", "bc"), default="mpc", help="调度策略"
    )
    parser.add_argument("--steps", type=int, default=5, help="闭环步数，范围 5–20")
    parser.add_argument("--seed", type=int, default=9191, help="环境随机种子")
    parser.add_argument(
        "--start-time",
        default="2023-08-01T05:00:00Z",
        help="UTC 起始时间",
    )
    parser.add_argument("--sustaincluster-root", type=Path, help="只读参考仓库目录")
    parser.add_argument("--checkpoint", type=Path, help="BC 检查点路径")
    return parser.parse_args()


def resources_within_bounds(env: Any) -> bool:
    """检查所有数据中心的可用资源是否仍位于物理边界内。"""
    tolerance = 1e-7
    return all(
        -tolerance <= value <= total + tolerance
        for dc in env.cluster_manager.datacenters.values()
        for value, total in (
            (dc.available_cores, dc.total_cores),
            (dc.available_gpus, dc.total_gpus),
            (dc.available_mem, dc.total_mem_GB),
        )
    )


def run_demo(args: argparse.Namespace) -> int:
    if not 5 <= args.steps <= 20:
        raise ValueError("--steps 必须在 5 到 20 之间")
    external_root = resolve_sustaincluster_root(args.sustaincluster_root)
    runtime = ExpertRuntimeConfig.from_yaml(
        PROJECT_ROOT / "configs" / "sustaincluster_mpc" / "h4_expert.yaml"
    )
    upstream_output = io.StringIO()
    with contextlib.redirect_stdout(upstream_output):
        env = build_sustaincluster_env(
            external_root,
            pd.Timestamp(args.start_time),
            args.steps,
            allow_defer=runtime.allow_defer,
            initial_seed=args.seed,
        )
    env.reset(seed=args.seed)
    action_adapter = SustainClusterActionAdapter.from_env(env)
    dc_ids = tuple(sorted(action_adapter.mapping.dc_id_to_action))
    state_adapter = HorizonStateAdapter()
    encoder = SustainClusterFeatureEncoder(SemanticActionSpace(dc_ids), horizon=4)
    forecaster = HistoricalArrivalForecaster(4, 16, args.seed)
    optimizer = RollingHorizonOptimizer()
    policy: BCPolicy | None = None
    if args.policy == "bc":
        policy = BCPolicy.load_checkpoint(resolve_bc_checkpoint(args.checkpoint))
        if policy.config.action_dc_ids != dc_ids:
            raise ValueError(
                f"BC 检查点数据中心编号 {policy.config.action_dc_ids} "
                f"与环境 {dc_ids} 不一致"
            )

    decision_ms: list[float] = []
    task_actions = 0
    completed_steps = 0
    solver_failures = 0
    resource_overflows = 0
    LOGGER.info("环境加载成功：%s", external_root)
    LOGGER.info(
        "策略=%s，数据中心=%d，计划步数=%d",
        args.policy.upper(),
        len(dc_ids),
        args.steps,
    )
    try:
        for step in range(args.steps):
            action_adapter.assert_matches_env(env)
            state = state_adapter.build_horizon_state(env, 4, "no_future_arrivals")
            started = time.perf_counter()
            if args.policy == "mpc":
                result = optimizer.solve(
                    state, runtime.optimizer_config(), action_adapter
                )
                if not result.feasible:
                    solver_failures += 1
                    raise RuntimeError(
                        f"第 {step + 1} 步 MPC 求解失败："
                        f"{result.status}，{result.message}"
                    )
                actions = list(result.environment_actions)
            else:
                assert policy is not None
                forecaster.observe(state.current.tasks)
                forecast = forecaster.predict(state.timestep_minutes)
                state = forecaster.apply(state, forecast)
                encoded = encoder.encode(state, forecast.uncertainties)
                decisions = policy.predict(encoded)
                actions = action_adapter.encode_assignments(
                    state.current.tasks, decisions
                )
            decision_ms.append((time.perf_counter() - started) * 1000.0)
            action_adapter.validate_actions(state.current.tasks, actions)
            task_actions += len(actions)
            before_resources = resources_within_bounds(env)
            _, _, terminated, truncated, _ = env.step(actions)
            after_resources = resources_within_bounds(env)
            resource_overflows += int(
                not before_resources or not after_resources
            )
            completed_steps += 1
            LOGGER.info(
                "完成第 %d 步：待决策任务=%d，动作=%d",
                step + 1,
                len(state.current.tasks),
                len(actions),
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    LOGGER.info("演示完成：闭环步数=%d，任务动作=%d", completed_steps, task_actions)
    LOGGER.info(
        "平均决策时延=%.3f ms，求解失败=%d，非法动作=0，资源越界=%d",
        sum(decision_ms) / max(1, len(decision_ms)),
        solver_failures,
        resource_overflows,
    )
    return 0


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        raise SystemExit(run_demo(parse_args()))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.error("演示失败：%s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
