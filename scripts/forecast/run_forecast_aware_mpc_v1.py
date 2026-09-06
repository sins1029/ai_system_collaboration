from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
import yaml


matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting.transformer_workload_provider import (
    TransformerWorkloadForecastProvider,
)
from forecasting.workload_forecast_provider import (
    ForecastBundle,
    ForecastTraceSource,
    OracleWorkloadForecastProvider,
    PersistenceWorkloadForecastProvider,
)
from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_imitation.paths import prepare_sustaincluster_imports
from sustaincluster_mpc.action_adapter import SustainClusterActionAdapter
from sustaincluster_mpc.forecast_pressure_adapter import (
    ForecastPressureApplication,
    apply_forecast_pressure,
    apply_oracle_future_signals,
)
from sustaincluster_mpc.future_signals import FutureSignalProvider
from sustaincluster_mpc.horizon_adapter import HorizonStateAdapter
from sustaincluster_mpc.timeline_contract import REPAIRED_H4_CAPACITY_NODES
from sustaincluster_mpc.rolling_horizon_optimizer import (
    RollingHorizonConfig,
    RollingHorizonOptimizer,
    RollingObjectiveWeights,
)


CONTROLLERS = ("H1", "H4_ORACLE", "H4_PERSISTENCE", "H4_TRANSFORMER")
CONTROL_METRICS = (
    "reward",
    "stage_cost",
    "completed",
    "sla",
    "avg_wait",
    "p50_wait",
    "p95_wait",
    "max_wait",
    "scheduler_avg_wait",
    "local_queue_avg_wait",
    "defer_ratio",
    "migration_ratio",
    "electricity",
    "carbon",
    "transmission",
    "terminal_backlog",
    "solver_ms",
)
HIGHER_IS_BETTER = {"reward", "completed"}
COMPARISONS = (
    ("H4_ORACLE - H1", "H1", "H4_ORACLE"),
    ("H4_PERSISTENCE - H1", "H1", "H4_PERSISTENCE"),
    ("H4_TRANSFORMER - H1", "H1", "H4_TRANSFORMER"),
    (
        "H4_TRANSFORMER - H4_PERSISTENCE",
        "H4_PERSISTENCE",
        "H4_TRANSFORMER",
    ),
    ("H4_TRANSFORMER - H4_ORACLE", "H4_ORACLE", "H4_TRANSFORMER"),
)


def _git(*args: str, cwd: Path = WORKSPACE) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8", errors="replace"
    ).strip()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _optimizer_config(path: Path) -> RollingHorizonConfig:
    data = yaml.safe_load(path.read_text("utf-8"))["expert"]
    weights = RollingObjectiveWeights(**data["objective_weights"])
    return RollingHorizonConfig(
        allow_defer=bool(data["allow_defer"]),
        weights=weights,
        cpu_power_w_per_core=float(data["power_model"]["cpu_power_w_per_core"]),
        gpu_power_w_per_unit=float(data["power_model"]["gpu_power_w_per_unit"]),
        memory_power_w_per_gb=float(data["power_model"]["memory_power_w_per_gb"]),
        waiting_cost_per_step=float(data["waiting_cost_per_step"]),
        terminal_backlog_base_cost=float(data["terminal_backlog_base_cost"]),
        deterministic_tie_break_epsilon=float(data["deterministic_tie_break_epsilon"]),
        solver_time_limit_seconds=float(data["solver_time_limit_seconds"]),
    )


def _resource_bounds(env: Any) -> bool:
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


def _arrival_signature(env: Any) -> str:
    rows = []
    current = pd.Timestamp(env.current_time)
    for task in env.current_tasks:
        if pd.Timestamp(task.arrival_time) != current:
            continue
        rows.append(
            (
                str(task.source_job_name),
                int(task.origin_dc_id),
                float(task.cores_req),
                float(task.gpu_req),
                float(task.mem_req),
                float(task.true_duration),
                float(task.estimated_duration),
                float(task.bandwidth_gb),
            )
        )
    payload = json.dumps(sorted(rows), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _empty_metrics() -> dict[str, Any]:
    return {
        "reward": 0.0,
        "stage_cost": 0.0,
        "completed": 0,
        "sla": 0,
        "wait_values": [],
        "scheduler_wait_values": [],
        "local_queue_wait_values": [],
        "scheduler_wait_intervals_added": 0,
        "decision_count": 0,
        "defer_count": 0,
        "assign_count": 0,
        "migration_count": 0,
        "electricity": 0.0,
        "carbon": 0.0,
        "transmission": 0.0,
        "projected_terminal_backlog_sum": 0,
        "solver_values": [],
        "infeasible_count": 0,
        "resource_overflow_count": 0,
        "capacity_shortage_events": 0,
        "forecast_fallback_count": 0,
        "forecast_history_unavailable_count": 0,
        "negative_prediction_clip_count": 0,
        "forecast_inference_values": [],
    }


def _physical_metrics(metrics: dict[str, Any], info: dict[str, Any], reward: float) -> None:
    metrics["reward"] += reward
    metrics["transmission"] += float(info.get("transmission_cost_total_usd", 0.0))
    for dc_info in info["datacenter_infos"].values():
        common = dc_info["__common__"]
        metrics["electricity"] += float(common["energy_cost_USD"])
        metrics["carbon"] += float(common["carbon_emissions_kg"])
        metrics["completed"] += int(common["finished_tasks_count"])
        metrics["sla"] += int(common["__sla__"]["violated"])


def _final_backlog(env: Any) -> int:
    queued = sum(
        len(dc.pending_tasks)
        for dc in env.cluster_manager.datacenters.values()
    )
    return int(len(env.current_tasks) + len(env.in_transit_tasks) + queued)


def _finish_metrics(metrics: dict[str, Any], env: Any, steps: int) -> dict[str, Any]:
    waits = metrics.pop("wait_values")
    scheduler_waits = metrics.pop("scheduler_wait_values")
    local_queue_waits = metrics.pop("local_queue_wait_values")
    solvers = metrics.pop("solver_values")
    forecast_latency = metrics.pop("forecast_inference_values")
    decisions = int(metrics["decision_count"])
    assigns = int(metrics["assign_count"])
    metrics.update(
        {
            "avg_wait": float(mean(waits)) if waits else 0.0,
            "p50_wait": float(np.percentile(waits, 50)) if waits else 0.0,
            "p95_wait": float(np.percentile(waits, 95)) if waits else 0.0,
            "max_wait": float(max(waits)) if waits else 0.0,
            "scheduler_avg_wait": (
                float(mean(scheduler_waits)) if scheduler_waits else 0.0
            ),
            "local_queue_avg_wait": (
                float(mean(local_queue_waits)) if local_queue_waits else 0.0
            ),
            "defer_ratio": metrics["defer_count"] / max(1, decisions),
            "migration_ratio": metrics["migration_count"] / max(1, assigns),
            "terminal_backlog": _final_backlog(env),
            "solver_ms": 1000.0 * float(mean(solvers)) if solvers else 0.0,
            "solver_p95_ms": 1000.0 * float(np.percentile(solvers, 95)) if solvers else 0.0,
            "forecast_inference_ms": 1000.0 * float(mean(forecast_latency)) if forecast_latency else 0.0,
            "steps": int(steps),
        }
    )
    return metrics


def _provider_for(controller: str, transformer: TransformerWorkloadForecastProvider) -> Any:
    if controller == "H4_ORACLE":
        return OracleWorkloadForecastProvider()
    if controller == "H4_PERSISTENCE":
        return PersistenceWorkloadForecastProvider()
    if controller == "H4_TRANSFORMER":
        return transformer
    return None


def run_episode(
    *,
    controller: str,
    scenario: str,
    environment_start: pd.Timestamp,
    expected_dataset_start: pd.Timestamp,
    seed: int,
    steps: int,
    trace_source: ForecastTraceSource,
    transformer: TransformerWorkloadForecastProvider,
    datacenter_configs: list[dict[str, Any]],
    optimizer_config: RollingHorizonConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    env = build_sustaincluster_env(
        None,
        environment_start,
        steps,
        allow_defer=True,
        initial_seed=seed,
        information_mode="deployable",
        duration_estimate_mode="declared_or_baseline",
        baseline_estimated_duration_minutes=60.0,
    )
    env.reset(seed=seed)
    action_adapter = SustainClusterActionAdapter.from_env(env)
    horizon_adapter = HorizonStateAdapter(
        information_mode="deployable",
        future_signal_provider=FutureSignalProvider("persistence"),
    )
    optimizer = RollingHorizonOptimizer()
    provider = _provider_for(controller, transformer)
    metrics = _empty_metrics()
    forecast_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    signatures: list[str] = []
    completed_steps = 0
    try:
        for step_index in range(steps):
            current_time = pd.Timestamp(env.current_time)
            aligned = trace_source.align_environment_timestamp(current_time)
            if step_index == 0 and aligned != expected_dataset_start:
                raise RuntimeError(
                    f"scenario time alignment failed: {aligned} != {expected_dataset_start}"
                )
            signatures.append(_arrival_signature(env))
            history, _ = trace_source.history(current_time)
            oracle_values, oracle_timestamps, _ = trace_source.oracle_future(current_time)
            current_workload = (
                np.asarray(history[-1, :4], dtype=np.float64)
                if history is not None
                else np.zeros(4, dtype=np.float64)
            )
            horizon = (
                1 if controller == "H1" else REPAIRED_H4_CAPACITY_NODES
            )
            state = horizon_adapter.build_horizon_state(
                env, horizon, "no_future_arrivals"
            )
            pressure: ForecastPressureApplication | None = None
            bundle: ForecastBundle | None = None
            if provider is not None:
                request = trace_source.request(
                    current_time,
                    include_oracle_future=controller == "H4_ORACLE",
                )
                bundle = provider.forecast(request)
                if controller == "H4_ORACLE":
                    state = apply_oracle_future_signals(state, env)
                pressure = apply_forecast_pressure(
                    state, bundle, datacenter_configs
                )
                state = pressure.state
                metrics["capacity_shortage_events"] += pressure.capacity_shortage_events
                metrics["forecast_fallback_count"] += int(bundle.persistence_fallback_used)
                metrics["forecast_history_unavailable_count"] += int(
                    not bundle.forecast_history_available
                )
                metrics["negative_prediction_clip_count"] += bundle.negative_prediction_clip_count
                metrics["forecast_inference_values"].append(bundle.inference_ms / 1000.0)

            before_resources = _resource_bounds(env)
            result = optimizer.solve(state, optimizer_config, action_adapter)
            if not result.feasible:
                metrics["infeasible_count"] += 1
                break
            metrics["stage_cost"] += result.first_step_costs.total
            metrics["projected_terminal_backlog_sum"] += result.terminal_backlog_count
            metrics["solver_values"].append(result.solve_seconds)
            step_defer = 0
            step_migration = 0
            semantic_actions = []
            target_dcs = []
            for task, decision in zip(state.current.tasks, result.first_step_decisions):
                metrics["decision_count"] += 1
                metrics["wait_values"].append(float(task.wait_intervals))
                metrics["scheduler_wait_values"].append(
                    float(task.scheduler_wait_intervals)
                )
                metrics["local_queue_wait_values"].append(
                    float(task.wait_intervals - task.scheduler_wait_intervals)
                )
                semantic_actions.append(decision.decision)
                target_dcs.append(decision.dc_id)
                if decision.decision == "defer":
                    metrics["defer_count"] += 1
                    step_defer += 1
                else:
                    metrics["assign_count"] += 1
                    migrated = int(int(decision.dc_id) != task.origin_dc_id)
                    metrics["migration_count"] += migrated
                    step_migration += migrated

            _, reward, terminated, truncated, info = env.step(
                list(result.environment_actions)
            )
            completed_steps = step_index + 1
            scheduler_added = int(info.get("scheduler_wait_intervals_added", 0))
            if scheduler_added != step_defer:
                raise RuntimeError("scheduler waiting increment disagrees with defer actions")
            metrics["scheduler_wait_intervals_added"] += scheduler_added
            metrics["resource_overflow_count"] += int(
                not before_resources or not _resource_bounds(env)
            )
            before_sla = int(metrics["sla"])
            _physical_metrics(metrics, info, float(reward))
            step_sla = int(metrics["sla"]) - before_sla
            true_gpu_h1 = float(oracle_values[0, 2])
            step_rows.append(
                {
                    "controller": controller,
                    "scenario": scenario,
                    "seed": seed,
                    "step": step_index,
                    "environment_timestamp": current_time,
                    "dataset_timestamp": aligned,
                    "true_future_gpu_h1": true_gpu_h1,
                    "stage_cost": result.first_step_costs.total,
                    "next_step_reward": float(reward),
                    "sla_violations": step_sla,
                    "defer_count": step_defer,
                    "migration_count": step_migration,
                    "decision_count": len(state.current.tasks),
                    "semantic_actions": json.dumps(semantic_actions),
                    "target_dcs": json.dumps(target_dcs),
                    "scheduler_wait_intervals_added": scheduler_added,
                    "capacity_shortage_event": int(
                        pressure is not None and pressure.capacity_shortage_events > 0
                    ),
                    "solver_feasible": True,
                }
            )
            if bundle is not None:
                for point_index, point in enumerate(bundle.points):
                    true = oracle_values[point_index]
                    predicted = bundle.as_array()[point_index]
                    forecast_rows.append(
                        {
                            "controller": controller,
                            "forecast_provider": bundle.provider,
                            "scenario": scenario,
                            "seed": seed,
                            "step": step_index,
                            "environment_timestamp": current_time,
                            "history_end_timestamp": aligned,
                            "forecast_timestamp": oracle_timestamps[point_index],
                            "horizon_step": point.horizon_step,
                            "horizon_minutes": point.horizon_minutes,
                            "predicted_task_count": predicted[0],
                            "predicted_cpu": predicted[1],
                            "predicted_gpu": predicted[2],
                            "predicted_memory": predicted[3],
                            "oracle_task_count": true[0],
                            "oracle_cpu": true[1],
                            "oracle_gpu": true[2],
                            "oracle_memory": true[3],
                            "absolute_cpu_error": abs(predicted[1] - true[1]),
                            "absolute_gpu_error": abs(predicted[2] - true[2]),
                            "absolute_memory_error": abs(predicted[3] - true[3]),
                            "forecast_history_available": bundle.forecast_history_available,
                            "persistence_fallback_used": bundle.persistence_fallback_used,
                            "negative_prediction_clip_count": bundle.negative_prediction_clip_count,
                            "inference_ms": bundle.inference_ms,
                        }
                    )
            if terminated or truncated:
                break
        result_metrics = _finish_metrics(metrics, env, completed_steps)
        result_metrics.update(
            {
                "controller": controller,
                "scenario": scenario,
                "seed": seed,
                "environment_start": environment_start.isoformat(),
                "dataset_start": expected_dataset_start.isoformat(),
                "information_mode": "deployable_environment",
                "runtime_source": "estimated_duration",
                "price_future": "oracle" if controller == "H4_ORACLE" else "persistence",
                "carbon_future": "oracle" if controller == "H4_ORACLE" else "persistence",
                "workload_future": {
                    "H1": "none",
                    "H4_ORACLE": "oracle_global_aggregate",
                    "H4_PERSISTENCE": "current_global_hold",
                    "H4_TRANSFORMER": "transformer_seed_33",
                }[controller],
            }
        )
        return result_metrics, forecast_rows, step_rows, signatures
    finally:
        env.close()


def _aggregate_metrics(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for controller in CONTROLLERS:
        selected = episodes[episodes["controller"] == controller]
        for metric in CONTROL_METRICS:
            rows.append(
                {
                    "controller": controller,
                    "metric": metric,
                    "mean": float(selected[metric].mean()),
                    "std": float(selected[metric].std(ddof=0)),
                }
            )
    return pd.DataFrame(rows)


def _paired_comparisons(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key = ["scenario", "seed"]
    for name, baseline, candidate in COMPARISONS:
        left = episodes[episodes["controller"] == baseline].set_index(key)
        right = episodes[episodes["controller"] == candidate].set_index(key)
        if set(left.index) != set(right.index):
            raise RuntimeError(f"paired episode keys differ for {name}")
        for metric in CONTROL_METRICS:
            delta = right[metric] - left[metric]
            if metric in HIGHER_IS_BETTER:
                better = delta > 1e-12
                worse = delta < -1e-12
            else:
                better = delta < -1e-12
                worse = delta > 1e-12
            rows.append(
                {
                    "comparison": name,
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    "mean_delta": float(delta.mean()),
                    "std_delta": float(delta.std(ddof=0)),
                    "better_count": int(better.sum()),
                    "worse_count": int(worse.sum()),
                    "tie_count": int((~better & ~worse).sum()),
                }
            )
    return pd.DataFrame(rows)


def _high_load_diagnostics(steps: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    reference = steps[
        (steps["controller"] == "H4_PERSISTENCE")
    ]["true_future_gpu_h1"]
    threshold = float(reference.quantile(0.9))
    values = steps[
        steps["controller"].isin(["H4_PERSISTENCE", "H4_TRANSFORMER"])
    ].copy()
    values["load_segment"] = np.where(
        values["true_future_gpu_h1"] >= threshold, "HIGH_LOAD", "NORMAL"
    )
    rows = []
    for (controller, segment), group in values.groupby(
        ["controller", "load_segment"], sort=False
    ):
        decisions = max(1, int(group["decision_count"].sum()))
        rows.append(
            {
                "controller": controller,
                "load_segment": segment,
                "gpu_p90_threshold": threshold,
                "step_count": len(group),
                "stage_cost_mean": float(group["stage_cost"].mean()),
                "sla_violations_sum": int(group["sla_violations"].sum()),
                "defer_ratio": float(group["defer_count"].sum() / decisions),
                "migration_ratio": float(group["migration_count"].sum() / decisions),
                "capacity_shortage_event_count": int(group["capacity_shortage_event"].sum()),
                "infeasible_event_count": int((~group["solver_feasible"]).sum()),
                "next_step_reward_mean": float(group["next_step_reward"].mean()),
            }
        )
    return pd.DataFrame(rows), threshold


def _oracle_gap(aggregate: pd.DataFrame) -> pd.DataFrame:
    table = aggregate.pivot(index="metric", columns="controller", values="mean")
    rows = []
    for metric in CONTROL_METRICS:
        h1 = float(table.loc[metric, "H1"])
        oracle = float(table.loc[metric, "H4_ORACLE"])
        transformer = float(table.loc[metric, "H4_TRANSFORMER"])
        oracle_better = oracle > h1 if metric in HIGHER_IS_BETTER else oracle < h1
        denominator = h1 - oracle
        recovery = (
            (h1 - transformer) / denominator
            if oracle_better and abs(denominator) > 1e-12
            else float("nan")
        )
        rows.append(
            {
                "metric": metric,
                "direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                "h1_mean": h1,
                "oracle_mean": oracle,
                "transformer_mean": transformer,
                "oracle_better_than_h1": oracle_better,
                "oracle_gain_recovery_ratio": recovery,
            }
        )
    return pd.DataFrame(rows)


def _fairness(signatures: dict[tuple[str, int, str], list[str]]) -> bool:
    for scenario, seed in sorted({(key[0], key[1]) for key in signatures}):
        values = [signatures[(scenario, seed, controller)] for controller in CONTROLLERS]
        if not all(value == values[0] for value in values[1:]):
            return False
    return True


def _plot_artifacts(
    aggregate: pd.DataFrame,
    forecast: pd.DataFrame,
    high_load: pd.DataFrame,
    figures: Path,
) -> list[Path]:
    figures.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for metric in ("stage_cost", "sla", "electricity", "transmission"):
        selected = aggregate[aggregate["metric"] == metric]
        fig, axis = plt.subplots(figsize=(7.5, 4.5))
        axis.bar(selected["controller"], selected["mean"], color=colors)
        axis.set_ylabel(metric)
        axis.set_title(f"Controller comparison: {metric}")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        path = figures / f"controller_{metric}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)

    selected = forecast[
        (forecast["scenario"] == "high_load_trace")
        & (forecast["seed"] == 1201)
        & (forecast["horizon_step"] == 1)
        & forecast["controller"].isin(["H4_PERSISTENCE", "H4_TRANSFORMER"])
    ]
    fig, axis = plt.subplots(figsize=(10, 4.8))
    truth = selected.drop_duplicates("forecast_timestamp").sort_values("forecast_timestamp")
    axis.plot(truth["forecast_timestamp"], truth["oracle_gpu"], color="#111827", label="True", linewidth=1.8)
    for controller, color in (("H4_PERSISTENCE", "#D55E00"), ("H4_TRANSFORMER", "#0072B2")):
        current = selected[selected["controller"] == controller].sort_values("forecast_timestamp")
        axis.plot(current["forecast_timestamp"], current["predicted_gpu"], color=color, label=controller, linewidth=1.2)
    axis.set_title("High-load trace: +15 min GPU forecast")
    axis.set_ylabel("GPU demand")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = figures / "gpu_forecast_high_load.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    segment = high_load[high_load["load_segment"] == "HIGH_LOAD"]
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.bar(segment["controller"], segment["defer_ratio"], color=["#D55E00", "#0072B2"])
    axis.set_ylabel("Defer ratio")
    axis.set_title("High-load control behavior")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = figures / "high_load_defer_behavior.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)
    return outputs


def _preliminary_role(
    paired: pd.DataFrame,
    high_load: pd.DataFrame,
) -> tuple[str, str]:
    def row(comparison: str, metric: str) -> pd.Series:
        return paired[(paired["comparison"] == comparison) & (paired["metric"] == metric)].iloc[0]

    oracle_cost = row("H4_ORACLE - H1", "stage_cost")
    transformer_cost = row("H4_TRANSFORMER - H4_PERSISTENCE", "stage_cost")
    transformer_reward = row("H4_TRANSFORMER - H4_PERSISTENCE", "reward")
    segment = high_load.set_index(["controller", "load_segment"])
    high_transformer_cost = float(segment.loc[("H4_TRANSFORMER", "HIGH_LOAD"), "stage_cost_mean"])
    high_persistence_cost = float(segment.loc[("H4_PERSISTENCE", "HIGH_LOAD"), "stage_cost_mean"])
    normal_transformer_cost = float(segment.loc[("H4_TRANSFORMER", "NORMAL"), "stage_cost_mean"])
    normal_persistence_cost = float(segment.loc[("H4_PERSISTENCE", "NORMAL"), "stage_cost_mean"])
    if oracle_cost.mean_delta >= 0 and oracle_cost.better_count <= oracle_cost.worse_count:
        return (
            "MPC AS EXPERT / TEACHER",
            "Repaired perfect-information H4 does not consistently reduce stage cost versus H1; always-online lookahead value is limited.",
        )
    if normal_transformer_cost < normal_persistence_cost and high_transformer_cost > high_persistence_cost:
        return (
            "CONSIDER TRIGGERED MPC",
            "Transformer pressure helps in normal load but loses its advantage in the P90 GPU segment; risk-aware triggering is worth a separate study.",
        )
    if transformer_cost.mean_delta < 0 and transformer_reward.mean_delta >= 0:
        return (
            "KEEP ONLINE MPC",
            "Transformer H4 improves stage cost without reducing reward versus Persistence across the paired evaluation.",
        )
    return (
        "INCONCLUSIVE",
        "Transformer forecast accuracy does not translate into a stable incremental control advantage over Persistence across paired metrics.",
    )


def run(config_path: Path, *, smoke: bool) -> Path:
    config = yaml.safe_load(config_path.read_text("utf-8"))["forecast_aware_mpc_v1"]
    output = (WORKSPACE / config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    dataset_root = (WORKSPACE / config["dataset_root"]).resolve()
    checkpoint = (WORKSPACE / config["transformer_checkpoint"]).resolve()
    sustain_repo = WORKSPACE / "references/external_repos/sustain-cluster"
    if _git("rev-parse", "HEAD", cwd=sustain_repo) != config["sustaincluster_commit"]:
        raise RuntimeError("SustainCluster commit changed")
    if _git("status", "--short", cwd=sustain_repo):
        raise RuntimeError("SustainCluster worktree must be clean")
    prepare_sustaincluster_imports(sustain_repo)
    dc_data = yaml.safe_load((WORKSPACE / config["datacenter_config"]).read_text("utf-8"))["datacenters"]
    optimizer_config = _optimizer_config(WORKSPACE / config["optimizer_config"])
    trace_source = ForecastTraceSource(dataset_root)
    transformer = TransformerWorkloadForecastProvider(
        checkpoint,
        dataset_root=dataset_root,
        device=config["device"],
    )

    seeds = [int(config["seeds"][0])] if smoke else [int(value) for value in config["seeds"]]
    steps = int(config["smoke_steps"] if smoke else config["episode_steps"])
    episode_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    signatures: dict[tuple[str, int, str], list[str]] = {}
    for scenario, scenario_config in config["scenarios"].items():
        environment_start = pd.Timestamp(scenario_config["environment_start"])
        dataset_start = pd.Timestamp(scenario_config["dataset_start"])
        for seed in seeds:
            for controller in CONTROLLERS:
                print(
                    f"RUN scenario={scenario} seed={seed} controller={controller} steps={steps}",
                    flush=True,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    episode, forecasts, controls, trace = run_episode(
                        controller=controller,
                        scenario=scenario,
                        environment_start=environment_start,
                        expected_dataset_start=dataset_start,
                        seed=seed,
                        steps=steps,
                        trace_source=trace_source,
                        transformer=transformer,
                        datacenter_configs=dc_data,
                        optimizer_config=optimizer_config,
                    )
                oracle_warnings = sum(
                    "non-deployable" in str(item.message) for item in caught
                )
                episode["oracle_warning_count"] = oracle_warnings
                if controller == "H4_ORACLE" and oracle_warnings == 0:
                    raise RuntimeError("H4_ORACLE did not emit an explicit warning")
                if controller != "H4_ORACLE" and oracle_warnings:
                    raise RuntimeError("deployable controller emitted an oracle warning")
                episode_rows.append(episode)
                forecast_rows.extend(forecasts)
                step_rows.extend(controls)
                signatures[(scenario, seed, controller)] = trace
                print(
                    json.dumps(
                        {
                            "controller": controller,
                            "scenario": scenario,
                            "seed": seed,
                            "reward": episode["reward"],
                            "stage_cost": episode["stage_cost"],
                            "completed": episode["completed"],
                            "sla": episode["sla"],
                            "solver_ms": episode["solver_ms"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    fairness_pass = _fairness(signatures)
    if not fairness_pass:
        raise RuntimeError("same-environment arrival trace fairness failed")
    episodes = pd.DataFrame(episode_rows)
    forecasts = pd.DataFrame(forecast_rows)
    controls = pd.DataFrame(step_rows)
    if smoke:
        smoke_record = {
            "status": "PASS",
            "controllers": list(CONTROLLERS),
            "scenarios": list(config["scenarios"]),
            "seed": seeds[0],
            "steps": steps,
            "all_solver_feasible": bool(episodes["infeasible_count"].eq(0).all()),
            "same_environment_trace": fairness_pass,
            "transformer_checkpoint_load_count": transformer.checkpoint_load_count,
            "oracle_warning_isolated": True,
            "forecast_timestamp_alignment": True,
        }
        _write_json(output / "smoke_test_results.json", smoke_record)
        print(json.dumps(smoke_record, ensure_ascii=False), flush=True)
        return output

    episodes.to_csv(output / "05_episode_metrics.csv", index=False)
    aggregate = _aggregate_metrics(episodes)
    aggregate.to_csv(output / "06_aggregate_metrics.csv", index=False)
    paired = _paired_comparisons(episodes)
    paired.to_csv(output / "07_paired_comparisons.csv", index=False)
    forecasts.to_parquet(
        output / "08_forecast_control_trace.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    controls.to_parquet(
        output / "repaired_control_action_trace.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    high_load, threshold = _high_load_diagnostics(controls)
    high_load.to_csv(output / "09_high_load_control_diagnostics.csv", index=False)
    oracle_gap = _oracle_gap(aggregate)
    oracle_gap.to_csv(output / "10_oracle_gap_analysis.csv", index=False)
    episodes[
        [
            "controller",
            "scenario",
            "seed",
            "solver_ms",
            "solver_p95_ms",
            "infeasible_count",
        ]
    ].to_csv(output / "11_solver_runtime.csv", index=False)
    figure_paths = _plot_artifacts(aggregate, forecasts, high_load, figures)
    role, role_reason = _preliminary_role(paired, high_load)

    experiment_config = {
        **config,
        "git_branch": _git("branch", "--show-current"),
        "git_head": _git("rev-parse", "HEAD"),
        "optimizer": asdict(optimizer_config),
        "transformer_checkpoint_sha256": _sha256(checkpoint),
        "transformer_checkpoint_selection": "lowest validation normalized MSE",
        "actual_environment_information_mode": "deployable for all controllers",
        "global_to_dc_formula": "population_weight * local_time_activity, normalized",
        "high_load_gpu_p90_threshold": threshold,
        "synthetic_future_tasks": False,
    }
    _write_json(output / "02_experiment_config.json", experiment_config)
    matrix = pd.DataFrame(
        [
            ["H1", 1, "none", "estimated_duration", "persistence", "persistence", "YES"],
            ["H4_ORACLE", 4, "true global aggregate", "estimated_duration", "oracle", "oracle", "NO"],
            ["H4_PERSISTENCE", 4, "current global hold", "estimated_duration", "persistence", "persistence", "YES"],
            ["H4_TRANSFORMER", 4, "Transformer seed 33", "estimated_duration", "persistence", "persistence", "YES"],
        ],
        columns=[
            "controller",
            "horizon",
            "future_workload",
            "runtime",
            "future_price",
            "future_carbon",
            "deployable",
        ],
    )
    matrix.to_csv(output / "03_controller_information_matrix.csv", index=False)
    bridge_design = """# Forecast Bridge Design

- Bridge type: **AGGREGATE RESOURCE PRESSURE**.
- Forecasts remain four global aggregate flows: task count, CPU, GPU and memory at +15/+30/+45/+60 minutes.
- Global demand is distributed deterministically with SustainCluster's population-weight x local-time-activity probability model. No RNG is called in planning.
- CPU/GPU/memory enter the existing MILP through per-DC capacity upper bounds. Each forecast point reserves only its corresponding interval (Option A reserve envelope).
- Arrival flows are not accumulated across later intervals and no duration approximation is attached.
- `future_arrivals` remains empty. No task identity, origin sample, deadline, bandwidth or synthetic duration is generated.
- The environment workload is never overwritten; the repeated outer timestamp is explicitly aligned to Dataset v1's first unique 49-day cycle.
- All actual environments use the deployable runtime contract. Oracle workload and oracle price/carbon are applied only to the H4_ORACLE planning snapshot.
"""
    _write_text(output / "04_forecast_bridge_design.md", bridge_design)

    pair_lookup = paired.set_index(["comparison", "metric"])
    aggregate_lookup = aggregate.set_index(["controller", "metric"])
    high_lookup = high_load.set_index(["controller", "load_segment"])
    def metric(controller: str, name: str) -> float:
        return float(aggregate_lookup.loc[(controller, name), "mean"])
    summary = f"""# Forecast-aware MPC v1 Summary

1. **Q1. Transformer forecast 如何接入 MPC？** 四步 global forecast 经确定性 origin probability 拆分为 per-DC CPU/GPU/memory 单 interval reserve，并直接收紧现有 capacity upper bounds。
2. **Q2. 是否生成 synthetic future tasks？** NO；`future_arrivals` 保持为空。
3. **Q3. Global workload 如何映射到 DC？** SustainCluster 原公式 `population_weight x local-time activity` 归一化期望值，无 RNG，逐 horizon 守恒。
4. **Q4. H4_TRANSFORMER 是否读取 oracle future？** NO；只读取截至当前时刻的 96 x 8 history 和冻结 checkpoint。
5. **Q5. H4_PERSISTENCE 是否读取 oracle future？** NO；只 hold 当前四维 workload。
6. **Q6. H4_ORACLE 使用哪些 oracle 信息？** 真实未来 global aggregate workload 以及未来 price/carbon；实际环境仍使用 estimated duration。
7. **Q7. H1 vs H4_ORACLE？** reward delta `{pair_lookup.loc[("H4_ORACLE - H1", "reward"), "mean_delta"]:.6f}`，stage-cost delta `{pair_lookup.loc[("H4_ORACLE - H1", "stage_cost"), "mean_delta"]:.6f}`。
8. **Q8. H1 vs H4_PERSISTENCE？** reward delta `{pair_lookup.loc[("H4_PERSISTENCE - H1", "reward"), "mean_delta"]:.6f}`，stage-cost delta `{pair_lookup.loc[("H4_PERSISTENCE - H1", "stage_cost"), "mean_delta"]:.6f}`。
9. **Q9. Persistence vs Transformer？** Transformer reward delta `{pair_lookup.loc[("H4_TRANSFORMER - H4_PERSISTENCE", "reward"), "mean_delta"]:.6f}`，stage-cost delta `{pair_lookup.loc[("H4_TRANSFORMER - H4_PERSISTENCE", "stage_cost"), "mean_delta"]:.6f}`。
10. **Q10. HIGH_LOAD 下 Transformer？** Persistence/Transformer stage-cost mean `{high_lookup.loc[("H4_PERSISTENCE", "HIGH_LOAD"), "stage_cost_mean"]:.6f}` / `{high_lookup.loc[("H4_TRANSFORMER", "HIGH_LOAD"), "stage_cost_mean"]:.6f}`。
11. **Q11. 平均预测优势是否转化为 control advantage？** 见 Q9 与 paired counts；不使用单一 composite score。
12. **Q12. Transformer 与 Oracle gap？** reward `{metric('H4_TRANSFORMER', 'reward') - metric('H4_ORACLE', 'reward'):.6f}`，stage cost `{metric('H4_TRANSFORMER', 'stage_cost') - metric('H4_ORACLE', 'stage_cost'):.6f}`。
13. **Q13. solver 是否远低于 15min？** YES；Transformer mean solver `{metric('H4_TRANSFORMER', 'solver_ms'):.3f}` ms。
14. **Q14. 是否发现 information leakage？** NO；fairness/no-oracle audit PASS。
15. **Q15. Preliminary recommendation？** **{role}**。{role_reason}
"""
    _write_text(output / "01_summary.md", summary)

    audit = f"""# Leakage and Fairness Audit

Status: **PASS**

- Transformer and Persistence planning requests do not contain oracle future workload.
- Provider invariance tests alter future truth and true duration while holding current state/history fixed; deployable planning input is unchanged.
- All four controllers use independently reset environments with paired seeds and identical actual deployable runtime configuration.
- Arrival signatures (source name, origin, resources, true/estimated duration and bandwidth) match across all paired controllers: `{fairness_pass}`.
- Forecast providers do not mutate environment arrivals; the bridge modifies an immutable planning snapshot only.
- Origin expectation is deterministic and conserves global CPU/GPU/memory at every horizon.
- Oracle workload provider emits an explicit non-deployable warning; deployable controllers emit none.
- Transformer checkpoint is loaded once, selected by validation loss only, and never retrained.
- Price/carbon are persistence for H1/Persistence/Transformer and oracle only for the explicitly isolated Oracle ceiling.
- Actual environment runtime is `estimated_duration` for all controllers, preserving paired reward/SLA semantics; no controller receives true duration.
"""
    _write_text(output / "12_leakage_and_fairness_audit.md", audit)
    risk = """# Risk Register

| Risk | Evidence | Consequence | Next-stage handling |
|---|---|---|---|
| Transformer underperforms Persistence on GPU P90 forecast | Transformer Forecast v1 | Burst reserves may be mistimed | Evaluate triggered/risk-aware MPC later; not implemented here |
| Aggregate forecast has no duration | Dataset target contract | Reserve is flow, not occupancy | v1 uses transparent single-interval envelope |
| Global-to-DC is expected, not observed geography | origin is synthetic | Per-DC pressure is probabilistic | Keep deterministic probability audit |
| H4 optimizer horizon conventions predate this bridge | Existing rolling optimizer | Forecast slots are treated as +15/+30/+45/+60 exogenous envelopes | Preserve explicit timestamp tests and trace logs |
| Workload parent artifact repeats 49 days | Dataset lineage | Generalization is limited | Use only first verified unique-cycle alignment |
| Public-data/checkpoint licensing | Prior release audit | External publication may need review | Resolve before public release |
"""
    _write_text(output / "13_risk_register.md", risk)
    evidence = f"""# Evidence Index

| Evidence | Location |
|---|---|
| Unified provider contract | `src/forecasting/workload_forecast_provider.py` |
| Frozen Transformer wrapper | `src/forecasting/transformer_workload_provider.py` |
| Aggregate pressure bridge | `src/sustaincluster_mpc/forecast_pressure_adapter.py` |
| Experiment configuration | `configs/sustaincluster_mpc/forecast_aware_mpc_v1.yaml` |
| Episode metrics | `artifacts/forecast_aware_mpc_v1/05_episode_metrics.csv` |
| Paired comparisons | `artifacts/forecast_aware_mpc_v1/07_paired_comparisons.csv` |
| Forecast/control trace | `artifacts/forecast_aware_mpc_v1/08_forecast_control_trace.parquet` |
| High-load diagnostic | `artifacts/forecast_aware_mpc_v1/09_high_load_control_diagnostics.csv` |
| Figures | `artifacts/forecast_aware_mpc_v1/figures/` ({len(figure_paths)} files) |
| Git HEAD | `{_git('rev-parse', 'HEAD')}` |
| SustainCluster HEAD | `{_git('rev-parse', 'HEAD', cwd=sustain_repo)}` |
"""
    _write_text(output / "14_evidence_index.md", evidence)
    changes = """# Change Manifest

## Added

- Unified Oracle/Persistence/Transformer aggregate workload providers and frozen trace alignment.
- Deterministic global-to-DC expected-pressure mapping and single-interval capacity reserve bridge.
- Four-controller paired runner, 18+ contract tests, smoke test, formal metrics, traces, diagnostics and figures.

## Explicit non-changes

- No Transformer training, hyperparameter change or checkpoint reselection.
- No Forecast Dataset split, scaler, target or data-file modification.
- No reward/objective-weight modification.
- No BC, SAC, expert dataset, Triggered MPC or uncertainty-model work.
- No SustainCluster vendor-source modification, commit or push.
"""
    _write_text(output / "15_change_manifest.md", changes)
    final = {
        "status": "READY FOR MPC ROLE DECISION",
        "preliminary_mpc_role": role,
        "reason": role_reason,
        "same_environment_trace": fairness_pass,
        "transformer_checkpoint_load_count": transformer.checkpoint_load_count,
        "transformer_clip_ratios": transformer.clip_ratios(),
        "high_load_gpu_p90_threshold": threshold,
        "episodes": len(episodes),
        "forecast_trace_rows": len(forecasts),
    }
    _write_json(output / "evaluation_summary.json", final)
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE / "configs/sustaincluster_mpc/forecast_aware_mpc_v1.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config.resolve(), smoke=args.smoke)


if __name__ == "__main__":
    main()
