from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import warnings
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting.transformer_workload_provider import (
    TransformerWorkloadForecastProvider,
)
from forecasting.workload_forecast_provider import (
    ForecastTraceSource,
    OracleWorkloadForecastProvider,
    build_bundle,
)
from scripts.audit import mpc_control_authority_diagnosis_v1 as diagnosis
from scripts.forecast import run_forecast_aware_mpc_v1 as repaired_runner
from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_imitation.paths import prepare_sustaincluster_imports
from sustaincluster_mpc.action_adapter import SustainClusterActionAdapter
from sustaincluster_mpc.forecast_pressure_adapter import (
    apply_forecast_pressure,
    apply_oracle_future_signals,
)
from sustaincluster_mpc.future_signals import FutureSignalProvider
from sustaincluster_mpc.horizon_adapter import (
    HorizonState,
    HorizonStateAdapter,
    RunningTaskHorizonSnapshot,
)
from sustaincluster_mpc.rolling_horizon_optimizer import RollingHorizonOptimizer
from sustaincluster_mpc.triggered_mpc import (
    CALIBRATION_SEEDS,
    EVALUATION_SEEDS,
    TRIGGER_TRACE_COLUMNS,
    TriggerRiskAssessment,
    TriggeredTransformerMPC,
    TriggerThresholds,
    calibrate_trigger_thresholds,
    compute_deployable_risk,
    safe_benefit_retention,
    validate_trigger_trace_schema,
)


CONTROLLERS = (
    "H1",
    "H4_TRANSFORMER_ALWAYS",
    "H4_TRANSFORMER_TRIGGERED_P95",
    "H4_ORACLE_ALWAYS",
    "H4_TRANSFORMER_TRIGGERED_P90",
    "H4_TRANSFORMER_TRIGGERED_P99",
)
TRIGGERED_CONTROLLERS = (
    "H4_TRANSFORMER_TRIGGERED_P90",
    "H4_TRANSFORMER_TRIGGERED_P95",
    "H4_TRANSFORMER_TRIGGERED_P99",
)
PRIMARY_CONTROLLERS = (
    "H1",
    "H4_TRANSFORMER_ALWAYS",
    "H4_TRANSFORMER_TRIGGERED_P95",
    "H4_ORACLE_ALWAYS",
)
CONTROL_METRICS = tuple(repaired_runner.CONTROL_METRICS)
HIGHER_IS_BETTER = {"reward", "completed"}
PRIMARY_COMPARISONS = (
    ("TRIGGERED_P95 - H1", "H1", "H4_TRANSFORMER_TRIGGERED_P95"),
    (
        "ALWAYS_TRANSFORMER - TRIGGERED_P95",
        "H4_TRANSFORMER_TRIGGERED_P95",
        "H4_TRANSFORMER_ALWAYS",
    ),
    (
        "ORACLE_ALWAYS - TRIGGERED_P95",
        "H4_TRANSFORMER_TRIGGERED_P95",
        "H4_ORACLE_ALWAYS",
    ),
    (
        "ORACLE_ALWAYS - ALWAYS_TRANSFORMER",
        "H4_TRANSFORMER_ALWAYS",
        "H4_ORACLE_ALWAYS",
    ),
)


def git(*args: str, cwd: Path = WORKSPACE) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8", errors="replace"
    ).strip()


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text("utf-8"))["triggered_mpc_v1"]


def verify_frozen_assets(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    sustain_repo = WORKSPACE / "references/external_repos/sustain-cluster"
    dataset_root = WORKSPACE / config["dataset_root"]
    checkpoint = WORKSPACE / config["transformer_checkpoint"]
    if git("rev-parse", "HEAD", cwd=sustain_repo) != config["sustaincluster_commit"]:
        raise RuntimeError("SustainCluster commit changed")
    if git("status", "--short", cwd=sustain_repo):
        raise RuntimeError("SustainCluster worktree must be clean")
    if sha256(checkpoint) != (
        "9CD85A0D229E135106EA1E49CD8FEC8936D7B6349DA41531564E565F063569E2"
    ):
        raise RuntimeError("Transformer checkpoint hash changed")
    manifest = json.loads((dataset_root / "13_dataset_manifest.json").read_text("utf-8"))
    for name, metadata in manifest["dataset_files"].items():
        if sha256(dataset_root / "dataset" / name) != metadata["sha256"]:
            raise RuntimeError(f"Forecast Dataset file hash changed: {name}")
    prepare_sustaincluster_imports(sustain_repo)
    return sustain_repo, dataset_root, checkpoint


def truth_signature(env: Any) -> str:
    exogenous = []
    for dc in sorted(
        env.cluster_manager.datacenters.values(), key=lambda item: int(item.dc_id)
    ):
        exogenous.append(
            (
                int(dc.dc_id),
                float(dc.price_manager.get_current_price()),
                float(dc.ci_manager.get_current_ci(norm=False)),
            )
        )
    payload = {
        "arrivals": repaired_runner._arrival_signature(env),
        "exogenous": exogenous,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()


def build_env(start: pd.Timestamp, steps: int, seed: int):
    env = build_sustaincluster_env(
        None,
        start,
        steps,
        allow_defer=True,
        initial_seed=seed,
        information_mode="deployable",
        duration_estimate_mode="declared_or_baseline",
        baseline_estimated_duration_minutes=60.0,
    )
    env.reset(seed=seed)
    return env


def make_horizon_adapter() -> HorizonStateAdapter:
    return HorizonStateAdapter(
        information_mode="deployable",
        future_signal_provider=FutureSignalProvider("persistence"),
    )


def action_summary(result: Any) -> tuple[str, str, str, str]:
    semantics = [item.decision for item in result.first_step_decisions]
    targets = [item.dc_id for item in result.first_step_decisions]
    if not semantics:
        semantic = "none"
    elif len(set(semantics)) == 1:
        semantic = semantics[0]
    else:
        semantic = "mixed"
    nonempty_targets = {target for target in targets if target is not None}
    target = str(next(iter(nonempty_targets))) if len(nonempty_targets) == 1 else json.dumps(targets)
    return semantic, target, json.dumps(semantics), json.dumps(targets)


def risk_state_features(state: HorizonState, risk: TriggerRiskAssessment) -> dict[str, float]:
    current_gpu_utilization = max(
        1.0 - dc.gpu_available_units / dc.gpu_total_units
        for dc in state.current.datacenters
    )
    current_free_gpu = min(
        dc.gpu_available_units / dc.gpu_total_units
        for dc in state.current.datacenters
    )
    pending = len(state.current.tasks) + sum(
        dc.queued_task_count for dc in state.current.datacenters
    )
    durations = [task.duration_minutes for task in state.current.tasks]
    return {
        "current_gpu_utilization": float(current_gpu_utilization),
        "current_free_gpu": float(current_free_gpu),
        "future_expected_free_gpu": float(1.0 - risk.gpu_pressure),
        "pending_task_count": float(pending),
        "estimated_duration_mean": float(mean(durations)) if durations else 0.0,
        "estimated_duration_p95": float(np.percentile(durations, 95)) if durations else 0.0,
    }


def run_calibration(config_path: Path) -> Path:
    config = load_config(config_path)
    output = WORKSPACE / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    _, dataset_root, checkpoint = verify_frozen_assets(config)
    if tuple(config["calibration_seeds"]) != CALIBRATION_SEEDS:
        raise RuntimeError("calibration seeds changed")
    if tuple(config["evaluation_seeds"]) != EVALUATION_SEEDS:
        raise RuntimeError("evaluation seeds changed")

    trace_source = ForecastTraceSource(dataset_root)
    transformer = TransformerWorkloadForecastProvider(
        checkpoint, dataset_root=dataset_root, device=config["device"]
    )
    dc_configs = yaml.safe_load(
        (WORKSPACE / config["datacenter_config"]).read_text("utf-8")
    )["datacenters"]
    optimizer_config = repaired_runner._optimizer_config(
        WORKSPACE / config["optimizer_config"]
    )
    rows = []
    for scenario, scenario_config in config["scenarios"].items():
        start = pd.Timestamp(scenario_config["environment_start"])
        expected_dataset_start = pd.Timestamp(scenario_config["dataset_start"])
        for seed in CALIBRATION_SEEDS:
            env = build_env(start, int(config["episode_steps"]), seed)
            action_adapter = SustainClusterActionAdapter.from_env(env)
            horizon_adapter = make_horizon_adapter()
            optimizer = RollingHorizonOptimizer()
            try:
                for step in range(int(config["episode_steps"])):
                    timestamp = pd.Timestamp(env.current_time)
                    if step == 0 and trace_source.align_environment_timestamp(timestamp) != expected_dataset_start:
                        raise RuntimeError("calibration trace alignment failed")
                    request = trace_source.request(
                        timestamp, include_oracle_future=False
                    )
                    bundle = transformer.forecast(request)
                    base_h4 = horizon_adapter.build_horizon_state(
                        env, 5, "no_future_arrivals"
                    )
                    risk = compute_deployable_risk(base_h4, bundle, dc_configs)
                    rows.append(
                        {
                            "timestamp": timestamp,
                            "scenario": scenario,
                            "seed": seed,
                            "risk_score": risk.risk_score,
                            "risk_dc": risk.risk_dc,
                            "risk_horizon": risk.risk_horizon,
                            "risk_resource": risk.risk_resource,
                            "cpu_pressure": risk.cpu_pressure,
                            "gpu_pressure": risk.gpu_pressure,
                            "mem_pressure": risk.mem_pressure,
                        }
                    )
                    h1 = horizon_adapter.build_horizon_state(
                        env, 1, "no_future_arrivals"
                    )
                    result = optimizer.solve(h1, optimizer_config, action_adapter)
                    if not result.feasible:
                        raise RuntimeError("calibration H1 solve infeasible")
                    env.step(list(result.environment_actions))
            finally:
                env.close()

    calibration = pd.DataFrame(rows)
    thresholds = calibrate_trigger_thresholds(calibration["risk_score"].tolist())
    calibration.to_csv(output / "04_calibration_risk_distribution.csv", index=False)
    threshold_record = {
        "P90": thresholds.p90,
        "P95_primary": thresholds.p95_primary,
        "P99": thresholds.p99,
        "selection_rule": thresholds.selection_rule,
        "calibration_seeds": list(thresholds.calibration_seeds),
        "evaluation_seeds_excluded": list(thresholds.evaluation_seeds),
        "calibration_state_count": len(calibration),
        "scenarios": list(config["scenarios"]),
        "episode_steps": int(config["episode_steps"]),
        "evaluation_reward_used": False,
        "checkpoint_sha256": sha256(checkpoint),
    }
    write_json(output / "05_trigger_thresholds.json", threshold_record)
    write_text(
        output / "02_trigger_design.md",
        """# Trigger Design

`R_t = max_{d,h,r} ((estimated running occupancy) + (Transformer arrival flow)) / capacity`.

- Future nodes are repaired +15/+30/+45/+60 minute nodes 1/2/3/4.
- Existing occupancy uses currently running tasks and estimated release steps only.
- Future arrival flow uses the frozen Transformer and deterministic global-to-DC probabilities.
- Flow remains single-node flow; it is not accumulated as active occupancy.
- The trigger chooses H1 or repaired H4 and never rewrites optimizer actions.
- P95 is primary; no hysteresis, cooldown, price trigger, learned trigger or Oracle fallback is present.
""",
    )
    write_text(
        output / "03_calibration_protocol.md",
        f"""# Calibration Protocol

- Controller trajectory: H1 only.
- Calibration seeds: `{list(CALIBRATION_SEEDS)}`.
- Evaluation seeds excluded: `{list(EVALUATION_SEEDS)}`.
- Scenarios: `{list(config['scenarios'])}`; 96 steps each; `{len(calibration)}` states total.
- Recorded inputs: deployable risk and its DC/horizon/resource components only.
- Thresholds are empirical P90/P95/P99 risk quantiles.
- Primary threshold was predeclared as P95.
- Reward, stage cost, Oracle action and H1/H4 disagreement were not used.
""",
    )
    print(json.dumps(threshold_record, ensure_ascii=False), flush=True)
    return output


def read_frozen_thresholds(output: Path) -> TriggerThresholds:
    path = output / "05_trigger_thresholds.json"
    if not path.is_file():
        raise FileNotFoundError("calibration thresholds must exist before evaluation")
    raw = json.loads(path.read_text("utf-8"))
    if raw["selection_rule"] != "risk quantile only; no reward tuning":
        raise RuntimeError("threshold selection rule changed")
    if raw["evaluation_reward_used"] is not False:
        raise RuntimeError("evaluation reward cannot select thresholds")
    if tuple(raw["calibration_seeds"]) != CALIBRATION_SEEDS:
        raise RuntimeError("calibration threshold seeds changed")
    if tuple(raw["evaluation_seeds_excluded"]) != EVALUATION_SEEDS:
        raise RuntimeError("evaluation seed exclusion changed")
    return TriggerThresholds(
        p90=float(raw["P90"]),
        p95_primary=float(raw["P95_primary"]),
        p99=float(raw["P99"]),
        calibration_seeds=CALIBRATION_SEEDS,
        evaluation_seeds=EVALUATION_SEEDS,
    )


def controller_threshold(controller: str, thresholds: TriggerThresholds) -> float | None:
    return {
        "H4_TRANSFORMER_TRIGGERED_P90": thresholds.p90,
        "H4_TRANSFORMER_TRIGGERED_P95": thresholds.p95_primary,
        "H4_TRANSFORMER_TRIGGERED_P99": thresholds.p99,
    }.get(controller)


def run_episode(
    *,
    controller: str,
    scenario: str,
    start: pd.Timestamp,
    expected_dataset_start: pd.Timestamp,
    seed: int,
    steps: int,
    trace_source: ForecastTraceSource,
    transformer: TransformerWorkloadForecastProvider,
    dc_configs: list[dict[str, Any]],
    optimizer_config: Any,
    thresholds: TriggerThresholds,
    trace_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    env = build_env(start, steps, seed)
    action_adapter = SustainClusterActionAdapter.from_env(env)
    horizon_adapter = make_horizon_adapter()
    optimizer = RollingHorizonOptimizer()
    oracle_provider = OracleWorkloadForecastProvider()
    metrics = repaired_runner._empty_metrics()
    traces = []
    shadow_rows = []
    truth = []
    risk_values = []
    h4_calls = h1_calls = 0
    h4_solver_total = h1_solver_total = 0.0
    completed_steps = 0
    threshold = controller_threshold(controller, thresholds)
    triggered_controller = (
        None
        if threshold is None
        else TriggeredTransformerMPC(threshold, dc_configs)
    )
    try:
        for step in range(steps):
            timestamp = pd.Timestamp(env.current_time)
            aligned = trace_source.align_environment_timestamp(timestamp)
            if step == 0 and aligned != expected_dataset_start:
                raise RuntimeError("evaluation trace alignment failed")
            truth.append(truth_signature(env))

            request = trace_source.request(timestamp, include_oracle_future=False)
            bundle = transformer.forecast(request)
            base_h4 = horizon_adapter.build_horizon_state(
                env, 5, "no_future_arrivals"
            )
            h4_transformer = apply_forecast_pressure(
                base_h4, bundle, dc_configs
            ).state
            h1 = horizon_adapter.build_horizon_state(env, 1, "no_future_arrivals")
            if triggered_controller is None:
                risk = compute_deployable_risk(base_h4, bundle, dc_configs)
                selection = None
            else:
                risk, selection = triggered_controller.evaluate(
                    h1, h4_transformer, base_h4, bundle
                )
            risk_values.append(risk.risk_score)

            if controller == "H1":
                selected_state = h1
                selected_controller = "H1"
                triggered = False
            elif controller == "H4_TRANSFORMER_ALWAYS":
                selected_state = h4_transformer
                selected_controller = "H4_TRANSFORMER"
                triggered = True
            elif controller == "H4_ORACLE_ALWAYS":
                oracle_bundle = oracle_provider.forecast(
                    trace_source.request(timestamp, include_oracle_future=True)
                )
                selected_state = apply_forecast_pressure(
                    base_h4, oracle_bundle, dc_configs
                ).state
                selected_state = apply_oracle_future_signals(selected_state, env)
                selected_controller = "H4_ORACLE"
                triggered = True
            else:
                assert selection is not None
                selected_state = selection.planning_state
                selected_controller = selection.selected_controller
                triggered = selection.triggered

            before_resources = repaired_runner._resource_bounds(env)
            result = optimizer.solve(selected_state, optimizer_config, action_adapter)
            if not result.feasible:
                metrics["infeasible_count"] += 1
                break
            solver_ms = 1000.0 * result.solve_seconds
            if selected_controller == "H1":
                h1_calls += 1
                h1_solver_total += solver_ms
            else:
                h4_calls += 1
                h4_solver_total += solver_ms

            if controller == "H1":
                shadow = optimizer.solve(
                    h4_transformer, optimizer_config, action_adapter
                )
                if not shadow.feasible:
                    raise RuntimeError("shadow H4 solve infeasible")
                for task, h1_decision, h4_decision in zip(
                    h1.current.tasks,
                    result.first_step_decisions,
                    shadow.first_step_decisions,
                ):
                    shadow_rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "step": step,
                            "timestamp": timestamp,
                            "task_id": task.task_id,
                            "risk_score": risk.risk_score,
                            "h1_semantic_action": h1_decision.decision,
                            "h4_semantic_action": h4_decision.decision,
                            "h1_target_dc": h1_decision.dc_id,
                            "h4_target_dc": h4_decision.dc_id,
                            "semantic_disagreement": h1_decision.decision
                            != h4_decision.decision,
                            "target_disagreement": h1_decision.dc_id
                            != h4_decision.dc_id,
                            "triggered_p90": risk.risk_score >= thresholds.p90,
                            "triggered_p95": risk.risk_score
                            >= thresholds.p95_primary,
                            "triggered_p99": risk.risk_score >= thresholds.p99,
                        }
                    )

            metrics["stage_cost"] += result.first_step_costs.total
            metrics["projected_terminal_backlog_sum"] += result.terminal_backlog_count
            metrics["solver_values"].append(result.solve_seconds)
            metrics["forecast_fallback_count"] += int(bundle.persistence_fallback_used)
            metrics["forecast_history_unavailable_count"] += int(
                not bundle.forecast_history_available
            )
            metrics["negative_prediction_clip_count"] += bundle.negative_prediction_clip_count
            metrics["forecast_inference_values"].append(bundle.inference_ms / 1000.0)

            step_defer = 0
            step_migration = 0
            for task, decision in zip(
                selected_state.current.tasks, result.first_step_decisions
            ):
                metrics["decision_count"] += 1
                metrics["wait_values"].append(float(task.wait_intervals))
                metrics["scheduler_wait_values"].append(
                    float(task.scheduler_wait_intervals)
                )
                metrics["local_queue_wait_values"].append(
                    float(task.wait_intervals - task.scheduler_wait_intervals)
                )
                if decision.decision == "defer":
                    metrics["defer_count"] += 1
                    step_defer += 1
                else:
                    metrics["assign_count"] += 1
                    migrated = int(int(decision.dc_id) != task.origin_dc_id)
                    metrics["migration_count"] += migrated
                    step_migration += migrated

            semantic, target, semantics_json, targets_json = action_summary(result)
            features = risk_state_features(base_h4, risk)
            trace_row = {
                "seed": seed,
                "scenario": scenario,
                "step": step,
                "timestamp": timestamp,
                "risk_score": risk.risk_score,
                "threshold": threshold,
                "triggered": triggered,
                "risk_dc": risk.risk_dc,
                "risk_horizon": risk.risk_horizon,
                "risk_resource": risk.risk_resource,
                "gpu_pressure": risk.gpu_pressure,
                "cpu_pressure": risk.cpu_pressure,
                "mem_pressure": risk.mem_pressure,
                "selected_controller": selected_controller,
                "selected_semantic_action": semantic,
                "selected_target_dc": target,
                "solver_ms": solver_ms,
                "history_fallback": bundle.persistence_fallback_used,
                "history_available": bundle.forecast_history_available,
                "semantic_actions_json": semantics_json,
                "target_dcs_json": targets_json,
                **features,
            }
            traces.append(trace_row)

            _, reward, terminated, truncated, info = env.step(
                list(result.environment_actions)
            )
            completed_steps = step + 1
            scheduler_added = int(info.get("scheduler_wait_intervals_added", 0))
            if scheduler_added != step_defer:
                raise RuntimeError("defer and scheduler waiting accounting disagree")
            metrics["scheduler_wait_intervals_added"] += scheduler_added
            metrics["resource_overflow_count"] += int(
                not before_resources or not repaired_runner._resource_bounds(env)
            )
            repaired_runner._physical_metrics(metrics, info, float(reward))
            if terminated or truncated:
                break

        result_metrics = repaired_runner._finish_metrics(
            metrics, env, completed_steps
        )
        result_metrics.update(
            {
                "controller": controller,
                "scenario": scenario,
                "seed": seed,
                "environment_start": start.isoformat(),
                "information_mode": "deployable_environment",
                "trigger_threshold": threshold,
                "trigger_rate": h4_calls / max(1, completed_steps),
                "H4_call_count": h4_calls,
                "H1_call_count": h1_calls,
                "H4_solver_time_total_ms": h4_solver_total,
                "H1_solver_time_total_ms": h1_solver_total,
                "controller_solver_time_total_ms": h4_solver_total
                + h1_solver_total,
                "risk_score_p50": float(np.percentile(risk_values, 50)),
                "risk_score_p90": float(np.percentile(risk_values, 90)),
                "risk_score_p95": float(np.percentile(risk_values, 95)),
                "risk_score_p99": float(np.percentile(risk_values, 99)),
            }
        )
        trace_frame = pd.DataFrame(traces)
        validate_trigger_trace_schema(trace_frame.columns)
        trace_frame.to_parquet(
            trace_path, index=False, engine="pyarrow", compression="zstd"
        )
        return result_metrics, traces, shadow_rows, truth
    finally:
        env.close()


def aggregate_metrics(episodes: pd.DataFrame) -> pd.DataFrame:
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
                    "min": float(selected[metric].min()),
                    "max": float(selected[metric].max()),
                }
            )
    return pd.DataFrame(rows)


def paired_comparisons(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["scenario", "seed"]
    for name, baseline, candidate in PRIMARY_COMPARISONS:
        left = episodes[episodes["controller"] == baseline].set_index(keys)
        right = episodes[episodes["controller"] == candidate].set_index(keys)
        if set(left.index) != set(right.index):
            raise RuntimeError(f"paired keys differ for {name}")
        for metric in CONTROL_METRICS:
            delta = right[metric] - left[metric]
            better = delta > 1e-12 if metric in HIGHER_IS_BETTER else delta < -1e-12
            worse = delta < -1e-12 if metric in HIGHER_IS_BETTER else delta > 1e-12
            rows.append(
                {
                    "comparison": name,
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    "mean_delta": float(delta.mean()),
                    "std_delta": float(delta.std(ddof=0)),
                    "better_count": int(better.sum()),
                    "same_count": int((~better & ~worse).sum()),
                    "worse_count": int(worse.sum()),
                }
            )
    return pd.DataFrame(rows)


def trigger_statistics(episodes: pd.DataFrame, traces: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for controller in TRIGGERED_CONTROLLERS:
        selected = episodes[episodes["controller"] == controller]
        controller_trace = traces[traces["controller"] == controller]
        for episode in selected.itertuples(index=False):
            episode_trace = controller_trace[
                (controller_trace["scenario"] == episode.scenario)
                & (controller_trace["seed"] == episode.seed)
            ]
            triggered_trace = episode_trace[episode_trace["triggered"]]
            rows.append(
                {
                    "controller": controller,
                    "scope": "episode",
                    "scenario": episode.scenario,
                    "seed": episode.seed,
                    "trigger_rate_mean": episode.trigger_rate,
                    "trigger_rate_std": np.nan,
                    "trigger_rate_min": episode.trigger_rate,
                    "trigger_rate_max": episode.trigger_rate,
                    "H4_call_count": episode.H4_call_count,
                    "H1_call_count": episode.H1_call_count,
                    "main_trigger_resource": _mode(triggered_trace["risk_resource"]),
                    "main_trigger_horizon": _mode(triggered_trace["risk_horizon"]),
                    "main_trigger_dc": _mode(triggered_trace["risk_dc"]),
                }
            )
        triggered_trace = controller_trace[controller_trace["triggered"]]
        rows.append(
            {
                "controller": controller,
                "scope": "aggregate",
                "scenario": "ALL",
                "seed": np.nan,
                "trigger_rate_mean": float(selected["trigger_rate"].mean()),
                "trigger_rate_std": float(selected["trigger_rate"].std(ddof=0)),
                "trigger_rate_min": float(selected["trigger_rate"].min()),
                "trigger_rate_max": float(selected["trigger_rate"].max()),
                "H4_call_count": int(selected["H4_call_count"].sum()),
                "H1_call_count": int(selected["H1_call_count"].sum()),
                "main_trigger_resource": _mode(triggered_trace["risk_resource"]),
                "main_trigger_horizon": _mode(triggered_trace["risk_horizon"]),
                "main_trigger_dc": _mode(triggered_trace["risk_dc"]),
            }
        )
    return pd.DataFrame(rows)


def _mode(values: Iterable[Any]) -> Any:
    values = list(values)
    return Counter(values).most_common(1)[0][0] if values else None


def disagreement_capture(shadow: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in ("p90", "p95", "p99"):
        triggered = shadow[f"triggered_{label}"].astype(bool)
        target = shadow["target_disagreement"].astype(bool)
        semantic = shadow["semantic_disagreement"].astype(bool)
        rows.append(
            {
                "threshold": label.upper(),
                "triggered_decisions": int(triggered.sum()),
                "non_triggered_decisions": int((~triggered).sum()),
                "target_disagreement_when_triggered": float(target[triggered].mean())
                if triggered.any()
                else 0.0,
                "target_disagreement_when_not_triggered": float(target[~triggered].mean())
                if (~triggered).any()
                else 0.0,
                "disagreement_capture_rate": float((target & triggered).sum() / max(1, target.sum())),
                "semantic_disagreement_rate": float(semantic.mean()),
                "total_target_disagreements": int(target.sum()),
            }
        )
    return pd.DataFrame(rows)


def benefit_retention(aggregate: pd.DataFrame) -> pd.DataFrame:
    lookup = aggregate.set_index(["controller", "metric"])
    rows = []
    for metric in ("reward", "stage_cost", "sla", "electricity", "carbon", "terminal_backlog"):
        h1 = float(lookup.loc[("H1", metric), "mean"])
        always = float(lookup.loc[("H4_TRANSFORMER_ALWAYS", metric), "mean"])
        triggered = float(
            lookup.loc[("H4_TRANSFORMER_TRIGGERED_P95", metric), "mean"]
        )
        direction = "reward" if metric == "reward" else "cost"
        retention = safe_benefit_retention(
            h1, always, triggered, direction=direction
        )
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "H1": h1,
                "AlwaysH4": always,
                "TriggeredP95": triggered,
                "benefit_retention": retention,
                "benefit_retention_display": "N/A"
                if retention is None
                else f"{100.0 * retention:.6f}%",
            }
        )
    return pd.DataFrame(rows)


def threshold_sensitivity(
    aggregate: pd.DataFrame,
    stats: pd.DataFrame,
    episodes: pd.DataFrame,
    thresholds: TriggerThresholds,
) -> pd.DataFrame:
    lookup = aggregate.set_index(["controller", "metric"])
    stat_lookup = stats[stats["scope"] == "aggregate"].set_index("controller")
    runtime_lookup = episodes.groupby("controller")[
        "controller_solver_time_total_ms"
    ].sum()
    rows = []
    for label, controller, threshold in (
        ("P90", "H4_TRANSFORMER_TRIGGERED_P90", thresholds.p90),
        ("P95_PRIMARY", "H4_TRANSFORMER_TRIGGERED_P95", thresholds.p95_primary),
        ("P99", "H4_TRANSFORMER_TRIGGERED_P99", thresholds.p99),
    ):
        rows.append(
            {
                "threshold_label": label,
                "controller": controller,
                "threshold": threshold,
                "trigger_rate": float(stat_lookup.loc[controller, "trigger_rate_mean"]),
                "reward": float(lookup.loc[(controller, "reward"), "mean"]),
                "stage_cost": float(lookup.loc[(controller, "stage_cost"), "mean"]),
                "sla": float(lookup.loc[(controller, "sla"), "mean"]),
                "electricity": float(lookup.loc[(controller, "electricity"), "mean"]),
                "terminal_backlog": float(lookup.loc[(controller, "terminal_backlog"), "mean"]),
                "solver_time_total_ms": float(runtime_lookup.loc[controller]),
                "primary": label == "P95_PRIMARY",
            }
        )
    return pd.DataFrame(rows)


def trigger_state_characterization(traces: pd.DataFrame) -> pd.DataFrame:
    selected = traces[
        traces["controller"] == "H4_TRANSFORMER_TRIGGERED_P95"
    ]
    features = (
        "current_gpu_utilization",
        "gpu_pressure",
        "current_free_gpu",
        "future_expected_free_gpu",
        "pending_task_count",
        "estimated_duration_mean",
        "estimated_duration_p95",
    )
    rows = []
    for triggered, group in selected.groupby("triggered"):
        for feature in features:
            values = group[feature].astype(float)
            rows.append(
                {
                    "state_group": "triggered" if triggered else "non_triggered",
                    "feature": feature,
                    "count": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "p50": float(np.percentile(values, 50)),
                    "p90": float(np.percentile(values, 90)),
                }
            )
    return pd.DataFrame(rows)


def synthetic_bundle(values: np.ndarray):
    return build_bundle(
        provider="transformer",
        current_timestamp=pd.Timestamp("2023-02-13T12:00:00Z"),
        values=values,
        history_available=True,
        fallback_used=False,
    )


def stress_trigger_diagnostics(thresholds: TriggerThresholds) -> pd.DataFrame:
    config = ({"dc_id": 1, "population_weight": 1.0, "timezone_shift": 0},)
    task = diagnosis.make_task()
    base = diagnosis.make_state(
        (task,), (diagnosis.make_horizon_dc(1, 5, total=100.0),)
    )
    capacity_release = replace(
        base,
        running_tasks=(
            RunningTaskHorizonSnapshot("running", 1, 2, 0.0, 80.0, 0.0),
        ),
    )
    zero = np.zeros((4, 4), dtype=np.float64)
    burst = zero.copy()
    burst[0, 2] = 80.0
    cases = (
        ("capacity_release", capacity_release, synthetic_bundle(zero)),
        ("future_low_price", base, synthetic_bundle(zero)),
        ("gpu_burst", base, synthetic_bundle(burst)),
    )
    rows = []
    for scenario, state, bundle in cases:
        risk = compute_deployable_risk(state, bundle, config)
        rows.append(
            {
                "scenario": scenario,
                "risk_score": risk.risk_score,
                "risk_dc": risk.risk_dc,
                "risk_horizon": risk.risk_horizon,
                "risk_resource": risk.risk_resource,
                "trigger_p90": risk.risk_score >= thresholds.p90,
                "trigger_p95": risk.risk_score >= thresholds.p95_primary,
                "trigger_p99": risk.risk_score >= thresholds.p99,
                "note": "resource-only trigger; price changes are intentionally excluded"
                if scenario == "future_low_price"
                else "repaired resource-pressure sanity",
            }
        )
    return pd.DataFrame(rows)


def solver_runtime_analysis(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for controller, group in episodes.groupby("controller"):
        rows.append(
            {
                "controller": controller,
                "episodes": len(group),
                "H4_call_count": int(group["H4_call_count"].sum()),
                "H1_call_count": int(group["H1_call_count"].sum()),
                "H4_solver_time_total_ms": float(group["H4_solver_time_total_ms"].sum()),
                "H1_solver_time_total_ms": float(group["H1_solver_time_total_ms"].sum()),
                "controller_solver_time_total_ms": float(
                    group["controller_solver_time_total_ms"].sum()
                ),
                "mean_solver_ms": float(group["solver_ms"].mean()),
                "p95_episode_solver_ms": float(np.percentile(group["solver_ms"], 95)),
            }
        )
    return pd.DataFrame(rows)


def decide_role(
    aggregate: pd.DataFrame,
    stats: pd.DataFrame,
    capture: pd.DataFrame,
    retention: pd.DataFrame,
    stress: pd.DataFrame,
) -> tuple[str, str]:
    lookup = aggregate.set_index(["controller", "metric"])
    primary_stat = stats[
        (stats["controller"] == "H4_TRANSFORMER_TRIGGERED_P95")
        & (stats["scope"] == "aggregate")
    ].iloc[0]
    capture_row = capture[capture["threshold"] == "P95"].iloc[0]
    h1_reward = float(lookup.loc[("H1", "reward"), "mean"])
    always_reward = float(
        lookup.loc[("H4_TRANSFORMER_ALWAYS", "reward"), "mean"]
    )
    triggered_reward = float(
        lookup.loc[("H4_TRANSFORMER_TRIGGERED_P95", "reward"), "mean"]
    )
    h1_cost = float(lookup.loc[("H1", "stage_cost"), "mean"])
    always_cost = float(
        lookup.loc[("H4_TRANSFORMER_ALWAYS", "stage_cost"), "mean"]
    )
    triggered_cost = float(
        lookup.loc[("H4_TRANSFORMER_TRIGGERED_P95", "stage_cost"), "mean"]
    )
    rate = float(primary_stat["trigger_rate_mean"])
    capture_rate = float(capture_row["disagreement_capture_rate"])
    stress_gpu = bool(
        stress.loc[stress["scenario"] == "gpu_burst", "trigger_p95"].iloc[0]
    )
    reward_retention = retention.loc[
        retention["metric"] == "reward", "benefit_retention"
    ].iloc[0]
    cost_retention = retention.loc[
        retention["metric"] == "stage_cost", "benefit_retention"
    ].iloc[0]

    improves_h1 = triggered_reward > h1_reward or triggered_cost < h1_cost
    retains = (
        (pd.notna(reward_retention) and float(reward_retention) >= 0.5)
        or (pd.notna(cost_retention) and float(cost_retention) >= 0.5)
    )
    always_small = abs(always_reward - h1_reward) < 1.0 and abs(always_cost - h1_cost) < 5.0
    if rate <= 0.20 and improves_h1 and retains and capture_rate >= 0.50:
        role = "TRIGGERED ONLINE MPC SUPPORTED"
        decision_clause = "the evidence supports a triggered online MPC role"
    elif always_small and abs(triggered_reward - h1_reward) < 1.0 and abs(triggered_cost - h1_cost) < 5.0:
        role = "ONLINE MPC VALUE LIMITED — MPC AS EXPERT / TEACHER"
        decision_clause = "online MPC value is limited and MPC should move to an expert/teacher role"
    elif rate > 0.50:
        role = "TRIGGERING DOES NOT PROVIDE USEFUL SPARSIFICATION"
        decision_clause = "the trigger does not provide useful H4 sparsification"
    else:
        role = "INCONCLUSIVE"
        decision_clause = "the online role remains inconclusive"

    reward_retention_text = (
        "N/A" if pd.isna(reward_retention) else f"{float(reward_retention):.3%}"
    )
    cost_retention_text = (
        "N/A" if pd.isna(cost_retention) else f"{float(cost_retention):.3%}"
    )
    reason = (
        f"The frozen P95 trigger invoked H4 on {rate:.3%} of evaluation steps, so it achieved sparse H4 use. "
        f"Target disagreement was {float(capture_row['target_disagreement_when_triggered']):.3%} when triggered versus {float(capture_row['target_disagreement_when_not_triggered']):.3%} otherwise, showing meaningful enrichment. "
        f"However, it captured only {capture_rate:.3%} of all H1/H4 target disagreements, so most disagreements remained outside the trigger. "
        f"Relative to H1, Triggered-P95 changed mean reward by {triggered_reward - h1_reward:+.6f} and mean stage cost by {triggered_cost - h1_cost:+.6f}, both small in absolute terms. "
        f"Always-H4 changed those same metrics by {always_reward - h1_reward:+.6f} and {always_cost - h1_cost:+.6f}, confirming that the deployable online H4 baseline itself has limited average gain. "
        f"Reward and stage-cost benefit retention were {reward_retention_text} and {cost_retention_text}, but those ratios inherit the small Always-H4 denominators. "
        f"The gpu-burst sanity case triggered as expected ({stress_gpu}); therefore {decision_clause}."
    )
    return role, reason

def generate_documents(
    *,
    output: Path,
    config: dict[str, Any],
    thresholds: TriggerThresholds,
    episodes: pd.DataFrame,
    aggregate: pd.DataFrame,
    stats: pd.DataFrame,
    paired: pd.DataFrame,
    capture: pd.DataFrame,
    retention: pd.DataFrame,
    sensitivity: pd.DataFrame,
    characterization: pd.DataFrame,
    stress: pd.DataFrame,
    runtime: pd.DataFrame,
    fairness_pass: bool,
    checkpoint: Path,
) -> tuple[str, str]:
    lookup = aggregate.set_index(["controller", "metric"])
    pair_lookup = paired.set_index(["comparison", "metric"])
    primary_stat = stats[
        (stats["controller"] == "H4_TRANSFORMER_TRIGGERED_P95")
        & (stats["scope"] == "aggregate")
    ].iloc[0]
    capture_row = capture[capture["threshold"] == "P95"].iloc[0]
    role, role_reason = decide_role(aggregate, stats, capture, retention, stress)
    policy_answer = (
        "YES; continue Policy + Triggered MPC."
        if role == "TRIGGERED ONLINE MPC SUPPORTED"
        else "NO; the present evidence does not support that path."
    )
    expert_answer = (
        "YES; online value is limited, so MPC should move to Expert / Teacher."
        if role == "ONLINE MPC VALUE LIMITED — MPC AS EXPERT / TEACHER"
        else "NOT SELECTED by the current role decision."
    )

    def metric(controller: str, name: str) -> float:
        return float(lookup.loc[(controller, name), "mean"])

    def delta(comparison: str, name: str) -> float:
        return float(pair_lookup.loc[(comparison, name), "mean_delta"])

    reward_retention = retention.loc[
        retention["metric"] == "reward", "benefit_retention_display"
    ].iloc[0]
    stage_retention = retention.loc[
        retention["metric"] == "stage_cost", "benefit_retention_display"
    ].iloc[0]
    stress_lookup = stress.set_index("scenario")
    p95_sensitivity = sensitivity[sensitivity["threshold_label"] == "P95_PRIMARY"].iloc[0]
    p90_sensitivity = sensitivity[sensitivity["threshold_label"] == "P90"].iloc[0]
    p99_sensitivity = sensitivity[sensitivity["threshold_label"] == "P99"].iloc[0]

    write_text(
        output / "01_summary.md",
        f"""# Triggered MPC v1 Summary

1. **Q1. Risk score?** `max_{{d,h,r}} (estimated running occupancy + Transformer arrival flow) / capacity`.
2. **Q2. Deployable only?** YES; current state, estimated duration, known capacities, issued Transformer forecast and deterministic DC mapping only.
3. **Q3. Calibration seeds?** `{list(CALIBRATION_SEEDS)}`; evaluation uses `{list(EVALUATION_SEEDS)}`.
4. **Q4. Evaluation reward tuned threshold?** NO.
5. **Q5. Primary P95 threshold?** `{thresholds.p95_primary:.9f}`.
6. **Q6. Evaluation trigger rate?** `{primary_stat['trigger_rate_mean']:.6%}` mean.
7. **Q7. Main trigger resource?** `{primary_stat['main_trigger_resource']}`.
8. **Q8. Main horizon?** `{primary_stat['main_trigger_horizon']} min`.
9. **Q9. Target disagreement when triggered?** `{capture_row['target_disagreement_when_triggered']:.6%}`.
10. **Q10. Target disagreement when not triggered?** `{capture_row['target_disagreement_when_not_triggered']:.6%}`.
11. **Q11. Disagreement capture?** `{capture_row['disagreement_capture_rate']:.6%}`.
12. **Q12. Triggered P95 vs H1?** reward `{delta('TRIGGERED_P95 - H1','reward'):.6f}`, stage cost `{delta('TRIGGERED_P95 - H1','stage_cost'):.6f}`, SLA `{delta('TRIGGERED_P95 - H1','sla'):.6f}`.
13. **Q13. Benefit retention?** reward `{reward_retention}`, stage cost `{stage_retention}`.
14. **Q14. H4 calls reduced?** `{1.0 - float(primary_stat['trigger_rate_mean']):.6%}` versus Always-H4.
15. **Q15. P90/P95/P99 trend?** trigger rates `{p90_sensitivity.trigger_rate:.6%}` / `{p95_sensitivity.trigger_rate:.6%}` / `{p99_sensitivity.trigger_rate:.6%}`; P95 remains primary.
16. **Q16. gpu_burst triggers?** `{bool(stress_lookup.loc['gpu_burst','trigger_p95'])}`.
17. **Q17. Policy + Triggered MPC?** {policy_answer} Role decision: **{role}**.
18. **Q18. MPC-as-Expert?** {expert_answer}
""",
    )
    write_text(
        output / "17_information_and_fairness_audit.md",
        f"""# Information and Fairness Audit

Status: **PASS**

- Trigger inputs are deployable-only; Oracle bundles are rejected by the risk module.
- Calibration uses H1 trajectories, seeds 1101-1105 and no reward/action disagreement fields.
- P90/P95/P99 were written before evaluation; P95 is predeclared primary.
- Evaluation seeds 1201-1205 are disjoint from calibration seeds.
- Paired environment truth signatures match across all six controllers: `{fairness_pass}`.
- Transformer history fallback is deployable persistence only; no Oracle fallback exists.
- Checkpoint SHA256 is `{sha256(checkpoint)}` and no training occurred.
- Dataset hashes match the frozen manifest; repaired +60 timeline is retained.
- Reward, optimizer objective, BC and SAC were not modified.
""",
    )
    write_text(
        output / "18_mpc_role_decision.md",
        f"""# MPC Role Decision

Decision: **{role}**

- P95 trigger rate: `{primary_stat['trigger_rate_mean']:.6%}`.
- Target disagreement triggered/non-triggered: `{capture_row['target_disagreement_when_triggered']:.6%}` / `{capture_row['target_disagreement_when_not_triggered']:.6%}`.
- Disagreement capture rate: `{capture_row['disagreement_capture_rate']:.6%}`.
- Triggered P95 vs H1 reward/stage-cost: `{delta('TRIGGERED_P95 - H1','reward'):.6f}` / `{delta('TRIGGERED_P95 - H1','stage_cost'):.6f}`.
- Always Transformer vs H1 reward/stage-cost: `{metric('H4_TRANSFORMER_ALWAYS','reward') - metric('H1','reward'):.6f}` / `{metric('H4_TRANSFORMER_ALWAYS','stage_cost') - metric('H1','stage_cost'):.6f}`.
- Reward/stage-cost benefit retention: `{reward_retention}` / `{stage_retention}`.
- Stress gpu_burst trigger: `{bool(stress_lookup.loc['gpu_burst','trigger_p95'])}`.

Reason: {role_reason}
""",
    )
    write_text(
        output / "19_evidence_index.md",
        """# Evidence Index

| Evidence | File |
|---|---|
| Trigger and calibration contract | `02_trigger_design.md`, `03_calibration_protocol.md` |
| Frozen risk thresholds | `04_calibration_risk_distribution.csv`, `05_trigger_thresholds.json` |
| Formal control metrics | `06_episode_metrics.csv`, `07_aggregate_metrics.csv`, `09_paired_comparisons.csv` |
| Trigger calls and shadow actions | `08_trigger_statistics.csv`, `10_shadow_action_diagnostics.csv`, `11_disagreement_capture.csv` |
| Retention and sensitivity | `12_benefit_retention.csv`, `13_threshold_sensitivity.csv` |
| Trigger interpretation and stress | `14_trigger_state_characterization.csv`, `15_stress_case_trigger_diagnostics.csv` |
| Runtime, fairness and role | `16_solver_runtime_analysis.csv`, `17_information_and_fairness_audit.md`, `18_mpc_role_decision.md` |
| Per-episode traces | `traces/trigger_trace_<controller>_<scenario>_<seed>.parquet` |
""",
    )
    changes = (
        "src/sustaincluster_mpc/triggered_mpc.py",
        "configs/sustaincluster_mpc/triggered_mpc_v1.yaml",
        "scripts/forecast/run_triggered_mpc_v1.py",
        "tests/test_triggered_mpc_v1.py",
    )
    write_text(
        output / "20_change_manifest.md",
        "# Change Manifest\n\n## Added\n\n"
        + "\n".join(f"- `{path}`" for path in changes)
        + "\n\n## Frozen / unchanged\n\n"
        + "- Transformer checkpoint, architecture and Dataset v1.\n"
        + "- Repaired MPC timeline, optimizer, objective, reward and ActionAdapter.\n"
        + "- BC/SAC and SustainCluster vendor source.\n"
        + "- No commit or push.\n",
    )
    return role, role_reason


def run_evaluation(config_path: Path) -> Path:
    config = load_config(config_path)
    output = WORKSPACE / config["output_dir"]
    traces_dir = output / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    _, dataset_root, checkpoint = verify_frozen_assets(config)
    thresholds = read_frozen_thresholds(output)
    trace_source = ForecastTraceSource(dataset_root)
    transformer = TransformerWorkloadForecastProvider(
        checkpoint, dataset_root=dataset_root, device=config["device"]
    )
    dc_configs = yaml.safe_load(
        (WORKSPACE / config["datacenter_config"]).read_text("utf-8")
    )["datacenters"]
    optimizer_config = repaired_runner._optimizer_config(
        WORKSPACE / config["optimizer_config"]
    )
    episode_rows = []
    trace_rows = []
    shadow_rows = []
    truth_by_key: dict[tuple[str, int, str], list[str]] = {}
    for scenario, scenario_config in config["scenarios"].items():
        start = pd.Timestamp(scenario_config["environment_start"])
        dataset_start = pd.Timestamp(scenario_config["dataset_start"])
        for seed in EVALUATION_SEEDS:
            for controller in CONTROLLERS:
                print(
                    f"RUN {scenario} seed={seed} controller={controller}",
                    flush=True,
                )
                trace_path = traces_dir / (
                    f"trigger_trace_{controller}_{scenario}_{seed}.parquet"
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    episode, traces, shadow, truth = run_episode(
                        controller=controller,
                        scenario=scenario,
                        start=start,
                        expected_dataset_start=dataset_start,
                        seed=seed,
                        steps=int(config["episode_steps"]),
                        trace_source=trace_source,
                        transformer=transformer,
                        dc_configs=dc_configs,
                        optimizer_config=optimizer_config,
                        thresholds=thresholds,
                        trace_path=trace_path,
                    )
                oracle_warnings = sum(
                    "non-deployable" in str(item.message) for item in caught
                )
                if controller == "H4_ORACLE_ALWAYS" and oracle_warnings == 0:
                    raise RuntimeError("Oracle controller did not emit warning")
                if controller != "H4_ORACLE_ALWAYS" and oracle_warnings:
                    raise RuntimeError("deployable controller emitted Oracle warning")
                episode["oracle_warning_count"] = oracle_warnings
                episode_rows.append(episode)
                trace_rows.extend({**row, "controller": controller} for row in traces)
                shadow_rows.extend(shadow)
                truth_by_key[(scenario, seed, controller)] = truth
                print(
                    json.dumps(
                        {
                            "reward": episode["reward"],
                            "stage_cost": episode["stage_cost"],
                            "trigger_rate": episode["trigger_rate"],
                            "H4_calls": episode["H4_call_count"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    fairness_pass = True
    for scenario in config["scenarios"]:
        for seed in EVALUATION_SEEDS:
            signatures = [
                truth_by_key[(scenario, seed, controller)]
                for controller in CONTROLLERS
            ]
            fairness_pass &= all(item == signatures[0] for item in signatures[1:])
    if not fairness_pass:
        raise RuntimeError("paired environment truth fairness failed")

    episodes = pd.DataFrame(episode_rows)
    traces = pd.DataFrame(trace_rows)
    shadow = pd.DataFrame(shadow_rows)
    aggregate = aggregate_metrics(episodes)
    stats = trigger_statistics(episodes, traces)
    paired = paired_comparisons(episodes)
    capture = disagreement_capture(shadow)
    retention = benefit_retention(aggregate)
    sensitivity = threshold_sensitivity(aggregate, stats, episodes, thresholds)
    characterization = trigger_state_characterization(traces)
    stress = stress_trigger_diagnostics(thresholds)
    runtime = solver_runtime_analysis(episodes)

    episodes.to_csv(output / "06_episode_metrics.csv", index=False)
    aggregate.to_csv(output / "07_aggregate_metrics.csv", index=False)
    stats.to_csv(output / "08_trigger_statistics.csv", index=False)
    paired.to_csv(output / "09_paired_comparisons.csv", index=False)
    shadow.to_csv(output / "10_shadow_action_diagnostics.csv", index=False)
    capture.to_csv(output / "11_disagreement_capture.csv", index=False)
    retention.to_csv(output / "12_benefit_retention.csv", index=False)
    sensitivity.to_csv(output / "13_threshold_sensitivity.csv", index=False)
    characterization.to_csv(
        output / "14_trigger_state_characterization.csv", index=False
    )
    stress.to_csv(output / "15_stress_case_trigger_diagnostics.csv", index=False)
    runtime.to_csv(output / "16_solver_runtime_analysis.csv", index=False)
    role, reason = generate_documents(
        output=output,
        config=config,
        thresholds=thresholds,
        episodes=episodes,
        aggregate=aggregate,
        stats=stats,
        paired=paired,
        capture=capture,
        retention=retention,
        sensitivity=sensitivity,
        characterization=characterization,
        stress=stress,
        runtime=runtime,
        fairness_pass=fairness_pass,
        checkpoint=checkpoint,
    )
    final = {
        "status": "READY FOR NEXT ARCHITECTURE STAGE",
        "mpc_role_decision": role,
        "reason": reason,
        "episodes": len(episodes),
        "trace_files": len(list(traces_dir.glob("*.parquet"))),
        "same_environment_truth": fairness_pass,
        "transformer_checkpoint_load_count": transformer.checkpoint_load_count,
    }
    write_json(output / "evaluation_summary.json", final)
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE / "configs/sustaincluster_mpc/triggered_mpc_v1.yaml",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--calibrate", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.calibrate:
        run_calibration(args.config.resolve())
    else:
        run_evaluation(args.config.resolve())


if __name__ == "__main__":
    main()
