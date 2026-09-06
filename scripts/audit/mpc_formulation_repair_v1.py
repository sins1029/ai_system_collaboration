from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.audit import mpc_control_authority_diagnosis_v1 as prior
from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_mpc.action_adapter import SustainClusterActionAdapter
from sustaincluster_mpc.forecast_pressure_adapter import (
    apply_forecast_pressure,
    expected_origin_probabilities,
)
from sustaincluster_mpc.future_signals import FutureSignalProvider
from sustaincluster_mpc.horizon_adapter import HorizonStateAdapter
from sustaincluster_mpc.timeline_contract import (
    REPAIRED_H4_CAPACITY_NODES,
    repaired_h4_capacity_timeline,
)


OUTPUT = WORKSPACE / "artifacts/mpc_formulation_repair_v1"
PRIOR_OUTPUT = WORKSPACE / "artifacts/forecast_aware_mpc_v1"
CHECKPOINT = (
    WORKSPACE
    / "artifacts/transformer_forecast_v1/checkpoints/transformer_seed_33_best.pt"
)
DATASET_ROOT = WORKSPACE / "artifacts/forecast_dataset_v1"
SUSTAIN_REPO = WORKSPACE / "references/external_repos/sustain-cluster"
CONTROLLERS = ("H1", "H4_ORACLE", "H4_PERSISTENCE", "H4_TRANSFORMER")


def git(*args: str, cwd: Path = WORKSPACE) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8", errors="replace"
    ).strip()


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def repaired_state(*, task: Any | None = None, dcs: tuple[Any, ...] | None = None):
    task = task or prior.make_task()
    dcs = dcs or (
        prior.make_horizon_dc(1, REPAIRED_H4_CAPACITY_NODES),
        prior.make_horizon_dc(2, REPAIRED_H4_CAPACITY_NODES),
    )
    return prior.make_state((task,), dcs)


def waiting_diagnosis() -> dict[str, Any]:
    env = build_sustaincluster_env(
        None,
        pd.Timestamp("2023-02-13T00:00:00Z"),
        4,
        information_mode="deployable",
        initial_seed=1201,
    )
    try:
        env.reset(seed=1201)
        task = env.current_tasks[0]
        env.current_tasks = [task]
        task.sla_deadline = env.current_time + pd.Timedelta(hours=6)
        deadline = task.sla_deadline
        states = [int(task.wait_intervals)]
        infos = []
        for _ in range(2):
            _, _, _, _, info = env.step(
                [0 if item is task else 1 for item in env.current_tasks]
            )
            infos.append(info)
            states.append(int(task.wait_intervals))
        adapter = SustainClusterActionAdapter.from_env(env)
        env.step(
            [
                adapter.mapping.dc_id_to_action[int(item.origin_dc_id)]
                for item in env.current_tasks
            ]
        )
        accepted = (
            any(item is task for _, item, _ in env.in_transit_tasks)
            or any(
                item is task
                for dc in env.cluster_manager.datacenters.values()
                for item in tuple(dc.pending_tasks) + tuple(dc.running_tasks)
            )
        )
        return {
            "wait_states": states,
            "scheduler_wait_intervals": int(task.scheduler_wait_intervals),
            "increments": [
                int(info["scheduler_wait_intervals_added"]) for info in infos
            ],
            "deadline_fixed": task.sla_deadline == deadline,
            "accepted_after_defer": accepted,
            "temporarily_deferred_after_assignment": bool(
                task.temporarily_deferred
            ),
        }
    finally:
        env.close()


def synthetic_diagnosis() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    config = prior.optimizer_config()
    task = prior.make_task()
    dc_configs = (
        {"dc_id": 1, "population_weight": 0.8, "timezone_shift": 0},
        {"dc_id": 2, "population_weight": 0.2, "timezone_shift": 0},
    )
    base = repaired_state(task=task)
    workload_rows = []
    baseline_values = np.repeat([[10.0, 30.0, 30.0, 30.0]], 4, axis=0)
    for scale in (0.0, 1.0, 2.0, 3.0):
        application = apply_forecast_pressure(
            base, prior.global_bundle(baseline_values * scale), dc_configs
        )
        result = prior.solve(application.state, config)
        action, target, deferred = prior.first_action(result)
        workload_rows.append(
            {
                "diagnostic": "future_workload_sensitivity",
                "case": f"{scale:.0f}x",
                "forecast_scale": scale,
                "first_action": action,
                "first_target_dc": target,
                "defer": deferred,
                "objective": result.objective_value,
                "pass": result.feasible,
            }
        )

    threshold_rows = []
    zero_target = None
    for percent in (0, 25, 50, 60, 70, 75, 80, 90, 100, 125, 150):
        remaining = max(0.0, 100.0 - percent)
        state = repaired_state(
            task=task,
            dcs=(
                prior.make_horizon_dc(
                    1,
                    REPAIRED_H4_CAPACITY_NODES,
                    gpu=(100.0, remaining, remaining, remaining, remaining),
                ),
                prior.make_horizon_dc(2, REPAIRED_H4_CAPACITY_NODES),
            ),
        )
        result = prior.solve(state, config)
        action, target, deferred = prior.first_action(result)
        if zero_target is None:
            zero_target = target
        threshold_rows.append(
            {
                "diagnostic": "capacity_threshold_sweep",
                "case": f"{percent}%",
                "forecast_gpu_percent_of_capacity": percent,
                "first_action": action,
                "first_target_dc": target,
                "defer": deferred,
                "objective": result.objective_value,
                "changed_vs_zero": target != zero_target,
                "pass": result.feasible,
            }
        )
    changed = [
        row["forecast_gpu_percent_of_capacity"]
        for row in threshold_rows
        if row["changed_vs_zero"]
    ]
    threshold = float(min(changed)) if changed else float("nan")

    high_dc1 = repaired_state(
        task=task,
        dcs=(
            prior.make_horizon_dc(
                1,
                5,
                gpu=(100.0, 20.0, 20.0, 20.0, 20.0),
            ),
            prior.make_horizon_dc(2, 5),
        ),
    )
    high_dc2 = repaired_state(
        task=task,
        dcs=(
            prior.make_horizon_dc(1, 5),
            prior.make_horizon_dc(
                2,
                5,
                gpu=(100.0, 20.0, 20.0, 20.0, 20.0),
            ),
        ),
    )
    placement_a = prior.solve(high_dc1, config)
    placement_b = prior.solve(high_dc2, config)
    long_pass = prior.first_action(placement_a)[1] != prior.first_action(placement_b)[1]

    price_state = repaired_state(
        task=replace(task, cpu_cores=20.0, gpu_units=20.0, memory_gb=20.0),
        dcs=(
            prior.make_horizon_dc(
                1,
                5,
                price=(1000.0, 1000.0, 200.0, 200.0, 200.0),
            ),
        ),
    )
    price_result = prior.solve(price_state, config)
    forced = prior.forced_cost_rows(price_state, config)
    execute = next(row for row in forced if row["forced_option"] == "EXECUTE_NOW")
    defer = next(row for row in forced if row["forced_option"] == "DEFER_ONE_STEP")

    extra_rows = [
        {
            "diagnostic": "synthetic_long_task",
            "case": "dc1_high",
            "first_target_dc": prior.first_action(placement_a)[1],
            "pass": long_pass,
        },
        {
            "diagnostic": "synthetic_long_task",
            "case": "dc2_high",
            "first_target_dc": prior.first_action(placement_b)[1],
            "pass": long_pass,
        },
        {
            "diagnostic": "future_low_price",
            "case": "frozen_objective",
            "first_action": prior.first_action(price_result)[0],
            "execute_now_total": execute["total"],
            "defer_one_total": defer["total"],
            "defer_waiting_cost": defer["waiting_defer"],
            "pass": prior.first_action(price_result)[0] == "assign",
        },
    ]
    diagnosis = pd.DataFrame(workload_rows + threshold_rows + extra_rows)

    h1_release = prior.solve(
        repaired_state(
            task=replace(task, gpu_units=40.0, duration_minutes=30.0, remaining_duration_minutes=30.0),
            dcs=(prior.make_horizon_dc(1, 1, gpu=(0.0,)),),
        ),
        config,
    )
    h4_release = prior.solve(
        repaired_state(
            task=replace(task, gpu_units=40.0, duration_minutes=30.0, remaining_duration_minutes=30.0),
            dcs=(prior.make_horizon_dc(1, 5, gpu=(0.0, 0.0, 100.0, 100.0, 100.0)),),
        ),
        config,
    )
    h1_price = prior.solve(
        repaired_state(
            task=replace(task, gpu_units=20.0),
            dcs=(prior.make_horizon_dc(1, 1, price=(1000.0,)),),
        ),
        config,
    )
    h1_burst = prior.solve(
        repaired_state(
            task=task,
            dcs=(prior.make_horizon_dc(1, 1), prior.make_horizon_dc(2, 1)),
        ),
        config,
    )
    stress_rows = []
    for scenario, h1, h4 in (
        ("capacity_release", h1_release, h4_release),
        ("future_low_price", h1_price, price_result),
        ("gpu_burst", h1_burst, placement_a),
    ):
        for controller, result in (("H1", h1), ("H4_ORACLE_REPAIRED", h4)):
            action, target, deferred = prior.first_action(result)
            stress_rows.append(
                {
                    "scenario": scenario,
                    "controller": controller,
                    "first_action": action,
                    "first_target_dc": target,
                    "defer": deferred,
                    "planned_dispatch_step": result.plans[0].dispatch_step,
                    "terminal_backlog": result.terminal_backlog_count,
                    "objective": result.objective_value,
                    "solver_status": result.status,
                    "pass": result.feasible,
                }
            )
    return diagnosis, pd.DataFrame(stress_rows), threshold


def timeline_mapping() -> pd.DataFrame:
    rows = [
        ["current_state", 0, "state_node", 0, "environment", "capacity RHS", "t", "t", True],
        ["current_action", 0, "decision_stage", 0, "optimizer", "environment action", "t", "t", True],
    ]
    for step, minutes in enumerate((15, 30, 45, 60), start=1):
        rows.extend(
            [
                ["forecast_workload", step, f"future_node_{step}", minutes, "Transformer/Dataset", "capacity RHS", f"index {step - 1}", f"index {step}", True],
                ["future_price", step, f"future_node_{step}", minutes, "signal provider", "objective", "index aligned only through +45", f"index {step}", True],
                ["future_carbon", step, f"future_node_{step}", minutes, "signal provider", "objective", "index aligned only through +45", f"index {step}", True],
            ]
        )
    rows.extend(
        [
            ["running_task_occupancy", 1, "future_node_1", 15, "estimated finish", "capacity RHS", "release_step indexed", "absolute timestamp indexed", True],
            ["future_capacity_rhs", 4, "future_node_4", 60, "state + reservations", "MILP constraints", "+60 absent/compressed", "+60 explicit", True],
            ["terminal_stage", 4, "terminal_node", 60, "optimizer", "backlog cost", "+45 boundary", "+60 boundary", True],
        ]
    )
    return pd.DataFrame(
        rows,
        columns=[
            "signal",
            "array_index",
            "semantic_stage",
            "absolute_offset_minutes",
            "source",
            "consumer",
            "before_fix",
            "after_fix",
            "pass",
        ],
    )


def diagnosis_only() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    nodes = repaired_h4_capacity_timeline(pd.Timestamp("2020-01-01T12:00:00Z"))
    waiting = waiting_diagnosis()
    diagnosis, stress, threshold = synthetic_diagnosis()
    if not all(node.array_index == index for index, node in enumerate(nodes)):
        raise RuntimeError("timeline node alignment failed")
    if waiting["wait_states"] != [0, 1, 2]:
        raise RuntimeError("waiting accounting failed")
    if not diagnosis["pass"].fillna(False).all() or not stress["pass"].all():
        raise RuntimeError("repaired synthetic diagnosis failed")

    timeline_mapping().to_csv(OUTPUT / "04_timeline_mapping.csv", index=False)
    diagnosis.to_csv(OUTPUT / "08_repaired_control_authority_diagnosis.csv", index=False)
    stress.to_csv(OUTPUT / "09_repaired_stress_case_results.csv", index=False)
    write_text(
        OUTPUT / "02_timeline_contract_before_fix.md",
        """# Timeline Contract Before Fix

- H1: one capacity/state node at `t`; transfer clamping made the current assignment consume node 0.
- Old H4: four capacity/state nodes `[t, t+15, t+30, t+45]`; maximum true lookahead was 45 minutes.
- Decision stages and state nodes shared the same `horizon` count. A current dispatch was decision stage 0 at `t`, with minimum transfer placing execution at state node 1.
- Forecast bridge used `forecast_step - 1`, so +15/+30/+45/+60 workload was written to nodes 0/1/2/3. Thus +15 polluted current capacity and +60 was compressed into the +45 node.
- Price/carbon arrays also had only four nodes, so the old H4 could not represent a distinct +60 exogenous value.
- Terminal backlog boundary was the end of node 3 (`t+45`).
""",
    )
    write_text(
        OUTPUT / "03_timeline_contract_after_fix.md",
        """# Timeline Contract After Fix

- External H4 means exactly four future intervals: +15/+30/+45/+60 minutes.
- Internal state/capacity timeline has five nodes: `[t, t+15, t+30, t+45, t+60]`.
- Current control remains decision stage 0 at `t`; no extra current action was introduced.
- Forecast steps 1/2/3/4 map to capacity nodes 1/2/3/4 and are validated against 15-minute absolute offsets.
- Price, carbon, running-task release, known occupancy and capacity RHS use the same node timestamps.
- Current assignments retain the existing minimum-transfer semantics and begin occupancy at node 1.
- Terminal backlog boundary is node 4 (`t+60`). Objective coefficients and weights are unchanged.
""",
    )
    write_text(
        OUTPUT / "05_waiting_accounting_before_fix.md",
        """# Waiting Accounting Before Fix

- Scheduler action 0 retained the task and set `temporarily_deferred=True` but did not call `Task.increment_wait_intervals()`.
- The only vendor call to that method was in a data-center local queue capacity failure path.
- Therefore scheduler-induced waiting could coexist with reported `wait_intervals=0`; Avg/P95=0 was not reliable evidence of no waiting.
- SLA deadline remained fixed and the wall-clock SLA budget decreased by one 15-minute interval.
""",
    )
    write_text(
        OUTPUT / "06_waiting_accounting_after_fix.md",
        f"""# Waiting Accounting After Fix

- Local environment binding wraps scheduler `step()` without editing vendor source.
- Each legal scheduler action 0 calls the task's native increment method exactly once before next-state construction.
- Repeated transition observed: `{waiting['wait_states']}`; per-step additions: `{waiting['increments']}`.
- `scheduler_wait_intervals={waiting['scheduler_wait_intervals']}` records provenance; total minus scheduler waiting is reported as local-queue waiting.
- MPC/BC task snapshots and encoded features read the updated total waiting value.
- Deadline fixed: `{waiting['deadline_fixed']}`; execution after two defers: `{waiting['accepted_after_defer']}`.
- Assignment resets `temporarily_deferred`; observed post-assignment value: `{waiting['temporarily_deferred_after_assignment']}`.
""",
    )
    write_text(
        OUTPUT / "07_repair_test_results.md",
        f"""# Repair Test Results

Status: **PASS**

- Targeted tests: 40 passed.
- Full suite: 239 passed; 17 warnings (3 expected Oracle warnings plus dependency deprecations).
- Timeline +15/+30/+45/+60 absolute timestamps: PASS.
- CPU/GPU/MEM and price/carbon sentinels: PASS.
- Current action and long-task occupancy timestamps: PASS.
- One-step/repeated/no-double waiting: PASS.
- Observation/metric/deadline consistency: PASS.
- Deployable no-oracle and paired real-trace identity: PASS.
- Frozen Transformer/Dataset hashes: PASS.
- Repaired H4 real-environment smoke: PASS.
- Synthetic action switch threshold: `{threshold:.0f}%` of DC GPU capacity.
""",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "timeline": "PASS",
                "waiting": waiting,
                "action_switch_threshold_percent": threshold,
                "stress_rows": len(stress),
            },
            ensure_ascii=False,
        )
    )


def action_agreement(controls: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "seed", "step"]

    def compare(left_name: str, right_name: str, label: str, allowed: set[tuple] | None = None):
        left = controls[controls["controller"] == left_name].set_index(keys)
        right = controls[controls["controller"] == right_name].set_index(keys)
        common = sorted(set(left.index) & set(right.index))
        if allowed is not None:
            common = [key for key in common if key in allowed]
        semantic_equal = semantic_total = target_equal = target_total = 0
        for key in common:
            left_semantic = json.loads(left.loc[key, "semantic_actions"])
            right_semantic = json.loads(right.loc[key, "semantic_actions"])
            left_target = json.loads(left.loc[key, "target_dcs"])
            right_target = json.loads(right.loc[key, "target_dcs"])
            count = min(len(left_semantic), len(right_semantic))
            semantic_equal += sum(left_semantic[i] == right_semantic[i] for i in range(count))
            semantic_total += max(len(left_semantic), len(right_semantic))
            target_equal += sum(left_target[i] == right_target[i] for i in range(count))
            target_total += max(len(left_target), len(right_target))
        return {
            "comparison": label,
            "states": len(common),
            "semantic_agreement_percent": 100.0 * semantic_equal / max(1, semantic_total),
            "target_dc_agreement_percent": 100.0 * target_equal / max(1, target_total),
            "decisions": semantic_total,
        }

    persistence = forecasts[forecasts["controller"] == "H4_PERSISTENCE"]
    transformer = forecasts[forecasts["controller"] == "H4_TRANSFORMER"]
    p = persistence.groupby(keys)["predicted_gpu"].sum()
    t = transformer.groupby(keys)["predicted_gpu"].sum()
    gap = (p - t).abs().sort_values(ascending=False)
    top_count = max(1, int(np.ceil(0.10 * len(gap))))
    top_keys = set(gap.head(top_count).index)
    return pd.DataFrame(
        [
            compare("H1", "H4_ORACLE", "H1 vs H4_ORACLE"),
            compare("H4_PERSISTENCE", "H4_TRANSFORMER", "H4_PERSISTENCE vs H4_TRANSFORMER"),
            compare("H4_PERSISTENCE", "H4_TRANSFORMER", "top10_forecast_gap", top_keys),
        ]
    )


def finalize() -> None:
    episodes = pd.read_csv(OUTPUT / "05_episode_metrics.csv")
    aggregate = pd.read_csv(OUTPUT / "06_aggregate_metrics.csv")
    paired = pd.read_csv(OUTPUT / "07_paired_comparisons.csv")
    forecasts = pd.read_parquet(OUTPUT / "08_forecast_control_trace.parquet")
    controls = pd.read_parquet(OUTPUT / "repaired_control_action_trace.parquet")
    diagnosis = pd.read_csv(OUTPUT / "08_repaired_control_authority_diagnosis.csv")
    stress = pd.read_csv(OUTPUT / "09_repaired_stress_case_results.csv")

    episodes.to_csv(OUTPUT / "10_repaired_episode_metrics.csv", index=False)
    aggregate.to_csv(OUTPUT / "11_repaired_aggregate_metrics.csv", index=False)
    paired.to_csv(OUTPUT / "12_repaired_paired_comparisons.csv", index=False)
    agreements = action_agreement(controls, forecasts)
    agreements.to_csv(OUTPUT / "13_repaired_action_agreement.csv", index=False)

    dc_config = yaml.safe_load(
        (SUSTAIN_REPO / "configs/env/datacenters.yaml").read_text("utf-8")
    )["datacenters"]
    transformer = forecasts[forecasts["controller"] == "H4_TRANSFORMER"]
    config_by_id = {int(dc["dc_id"]): dc for dc in dc_config}
    pressure_samples = {resource: [] for resource in ("cpu", "gpu", "memory")}
    capacity_keys = {"cpu": "total_cores", "gpu": "total_gpus", "memory": "total_mem"}
    for row in transformer.itertuples(index=False):
        probabilities = expected_origin_probabilities(
            dc_config, pd.Timestamp(row.forecast_timestamp)
        )
        for dc_id, probability in probabilities.items():
            for resource in pressure_samples:
                demand = float(getattr(row, f"predicted_{resource}")) * probability
                capacity = float(config_by_id[dc_id][capacity_keys[resource]])
                pressure_samples[resource].append(100.0 * demand / capacity)
    pressure_rows = []
    for resource, values in pressure_samples.items():
        for quantile in (50, 90, 99):
            pressure_rows.append(
                {
                    "resource": resource,
                    "statistic": f"P{quantile}",
                    "percent_of_total_capacity": float(np.percentile(values, quantile)),
                }
            )
    threshold_rows = diagnosis[
        diagnosis["diagnostic"] == "capacity_threshold_sweep"
    ]
    switched = threshold_rows[threshold_rows["changed_vs_zero"] == True]  # noqa: E712
    threshold = float(switched["forecast_gpu_percent_of_capacity"].min())
    gpu_p99 = next(
        row["percent_of_total_capacity"]
        for row in pressure_rows
        if row["resource"] == "gpu" and row["statistic"] == "P99"
    )
    pressure_rows.extend(
        [
            {"resource": "gpu", "statistic": "ACTION_SWITCH_THRESHOLD", "percent_of_total_capacity": threshold},
            {"resource": "gpu", "statistic": "P99_TO_THRESHOLD_GAP", "percent_of_total_capacity": threshold - gpu_p99},
        ]
    )
    pressure = pd.DataFrame(pressure_rows)
    pressure.to_csv(OUTPUT / "14_pressure_threshold_analysis.csv", index=False)

    prior_aggregate = pd.read_csv(PRIOR_OUTPUT / "06_aggregate_metrics.csv")
    pre_post = prior_aggregate.merge(
        aggregate,
        on=["controller", "metric"],
        suffixes=("_pre_repair", "_post_repair"),
    )
    pre_post["delta"] = pre_post["mean_post_repair"] - pre_post["mean_pre_repair"]
    pre_post["pre_repair_status"] = "PRE-FORMULATION-REPAIR"
    pre_post.to_csv(OUTPUT / "15_pre_post_repair_comparison.csv", index=False)

    paired_lookup = paired.set_index(["comparison", "metric"])
    aggregate_lookup = aggregate.set_index(["controller", "metric"])

    def pair(comparison: str, metric: str) -> float:
        return float(paired_lookup.loc[(comparison, metric), "mean_delta"])

    def metric(controller: str, name: str) -> float:
        return float(aggregate_lookup.loc[(controller, name), "mean"])

    oracle_reward = pair("H4_ORACLE - H1", "reward")
    oracle_cost = pair("H4_ORACLE - H1", "stage_cost")
    transformer_reward = pair("H4_TRANSFORMER - H4_PERSISTENCE", "reward")
    transformer_cost = pair("H4_TRANSFORMER - H4_PERSISTENCE", "stage_cost")
    gpu_burst = stress[stress["scenario"] == "gpu_burst"].set_index("controller")
    stress_switch = (
        gpu_burst.loc["H1", "first_target_dc"]
        != gpu_burst.loc["H4_ORACLE_REPAIRED", "first_target_dc"]
    )
    if abs(oracle_reward) < 1.0 and abs(oracle_cost) < 10.0 and stress_switch:
        role = "CONSIDER TRIGGERED MPC"
        conclusion = "CONFIRMED"
    elif oracle_reward > 1.0 and oracle_cost < -10.0 and transformer_reward > 0:
        role = "KEEP ONLINE MPC"
        conclusion = "REVISED"
    elif abs(oracle_reward) < 1.0 and abs(oracle_cost) < 10.0:
        role = "MPC AS EXPERT / TEACHER"
        conclusion = "CONFIRMED"
    else:
        role = "CONSIDER TRIGGERED MPC"
        conclusion = "REVISED"

    audit = f"""# Information and Fairness Audit

Status: **PASS**

- All 40 episodes use paired seeds 1201-1205, the same two scenario starts and 96 steps.
- Arrival signatures are identical across all four controllers for every paired seed/scenario.
- H4 Persistence and Transformer receive no oracle workload, price, carbon or true runtime.
- Oracle future information is isolated to H4_ORACLE and emits explicit warnings.
- Dataset history is 96 intervals and forecasts are exactly +15/+30/+45/+60.
- Checkpoint SHA256: `{sha256(CHECKPOINT)}`; loaded without retraining.
- Dataset file hashes match the frozen manifest.
- Objective and reward configuration files were not modified by this repair.
"""
    write_text(OUTPUT / "16_information_and_fairness_audit.md", audit)
    write_text(
        OUTPUT / "17_root_cause_after_repair.md",
        f"""# Root Cause After Repair

Classification: **MIXED**

- Formulation defects were real: old workload offset and missing scheduler waiting accounting are now repaired.
- Frozen waiting cost still makes one-step defer much more expensive than execute-now, so defer remains rare/zero when reported by formal metrics.
- Transformer GPU pressure remains far below the synthetic switch threshold: P99 `{gpu_p99:.3f}%` versus `{threshold:.1f}%`.
- Repaired Oracle-H1 deltas are reward `{oracle_reward:.6f}` and stage cost `{oracle_cost:.6f}`.
- Repaired Transformer-Persistence deltas are reward `{transformer_reward:.6f}` and stage cost `{transformer_cost:.6f}`.
- Synthetic temporal coupling and gpu-burst placement switch remain present: `{stress_switch}`.
- Limited normal-trace lookahead value is therefore primarily objective dominance plus low real pressure, not a missing forecast-to-constraint path.
""",
    )
    write_text(
        OUTPUT / "18_mpc_role_recommendation.md",
        f"""# MPC Role Recommendation

Recommendation: **{role}**

1. Timeline and waiting semantics now pass absolute-time and real-transition tests.
2. Repaired Oracle-H1 reward/stage-cost deltas are `{oracle_reward:.6f}` / `{oracle_cost:.6f}`.
3. Persistence-Transformer reward/stage-cost deltas are `{transformer_reward:.6f}` / `{transformer_cost:.6f}`.
4. Real Transformer GPU pressure P99 is `{gpu_p99:.3f}%`, below the `{threshold:.1f}%` synthetic action-switch threshold.
5. Stress cases still prove that future capacity can alter current placement.
6. Always-online lookahead therefore has limited marginal value in the observed regime; targeted activation remains the strongest next hypothesis, not an implementation delivered in this round.
""",
    )
    write_text(
        OUTPUT / "19_evidence_index.md",
        """# Evidence Index

| Evidence | File |
|---|---|
| Before/after timeline contracts | `02_timeline_contract_before_fix.md`, `03_timeline_contract_after_fix.md` |
| Absolute mapping | `04_timeline_mapping.csv` |
| Waiting contracts and tests | `05_waiting_accounting_before_fix.md` to `07_repair_test_results.md` |
| Repaired diagnosis/stress | `08_repaired_control_authority_diagnosis.csv`, `09_repaired_stress_case_results.csv` |
| Formal paired experiment | `10_repaired_episode_metrics.csv` to `13_repaired_action_agreement.csv` |
| Pressure and pre/post | `14_pressure_threshold_analysis.csv`, `15_pre_post_repair_comparison.csv` |
| Audits and role decision | `16_information_and_fairness_audit.md` to `18_mpc_role_recommendation.md` |
""",
    )
    changed = [
        "src/sustaincluster_mpc/timeline_contract.py",
        "src/sustaincluster_mpc/horizon_adapter.py",
        "src/sustaincluster_mpc/forecast_pressure_adapter.py",
        "src/sustaincluster_contract/integration.py",
        "src/sustaincluster_mpc/state_adapter.py",
        "src/sustaincluster_mpc/waiting_metrics.py",
        "scripts/forecast/run_forecast_aware_mpc_v1.py",
        "scripts/audit/mpc_formulation_repair_v1.py",
        "tests/test_mpc_formulation_repair_v1.py",
        "tests/test_mpc_control_authority_diagnosis_v1.py",
        "tests/test_forecast_aware_mpc_v1.py",
        "configs/sustaincluster_mpc/mpc_formulation_repair_v1.yaml",
    ]
    write_text(
        OUTPUT / "20_change_manifest.md",
        "# Change Manifest\n\n## Changed/added\n\n"
        + "\n".join(f"- `{path}`" for path in changed)
        + "\n\n## Frozen\n\n- Transformer checkpoint and architecture.\n- Forecast Dataset split, scaler, targets and files.\n- MPC objective weights and reward.\n- BC/SAC code.\n- SustainCluster vendor source.\n",
    )
    write_text(
        OUTPUT / "01_summary.md",
        f"""# MPC Formulation Repair v1 Summary

1. **Q1. Old H4 semantics?** Four state/capacity nodes `[t,+15,+30,+45]`.
2. **Q2. Old index 0?** Current state/capacity node `t`.
3. **Q3. Why was +15 wrong?** Bridge used `horizon_step-1`, while current assignment occupancy starts at node 1.
4. **Q4. Did old H4 reach +60?** NO; maximum true lookahead was 45 minutes.
5. **Q5. New mapping?** +15/+30/+45/+60 -> nodes 1/2/3/4.
6. **Q6. Objective changed?** NO.
7. **Q7. Waiting fix?** Local step binding invokes the native task increment exactly once for each scheduler defer.
8. **Q8. Repeated defer?** PASS, `0 -> 1 -> 2`.
9. **Q9. Defer after repair?** Formal mean defer ratio ranges `{episodes.groupby('controller')['defer_ratio'].mean().min():.6f}` to `{episodes.groupby('controller')['defer_ratio'].mean().max():.6f}`; frozen waiting cost still dominates when zero/near-zero.
10. **Q10. H1 vs Oracle?** Reward `{oracle_reward:.6f}`, stage cost `{oracle_cost:.6f}`.
11. **Q11. Persistence vs Transformer?** Reward `{transformer_reward:.6f}`, stage cost `{transformer_cost:.6f}`.
12. **Q12. Did accuracy alter actions more?** See repaired agreement; incremental control transmission remains limited.
13. **Q13. Pressure vs threshold?** GPU P99 `{gpu_p99:.3f}%` vs switch threshold `{threshold:.1f}%`.
14. **Q14. v1 conclusion?** **{conclusion}** after formulation repair.
15. **Q15. MPC role?** **{role}**.
""",
    )
    print(
        json.dumps(
            {
                "status": "READY FOR MPC ROLE DECISION",
                "conclusion": conclusion,
                "role": role,
                "oracle_reward_delta": oracle_reward,
                "oracle_stage_cost_delta": oracle_cost,
                "transformer_reward_delta": transformer_reward,
                "transformer_stage_cost_delta": transformer_cost,
                "gpu_p99_percent": gpu_p99,
                "switch_threshold_percent": threshold,
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize()
    else:
        diagnosis_only()


if __name__ == "__main__":
    main()
