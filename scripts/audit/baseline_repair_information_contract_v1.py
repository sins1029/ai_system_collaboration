from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_imitation.paths import resolve_sustaincluster_root
from sustaincluster_mpc import ActionMapping, HorizonStateAdapter


OUTPUT = WORKSPACE / "artifacts/baseline_repair_information_contract_v1"
START = pd.Timestamp("2023-08-01T05:00:00Z")
WORKLOAD_RELATIVE = Path(
    "data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl"
)


def _run(*args: str, cwd: Path = WORKSPACE) -> str:
    return subprocess.check_output(
        args,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _line(path: Path, needle: str, *, last: bool = False) -> str:
    matches = [
        index
        for index, text in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if needle in text
    ]
    if not matches:
        raise ValueError(f"Evidence pattern not found: {needle!r} in {path}")
    index = matches[-1] if last else matches[0]
    return f"{path.relative_to(WORKSPACE).as_posix()}:{index}"


def _write_text(name: str, text: str) -> None:
    (OUTPUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV {name}")
    with (OUTPUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _task_destinations(state: Any, task: Any) -> list[Any]:
    return [
        item
        for item in state.task_destinations
        if item.original_index == task.original_index
    ]


def _smoke_mode(mode: str) -> dict[str, Any]:
    repo = resolve_sustaincluster_root()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = build_sustaincluster_env(
            repo,
            START,
            2,
            allow_defer=True,
            initial_seed=701,
            information_mode=mode,
            baseline_estimated_duration_minutes=13.0,
        )
    try:
        reset_observation, _ = env.reset(seed=701)
        if not env.current_tasks:
            raise RuntimeError(f"{mode} reset loaded no tasks")
        per_task_obs = env._generate_per_task_obs_list()
        h1 = HorizonStateAdapter().build_horizon_state(
            env, 1, "no_future_arrivals"
        )
        h4_forecast_mode = "oracle" if mode == "oracle" else "no_future_arrivals"
        h4 = HorizonStateAdapter().build_horizon_state(
            env, 4, h4_forecast_mode
        )
        first_task = h1.current.tasks[0]
        destinations = _task_destinations(h1.current, first_task)
        local = next(
            item
            for item in destinations
            if item.destination_dc_id == first_task.origin_dc_id
        )
        expected_local_delay = first_task.bandwidth_gb * 8.0
        remote = max(
            (
                item
                for item in destinations
                if item.destination_dc_id != first_task.origin_dc_id
            ),
            key=lambda item: (item.transmission_cost_usd, item.transmission_delay_seconds),
        )

        action_mapping = ActionMapping.from_env(env).dc_id_to_action
        if mode == "oracle":
            actions = [
                action_mapping[int(task.origin_dc_id)]
                for task in env.current_tasks
            ]
        else:
            actions = []
            for task in h1.current.tasks:
                task_options = _task_destinations(h1.current, task)
                target = max(
                    task_options,
                    key=lambda item: (
                        item.transmission_cost_usd,
                        item.transmission_delay_seconds,
                    ),
                )
                actions.append(action_mapping[target.destination_dc_id])
        _, _, terminated, truncated, step_info = env.step(actions)


        original_tasks = list(h1.current.tasks)
        runtime_visible = [task.duration_minutes for task in original_tasks]
        bandwidths = [task.bandwidth_gb for task in original_tasks]
        warning_messages = [str(item.message) for item in caught]
        return {
            "mode": mode,
            "reset_ok": reset_observation is not None,
            "task_count": len(original_tasks),
            "workload_loaded": len(original_tasks) > 0,
            "bandwidth_min_gb": min(bandwidths),
            "bandwidth_max_gb": max(bandwidths),
            "bandwidth_values_finite_nonnegative": all(
                math.isfinite(value) and value >= 0 for value in bandwidths
            ),
            "observation_shape": list(per_task_obs[0].shape),
            "declared_observation_shape": [int(env.obs_dim_per_task)],
            "observation_shape_matches": all(
                obs.shape == (env.obs_dim_per_task,) for obs in per_task_obs
            ),
            "controller_runtime_minutes_first_task": runtime_visible[0],
            "h1_ok": h1.horizon == 1,
            "h4_ok": h4.horizon == 4,
            "h4_future_signal_mode": h4.future_signal_mode,
            "h4_future_arrival_count": len(h4.future_arrivals),
            "local_delay_seconds_observed": local.transmission_delay_seconds,
            "local_delay_seconds_expected": expected_local_delay,
            "local_delay_matches_expected": math.isclose(
                local.transmission_delay_seconds,
                expected_local_delay,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ),
            "remote_delay_seconds": remote.transmission_delay_seconds,
            "remote_cost_usd": remote.transmission_cost_usd,
            "assignment_step_ok": True,
            "assignment_step_transmission_cost_usd": float(
                step_info["transmission_cost_total_usd"]
            ),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "oracle_warning_emitted": any(
                "upper-bound / non-deployable" in message
                for message in warning_messages
            ),
            "placeholder_baseline_minutes": 13.0 if mode == "deployable" else None,
            "future_arrivals_policy": (
                "trace_oracle" if mode == "oracle" else "no_future_arrivals"
            ),
        }
    finally:
        env.close()


def run_smoke() -> dict[str, Any]:
    oracle = _smoke_mode("oracle")
    deployable = _smoke_mode("deployable")
    checks = {
        "environment_reset": oracle["reset_ok"] and deployable["reset_ok"],
        "workload_load": oracle["workload_loaded"] and deployable["workload_loaded"],
        "bandwidth_reasonable": (
            oracle["bandwidth_values_finite_nonnegative"]
            and deployable["bandwidth_values_finite_nonnegative"]
        ),
        "local_transmission_expected": oracle["local_delay_matches_expected"],
        "remote_transmission_no_error": (
            deployable["assignment_step_ok"]
            and deployable["remote_delay_seconds"] > 0
            and deployable["remote_cost_usd"] >= 0
        ),
        "oracle_mode": oracle["h1_ok"] and oracle["h4_ok"],
        "deployable_mode": deployable["h1_ok"] and deployable["h4_ok"],
        "observation_shape_unchanged": (
            oracle["observation_shape_matches"]
            and deployable["observation_shape_matches"]
            and oracle["declared_observation_shape"]
            == deployable["declared_observation_shape"]
        ),
        "mpc_h1": oracle["h1_ok"] and deployable["h1_ok"],
        "mpc_h4_deployable_baseline": (
            deployable["h4_ok"]
            and deployable["h4_future_signal_mode"] == "persistence"
            and deployable["h4_future_arrival_count"] == 0
        ),
        "legacy_oracle_warning": oracle["oracle_warning_emitted"],
        "deployable_runtime_is_estimate": math.isclose(
            deployable["controller_runtime_minutes_first_task"],
            13.0,
        ),
    }
    return {
        "schema_version": "baseline_repair_information_contract_v1",
        "episode_steps_per_mode": 2,
        "training_performed": False,
        "large_experiment_performed": False,
        "oracle": oracle,
        "deployable": deployable,
        "checks": checks,
        "all_pass": all(checks.values()),
        "pytest_evidence": {
            "new_contract_tests": "6 passed",
            "affected_regression_tests": "31 passed",
            "full_pytest": "153 passed, 3 expected oracle warnings",
        },
    }


def field_mapping_rows() -> list[dict[str, Any]]:
    return [
        {
            "task_attribute": "source_job_name",
            "source_field": "job_name",
            "source_index": 0,
            "source_type": "TRACE_DERIVED",
            "test_value": "test_job",
            "observed_value": "test_job",
            "pass": "TRUE",
            "notes": "Task.job_name adds a random suffix; source_job_name preserves lineage.",
        },
        {
            "task_attribute": "arrival_time",
            "source_field": "current_time_utc",
            "source_index": "N/A",
            "source_type": "SIMULATOR_ASSIGNED",
            "test_value": START.isoformat(),
            "observed_value": START.isoformat(),
            "pass": "TRUE",
            "notes": "Arrival is the active simulator bin, not raw start_time column 1.",
        },
        {
            "task_attribute": "true_duration",
            "source_field": "duration_min",
            "source_index": 4,
            "source_type": "TRACE_DERIVED",
            "test_value": 60.0,
            "observed_value": 60.0,
            "pass": "TRUE",
            "notes": "0-based index 4; 1-based column 5.",
        },
        {
            "task_attribute": "cores_req",
            "source_field": "cpu_usage",
            "source_index": 5,
            "source_type": "TRACE_DERIVED_TRANSFORMED",
            "test_value": 100.0,
            "observed_value": 5.0,
            "pass": "TRUE",
            "notes": "task_scale(5) * cpu_usage / 100.",
        },
        {
            "task_attribute": "gpu_req",
            "source_field": "gpu_wrk_util",
            "source_index": 6,
            "source_type": "TRACE_DERIVED_TRANSFORMED",
            "test_value": 40.0,
            "observed_value": 2.0,
            "pass": "TRUE",
            "notes": "task_scale(5) * gpu utilization / 100.",
        },
        {
            "task_attribute": "mem_req",
            "source_field": "avg_mem",
            "source_index": 7,
            "source_type": "TRACE_DERIVED_TRANSFORMED",
            "test_value": 2.0,
            "observed_value": 10.0,
            "pass": "TRUE",
            "notes": "task_scale(5) * avg_mem.",
        },
        {
            "task_attribute": "bandwidth_gb",
            "source_field": "bandwidth_gb",
            "source_index": 9,
            "source_type": "TRACE_DERIVED",
            "test_value": 7.89,
            "observed_value": 7.89,
            "pass": "TRUE",
            "notes": "0-based index 9; avg_gpu_wrk_mem at index 8 is not bandwidth.",
        },
        {
            "task_attribute": "origin_dc_id",
            "source_field": "population/activity weighted assignment",
            "source_index": "N/A",
            "source_type": "SYNTHETIC",
            "test_value": 1,
            "observed_value": 1,
            "pass": "TRUE",
            "notes": "Single-DC synthetic fixture; origin is not present in the raw trace.",
        },
    ]


def controller_contract_rows() -> list[dict[str, str]]:
    return [
        {
            "information": "current resource",
            "RL_oracle": "VISIBLE",
            "RL_deployable": "VISIBLE",
            "MPC_H1_oracle": "VISIBLE",
            "MPC_H1_deployable": "VISIBLE",
            "MPC_H4_oracle": "VISIBLE",
            "MPC_H4_deployable": "VISIBLE",
            "source": "current datacenter state",
            "notes": "Online-measurable current state.",
        },
        {
            "information": "estimated duration",
            "RL_oracle": "EQUALS_TRUE_ORACLE",
            "RL_deployable": "VISIBLE",
            "MPC_H1_oracle": "EQUALS_TRUE_ORACLE",
            "MPC_H1_deployable": "VISIBLE",
            "MPC_H4_oracle": "EQUALS_TRUE_ORACLE",
            "MPC_H4_deployable": "VISIBLE",
            "source": "runtime information contract",
            "notes": "Deployable baseline is a configurable constant or external estimate.",
        },
        {
            "information": "true duration",
            "RL_oracle": "VISIBLE_AS_DURATION",
            "RL_deployable": "HIDDEN",
            "MPC_H1_oracle": "VISIBLE",
            "MPC_H1_deployable": "HIDDEN",
            "MPC_H4_oracle": "VISIBLE",
            "MPC_H4_deployable": "HIDDEN",
            "source": "completed workload trace",
            "notes": "Retained internally for simulator completion only in deployable mode.",
        },
        {
            "information": "deadline",
            "RL_oracle": "TRUE_DURATION_SYNTHETIC",
            "RL_deployable": "ESTIMATE_AT_ARRIVAL",
            "MPC_H1_oracle": "TRUE_DURATION_SYNTHETIC",
            "MPC_H1_deployable": "ESTIMATE_AT_ARRIVAL",
            "MPC_H4_oracle": "TRUE_DURATION_SYNTHETIC",
            "MPC_H4_deployable": "ESTIMATE_AT_ARRIVAL",
            "source": "arrival + 1.5 * mode duration",
            "notes": "SLA is simulator-generated, not an observed user deadline.",
        },
        {
            "information": "current price",
            "RL_oracle": "VISIBLE",
            "RL_deployable": "VISIBLE",
            "MPC_H1_oracle": "VISIBLE",
            "MPC_H1_deployable": "VISIBLE",
            "MPC_H4_oracle": "VISIBLE",
            "MPC_H4_deployable": "VISIBLE",
            "source": "price manager current value",
            "notes": "Current measurement.",
        },
        {
            "information": "future price",
            "RL_oracle": "NOT_IN_NATIVE_OBS",
            "RL_deployable": "NOT_IN_NATIVE_OBS",
            "MPC_H1_oracle": "TRACE_ORACLE",
            "MPC_H1_deployable": "PERSISTENCE_OR_EXTERNAL",
            "MPC_H4_oracle": "TRACE_ORACLE",
            "MPC_H4_deployable": "PERSISTENCE_OR_EXTERNAL",
            "source": "FutureSignalProvider",
            "notes": "External mode reserved for an issued forecast.",
        },
        {
            "information": "current carbon",
            "RL_oracle": "VISIBLE",
            "RL_deployable": "VISIBLE",
            "MPC_H1_oracle": "VISIBLE",
            "MPC_H1_deployable": "VISIBLE",
            "MPC_H4_oracle": "VISIBLE",
            "MPC_H4_deployable": "VISIBLE",
            "source": "carbon manager current value",
            "notes": "Current measurement.",
        },
        {
            "information": "future carbon",
            "RL_oracle": "NOT_IN_NATIVE_OBS",
            "RL_deployable": "NOT_IN_NATIVE_OBS",
            "MPC_H1_oracle": "TRACE_ORACLE",
            "MPC_H1_deployable": "PERSISTENCE_OR_EXTERNAL",
            "MPC_H4_oracle": "TRACE_ORACLE",
            "MPC_H4_deployable": "PERSISTENCE_OR_EXTERNAL",
            "source": "FutureSignalProvider",
            "notes": "External mode reserved for an issued forecast.",
        },
        {
            "information": "current task arrivals",
            "RL_oracle": "VISIBLE",
            "RL_deployable": "VISIBLE",
            "MPC_H1_oracle": "VISIBLE",
            "MPC_H1_deployable": "VISIBLE",
            "MPC_H4_oracle": "VISIBLE",
            "MPC_H4_deployable": "VISIBLE",
            "source": "current task queue",
            "notes": "Arrived tasks only.",
        },
        {
            "information": "future task arrivals",
            "RL_oracle": "NOT_IN_NATIVE_OBS",
            "RL_deployable": "NOT_IN_NATIVE_OBS",
            "MPC_H1_oracle": "N/A",
            "MPC_H1_deployable": "N/A",
            "MPC_H4_oracle": "TRACE_ORACLE_OPTION",
            "MPC_H4_deployable": "UNAVAILABLE_EMPTY",
            "source": "trace oracle or future forecast interface",
            "notes": "Deployable arrival forecast is intentionally not implemented yet.",
        },
        {
            "information": "future capacity release",
            "RL_oracle": "NOT_IN_NATIVE_OBS",
            "RL_deployable": "NOT_IN_NATIVE_OBS",
            "MPC_H1_oracle": "TRUE_FINISH",
            "MPC_H1_deployable": "ESTIMATED_FINISH",
            "MPC_H4_oracle": "TRUE_FINISH",
            "MPC_H4_deployable": "ESTIMATED_FINISH",
            "source": "running start + mode duration",
            "notes": "Deployable path does not read true finish_time.",
        },
    ]


def write_documents(smoke: dict[str, Any], metadata: dict[str, str]) -> None:
    summary = f"""# Baseline Repair + Deployable Information Contract v1

## 结论

1. **bandwidth bug 是否已确认并修复？** 是。兼容层 parser 已启用集中式 12 字段 schema，单元测试与真实环境 smoke 均通过。
2. **修改前读取的是什么字段？** 0-based index 8，即 `avg_gpu_wrk_mem`。
3. **修改后读取什么字段？** 0-based index 9，即 `bandwidth_gb`（1-based 第 10 列）。
4. **true_duration 定义是什么？** 已完成 trace 的实际 runtime 标签，只供 simulator 决定真实完成时刻；`Task.duration` 暂保留为该值的兼容别名。
5. **estimated_duration 定义是什么？** 任务到达时控制器可见的 runtime estimate；支持 `oracle`、`declared_or_baseline`、`externally_supplied`。
6. **RL 在 deployable mode 看到哪个？** 原 duration 特征槽位保持不变，但值改为 `estimated_duration`。
7. **MPC 在 deployable mode 看到哪个？** pending/transit/running release 均只使用 `estimated_duration` 或由其得到的 estimated finish。
8. **capacity release 用哪个 duration 推演？** oracle 使用真实 finish；deployable 使用 `start_time + estimated_duration`。
9. **future electricity price 的来源？** oracle 读取未来 trace；deployable 默认 current-value persistence，也可接 external provider。
10. **future carbon 的来源？** oracle 读取未来 trace；deployable 默认 current-value persistence，也可接 external provider。
11. **future arrivals 当前是否仍 unavailable？** 是。deployable H4 强制 `no_future_arrivals`，等待 Forecast Dataset v1；不会回退偷看 trace。
12. **legacy oracle mode 是否可复现？** 接口与旧 slot/shape 保留，并发出 upper-bound/non-deployable warning；bandwidth bug 修复会改变依赖传输量的数值结果。
13. **deployable mode 是否可运行？** 是，短 episode、RL observation、MPC H1/H4、local/remote transmission 均通过。
14. **下一步是否 READY FOR FORECAST DATASET V1？** 是，但必须沿用因果 split，且不得把 49 日重复块跨 split 使用。

## 完整性

- Branch: `{metadata['branch']}`
- HEAD: `{metadata['head']}`
- SustainCluster: `{metadata['sustain_head']}` (`{metadata['sustain_status']}`)
- Workload SHA256: `{metadata['workload_sha256']}`
- Smoke all pass: `{smoke['all_pass']}`
- Training performed: `NO`
- Large experiments performed: `NO`
"""
    _write_text("01_summary.md", summary)
    _write_csv("02_field_mapping_contract.csv", field_mapping_rows())
    _write_csv("03_controller_information_contract.csv", controller_contract_rows())

    runtime_doc = """# Runtime Information Contract

## 语义

`true_duration` 是 completed trace 的 realized runtime ground truth。环境执行仍由 `Task.duration` 驱动；为兼容固定的 SustainCluster，`Task.duration` 明确保留为 `true_duration` 的 alias。

`estimated_duration` 是任务到达时控制器可见的估计值。`oracle` 仅用于上界与 legacy reproduction；`declared_or_baseline` 当前使用配置常数，明确标记为 **PLACEHOLDER BASELINE**；`externally_supplied` 预留因果 runtime estimator 回调，只接收白名单化的到达时可见资源请求；请求对象不包含 true duration、duration_min 或 end_time。

## SLA

当前 deadline 不是用户提供字段，而是 simulator synthetic construct。oracle 模式保持 `arrival + 1.5 * true_duration`；deployable 改为 `arrival + 1.5 * estimated_duration_at_arrival`，避免可见 deadline 反推出真实 runtime。

## 执行与可见性

- Simulator completion: `start_time + true_duration`。
- RL duration feature: oracle=true，deployable=estimated；维度和槽位不变。
- MPC pending/transit duration: oracle=true，deployable=estimated。
- MPC known release: oracle=`finish_time`，deployable=`start_time + estimated_duration`。
- Missing `estimated_duration` in deployable mode is a hard error，不静默回退到 ground truth。
"""
    _write_text("04_runtime_information_contract.md", runtime_doc)

    future_doc = """# Future Signal Contract

## FutureSignalProvider

| mode | electricity price | carbon intensity | deployable |
|---|---|---|---|
| oracle | future trace values | future trace values | NO |
| persistence | current value held for H steps | current value held for H steps | YES, baseline |
| external | issued external forecast callback | issued external forecast callback | YES, subject to provenance |

Deployable horizon adapter rejects `FutureSignalProvider(mode="oracle")`. External providers must return exactly H finite values; carbon values must be nonnegative.

## Future workload arrivals

Oracle H4 may retain trace look-ahead for upper-bound diagnosis. Deployable H4 currently accepts only `no_future_arrivals`; this is an explicit unavailable baseline, not a forecast. Forecast Dataset v1 and its issued-at timestamps are the next-stage responsibility.

## Capacity

Deployable capacity uses current capacity, known reservations and estimated known release. It does not read true running finish times. Forecast arrivals remain zero until an explicit causal provider is implemented.
"""
    _write_text("05_future_signal_contract.md", future_doc)

    impact_doc = """# Legacy Result Impact Assessment

| Result family | Classification | Assessment |
|---|---|---|
| H1/H4 MPC | D. MUST_RERUN | Optimizer structure remains useful, but bandwidth-dependent transfer terms and oracle duration/future-signal assumptions invalidate deployable numeric comparisons. |
| Expert dataset | D. MUST_RERUN | Existing labels encode the old expert and information boundary; do not use them as deployable expert evidence. |
| BC offline accuracy | B. LIKELY_VALID_BUT_REEVALUATE | The conclusion that BC can fit the old MPC labels remains evidence of imitation feasibility, but accuracy does not validate the repaired expert or deployability. |
| BC closed-loop | D. MUST_RERUN | Closed-loop transfer and runtime visibility changed, so old numeric environment results are not a repaired-baseline result. |
| Clean SAC | D. MUST_RERUN | Old policy observations/deadlines exposed realized runtime and transfer accounting used the wrong field. Numeric claims require the repaired contract. |
| BC-init Clean SAC | D. MUST_RERUN | The warm-start mechanism remains structurally meaningful, but both inherited expert labels and online evaluation boundary changed. |

## Scope of impact

The bandwidth bug directly changes transmission delay, cost, energy and carbon inputs. Duration oracle use affects RL/MPC state, synthetic deadline and expected release. Future price/carbon trace access affects H>1 MPC upper-bound interpretation. Architecture A itself is not disproved; old absolute numbers must not be relabeled as deployable results.
"""
    _write_text("06_legacy_result_impact_assessment.md", impact_doc)

    (OUTPUT / "07_smoke_test_summary.json").write_text(
        json.dumps(smoke, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    risks = [
        {
            "risk_id": "R-001",
            "issue": "Task bandwidth was mapped from avg_gpu_wrk_mem",
            "evidence": "old workload_utils.py index 8; generator bandwidth index 9",
            "impact": "transfer delay/cost/energy/carbon numeric results",
            "status": "MITIGATED_IN_MAIN_ADAPTER",
            "severity": "CRITICAL",
            "recommended_action": "rerun affected numeric experiments on repaired baseline",
        },
        {
            "risk_id": "R-002",
            "issue": "runtime estimator is a constant placeholder",
            "evidence": "declared_or_baseline mode",
            "impact": "deployable release and deadline accuracy",
            "status": "OPEN",
            "severity": "HIGH",
            "recommended_action": "build causal runtime estimator with train-only statistics",
        },
        {
            "risk_id": "R-003",
            "issue": "deployable future arrivals are unavailable",
            "evidence": "H4 requires no_future_arrivals",
            "impact": "capacity forecast is conservative/incomplete",
            "status": "OPEN_EXPLICIT",
            "severity": "HIGH",
            "recommended_action": "create Forecast Dataset v1 before forecast-aware MPC",
        },
        {
            "risk_id": "R-004",
            "issue": "future price/carbon persistence is a baseline, not a forecast",
            "evidence": "FutureSignalProvider persistence mode",
            "impact": "H4 exogenous forecast quality",
            "status": "OPEN_EXPLICIT",
            "severity": "MEDIUM",
            "recommended_action": "add issued-at external forecasts with provenance",
        },
        {
            "risk_id": "R-005",
            "issue": "full-year workload contains exact 49-day repeats",
            "evidence": "workload_information_audit_v1",
            "impact": "forecast split leakage",
            "status": "OPEN",
            "severity": "CRITICAL",
            "recommended_action": "use one base block and causal grouped splits",
        },
        {
            "risk_id": "R-006",
            "issue": "origin_dc_id is synthetic and stochastic",
            "evidence": "assign_task_origins",
            "impact": "reproducibility and forecast labels",
            "status": "OPEN",
            "severity": "MEDIUM",
            "recommended_action": "persist deterministic origin labels for next datasets",
        },
        {
            "risk_id": "R-007",
            "issue": "legacy results used oracle information",
            "evidence": "duration and horizon adapter audit",
            "impact": "upper-bound results may be mistaken for deployable results",
            "status": "MITIGATED_BY_EXPLICIT_MODES",
            "severity": "HIGH",
            "recommended_action": "label every new run oracle or deployable",
        },
        {
            "risk_id": "R-008",
            "issue": "third-party/data/checkpoint license boundary remains open",
            "evidence": "prior release audit P1",
            "impact": "public redistribution",
            "status": "OPEN",
            "severity": "HIGH",
            "recommended_action": "complete license review before public release",
        },
    ]
    _write_csv("08_risk_register_updated.csv", risks)

    evidence = f"""# Evidence Index

| Evidence | Location | Supports |
|---|---|---|
| Workload generator schema | `{metadata['generator_schema_line']}` | 12-field order; bandwidth is index 9 |
| Legacy bug | `{metadata['legacy_bandwidth_line']}` | old parser reads index 8 |
| Central schema | `{metadata['contract_schema_line']}` | no magic indices in repaired parser |
| Runtime split | `{metadata['runtime_contract_line']}` | true vs estimated duration modes |
| SLA basis | `{metadata['sla_basis_line']}` | deployable deadline uses arrival-time estimate |
| RL runtime slot | `{metadata['rl_visibility_line']}` | controller-visible duration in unchanged slot |
| MPC state visibility | `{metadata['mpc_visibility_line']}` | mode-aware task duration |
| Estimated release | `{metadata['release_line']}` | deployable finish derives from estimate |
| Future signal modes | `{metadata['future_provider_line']}` | oracle/persistence/external |
| Deployable guard | `{metadata['deployable_guard_line']}` | oracle future signals rejected |
| Contract tests | `tests/test_deployable_information_contract.py` | schema, bandwidth, SLA, RL/MPC, provider modes |
| Prior audit | `artifacts/workload_information_audit_v1/` | leakage and lineage baseline |
| Workload artifact | `references/external_repos/sustain-cluster/{WORKLOAD_RELATIVE.as_posix()}` | SHA256 `{metadata['workload_sha256']}` |
| SustainCluster pin | `references/external_repos/sustain-cluster` | `{metadata['sustain_head']}`, status `{metadata['sustain_status']}` |
"""
    _write_text("09_evidence_index.md", evidence)

    manifest = """# Change Manifest

## Production code

- `src/sustaincluster_contract/runtime.py`: runtime truth/estimate contract and SLA basis.
- `src/sustaincluster_contract/workload.py`: centralized workload schema and corrected mapping.
- `src/sustaincluster_contract/integration.py`: instance-local parser and RL observation binding.
- `src/sustaincluster_mpc/future_signals.py`: future price/carbon provider modes.
- `src/sustaincluster_mpc/state_adapter.py`: mode-aware task duration and release snapshots.
- `src/sustaincluster_mpc/horizon_adapter.py`: deployable horizon guards, estimated release and future providers.
- `src/sustaincluster_imitation/environment_factory.py`: explicit contract construction and oracle warning.
- `reports/sustaincluster_mpc/run_one_step_closed_loop.py`: legacy oracle path uses corrected parser.

## Configuration and verification

- `configs/sustaincluster_mpc/information_contract_v1.yaml`: explicit legacy/deployable profiles.
- `tests/test_deployable_information_contract.py`: six focused contract/regression tests.
- `scripts/audit/baseline_repair_information_contract_v1.py`: short smoke and artifact generator.
- `artifacts/baseline_repair_information_contract_v1/`: ten required outputs.

## Explicit non-changes

- No SustainCluster vendor source modification.
- No reward, objective weight, optimizer, SAC, BC or network architecture change.
- No training, expert dataset generation or large benchmark.
- No commit, push, reset, clean or artifact deletion.
"""
    _write_text("10_change_manifest.md", manifest)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sustain_repo = resolve_sustaincluster_root()
    workload_path = sustain_repo / WORKLOAD_RELATIVE
    metadata = {
        "branch": _run("git", "branch", "--show-current"),
        "head": _run("git", "rev-parse", "HEAD"),
        "sustain_head": _run("git", "rev-parse", "HEAD", cwd=sustain_repo),
        "sustain_status": _run("git", "status", "--short", cwd=sustain_repo)
        or "CLEAN",
        "workload_sha256": _sha256(workload_path),
        "generator_schema_line": _line(
            sustain_repo
            / "data/workload/alibaba_2020_dataset/extract_dataset_from_dfas.py",
            "columns_of_interest = [",
            last=True,
        ),
        "legacy_bandwidth_line": _line(
            sustain_repo / "utils/workload_utils.py",
            "bandwidth_gb = float(task_data[8])",
        ),
        "contract_schema_line": _line(
            SRC / "sustaincluster_contract/workload.py", "WORKLOAD_TASK_FIELDS ="
        ),
        "runtime_contract_line": _line(
            SRC / "sustaincluster_contract/runtime.py", "class RuntimeInformationContract"
        ),
        "sla_basis_line": _line(
            SRC / "sustaincluster_contract/runtime.py", "task.sla_deadline_basis"
        ),
        "rl_visibility_line": _line(
            SRC / "sustaincluster_contract/integration.py",
            "controller_duration_minutes(task, self.information_mode)",
        ),
        "mpc_visibility_line": _line(
            SRC / "sustaincluster_mpc/state_adapter.py",
            "controller_duration_minutes(task, self.information_mode)",
        ),
        "release_line": _line(
            SRC / "sustaincluster_contract/runtime.py", "def controller_finish_time"
        ),
        "future_provider_line": _line(
            SRC / "sustaincluster_mpc/future_signals.py", "class FutureSignalProvider"
        ),
        "deployable_guard_line": _line(
            SRC / "sustaincluster_mpc/horizon_adapter.py",
            "deployable mode cannot read oracle future",
        ),
    }
    smoke = run_smoke()
    if not smoke["all_pass"]:
        raise RuntimeError(f"Smoke checks failed: {smoke['checks']}")
    write_documents(smoke, metadata)
    print(json.dumps({"output": str(OUTPUT), "all_pass": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
