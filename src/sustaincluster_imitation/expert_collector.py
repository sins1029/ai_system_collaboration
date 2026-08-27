from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sustaincluster_imitation.dataset_schema import (
    EpisodeRecord,
    StepRecord,
    TaskActionRecord,
)
from sustaincluster_imitation.environment_factory import build_sustaincluster_env
from sustaincluster_imitation.feature_encoder import (
    SemanticActionSpace,
    SustainClusterFeatureEncoder,
)
from sustaincluster_imitation.forecast_baseline import (
    BaselineForecast,
    HistoricalArrivalForecaster,
)
from sustaincluster_mpc import (
    ActionMapping,
    FutureArrivalAggregate,
    HorizonState,
    HorizonStateAdapter,
    RollingHorizonConfig,
    RollingHorizonOptimizer,
    RollingObjectiveWeights,
    SustainClusterActionAdapter,
)

from sustaincluster_imitation.synthetic_scenarios import (
    SyntheticEpisodeSimulator,
    SyntheticScenario,
)


@dataclass(frozen=True)
class ExpertRuntimeConfig:
    horizon: int
    forecast_mode: str
    allow_defer: bool
    solver_time_limit_seconds: float
    objective_weights: RollingObjectiveWeights
    cpu_power_w_per_core: float
    gpu_power_w_per_unit: float
    memory_power_w_per_gb: float
    waiting_cost_per_step: float
    terminal_backlog_base_cost: float
    deterministic_tie_break_epsilon: float

    @classmethod
    def from_yaml(cls, path: Path) -> "ExpertRuntimeConfig":
        with Path(path).open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)["expert"]
        power = values["power_model"]
        return cls(
            horizon=int(values["horizon"]),
            forecast_mode=str(values["forecast_mode"]),
            allow_defer=bool(values["allow_defer"]),
            solver_time_limit_seconds=float(values["solver_time_limit_seconds"]),
            objective_weights=RollingObjectiveWeights(
                **{key: float(value) for key, value in values["objective_weights"].items()}
            ),
            cpu_power_w_per_core=float(power["cpu_power_w_per_core"]),
            gpu_power_w_per_unit=float(power["gpu_power_w_per_unit"]),
            memory_power_w_per_gb=float(power["memory_power_w_per_gb"]),
            waiting_cost_per_step=float(values["waiting_cost_per_step"]),
            terminal_backlog_base_cost=float(values["terminal_backlog_base_cost"]),
            deterministic_tie_break_epsilon=float(
                values["deterministic_tie_break_epsilon"]
            ),
        )

    def optimizer_config(self) -> RollingHorizonConfig:
        return RollingHorizonConfig(
            allow_defer=self.allow_defer,
            weights=self.objective_weights,
            cpu_power_w_per_core=self.cpu_power_w_per_core,
            gpu_power_w_per_unit=self.gpu_power_w_per_unit,
            memory_power_w_per_gb=self.memory_power_w_per_gb,
            waiting_cost_per_step=self.waiting_cost_per_step,
            terminal_backlog_base_cost=self.terminal_backlog_base_cost,
            deterministic_tie_break_epsilon=self.deterministic_tie_break_epsilon,
            solver_time_limit_seconds=self.solver_time_limit_seconds,
        )


@dataclass
class CollectedEpisode:
    episode: EpisodeRecord
    steps: list[StepRecord]
    tasks: list[TaskActionRecord]
    summary: dict[str, Any]


class ExpertCollector:
    """运行冻结 MPC 专家并记录带数据谱系的训练样本。"""
    def __init__(
        self,
        repo: Path,
        project_commit: str,
        sustaincluster_commit: str,
        config: ExpertRuntimeConfig,
        canonical_dc_ids: tuple[int, ...] = (1, 2, 3, 4, 5),
        baseline_history_window: int = 16,
    ) -> None:
        self.repo = Path(repo)
        self.project_commit = project_commit
        self.sustaincluster_commit = sustaincluster_commit
        self.config = config
        self.semantic_actions = SemanticActionSpace(canonical_dc_ids, True)
        self.encoder = SustainClusterFeatureEncoder(
            self.semantic_actions, config.horizon
        )
        self.baseline_history_window = baseline_history_window
        self.optimizer = RollingHorizonOptimizer()

    def collect_real_episode(
        self,
        *,
        episode_id: str,
        scenario_name: str,
        seed: int,
        start_time: pd.Timestamp,
        episode_steps: int,
        dataset_variant: str,
        split_hint: str,
    ) -> CollectedEpisode:
        env = build_sustaincluster_env(
            self.repo,
            start_time,
            episode_steps,
            allow_defer=self.config.allow_defer,
            initial_seed=seed,
        )
        env.reset(seed=seed)
        action_adapter = SustainClusterActionAdapter.from_env(env)
        horizon_adapter = HorizonStateAdapter()
        forecaster = HistoricalArrivalForecaster(
            self.config.horizon, self.baseline_history_window, seed
        )
        episode = EpisodeRecord(
            episode_id,
            scenario_name,
            seed,
            f"{start_time.isoformat()}/{episode_steps}",
            self.sustaincluster_commit,
            self.project_commit,
            self.config.horizon,
            dataset_variant,
            json.dumps(asdict(self.config.objective_weights), sort_keys=True),
            15.0,
            dataset_variant,
            split_hint,
            1.0,
        )
        step_rows: list[StepRecord] = []
        task_rows: list[TaskActionRecord] = []
        metrics = self._empty_metrics()
        for step_index in range(episode_steps):
            action_adapter.assert_matches_env(env)
            horizon_state, forecast = self._build_state(
                env, horizon_adapter, forecaster, dataset_variant
            )
            encoded = self.encoder.encode(
                horizon_state,
                forecast.uncertainties if forecast else (),
            )
            result = self.optimizer.solve(
                horizon_state,
                self.config.optimizer_config(),
                action_adapter,
            )
            if not result.feasible:
                metrics["infeasible_count"] += 1
                break
            action_adapter.validate_actions(
                horizon_state.current.tasks, result.environment_actions
            )
            plans = {
                (plan.original_index, plan.task_id): plan for plan in result.plans
            }
            decisions = {
                (item.original_index, item.task_id): item
                for item in result.first_step_decisions
            }
            timestamp = horizon_state.current.exogenous.current_time_utc
            for task_index, task in enumerate(horizon_state.current.tasks):
                plan = plans[(task.original_index, task.task_id)]
                decision = decisions[(task.original_index, task.task_id)]
                label = self.semantic_actions.encode_decision(decision)
                if not bool(encoded.feasible_action_mask[task_index, label]):
                    raise RuntimeError("专家生成了不可行的语义标签")
                destinations = [
                    item
                    for item in horizon_state.current.task_destinations
                    if item.original_index == task.original_index
                ]
                task_rows.append(
                    TaskActionRecord(
                        episode_id,
                        scenario_name,
                        seed,
                        dataset_variant,
                        step_index,
                        timestamp,
                        task.task_id,
                        task.original_index,
                        task.origin_dc_id,
                        task.cpu_cores,
                        task.gpu_units,
                        task.memory_gb,
                        task.duration_minutes,
                        task.remaining_duration_minutes,
                        task.remaining_sla_minutes,
                        task.bandwidth_gb,
                        task.wait_intervals,
                        task.was_deferred,
                        json.dumps([asdict(item) for item in destinations], sort_keys=True),
                        decision.decision,
                        decision.dc_id,
                        label,
                        plan.dispatch_step,
                        plan.execution_start_step,
                        None
                        if plan.execution_start_step is None
                        else plan.execution_start_step + plan.duration_steps,
                        encoded.features[task_index].astype(float).tolist(),
                        encoded.feasible_action_mask[task_index].tolist(),
                        result.status,
                        result.solve_seconds,
                        result.integer_variable_count,
                        float(result.objective_value or 0.0),
                        result.costs.electricity,
                        result.costs.carbon,
                        result.costs.transmission,
                        result.costs.waiting_defer,
                        result.costs.sla_risk,
                        result.costs.terminal_backlog,
                    )
                )
                metrics["decision_count"] += 1
                metrics["defer_count"] += int(decision.decision == "defer")
                if decision.decision == "assign":
                    metrics["assign_by_dc"][str(decision.dc_id)] += 1
                    metrics["migration_count"] += int(
                        int(decision.dc_id) != task.origin_dc_id
                    )
            before_resources = self._resource_bounds(env)
            _, reward, terminated, truncated, info = env.step(
                list(result.environment_actions)
            )
            after_resources = self._resource_bounds(env)
            metrics["resource_overflow_count"] += int(
                not before_resources or not after_resources
            )
            self._update_runtime_metrics(
                metrics, info, result, horizon_state, float(reward)
            )
            step_rows.append(
                StepRecord(
                    episode_id,
                    step_index,
                    timestamp,
                    len(horizon_state.current.tasks),
                    self._datacenters_json(horizon_state),
                    self._horizon_json(horizon_state),
                    json.dumps(
                        [asdict(item) for item in (forecast.uncertainties if forecast else ())],
                        sort_keys=True,
                    ),
                    result.status,
                    result.solve_seconds,
                    result.integer_variable_count,
                    float(result.objective_value or 0.0),
                    result.first_step_costs.electricity,
                    result.first_step_costs.carbon,
                    result.first_step_costs.transmission,
                    result.first_step_costs.waiting_defer,
                    result.first_step_costs.sla_risk,
                    result.first_step_costs.terminal_backlog,
                    float(reward),
                    bool(terminated),
                    bool(truncated),
                )
            )
            if terminated or truncated:
                break
        return CollectedEpisode(
            episode, step_rows, task_rows, self._finish_metrics(metrics)
        )

    def collect_synthetic_episode(
        self,
        *,
        episode_id: str,
        scenario: SyntheticScenario,
        seed: int,
        dataset_variant: str,
        split_hint: str,
    ) -> CollectedEpisode:
        simulator = SyntheticEpisodeSimulator(scenario)
        simulator.reset()
        dc_ids = tuple(dc.dc_id for dc in scenario.datacenters)
        action_adapter = SustainClusterActionAdapter(
            ActionMapping(
                tuple((dc_id, index + 1) for index, dc_id in enumerate(dc_ids)),
                0,
                len(dc_ids) + 1,
            )
        )
        forecaster = HistoricalArrivalForecaster(
            self.config.horizon, self.baseline_history_window, seed
        )
        episode = EpisodeRecord(
            episode_id,
            scenario.name,
            seed,
            f"synthetic:0/{scenario.steps}",
            self.sustaincluster_commit,
            self.project_commit,
            self.config.horizon,
            dataset_variant,
            json.dumps(asdict(self.config.objective_weights), sort_keys=True),
            15.0,
            dataset_variant,
            split_hint,
            scenario.burst_intensity,
        )
        step_rows: list[StepRecord] = []
        task_rows: list[TaskActionRecord] = []
        metrics = self._empty_metrics()
        forecast_errors: list[dict[str, float]] = []
        for step_index in range(scenario.steps):
            simulator.admit_current_arrivals()
            base_state = simulator.build_state(self.config.horizon)
            state, forecast = self._synthetic_forecast_state(
                simulator, base_state, forecaster, dataset_variant
            )
            if forecast is not None:
                forecast_errors.extend(
                    self._forecast_errors(
                        simulator, forecast, self.config.horizon
                    )
                )
            encoded = self.encoder.encode(
                state, forecast.uncertainties if forecast else ()
            )
            result = self.optimizer.solve(
                state, self.config.optimizer_config(), action_adapter
            )
            if not result.feasible:
                metrics["infeasible_count"] += 1
                break
            action_adapter.validate_actions(
                state.current.tasks, result.environment_actions
            )
            plans = {
                (plan.original_index, plan.task_id): plan for plan in result.plans
            }
            decisions = {
                (item.original_index, item.task_id): item
                for item in result.first_step_decisions
            }
            for task_index, task in enumerate(state.current.tasks):
                plan = plans[(task.original_index, task.task_id)]
                decision = decisions[(task.original_index, task.task_id)]
                label = self.semantic_actions.encode_decision(decision)
                if not bool(encoded.feasible_action_mask[task_index, label]):
                    raise RuntimeError(
                        "synthetic 专家生成了不可行的语义标签"
                    )
                destinations = [
                    item
                    for item in state.current.task_destinations
                    if item.original_index == task.original_index
                    and item.task_id == task.task_id
                ]
                task_rows.append(
                    TaskActionRecord(
                        episode_id,
                        scenario.name,
                        seed,
                        dataset_variant,
                        step_index,
                        state.current.exogenous.current_time_utc,
                        task.task_id,
                        task.original_index,
                        task.origin_dc_id,
                        task.cpu_cores,
                        task.gpu_units,
                        task.memory_gb,
                        task.duration_minutes,
                        task.remaining_duration_minutes,
                        task.remaining_sla_minutes,
                        task.bandwidth_gb,
                        task.wait_intervals,
                        task.was_deferred,
                        json.dumps(
                            [asdict(item) for item in destinations], sort_keys=True
                        ),
                        decision.decision,
                        decision.dc_id,
                        label,
                        plan.dispatch_step,
                        plan.execution_start_step,
                        None
                        if plan.execution_start_step is None
                        else plan.execution_start_step + plan.duration_steps,
                        encoded.features[task_index].astype(float).tolist(),
                        encoded.feasible_action_mask[task_index].tolist(),
                        result.status,
                        result.solve_seconds,
                        result.integer_variable_count,
                        float(result.objective_value or 0.0),
                        result.costs.electricity,
                        result.costs.carbon,
                        result.costs.transmission,
                        result.costs.waiting_defer,
                        result.costs.sla_risk,
                        result.costs.terminal_backlog,
                    )
                )
                metrics["decision_count"] += 1
                metrics["defer_count"] += int(decision.decision == "defer")
                if decision.decision == "assign":
                    metrics["assign_by_dc"][str(decision.dc_id)] += 1
                    metrics["migration_count"] += int(
                        int(decision.dc_id) != task.origin_dc_id
                    )
            metrics["solve_seconds"].append(result.solve_seconds)
            metrics["integer_variables"].append(result.integer_variable_count)
            metrics["stage_cost"] += result.first_step_costs.total
            metrics["electricity"] += result.first_step_costs.electricity
            metrics["carbon"] += result.first_step_costs.carbon
            metrics["transmission"] += result.first_step_costs.transmission
            metrics["wait_steps"].extend(
                float(task.wait_intervals) for task in state.current.tasks
            )
            simulator.apply(result.first_step_decisions)
            step_rows.append(
                StepRecord(
                    episode_id,
                    step_index,
                    state.current.exogenous.current_time_utc,
                    len(state.current.tasks),
                    self._datacenters_json(state),
                    self._horizon_json(state),
                    json.dumps(
                        [asdict(item) for item in (forecast.uncertainties if forecast else ())],
                        sort_keys=True,
                    ),
                    result.status,
                    result.solve_seconds,
                    result.integer_variable_count,
                    float(result.objective_value or 0.0),
                    result.first_step_costs.electricity,
                    result.first_step_costs.carbon,
                    result.first_step_costs.transmission,
                    result.first_step_costs.waiting_defer,
                    result.first_step_costs.sla_risk,
                    result.first_step_costs.terminal_backlog,
                    -result.first_step_costs.total,
                    False,
                    False,
                )
            )
        summary = self._finish_metrics(metrics)
        summary.update(simulator.summary())
        summary.update(self._mean_forecast_errors(forecast_errors))
        return CollectedEpisode(episode, step_rows, task_rows, summary)

    def _build_state(
        self,
        env: Any,
        adapter: HorizonStateAdapter,
        forecaster: HistoricalArrivalForecaster,
        variant: str,
    ) -> tuple[HorizonState, BaselineForecast | None]:
        if variant == "oracle_upper_bound":
            return (
                adapter.build_horizon_state(env, self.config.horizon, "oracle"),
                None,
            )
        base = adapter.build_horizon_state(
            env, self.config.horizon, "no_future_arrivals"
        )
        if variant == "deployable_no_future":
            return base, None
        if variant != "deployable_baseline_forecast":
            raise ValueError(f"未知数据集变体 {variant!r}")
        forecaster.observe(base.current.tasks)
        forecast = forecaster.predict(base.timestep_minutes)
        return forecaster.apply(base, forecast), forecast

    def _synthetic_forecast_state(
        self,
        simulator: SyntheticEpisodeSimulator,
        base: HorizonState,
        forecaster: HistoricalArrivalForecaster,
        variant: str,
    ) -> tuple[HorizonState, BaselineForecast | None]:
        if variant == "deployable_no_future":
            return base, None
        if variant == "deployable_baseline_forecast":
            forecaster.observe(base.current.tasks)
            forecast = forecaster.predict(base.timestep_minutes)
            return forecaster.apply(base, forecast), forecast
        if variant != "oracle_upper_bound":
            raise ValueError(f"未知数据集变体 {variant!r}")
        grouped: dict[tuple[int, int], list[Any]] = {}
        for task in simulator.oracle_future_tasks(self.config.horizon):
            relative_step = task.arrival_step - simulator.current_step
            grouped.setdefault((relative_step, task.origin_dc_id), []).append(task)
        arrivals = tuple(
            FutureArrivalAggregate(
                arrival_step,
                origin_dc_id,
                len(tasks),
                sum(task.cpu_cores for task in tasks),
                sum(task.gpu_units for task in tasks),
                sum(task.memory_gb for task in tasks),
                max(task.duration_steps for task in tasks),
                min(task.deadline_step - simulator.current_step for task in tasks),
            )
            for (arrival_step, origin_dc_id), tasks in sorted(grouped.items())
        )
        return replace(base, forecast_mode="oracle", future_arrivals=arrivals), None

    @staticmethod
    def _forecast_errors(
        simulator: SyntheticEpisodeSimulator,
        forecast: BaselineForecast,
        horizon: int,
    ) -> list[dict[str, float]]:
        actual: dict[int, dict[str, float]] = {}
        for task in simulator.oracle_future_tasks(horizon):
            step = task.arrival_step - simulator.current_step
            totals = actual.setdefault(
                step, {"tasks": 0.0, "cpu": 0.0, "gpu": 0.0, "memory": 0.0}
            )
            totals["tasks"] += 1.0
            totals["cpu"] += task.cpu_cores
            totals["gpu"] += task.gpu_units
            totals["memory"] += task.memory_gb
        predicted: dict[int, dict[str, float]] = {}
        for item in forecast.aggregates:
            totals = predicted.setdefault(
                item.arrival_step,
                {"tasks": 0.0, "cpu": 0.0, "gpu": 0.0, "memory": 0.0},
            )
            totals["tasks"] += item.task_count
            totals["cpu"] += item.cpu_cores
            totals["gpu"] += item.gpu_units
            totals["memory"] += item.memory_gb
        return [
            {
                key: abs(
                    predicted.get(step, {}).get(key, 0.0)
                    - actual.get(step, {}).get(key, 0.0)
                )
                for key in ("tasks", "cpu", "gpu", "memory")
            }
            for step in range(1, horizon)
        ]

    @staticmethod
    def _mean_forecast_errors(
        errors: list[dict[str, float]],
    ) -> dict[str, float]:
        return {
            f"forecast_{key}_mae": mean(item[key] for item in errors)
            if errors
            else 0.0
            for key in ("tasks", "cpu", "gpu", "memory")
        }

    @staticmethod
    def _empty_metrics() -> dict[str, Any]:
        return {
            "decision_count": 0,
            "defer_count": 0,
            "migration_count": 0,
            "assign_by_dc": {str(dc): 0 for dc in range(1, 6)},
            "completed_tasks": 0,
            "sla_met": 0,
            "sla_violations": 0,
            "wait_steps": [],
            "rewards": [],
            "solve_seconds": [],
            "integer_variables": [],
            "stage_cost": 0.0,
            "electricity": 0.0,
            "carbon": 0.0,
            "transmission": 0.0,
            "infeasible_count": 0,
            "resource_overflow_count": 0,
            "illegal_action_count": 0,
        }

    @staticmethod
    def _update_runtime_metrics(
        metrics: dict[str, Any],
        info: dict,
        result: Any,
        state: HorizonState,
        reward: float,
    ) -> None:
        metrics["rewards"].append(reward)
        metrics["solve_seconds"].append(result.solve_seconds)
        metrics["integer_variables"].append(result.integer_variable_count)
        metrics["stage_cost"] += result.first_step_costs.total
        metrics["transmission"] += float(
            info.get("transmission_cost_total_usd", 0.0)
        )
        metrics["wait_steps"].extend(
            float(task.wait_intervals) for task in state.current.tasks
        )
        for dc_info in info["datacenter_infos"].values():
            common = dc_info["__common__"]
            metrics["electricity"] += float(common["energy_cost_USD"])
            metrics["carbon"] += float(common["carbon_emissions_kg"])
            metrics["completed_tasks"] += int(common["finished_tasks_count"])
            metrics["sla_met"] += int(common["__sla__"]["met"])
            metrics["sla_violations"] += int(common["__sla__"]["violated"])

    @staticmethod
    def _finish_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        waits = metrics.pop("wait_steps")
        solve = metrics.pop("solve_seconds")
        variables = metrics.pop("integer_variables")
        rewards = metrics.pop("rewards")
        decisions = metrics["decision_count"]
        assigned = decisions - metrics["defer_count"]
        metrics.update(
            {
                "defer_ratio": metrics["defer_count"] / max(1, decisions),
                "migration_ratio": metrics["migration_count"] / max(1, assigned),
                "average_wait_steps": mean(waits) if waits else 0.0,
                "p95_wait_steps": float(np.percentile(waits, 95)) if waits else 0.0,
                "total_reward": sum(rewards),
                "average_solve_seconds": mean(solve) if solve else 0.0,
                "maximum_solve_seconds": max(solve, default=0.0),
                "average_integer_variables": mean(variables) if variables else 0.0,
                "maximum_integer_variables": max(variables, default=0),
            }
        )
        return metrics

    @staticmethod
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

    @staticmethod
    def _datacenters_json(state: HorizonState) -> str:
        return json.dumps(
            [asdict(dc) for dc in state.current.datacenters], sort_keys=True
        )

    @staticmethod
    def _horizon_json(state: HorizonState) -> str:
        return json.dumps(
            {
                "datacenters": [asdict(dc) for dc in state.datacenters],
                "future_arrivals": [asdict(item) for item in state.future_arrivals],
                "running_tasks": [asdict(item) for item in state.running_tasks],
                "transit_tasks": [asdict(item) for item in state.transit_tasks],
            },
            sort_keys=True,
        )
