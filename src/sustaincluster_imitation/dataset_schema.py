from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


DATASET_SCHEMA_VERSION = "1.0.0"
SemanticDecision = Literal["defer", "assign"]


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    scenario_name: str
    seed: int
    workload_window: str
    sustaincluster_commit: str
    project_commit: str
    horizon: int
    forecast_mode: str
    objective_weights_json: str
    timestep_minutes: float
    dataset_variant: str
    split_hint: str
    burst_intensity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StepRecord:
    episode_id: str
    step_index: int
    timestamp: str
    pending_task_count: int
    datacenters_json: str
    horizon_summary_json: str
    forecast_uncertainty_json: str
    solver_status: str
    solve_seconds: float
    integer_variable_count: int
    mpc_objective: float
    electricity_component: float
    carbon_component: float
    transmission_component: float
    waiting_component: float
    sla_component: float
    terminal_backlog_component: float
    reward: float
    terminated: bool
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskActionRecord:
    episode_id: str
    scenario_name: str
    seed: int
    dataset_variant: str
    step_index: int
    timestamp: str
    task_id: str
    original_index: int
    origin_dc_id: int
    cpu_cores: float
    gpu_units: float
    memory_gb: float
    duration_minutes: float
    remaining_duration_minutes: float
    remaining_sla_minutes: float
    bandwidth_gb: float
    wait_intervals: int
    was_deferred: bool
    destinations_json: str
    semantic_decision: SemanticDecision
    destination_dc_id: int | None
    semantic_label_index: int
    dispatch_step: int | None
    execution_start_step: int | None
    projected_completion_step: int | None
    feature_vector: list[float]
    feasible_action_mask: list[bool]
    solver_status: str
    solve_seconds: float
    integer_variable_count: int
    mpc_objective: float
    electricity_component: float
    carbon_component: float
    transmission_component: float
    waiting_component: float
    sla_component: float
    terminal_backlog_component: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
