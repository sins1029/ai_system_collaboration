from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from sustaincluster_imitation.forecast_baseline import ForecastUncertainty
from sustaincluster_mpc import (
    AssignmentDecision,
    HorizonState,
    TaskSnapshot,
)


@dataclass(frozen=True)
class EncodedTaskBatch:
    features: np.ndarray
    feasible_action_mask: np.ndarray
    task_ids: tuple[str, ...]
    original_indices: tuple[int, ...]


@dataclass(frozen=True)
class SemanticActionSpace:
    dc_ids: tuple[int, ...]
    allow_defer: bool = True

    def __post_init__(self) -> None:
        if not self.dc_ids or len(set(self.dc_ids)) != len(self.dc_ids):
            raise ValueError("dc_ids 不能为空且必须唯一")
        if tuple(sorted(self.dc_ids)) != self.dc_ids:
            raise ValueError("dc_ids 必须使用稳定升序")

    @property
    def size(self) -> int:
        return len(self.dc_ids) + int(self.allow_defer)

    @property
    def defer_index(self) -> int | None:
        return 0 if self.allow_defer else None

    def index_for_dc(self, dc_id: int) -> int:
        if dc_id not in self.dc_ids:
            raise ValueError(f"未知语义 dc_id={dc_id}")
        return self.dc_ids.index(dc_id) + int(self.allow_defer)

    def dc_for_index(self, index: int) -> int | None:
        if self.allow_defer and index == 0:
            return None
        position = index - int(self.allow_defer)
        if position < 0 or position >= len(self.dc_ids):
            raise ValueError(f"语义动作索引超出范围： {index}")
        return self.dc_ids[position]

    def encode_decision(self, decision: AssignmentDecision) -> int:
        if decision.decision == "defer":
            if not self.allow_defer:
                raise ValueError("延后动作已禁用")
            return 0
        return self.index_for_dc(int(decision.dc_id))


class SustainClusterFeatureEncoder:
    """按语义数据中心编号稳定排序任务特征，不依赖环境位置。"""

    def __init__(
        self,
        action_space: SemanticActionSpace,
        horizon: int = 4,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon 必须为正数")
        self.action_space = action_space
        self.horizon = horizon

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names())

    def feature_names(self) -> tuple[str, ...]:
        names = [
            "task_cpu_cores_scaled",
            "task_gpu_units_scaled",
            "task_memory_gb_scaled",
            "task_duration_steps_scaled",
            "task_remaining_duration_steps_scaled",
            "task_remaining_sla_steps_scaled",
            "task_bandwidth_gb_scaled",
            "task_wait_intervals_scaled",
            "task_was_deferred",
        ]
        names.extend(f"task_origin_dc_{dc_id}" for dc_id in self.action_space.dc_ids)
        for dc_id in self.action_space.dc_ids:
            names.extend(
                f"dc{dc_id}_{name}"
                for name in (
                    "present",
                    "cpu_schedulable_ratio",
                    "gpu_schedulable_ratio",
                    "memory_schedulable_ratio",
                    "running_scaled",
                    "queued_scaled",
                    "transit_scaled",
                    "price_scaled",
                    "carbon_scaled",
                    "total_power_scaled",
                    "temperature_scaled",
                )
            )
            for step in range(self.horizon):
                names.extend(
                    f"dc{dc_id}_h{step}_{name}"
                    for name in (
                        "cpu_available_ratio",
                        "gpu_available_ratio",
                        "memory_available_ratio",
                        "price_scaled",
                        "carbon_scaled",
                        "known_gpu_reservation_ratio",
                        "forecast_gpu_reservation_ratio",
                    )
                )
        for step in range(self.horizon):
            names.extend(
                f"arrival_h{step}_{name}"
                for name in (
                    "task_count_scaled",
                    "cpu_scaled",
                    "gpu_scaled",
                    "memory_scaled",
                    "count_uncertainty_scaled",
                    "gpu_uncertainty_scaled",
                )
            )
        return tuple(names)

    def encode(
        self,
        state: HorizonState,
        uncertainties: Sequence[ForecastUncertainty] = (),
    ) -> EncodedTaskBatch:
        if state.horizon != self.horizon:
            raise ValueError("状态 horizon 与特征编码器不一致")
        global_features = self._global_features(state, uncertainties)
        features = []
        masks = []
        for task in state.current.tasks:
            features.append(self._task_features(task, state) + global_features)
            masks.append(self._feasible_mask(task, state))
        if features:
            matrix = np.asarray(features, dtype=np.float32)
            mask = np.asarray(masks, dtype=bool)
        else:
            matrix = np.empty((0, self.feature_dim), dtype=np.float32)
            mask = np.empty((0, self.action_space.size), dtype=bool)
        if matrix.shape[1] != self.feature_dim:
            raise RuntimeError("特征维度与声明的模式不一致")
        return EncodedTaskBatch(
            matrix,
            mask,
            tuple(task.task_id for task in state.current.tasks),
            tuple(task.original_index for task in state.current.tasks),
        )

    def _task_features(
        self, task: TaskSnapshot, state: HorizonState
    ) -> list[float]:
        step_minutes = state.timestep_minutes
        values = [
            task.cpu_cores / 1000.0,
            task.gpu_units / 100.0,
            task.memory_gb / 10000.0,
            task.duration_minutes / step_minutes / 100.0,
            task.remaining_duration_minutes / step_minutes / 100.0,
            task.remaining_sla_minutes / step_minutes / 100.0,
            task.bandwidth_gb / 100.0,
            task.wait_intervals / 100.0,
            float(task.was_deferred),
        ]
        values.extend(float(task.origin_dc_id == dc_id) for dc_id in self.action_space.dc_ids)
        return values

    def _global_features(
        self,
        state: HorizonState,
        uncertainties: Sequence[ForecastUncertainty],
    ) -> list[float]:
        current = {dc.dc_id: dc for dc in state.current.datacenters}
        horizon = {dc.dc_id: dc for dc in state.datacenters}
        values: list[float] = []
        for dc_id in self.action_space.dc_ids:
            dc = current.get(dc_id)
            hdc = horizon.get(dc_id)
            if dc is None or hdc is None:
                values.extend([0.0] * (11 + self.horizon * 7))
                continue
            values.extend(
                [
                    1.0,
                    dc.cpu_schedulable_cores / max(dc.cpu_total_cores, 1.0),
                    dc.gpu_schedulable_units / max(dc.gpu_total_units, 1.0),
                    dc.memory_schedulable_gb / max(dc.memory_total_gb, 1.0),
                    dc.running_task_count / 1000.0,
                    dc.queued_task_count / 1000.0,
                    dc.in_transit_task_count / 1000.0,
                    dc.electricity_price_usd_per_mwh / 1000.0,
                    dc.carbon_intensity_gco2_per_kwh / 1000.0,
                    (dc.total_power_kw or 0.0) / 100000.0,
                    (dc.internal_temperature_c or 0.0) / 100.0,
                ]
            )
            for step in range(self.horizon):
                values.extend(
                    [
                        hdc.cpu_available_cores[step] / max(hdc.cpu_total_cores, 1.0),
                        hdc.gpu_available_units[step] / max(hdc.gpu_total_units, 1.0),
                        hdc.memory_available_gb[step] / max(hdc.memory_total_gb, 1.0),
                        hdc.electricity_price_usd_per_mwh[step] / 1000.0,
                        hdc.carbon_intensity_gco2_per_kwh[step] / 1000.0,
                        hdc.known_gpu_reservations[step] / max(hdc.gpu_total_units, 1.0),
                        hdc.forecast_gpu_reservations[step] / max(hdc.gpu_total_units, 1.0),
                    ]
                )
        uncertainty_by_step = {
            step: [item for item in uncertainties if item.arrival_step == step]
            for step in range(self.horizon)
        }
        for step in range(self.horizon):
            arrivals = [
                item for item in state.future_arrivals if item.arrival_step == step
            ]
            uncertainty = uncertainty_by_step[step]
            values.extend(
                [
                    sum(item.task_count for item in arrivals) / 1000.0,
                    sum(item.cpu_cores for item in arrivals) / 10000.0,
                    sum(item.gpu_units for item in arrivals) / 1000.0,
                    sum(item.memory_gb for item in arrivals) / 100000.0,
                    sum(item.task_count_std for item in uncertainty) / 1000.0,
                    sum(item.gpu_units_std for item in uncertainty) / 1000.0,
                ]
            )
        return values

    def _feasible_mask(
        self, task: TaskSnapshot, state: HorizonState
    ) -> list[bool]:
        mask = [True] if self.action_space.allow_defer else []
        by_id = {dc.dc_id: dc for dc in state.datacenters}
        destinations = {
            item.destination_dc_id: item
            for item in state.current.task_destinations
            if item.original_index == task.original_index
            and item.task_id == task.task_id
        }
        duration_steps = max(
            1,
            int(
                math.ceil(
                    max(0.0, task.remaining_duration_minutes)
                    / state.timestep_minutes
                )
            ),
        )
        deadline_step = max(
            0,
            int(
                math.floor(
                    task.remaining_sla_minutes / state.timestep_minutes
                )
            ),
        )
        for dc_id in self.action_space.dc_ids:
            dc = by_id.get(dc_id)
            destination = destinations.get(dc_id)
            if dc is None or destination is None:
                mask.append(False)
                continue
            transfer_steps = max(
                1,
                int(
                    math.ceil(
                        max(0.0, destination.transmission_delay_seconds)
                        / 60.0
                        / state.timestep_minutes
                    )
                ),
            )
            execution_step = transfer_steps
            completion_step = execution_step + duration_steps
            effective_start = min(execution_step, state.horizon - 1)
            effective_stop = min(
                state.horizon, effective_start + duration_steps
            )
            feasible = not (
                (state.horizon > 1 and execution_step >= state.horizon)
                or completion_step > deadline_step
            ) and all(
                task.cpu_cores <= dc.cpu_available_cores[step] + 1e-9
                and task.gpu_units <= dc.gpu_available_units[step] + 1e-9
                and task.memory_gb <= dc.memory_available_gb[step] + 1e-9
                for step in range(effective_start, effective_stop)
            )
            mask.append(bool(feasible))
        return mask
