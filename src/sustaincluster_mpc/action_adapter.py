from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from sustaincluster_mpc.state_adapter import TaskSnapshot


DecisionKind = Literal["assign", "defer"]


@dataclass(frozen=True)
class AssignmentDecision:
    task_id: str
    original_index: int
    decision: DecisionKind
    dc_id: int | None = None


@dataclass(frozen=True)
class ActionMapping:
    dc_id_to_action_items: tuple[tuple[int, int], ...]
    defer_action: int | None
    action_space_n: int

    @property
    def dc_id_to_action(self) -> Mapping[int, int]:
        return dict(self.dc_id_to_action_items)

    @property
    def allow_defer(self) -> bool:
        return self.defer_action is not None

    @classmethod
    def from_env(cls, env: Any) -> "ActionMapping":
        if not hasattr(env, "cluster_manager") or not hasattr(env, "action_space"):
            raise TypeError("env 必须提供 cluster_manager 和 action_space")
        ordered_dc_ids = tuple(
            int(dc.dc_id) for dc in env.cluster_manager.datacenters.values()
        )
        if not ordered_dc_ids or len(set(ordered_dc_ids)) != len(ordered_dc_ids):
            raise ValueError("数据中心编号不能为空且必须唯一")
        allow_defer = not bool(env.disable_defer_action)
        offset = 1 if allow_defer else 0
        items = tuple(
            (dc_id, position + offset)
            for position, dc_id in enumerate(ordered_dc_ids)
        )
        expected_n = len(ordered_dc_ids) + offset
        action_space_n = getattr(env.action_space, "n", None)
        if action_space_n is None or int(action_space_n) != expected_n:
            raise ValueError(
                f"action space mismatch: expected Discrete({expected_n}), "
                f"got {env.action_space!s}"
            )
        return cls(
            dc_id_to_action_items=items,
            defer_action=0 if allow_defer else None,
            action_space_n=expected_n,
        )


class SustainClusterActionAdapter:
    """在稳定语义决策与环境离散动作之间进行编码和校验。"""
    def __init__(self, action_mapping: ActionMapping) -> None:
        if not isinstance(action_mapping, ActionMapping):
            raise TypeError("action_mapping 必须是 ActionMapping 实例")
        self._mapping = action_mapping
        actions = tuple(action for _, action in action_mapping.dc_id_to_action_items)
        if len(set(actions)) != len(actions):
            raise ValueError("数据中心动作值必须唯一")
        for dc_id, action in action_mapping.dc_id_to_action_items:
            if dc_id <= 0:
                raise ValueError(f"dc_id 必须为正数，实际为 {dc_id}")
            self._validate_action_value(action, "mapping")
        if action_mapping.defer_action is not None:
            self._validate_action_value(action_mapping.defer_action, "defer")
            if action_mapping.defer_action in actions:
                raise ValueError("延后动作与数据中心动作冲突")

    @classmethod
    def from_env(cls, env: Any) -> "SustainClusterActionAdapter":
        return cls(ActionMapping.from_env(env))

    @property
    def mapping(self) -> ActionMapping:
        return self._mapping

    def assert_matches_env(self, env: Any) -> None:
        current = ActionMapping.from_env(env)
        if current != self._mapping:
            raise RuntimeError(
                "动作映射已失效；请在 env.reset() 后重建适配器"
            )

    def encode_assignments(
        self,
        task_snapshots: Sequence[TaskSnapshot],
        assignments: Sequence[AssignmentDecision],
    ) -> list[int]:
        if len(task_snapshots) != len(assignments):
            raise ValueError(
                f"assignment count mismatch: {len(assignments)} decisions for "
                f"{len(task_snapshots)} tasks"
            )
        if not task_snapshots:
            return []

        dc_actions = self._mapping.dc_id_to_action
        actions: list[int] = []
        for task, assignment in zip(task_snapshots, assignments):
            self._validate_correspondence(task, assignment)
            if assignment.decision == "defer":
                if assignment.dc_id is not None:
                    raise ValueError(
                        self._task_error(task, "延后决策不能包含 dc_id")
                    )
                if self._mapping.defer_action is None:
                    raise ValueError(
                        self._task_error(
                            task, "环境动作映射已禁用延后动作"
                        )
                    )
                action = self._mapping.defer_action
            elif assignment.decision == "assign":
                if assignment.dc_id is None:
                    raise ValueError(
                        self._task_error(task, "分配决策必须包含 dc_id")
                    )
                if assignment.dc_id not in dc_actions:
                    raise ValueError(
                        self._task_error(
                            task, f"unknown destination dc_id={assignment.dc_id}"
                        )
                    )
                action = dc_actions[assignment.dc_id]
            else:
                raise ValueError(
                    self._task_error(
                        task, f"unknown decision={assignment.decision!r}"
                    )
                )
            actions.append(action)

        self.validate_actions(task_snapshots, actions)
        return actions

    def validate_actions(
        self,
        task_snapshots: Sequence[TaskSnapshot],
        actions: Sequence[int],
    ) -> None:
        if len(actions) != len(task_snapshots):
            raise ValueError(
                f"action count mismatch: {len(actions)} actions for "
                f"{len(task_snapshots)} tasks"
            )
        legal_values = {action for _, action in self._mapping.dc_id_to_action_items}
        if self._mapping.defer_action is not None:
            legal_values.add(self._mapping.defer_action)
        for task, action in zip(task_snapshots, actions):
            if isinstance(action, bool) or not isinstance(action, int):
                raise TypeError(
                    self._task_error(task, f"action must be int, got {action!r}")
                )
            if action not in legal_values:
                raise ValueError(
                    self._task_error(
                        task,
                        f"illegal action={action}; legal values={sorted(legal_values)}",
                    )
                )
            self._validate_action_value(action, self._task_error(task, "action"))

    def _validate_action_value(self, action: int, label: str) -> None:
        if isinstance(action, bool) or not isinstance(action, int):
            raise TypeError(f"{label} 动作必须为整数")
        if action < 0 or action >= self._mapping.action_space_n:
            raise ValueError(
                f"{label} action={action} is outside Discrete"
                f"({self._mapping.action_space_n})"
            )

    @staticmethod
    def _validate_correspondence(
        task: TaskSnapshot, assignment: AssignmentDecision
    ) -> None:
        if assignment.original_index != task.original_index:
            raise ValueError(
                SustainClusterActionAdapter._task_error(
                    task,
                    f"decision original_index={assignment.original_index} 未对齐",
                )
            )
        if assignment.task_id != task.task_id:
            raise ValueError(
                SustainClusterActionAdapter._task_error(
                    task, f"decision task_id={assignment.task_id!r} 未对齐"
                )
            )

    @staticmethod
    def _task_error(task: TaskSnapshot, message: str) -> str:
        return (
            f"task_id={task.task_id!r}, original_index={task.original_index}: "
            f"{message}"
        )
