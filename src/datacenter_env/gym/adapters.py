from __future__ import annotations

from datacenter_env.contracts import TaskScheduler
from datacenter_env.gym.action import DEFER_CURRENT_TASK, START_CURRENT_TASK


class TaskSchedulerGymAdapter:
    def __init__(self, scheduler: TaskScheduler):
        self.scheduler = scheduler
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self.scheduler.reset(seed)
        self._simulation_step_index = -1
        self._planned_start_ids: frozenset[str] = frozenset()

    def action(self, env, observation, info) -> int:
        del observation, info
        if env.simulation_step_index != self._simulation_step_index:
            decision = self.scheduler.schedule(
                env.scheduling_observation,
                env.forecast_for_scheduler(self.scheduler.max_forecast_steps),
            )
            self._planned_start_ids = frozenset(decision.start_task_ids)
            env.set_planned_start_ids(decision.start_task_ids)
            self._simulation_step_index = env.simulation_step_index
        candidate = env.current_candidate
        if candidate is not None and candidate.spec.task_id in self._planned_start_ids:
            return START_CURRENT_TASK
        return DEFER_CURRENT_TASK


class RuleBasedGymPolicy:
    def __init__(self, env, scheduler: TaskScheduler):
        self.env = env
        self.adapter = TaskSchedulerGymAdapter(scheduler)

    def reset(self, seed: int | None = None) -> None:
        self.adapter.reset(seed)

    def action(self, observation, info) -> int:
        return self.adapter.action(self.env, observation, info)
