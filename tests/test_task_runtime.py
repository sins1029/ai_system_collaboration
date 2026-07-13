from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import unittest

from datacenter_env import (
    DataCenterAction,
    DataCenterEnvironment,
    DataCenterSystem,
    DataCenterSystemConfig,
    ExogenousInput,
    NullRunStore,
    RunMetadata,
    TaskArrivalBatch,
    TaskSchedulingDecision,
    TaskSpec,
    TaskStatus,
)
from datacenter_env.exceptions import (
    InputValidationError,
    SchedulingError,
    UnschedulableTaskError,
)
from models.config_loader import load_simple_yaml


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 1, 1)


def config(
    scheduler: str = "fifo_immediate", policy: str = "record"
) -> DataCenterSystemConfig:
    return DataCenterSystemConfig.from_dict(
        load_simple_yaml(ROOT / "configs" / "datacenter.yaml"),
        load_simple_yaml(ROOT / "configs" / "optimization.yaml"),
        workload={"mode": "task_queue"},
        capacity={
            "total_cpu_cores": 100,
            "total_gpu_units": 10,
            "total_memory_gb": 200,
        },
        tasks={"unschedulable_policy": policy},
        scheduler={"name": scheduler, "forecast_steps": 4},
    )


def signal(index: int, workload: float = 0.0) -> ExogenousInput:
    return ExogenousInput(
        NOW + timedelta(minutes=15 * index), workload, 0.5, 0.4, 25.0, 20.0
    )


def task(
    task_id: str,
    arrival_step: int = 0,
    duration: int = 1,
    cpu: float = 50,
    deadline_steps: int = 4,
    deferrable: bool = True,
) -> TaskSpec:
    arrival = NOW + timedelta(minutes=15 * arrival_step)
    return TaskSpec(
        task_id,
        arrival,
        duration,
        cpu,
        0,
        20,
        arrival + timedelta(minutes=15 * deadline_steps),
        deferrable=deferrable,
    )


class TaskRuntimeTest(unittest.TestCase):
    def test_duration_one_completes_at_end_of_first_interval(self) -> None:
        system = DataCenterSystem.from_config(config(), store=NullRunStore())
        system.start_run(RunMetadata())
        result = system.step(signal(0), task_arrivals=TaskArrivalBatch(NOW, (task("one"),)))
        self.assertEqual(result.tasking.completed_task_ids, ("one",))
        outcome = system.environment.task_outcomes()[0]
        self.assertEqual(outcome.completion_time, NOW + timedelta(minutes=15))
        self.assertEqual(outcome.final_status, TaskStatus.COMPLETED)

    def test_multistep_task_is_nonpreemptive_and_releases_resources(self) -> None:
        system = DataCenterSystem.from_config(config(), store=NullRunStore())
        system.start_run(RunMetadata())
        first = system.step(
            signal(0), task_arrivals=TaskArrivalBatch(NOW, (task("long", duration=2, cpu=100),))
        )
        second_task = task("blocked", arrival_step=1, cpu=100)
        second = system.step(
            signal(1),
            task_arrivals=TaskArrivalBatch(signal(1).timestamp, (second_task,)),
        )
        third = system.step(
            signal(2), task_arrivals=TaskArrivalBatch(signal(2).timestamp, ())
        )
        self.assertEqual(first.tasking.running_count, 1)
        self.assertEqual(second.tasking.completed_task_ids, ("long",))
        self.assertEqual(second.tasking.waiting_count, 1)
        self.assertEqual(third.tasking.started_task_ids, ("blocked",))

    def test_external_aggregate_workload_is_rejected_in_task_mode(self) -> None:
        env = DataCenterEnvironment.from_config(config())
        with self.assertRaises(InputValidationError):
            env.step(
                DataCenterAction(100, TaskSchedulingDecision()),
                signal(0, workload=0.2),
                task_arrivals=TaskArrivalBatch(NOW, ()),
            )

    def test_unknown_or_overallocating_decision_is_explicit(self) -> None:
        env = DataCenterEnvironment.from_config(config())
        batch = TaskArrivalBatch(NOW, (task("a", cpu=60), task("b", cpu=60)))
        env.prepare_task_step(signal(0), batch)
        with self.assertRaises(SchedulingError):
            env.apply_task_decision(NOW, TaskSchedulingDecision(("unknown",)))

        env = DataCenterEnvironment.from_config(config())
        env.prepare_task_step(signal(0), batch)
        with self.assertRaises(SchedulingError):
            env.apply_task_decision(NOW, TaskSchedulingDecision(("a", "b")))

    def test_unschedulable_record_and_raise_policies(self) -> None:
        oversized = task("oversized", cpu=101)
        system = DataCenterSystem.from_config(config(policy="record"), store=NullRunStore())
        result = system.step(
            signal(0), task_arrivals=TaskArrivalBatch(NOW, (oversized,))
        )
        self.assertEqual(result.tasking.unschedulable_task_ids, ("oversized",))
        raising = DataCenterSystem.from_config(config(policy="raise"), store=NullRunStore())
        with self.assertRaises(UnschedulableTaskError):
            raising.step(signal(0), task_arrivals=TaskArrivalBatch(NOW, (oversized,)))

    def test_sla_completion_boundary_and_late_completion(self) -> None:
        on_time = task("on-time", deadline_steps=1)
        late = TaskSpec("late", NOW, 1, 40, 0, 20, NOW + timedelta(minutes=10))
        system = DataCenterSystem.from_config(config(), store=NullRunStore())
        result = system.step(
            signal(0), task_arrivals=TaskArrivalBatch(NOW, (on_time, late))
        )
        self.assertEqual(result.tasking.newly_sla_violated_task_ids, ("late",))
        outcomes = {item.task_id: item for item in system.environment.task_outcomes()}
        self.assertFalse(outcomes["on-time"].sla_violated)
        self.assertEqual(outcomes["late"].lateness_minutes, 5.0)

    def test_duplicate_arrival_across_steps_is_rejected(self) -> None:
        system = DataCenterSystem.from_config(config(), store=NullRunStore())
        original = task("same")
        system.step(signal(0), task_arrivals=TaskArrivalBatch(NOW, (original,)))
        duplicate = task("same", arrival_step=1)
        with self.assertRaises(InputValidationError):
            system.step(
                signal(1),
                task_arrivals=TaskArrivalBatch(signal(1).timestamp, (duplicate,)),
            )

    def test_reset_and_multiple_instances_isolate_task_state(self) -> None:
        a = DataCenterSystem.from_config(config(), store=NullRunStore())
        b = DataCenterSystem.from_config(config(), store=NullRunStore())
        a.step(signal(0), task_arrivals=TaskArrivalBatch(NOW, (task("a"),)))
        self.assertEqual(len(a.environment.task_outcomes()), 1)
        self.assertEqual(len(b.environment.task_outcomes()), 0)
        a.reset(seed=2)
        self.assertEqual(len(a.environment.task_outcomes()), 0)

    def test_unfinished_tasks_remain_unfinished_and_zero_throughput_ratios_are_none(self) -> None:
        system = DataCenterSystem.from_config(config(), store=NullRunStore())
        system.start_run(RunMetadata())
        system.step(
            signal(0),
            task_arrivals=TaskArrivalBatch(NOW, (task("unfinished", duration=4),)),
        )
        summary = system.finish_run()
        self.assertEqual(summary.metrics["tasks_unfinished"], 1.0)
        self.assertIsNone(summary.metrics["cost_per_completed_task"])
