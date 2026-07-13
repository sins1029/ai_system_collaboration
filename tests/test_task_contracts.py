from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from datacenter_env import (
    ExogenousInput,
    ForecastWindow,
    TaskArrivalBatch,
    TaskSchedulingDecision,
    TaskSpec,
)
from datacenter_env.contracts.tasks import (
    EnvironmentalView,
    ResourceAvailability,
    ResourceCapacity,
    ResourceUsage,
    TaskSchedulingObservation,
    TaskStatus,
    TaskView,
)
from datacenter_env.exceptions import InputValidationError
from datacenter_env.tasking import (
    EarliestDeadlineFirstScheduler,
    EnergyAwareDeferralScheduler,
    FifoImmediateScheduler,
    TaskLoadAggregator,
)


NOW = datetime(2026, 1, 1)


def task(
    task_id: str,
    *,
    arrival: datetime = NOW,
    duration: int = 1,
    cpu: float = 10.0,
    gpu: float = 0.0,
    memory: float = 10.0,
    deadline_steps: int = 4,
    priority: int = 0,
    deferrable: bool = True,
) -> TaskSpec:
    return TaskSpec(
        task_id,
        arrival,
        duration,
        cpu,
        gpu,
        memory,
        arrival + timedelta(minutes=15 * deadline_steps),
        priority,
        deferrable,
    )


def observation(*tasks: TaskSpec) -> TaskSchedulingObservation:
    views = tuple(
        TaskView(
            item,
            TaskStatus.WAITING,
            item.duration_steps,
            0,
            0,
            item.deadline_time - timedelta(minutes=15 * item.duration_steps),
        )
        for item in tasks
    )
    return TaskSchedulingObservation(
        NOW,
        views,
        (),
        ResourceCapacity(100.0, 10.0, 200.0),
        ResourceUsage(),
        ResourceAvailability(100.0, 10.0, 200.0),
        EnvironmentalView(0.8, 0.6, 25.0, 0.0),
    )


def signal(index: int, price: float, carbon: float, renewable: float) -> ExogenousInput:
    return ExogenousInput(
        NOW + timedelta(minutes=15 * index), 0.0, price, carbon, 25.0, renewable
    )


class TaskContractTest(unittest.TestCase):
    def test_task_spec_rejects_invalid_duration_resources_and_deadline(self) -> None:
        with self.assertRaises(InputValidationError):
            task("bad-duration", duration=0)
        with self.assertRaises(InputValidationError):
            task("bad-resource", cpu=-1)
        with self.assertRaises(InputValidationError):
            task("no-compute", cpu=0, gpu=0)
        with self.assertRaises(InputValidationError):
            task("bad-deadline", deadline_steps=0)

    def test_task_spec_rejects_timezone_mismatch(self) -> None:
        with self.assertRaises(InputValidationError):
            TaskSpec(
                "mixed-timezone",
                NOW,
                1,
                1,
                0,
                1,
                NOW.replace(tzinfo=timezone.utc) + timedelta(minutes=15),
            )

    def test_arrival_batch_rejects_duplicate_ids_and_timestamp_mismatch(self) -> None:
        item = task("duplicate")
        with self.assertRaises(InputValidationError):
            TaskArrivalBatch(NOW, (item, item))
        with self.assertRaises(InputValidationError):
            TaskArrivalBatch(NOW + timedelta(minutes=15), (item,))

    def test_decision_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(InputValidationError):
            TaskSchedulingDecision(("a", "a"))

    def test_fifo_orders_by_arrival_then_id_and_packs_capacity(self) -> None:
        first = task("b", cpu=70)
        second = task("a", cpu=40)
        decision = FifoImmediateScheduler().schedule(observation(first, second))
        self.assertEqual(decision.start_task_ids, ("a",))

    def test_edf_orders_deadline_priority_arrival_and_id(self) -> None:
        later = task("later", deadline_steps=4, priority=5, cpu=100)
        urgent = task("urgent", deadline_steps=2, priority=0, cpu=100)
        decision = EarliestDeadlineFirstScheduler().schedule(observation(later, urgent))
        self.assertEqual(decision.start_task_ids, ("urgent",))

    def test_energy_aware_defers_only_deferrable_work(self) -> None:
        flexible = task("flexible", deadline_steps=6, deferrable=True, cpu=40)
        fixed = task("fixed", deadline_steps=6, deferrable=False, cpu=40)
        window = ForecastWindow(
            (signal(0, 0.9, 0.7, 0), signal(1, 0.2, 0.3, 100))
        )
        decision = EnergyAwareDeferralScheduler().schedule(
            observation(flexible, fixed), window
        )
        self.assertEqual(decision.start_task_ids, ("fixed",))

    def test_energy_aware_never_defers_past_latest_start(self) -> None:
        due = task("due", duration=2, deadline_steps=2, deferrable=True)
        window = ForecastWindow(
            (signal(0, 0.9, 0.7, 0), signal(1, 0.1, 0.2, 100))
        )
        decision = EnergyAwareDeferralScheduler().schedule(observation(due), window)
        self.assertEqual(decision.start_task_ids, ("due",))

    def test_weighted_and_max_aggregation_match_hand_calculation(self) -> None:
        capacity = ResourceCapacity(100, 10, 200)
        usage = ResourceUsage(50, 5, 100)
        weighted = TaskLoadAggregator(capacity, "weighted_sum", 0.4, 0.4, 0.2)
        maximum = TaskLoadAggregator(capacity, "max_utilization")
        self.assertAlmostEqual(weighted.aggregate(usage), 0.5)
        self.assertAlmostEqual(maximum.aggregate(usage), 0.5)
