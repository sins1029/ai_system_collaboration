from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioTask:
    task_id: str
    arrival_step: int
    deadline_step: int
    duration_steps: int
    origin_dc_id: int = 1
    cpu_cores: float = 1.0
    gpu_units: float = 0.0
    memory_gb: float = 1.0
    bandwidth_gb: float = 1.0
    priority: str = "normal"


@dataclass(frozen=True)
class ScenarioDataCenter:
    dc_id: int
    cpu_capacity: tuple[float, ...]
    gpu_capacity: tuple[float, ...]
    memory_capacity: tuple[float, ...]
    electricity_price: tuple[float, ...]
    carbon_intensity: tuple[float, ...]


@dataclass(frozen=True)
class StressScenario:
    key: str
    name: str
    description: str
    steps: int
    datacenters: tuple[ScenarioDataCenter, ...]
    tasks: tuple[ScenarioTask, ...]


def _constant(value: float, steps: int) -> tuple[float, ...]:
    return (value,) * steps


def build_stress_scenarios() -> tuple[StressScenario, ...]:
    steps = 8
    gpu_burst = StressScenario(
        key="A_gpu_burst",
        name="GPU burst",
        description=(
            "Four flexible GPU tasks are pending before two urgent GPU tasks "
            "arrive at the next step."
        ),
        steps=steps,
        datacenters=(
            ScenarioDataCenter(
                1,
                _constant(32.0, steps),
                _constant(8.0, steps),
                _constant(128.0, steps),
                _constant(100.0, steps),
                _constant(300.0, steps),
            ),
        ),
        tasks=tuple(
            ScenarioTask(
                f"flex-gpu-{index}", 0, 8, 3, gpu_units=2.0, priority="low"
            )
            for index in range(4)
        )
        + tuple(
            ScenarioTask(
                f"urgent-gpu-{index}",
                1,
                4,
                2,
                gpu_units=4.0,
                priority="high",
            )
            for index in range(2)
        ),
    )

    future_low_price = StressScenario(
        key="B_future_low_price",
        name="Future low price",
        description=(
            "Flexible work can wait until the electricity price drops after "
            "two forecast steps."
        ),
        steps=steps,
        datacenters=(
            ScenarioDataCenter(
                1,
                _constant(20.0, steps),
                _constant(4.0, steps),
                _constant(128.0, steps),
                (1000.0, 1000.0, 20.0, 20.0, 20.0, 100.0, 100.0, 100.0),
                _constant(300.0, steps),
            ),
        ),
        tasks=tuple(
            ScenarioTask(
                f"flex-price-{index}",
                0,
                6,
                1,
                cpu_cores=5.0,
                gpu_units=1.0,
            )
            for index in range(3)
        ),
    )

    capacity_release = StressScenario(
        key="C_capacity_release",
        name="Capacity release",
        description=(
            "GPU capacity is unavailable now and at the next scheduling epoch, "
            "then becomes available."
        ),
        steps=steps,
        datacenters=(
            ScenarioDataCenter(
                1,
                _constant(32.0, steps),
                (0.0, 0.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0),
                _constant(128.0, steps),
                _constant(100.0, steps),
                _constant(300.0, steps),
            ),
        ),
        tasks=(
            ScenarioTask("release-wait", 0, 7, 2, gpu_units=4.0),
        ),
    )

    sla_conflict = StressScenario(
        key="D_sla_conflict",
        name="SLA conflict",
        description=(
            "A low-price interval is visible, but the urgent task must be "
            "dispatched immediately to finish by its deadline."
        ),
        steps=steps,
        datacenters=(
            ScenarioDataCenter(
                1,
                _constant(32.0, steps),
                _constant(8.0, steps),
                _constant(128.0, steps),
                (1000.0, 1000.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0),
                _constant(300.0, steps),
            ),
        ),
        tasks=(
            ScenarioTask(
                "urgent-price", 0, 2, 1, cpu_cores=8.0, gpu_units=2.0,
                priority="high"
            ),
        ),
    )

    center_difference = StressScenario(
        key="E_center_difference",
        name="Center capacity difference",
        description=(
            "A small inexpensive center and a larger expensive center require "
            "capacity-aware geographic placement."
        ),
        steps=steps,
        datacenters=(
            ScenarioDataCenter(
                1,
                _constant(8.0, steps),
                _constant(2.0, steps),
                _constant(32.0, steps),
                _constant(20.0, steps),
                _constant(100.0, steps),
            ),
            ScenarioDataCenter(
                2,
                _constant(32.0, steps),
                _constant(8.0, steps),
                _constant(128.0, steps),
                _constant(300.0, steps),
                _constant(500.0, steps),
            ),
        ),
        tasks=tuple(
            ScenarioTask(
                f"placement-{index}", 0, 6, 2, cpu_cores=6.0, gpu_units=2.0
            )
            for index in range(3)
        ),
    )
    return (
        gpu_burst,
        future_low_price,
        capacity_release,
        sla_conflict,
        center_difference,
    )


if __name__ == "__main__":
    for scenario in build_stress_scenarios():
        print(
            f"{scenario.key}: steps={scenario.steps}, "
            f"datacenters={len(scenario.datacenters)}, tasks={len(scenario.tasks)}"
        )
