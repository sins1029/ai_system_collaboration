from __future__ import annotations

import inspect
from types import MappingProxyType

import numpy as np

from scripts.competition import run_h60_common_counterfactual_value as audit
from scripts.imitation import build_spotgpu2026_expert_dataset_v3 as spot_v3


def _runtime_item(*, task_id: str, completion: int) -> dict[str, object]:
    return {
        "task_id": task_id,
        "row_index": int(task_id),
        "dc_id": 1,
        "planned_execution_start_step": 1,
        "estimated_completion_step": completion,
        "true_completion_step": completion,
    }


def test_runtime_clone_is_independent_for_mutable_branch_state() -> None:
    near = _runtime_item(task_id="1", completion=3)
    far = MappingProxyType(_runtime_item(task_id="2", completion=20))
    transit = _runtime_item(task_id="3", completion=8)
    base = spot_v3.RuntimeState(
        next_step=1,
        pending=[10, 11],
        running={1: near, 2: far},
        transit={3: transit},
        used={1: [1.0, 2.0, 3.0]},
        reserved={1: [4.0, 5.0, 6.0]},
        completed_count=7,
    )

    branch = audit.clone_runtime_state(base, last_advance_step=4)
    branch.pending.append(12)
    branch.running[1]["true_completion_step"] = 4
    branch.transit[3]["planned_execution_start_step"] = 2
    branch.used[1][0] = 99.0
    branch.reserved[1][0] = 88.0

    assert base.pending == [10, 11]
    assert base.running[1]["true_completion_step"] == 3
    assert base.transit[3]["planned_execution_start_step"] == 1
    assert base.used[1][0] == 1.0
    assert base.reserved[1][0] == 4.0
    assert branch.running[2] is base.running[2]


def test_relative_tolerance_classification() -> None:
    assert audit.classify_delta(-0.002, 1000.0)[0] == "POSITIVE"
    assert audit.classify_delta(0.002, 1000.0)[0] == "NEGATIVE"
    assert audit.classify_delta(0.0005, 1000.0)[0] == "EQUIVALENT"
    assert audit.classify_delta(5e-9, 0.0)[0] == "EQUIVALENT"


def test_paired_bootstrap_is_deterministic() -> None:
    savings = np.asarray([1.0, -0.5, 0.25, 2.0])
    first = audit.bootstrap_mean_saving_ci(savings, seed=22, repetitions=1000)
    second = audit.bootstrap_mean_saving_ci(savings, seed=22, repetitions=1000)
    assert first == second


def test_common_continuation_and_learned_information_boundary_are_explicit() -> None:
    continuation_source = inspect.getsource(audit.run_branch)
    learned_source = inspect.getsource(audit.build_learned_terminal_inputs)

    assert 'generator._solve(state, "h1")' in continuation_source
    assert "range(step + 1, step + K)" in continuation_source
    assert "MpcH60" not in continuation_source
    assert "PRIVILEGED" not in learned_source
    assert "true_duration" not in learned_source
    assert 'source="DEPLOYABLE_LEARNED_T60"' in learned_source
