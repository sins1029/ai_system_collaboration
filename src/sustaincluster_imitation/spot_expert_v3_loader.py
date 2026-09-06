from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


VALID_SPLITS = frozenset({"train", "validation", "test", "full"})


@dataclass(frozen=True)
class SpotExpertDatasetV3:
    states: pd.DataFrame
    tasks: pd.DataFrame
    labels: pd.DataFrame | None = None
    simulator_truth: pd.DataFrame | None = None
    privileged_future: pd.DataFrame | None = None


def _suffix(split: str) -> str:
    return "val" if split == "validation" else split


def default_dataset_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "artifacts/mpc_expert_dataset_v3"
    )


def load_spot_expert_v3(
    split: str = "train",
    *,
    root: Path | None = None,
    include_labels: bool = False,
    include_simulator_truth: bool = False,
    include_privileged_future: bool = False,
) -> SpotExpertDatasetV3:
    """Load Spot v3 with deployable-current information as the safe default."""
    if split not in VALID_SPLITS:
        raise ValueError(f"unsupported split={split!r}")
    base = Path(root) if root is not None else default_dataset_root()
    suffix = _suffix(split)
    scenario = base / "dataset/scenario_b"
    states = pd.read_parquet(
        scenario / "deployable_current" / f"states_{suffix}.parquet"
    )
    tasks = pd.read_parquet(
        scenario / "deployable_current" / f"tasks_{suffix}.parquet"
    )
    labels = None
    truth = None
    future = None
    if include_labels:
        labels = pd.read_parquet(
            scenario / "labels" / f"expert_actions_{suffix}.parquet"
        )
    if include_simulator_truth:
        truth = pd.read_parquet(
            scenario
            / "simulator_only"
            / f"task_truth_decisions_{suffix}.parquet"
        )
    if include_privileged_future:
        future = pd.read_parquet(
            base
            / "privileged_future/scenario_b"
            / f"oracle_future_{suffix}.parquet"
        )
    return SpotExpertDatasetV3(states, tasks, labels, truth, future)
