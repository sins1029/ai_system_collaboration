from __future__ import annotations

import pandas as pd

from models.exogenous_signals import ExogenousSignals


def immediate_dispatch(signals: ExogenousSignals) -> pd.Series:
    return signals.to_frame()["batch_workload"].copy().rename("batch_service")
