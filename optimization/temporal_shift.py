from __future__ import annotations

import pandas as pd

from models.exogenous_signals import ExogenousSignals


def price_aware_dispatch(signals: ExogenousSignals, config: dict) -> pd.Series:
    batch_cfg = config["batch"]
    max_service = float(batch_cfg["max_service_per_step"])
    low_price = float(batch_cfg["low_price_threshold"])
    high_price = float(batch_cfg["high_price_threshold"])
    lookahead = int(batch_cfg["lookahead_steps"])

    backlog = 0.0
    dispatch: list[float] = []
    for step_index in range(len(signals)):
        row = signals.current(step_index)
        backlog += float(row["batch_workload"])
        price = float(row["electricity_price"])
        online = float(row["online_workload"])
        capacity_left = max(0.0, max_service - online)
        window_prices = signals.window(step_index, lookahead).values("electricity_price")

        if price <= low_price or price <= float(window_prices.min()):
            service = min(backlog, capacity_left)
        elif price >= high_price:
            service = min(backlog, capacity_left * 0.25)
        else:
            service = min(backlog, capacity_left * 0.55)

        remaining_steps = len(signals) - step_index
        if remaining_steps <= lookahead:
            service = min(backlog, capacity_left, max(service, backlog / remaining_steps))

        backlog -= service
        dispatch.append(service)

    return pd.Series(dispatch, name="batch_service")
