from __future__ import annotations

from collections.abc import Callable, Iterable

from datacenter_env.config import DataCenterSystemConfig
from datacenter_env.contracts import (
    ExogenousInput,
    ForecastWindow,
    RunMetadata,
    RunStore,
    RunSummary,
)
from datacenter_env.runtime.system import DataCenterSystem


ForecastProvider = Callable[[int, ExogenousInput], ForecastWindow | None]


def run_single_center(
    config: DataCenterSystemConfig,
    signal_provider: Iterable[ExogenousInput],
    forecast_provider: ForecastProvider | None = None,
    metadata: RunMetadata | None = None,
    store: RunStore | None = None,
) -> RunSummary:
    system = DataCenterSystem.from_config(config, store=store)
    run_metadata = metadata or RunMetadata(
        controller_name=config.controller_name,
        condition_name=config.condition_name,
    )
    system.start_run(run_metadata, seed=run_metadata.seed)
    for index, current_input in enumerate(signal_provider):
        window = forecast_provider(index, current_input) if forecast_provider else None
        system.step(current_input=current_input, forecast_window=window)
    return system.finish_run()
