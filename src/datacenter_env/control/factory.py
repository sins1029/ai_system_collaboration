from __future__ import annotations

from datacenter_env.config import DataCenterSystemConfig
from datacenter_env.contracts.protocols import DataCenterController
from datacenter_env.control.controllers import FeedbackController, FiniteHorizonController


def build_controller(config: DataCenterSystemConfig) -> DataCenterController:
    if config.controller_name == "baseline":
        return FeedbackController(config, "load_following")
    if config.controller_name == "heuristic":
        return FeedbackController(config, "price_aware_precooling")
    return FiniteHorizonController(config)
