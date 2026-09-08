from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sustaincluster_mpc.action_adapter import SustainClusterActionAdapter
from sustaincluster_mpc.horizon_adapter import HorizonState
from sustaincluster_mpc.rolling_horizon_optimizer import RollingHorizonConfig
from sustaincluster_mpc.terminal_h60_optimizer import (
    TerminalH60Config,
    TerminalH60DataCenter,
    TerminalH60Optimizer,
    TerminalH60Result,
)


@dataclass(frozen=True)
class H60ControllerContract:
    name: str
    information_class: str
    required_terminal_source: str
    deployable: bool


class H60Controller:
    """Typed controller boundary around the shared MPC-H60 optimizer."""

    contract: H60ControllerContract

    def __init__(self, optimizer: TerminalH60Optimizer | None = None) -> None:
        self.optimizer = optimizer or TerminalH60Optimizer()

    def solve(
        self,
        current_state: HorizonState,
        terminal_datacenters: Sequence[TerminalH60DataCenter],
        *,
        current_config: RollingHorizonConfig,
        terminal_config: TerminalH60Config,
        action_adapter: SustainClusterActionAdapter,
    ) -> TerminalH60Result:
        sources = {item.source for item in terminal_datacenters}
        expected = {self.contract.required_terminal_source}
        if sources != expected:
            raise ValueError(
                f"{self.contract.name} requires terminal source {expected}, got {sources}"
            )
        if self.contract.deployable and any("ORACLE" in source for source in sources):
            raise ValueError(f"{self.contract.name} rejected privileged Oracle input")
        return self.optimizer.solve(
            current_state,
            terminal_datacenters,
            current_config=current_config,
            terminal_config=terminal_config,
            action_adapter=action_adapter,
        )


class MpcH60PersistenceController(H60Controller):
    contract = H60ControllerContract(
        name="MPC-H60-Persistence",
        information_class="DEPLOYABLE_HISTORY_CURRENT_PREDICTED_T60",
        required_terminal_source="DEPLOYABLE_PERSISTENCE_T60",
        deployable=True,
    )


class MpcH60LearnedController(H60Controller):
    contract = H60ControllerContract(
        name="MPC-H60-Learned",
        information_class="DEPLOYABLE_HISTORY_CURRENT_PREDICTED_T60",
        required_terminal_source="DEPLOYABLE_LEARNED_T60",
        deployable=True,
    )


class MpcH60OracleController(H60Controller):
    contract = H60ControllerContract(
        name="MPC-H60-Oracle",
        information_class="PRIVILEGED_ORACLE_T60",
        required_terminal_source="PRIVILEGED_ORACLE_T60",
        deployable=False,
    )
