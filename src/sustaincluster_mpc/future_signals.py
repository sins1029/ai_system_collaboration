from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence


FutureSignalMode = Literal["oracle", "persistence", "external"]
ExternalSignalSource = Callable[
    [str, int, Any, int, float], Sequence[float]
]


@dataclass(frozen=True)
class FutureSignalProvider:
    """Provides future exogenous signals with explicit provenance."""

    mode: FutureSignalMode
    external_source: ExternalSignalSource | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("oracle", "persistence", "external"):
            raise ValueError(f"Unsupported future signal mode={self.mode!r}")
        if self.mode == "external" and self.external_source is None:
            raise ValueError("external mode requires external_source")

    def electricity_price(
        self,
        dc: Any,
        current_time: Any,
        horizon: int,
    ) -> tuple[float, ...]:
        current = float(dc.price_manager.get_current_price())
        if self.mode == "oracle":
            return self._cyclic(
                dc.price_manager.prices,
                int(dc.price_manager.index),
                horizon,
                "electricity_price",
            )
        return self._deployable_values(
            "electricity_price", dc, current_time, horizon, current
        )

    def carbon_intensity(
        self,
        dc: Any,
        current_time: Any,
        horizon: int,
    ) -> tuple[float, ...]:
        current = float(dc.ci_manager.get_current_ci(norm=False))
        if self.mode == "oracle":
            return self._cyclic(
                dc.ci_manager.carbon_smooth,
                int(dc.ci_manager.time_step),
                horizon,
                "carbon_intensity",
                nonnegative=True,
            )
        return self._deployable_values(
            "carbon_intensity", dc, current_time, horizon, current,
            nonnegative=True,
        )

    def _deployable_values(
        self,
        signal: str,
        dc: Any,
        current_time: Any,
        horizon: int,
        current: float,
        *,
        nonnegative: bool = False,
    ) -> tuple[float, ...]:
        if self.mode == "persistence":
            raw = [current] * horizon
        else:
            assert self.external_source is not None
            raw = list(
                self.external_source(
                    signal,
                    int(dc.dc_id),
                    current_time,
                    horizon,
                    current,
                )
            )
        if len(raw) != horizon:
            raise ValueError(
                f"{signal} provider returned {len(raw)} values, expected {horizon}"
            )
        return tuple(
            self._validated(value, signal, nonnegative) for value in raw
        )

    @classmethod
    def _cyclic(
        cls,
        values: Sequence[float],
        index: int,
        horizon: int,
        signal: str,
        *,
        nonnegative: bool = False,
    ) -> tuple[float, ...]:
        if len(values) == 0:
            raise ValueError(f"Oracle {signal} source is empty")
        return tuple(
            cls._validated(
                values[(index + offset) % len(values)],
                signal,
                nonnegative,
            )
            for offset in range(horizon)
        )

    @staticmethod
    def _validated(value: Any, signal: str, nonnegative: bool) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{signal} values must be finite")
        if nonnegative and result < 0:
            raise ValueError(f"{signal} values must be nonnegative")
        return result
