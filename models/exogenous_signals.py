"""Deprecated compatibility imports for the package-external signal layer."""

from external_signals.models import (
    SIGNAL_COLUMNS,
    SIGNAL_UNITS,
    ExogenousSignals,
    SignalMetadata,
    SignalWindow,
    load_exogenous_signals,
)

__all__ = [
    "SIGNAL_COLUMNS",
    "SIGNAL_UNITS",
    "ExogenousSignals",
    "SignalMetadata",
    "SignalWindow",
    "load_exogenous_signals",
]
