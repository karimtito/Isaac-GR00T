from __future__ import annotations

from .base import MonitorResult, STLMonitor, TimeSeries, Trace
from .rtamt_monitor import RTAMTMonitor
from .toy_monitor import ToySTLMonitor


def create_monitor(
    formula: str,
    *,
    variables: dict[str, str] | None = None,
    backend: str = "rtamt",
) -> STLMonitor:
    selected = backend.lower()
    if selected == "rtamt":
        return RTAMTMonitor(formula=formula, variables=variables)
    if selected == "toy":
        return ToySTLMonitor(formula=formula)
    raise ValueError(f"Unsupported monitor backend: {backend}")


__all__ = [
    "MonitorResult",
    "STLMonitor",
    "TimeSeries",
    "Trace",
    "RTAMTMonitor",
    "ToySTLMonitor",
    "create_monitor",
]
