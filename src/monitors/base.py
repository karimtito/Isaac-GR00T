from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


TimeSeries = list[tuple[float, float]]
Trace = dict[str, TimeSeries]


@dataclass
class MonitorResult:
    satisfied: bool
    robustness: float
    time: float | None
    formula: str
    signals: list[str]
    raw: Any = None


class STLMonitor(Protocol):
    def evaluate(self, trace: Trace) -> MonitorResult: ...
