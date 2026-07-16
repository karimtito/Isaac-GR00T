from __future__ import annotations

from collections.abc import Callable
from typing import Any

from monitors.base import Trace


SignalExtractor = Callable[[Any, Any], float]


class RolloutTraceLogger:
    def __init__(self, signal_extractors: dict[str, SignalExtractor]):
        self.signal_extractors = signal_extractors
        self.trace: Trace = {}
        self.reset()

    def reset(self) -> None:
        self.trace = {name: [] for name in self.signal_extractors}

    def log_step(self, t: float, sim_state: Any, groot_action: Any = None) -> None:
        for name, extractor in self.signal_extractors.items():
            value = extractor(sim_state, groot_action)
            self.trace[name].append((float(t), float(value)))

    def merge_signals(self, t: float, extra_signals: dict[str, float]) -> None:
        for name, value in extra_signals.items():
            if name not in self.trace:
                self.trace[name] = []
            self.trace[name].append((float(t), float(value)))

    def get_trace(self) -> Trace:
        return self.trace
