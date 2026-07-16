from __future__ import annotations

import re
import warnings

from .base import MonitorResult, Trace
from .trace import canonicalize_trace


_TOY_FORMULA_PATTERN = re.compile(
    r"^(always|eventually)\((?P<signal>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<comparator><=|>=|<|>|==|!=)\s*(?P<threshold>-?\d+(?:\.\d+)?)\)$"
)


class ToySTLMonitor:
    def __init__(self, formula: str):
        self.formula = formula
        warnings.warn(
            "ToySTLMonitor is deprecated and kept only as a fallback. Prefer backend='rtamt'.",
            DeprecationWarning,
            stacklevel=2,
        )

    def evaluate(self, trace: Trace) -> MonitorResult:
        canonical = canonicalize_trace(trace)
        match = _TOY_FORMULA_PATTERN.match(self.formula.replace(" ", ""))
        if not match:
            raise ValueError(
                "Toy monitor only supports 'always(...)' or 'eventually(...)' single-signal formulas."
            )

        operator = match.group(1)
        signal = match.group("signal")
        comparator = match.group("comparator")
        threshold = float(match.group("threshold"))

        if signal not in canonical:
            raise ValueError(f"Signal '{signal}' not found in trace.")

        values = [value for _, value in canonical[signal]]
        predicate_values = [_predicate(value, comparator, threshold) for value in values]

        satisfied = all(predicate_values) if operator == "always" else any(predicate_values)
        if comparator in {"<=", "<"}:
            robustness = -max(value - threshold for value in values)
        else:
            robustness = min(value - threshold for value in values)

        return MonitorResult(
            satisfied=satisfied,
            robustness=float(robustness),
            time=canonical[signal][0][0],
            formula=self.formula,
            signals=[signal],
            raw=predicate_values,
        )


def _predicate(value: float, comparator: str, threshold: float) -> bool:
    if comparator == "<":
        return value < threshold
    if comparator == "<=":
        return value <= threshold
    if comparator == ">":
        return value > threshold
    if comparator == ">=":
        return value >= threshold
    if comparator == "==":
        return value == threshold
    if comparator == "!=":
        return value != threshold
    raise ValueError(f"Unsupported comparator: {comparator}")
