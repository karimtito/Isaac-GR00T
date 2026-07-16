from __future__ import annotations

from numbers import Real

from .base import Trace


def canonicalize_trace(trace: dict[str, list[tuple[float, float]]]) -> Trace:
    canonical: Trace = {}
    for signal, series in trace.items():
        if not isinstance(series, list):
            raise ValueError(f"Signal '{signal}' must map to a list of (time, value) tuples.")
        if not series:
            raise ValueError(f"Signal '{signal}' must not be empty.")

        canonical_series: list[tuple[float, float]] = []
        previous_time: float | None = None
        for index, point in enumerate(series):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(
                    f"Signal '{signal}' point at index {index} is not a (time, value) tuple."
                )
            t_raw, v_raw = point
            if not isinstance(t_raw, Real) or not isinstance(v_raw, Real):
                raise ValueError(f"Signal '{signal}' has non-numeric values at index {index}.")
            t = float(t_raw)
            v = float(v_raw)
            if previous_time is not None and t < previous_time:
                raise ValueError(f"Signal '{signal}' timestamps must be monotonic non-decreasing.")
            previous_time = t
            canonical_series.append((t, v))

        canonical[signal] = canonical_series

    if not canonical:
        raise ValueError("Trace must contain at least one signal.")
    return canonical


def validate_trace(trace: Trace) -> None:
    canonicalize_trace(trace)
