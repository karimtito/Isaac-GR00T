from monitors.trace import canonicalize_trace
import pytest


def test_trace_values_must_be_numeric() -> None:
    with pytest.raises(ValueError, match="non-numeric"):
        canonicalize_trace({"ee_x": [(0.0, "bad")]})


def test_trace_timestamps_must_be_monotonic() -> None:
    with pytest.raises(ValueError, match="monotonic"):
        canonicalize_trace({"ee_x": [(0.1, 1.0), (0.0, 2.0)]})


def test_trace_signals_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        canonicalize_trace({"ee_x": []})
