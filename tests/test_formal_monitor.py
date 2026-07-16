import sys
from types import SimpleNamespace

import pytest
from rfm_validator.formal_monitor import SpeedLimitMonitor, STLMonitor, run_formal_monitors
from rfm_validator.pipeline import load_trace
from rfm_validator.types import FormalRule


def test_speed_limit_monitor_passes_when_within_limit() -> None:
    trace = load_trace("examples/toy_trace.json")
    rule = FormalRule(
        rule_id="speed-001",
        description="Speed under 0.8",
        rule_type="speed_limit",
        parameters={"max_speed": 0.8},
    )

    verdict = SpeedLimitMonitor().evaluate(trace, rule)

    assert verdict.passed is True


def test_stl_monitor_always_over_observation_signal() -> None:
    trace = load_trace("examples/toy_trace.json")
    rule = FormalRule(
        rule_id="stl-obs-001",
        description="Always keep speed under 0.8",
        rule_type="stl",
        parameters={
            "temporal_operator": "always",
            "safety_function": {
                "source": "observation",
                "key": "speed",
                "comparator": "<=",
                "threshold": 0.8,
            },
        },
    )

    verdict = STLMonitor().evaluate(trace, rule)

    assert verdict.passed is True


def test_stl_monitor_eventually_over_action_signal() -> None:
    trace = load_trace("examples/toy_trace.json")
    rule = FormalRule(
        rule_id="stl-act-001",
        description="Eventually action delta reaches at least 0.05",
        rule_type="stl",
        parameters={
            "temporal_operator": "eventually",
            "safety_function": {
                "source": "action",
                "key": "delta",
                "comparator": ">=",
                "threshold": 0.05,
            },
        },
    )

    verdict = run_formal_monitors(trace, [rule])[0]

    assert verdict.passed is True


def test_stl_monitor_rtamt_engine_with_mocked_rtamt(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSpecification:
        def __init__(self) -> None:
            self.name = ""
            self.spec = ""
            self.declared: list[tuple[str, str]] = []

        def declare_var(self, name: str, data_type: str) -> None:
            self.declared.append((name, data_type))

        def parse(self) -> None:
            return None

        def evaluate(self, dataset):
            assert dataset["time"] == [0, 1]
            assert dataset["speed"] == [0.2, 0.7]
            return [(0, 0.2), (1, 0.3)]

    fake_rtamt = SimpleNamespace(StlDiscreteTimeSpecification=FakeSpecification)
    monkeypatch.setitem(sys.modules, "rtamt", fake_rtamt)

    trace = load_trace("examples/toy_trace.json")
    rule = FormalRule(
        rule_id="stl-rtamt-001",
        description="RTAMT-backed always check",
        rule_type="stl",
        parameters={
            "engine": "rtamt",
            "rtamt_variable": "speed",
            "rtamt_spec": "always(speed <= 0.8)",
            "safety_function": {
                "source": "observation",
                "key": "speed",
                "comparator": "<=",
                "threshold": 0.8,
            },
        },
    )

    verdict = STLMonitor().evaluate(trace, rule)

    assert verdict.passed is True
    assert verdict.metadata["engine"] == "rtamt"
    assert verdict.metadata["robustness"] == 0.3


def test_stl_monitor_rtamt_engine_without_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "rtamt", raising=False)
    import builtins

    original_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "rtamt":
            raise ImportError("rtamt not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)

    trace = load_trace("examples/toy_trace.json")
    rule = FormalRule(
        rule_id="stl-rtamt-002",
        description="RTAMT missing dependency path",
        rule_type="stl",
        parameters={
            "engine": "rtamt",
            "safety_function": {
                "source": "observation",
                "key": "speed",
                "comparator": "<=",
                "threshold": 0.8,
            },
        },
    )

    verdict = STLMonitor().evaluate(trace, rule)

    assert verdict.passed is False
    assert "not installed" in verdict.rationale.lower()
    assert verdict.metadata["engine"] == "rtamt"
