import sys
from types import SimpleNamespace

from monitors.rtamt_monitor import RTAMTMonitor


class _FakeSpec:
    def __init__(self) -> None:
        self.spec = ""
        self.declared: list[tuple[str, str]] = []

    def declare_var(self, name: str, typ: str) -> None:
        self.declared.append((name, typ))

    def parse(self) -> None:
        return None

    def evaluate(self, dataset):
        values = dataset["x"]
        rho = min(values)
        return [(0.0, rho)]


def _install_fake_rtamt(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "rtamt", SimpleNamespace(StlDiscreteTimeSpecification=_FakeSpec)
    )


def test_rtamt_monitor_always_nonnegative_pass(monkeypatch) -> None:
    _install_fake_rtamt(monkeypatch)
    monitor = RTAMTMonitor("always(x >= 0)", variables={"x": "float"})
    result = monitor.evaluate({"x": [(0.0, 1.0), (0.1, 0.2)]})
    assert result.satisfied is True
    assert result.robustness >= 0


def test_rtamt_monitor_always_nonnegative_fail(monkeypatch) -> None:
    _install_fake_rtamt(monkeypatch)
    monitor = RTAMTMonitor("always(x >= 0)", variables={"x": "float"})
    result = monitor.evaluate({"x": [(0.0, -0.1), (0.1, 0.2)]})
    assert result.satisfied is False
    assert result.robustness < 0
