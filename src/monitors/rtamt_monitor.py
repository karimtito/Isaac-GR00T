from __future__ import annotations

from .base import MonitorResult, Trace
from .trace import canonicalize_trace


class RTAMTMonitor:
    def __init__(self, formula: str, variables: dict[str, str] | None = None):
        self.formula = formula
        self.variables = variables or {}

        try:
            import rtamt  # type: ignore
        except ImportError as error:  # pragma: no cover - handled by tests with monkeypatch
            raise RuntimeError("RTAMT package is not installed.") from error

        self.spec = rtamt.StlDiscreteTimeSpecification()
        for name, typ in self.variables.items():
            self.spec.declare_var(name, typ)
        self.spec.spec = formula
        self.spec.parse()

    def evaluate(self, trace: Trace) -> MonitorResult:
        canonical = canonicalize_trace(trace)
        any_series = next(iter(canonical.values()))
        dataset = {"time": [t for t, _ in any_series]}
        for name, series in canonical.items():
            dataset[name] = [v for _, v in series]

        raw = self.spec.evaluate(dataset)

        if isinstance(raw, list) and raw:
            t0, rho0 = raw[0]
            time = float(t0)
            robustness = float(rho0)
        else:
            time = None
            robustness = float(raw)

        return MonitorResult(
            satisfied=robustness >= 0.0,
            robustness=robustness,
            time=time,
            formula=self.formula,
            signals=list(canonical.keys()),
            raw=raw,
        )
