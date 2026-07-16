from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import ExecutionTrace, FormalRule, ValidatorVerdict


class FormalMonitor(ABC):
    """Placeholder interface for formal rule monitoring (e.g., STL/LTL in future)."""

    @abstractmethod
    def evaluate(self, trace: ExecutionTrace, rule: FormalRule) -> ValidatorVerdict:
        raise NotImplementedError


class SpeedLimitMonitor(FormalMonitor):
    """Toy monitor: checks observed speed values against a max_speed threshold."""

    def evaluate(self, trace: ExecutionTrace, rule: FormalRule) -> ValidatorVerdict:
        max_speed = float(rule.parameters.get("max_speed", 1.0))
        speeds = [float(obs.data["speed"]) for obs in trace.observations if "speed" in obs.data]

        if not speeds:
            return ValidatorVerdict(
                rule_id=rule.rule_id,
                passed=False,
                confidence=0.5,
                rationale="No speed observations available for speed-limit check.",
            )

        peak_speed = max(speeds)
        passed = peak_speed <= max_speed
        rationale = (
            f"Peak speed {peak_speed:.3f} within limit {max_speed:.3f}."
            if passed
            else f"Peak speed {peak_speed:.3f} exceeds limit {max_speed:.3f}."
        )
        return ValidatorVerdict(
            rule_id=rule.rule_id, passed=passed, confidence=1.0, rationale=rationale
        )


class STLMonitor(FormalMonitor):
    """Initial STL monitor over safety functions derived from trace signals.

    For action-sourced safety functions, `key="name"` maps to a binary signal:
    1.0 when the action name matches `expected_name` (if provided), otherwise
    1.0 when any action name is present and 0.0 when missing.
    """

    def evaluate(self, trace: ExecutionTrace, rule: FormalRule) -> ValidatorVerdict:
        operator = str(
            rule.parameters.get("temporal_operator", rule.parameters.get("operator", "always"))
        ).lower()
        safety_function = rule.parameters.get("safety_function", {})

        if not isinstance(safety_function, dict):
            return ValidatorVerdict(
                rule_id=rule.rule_id,
                passed=False,
                confidence=0.0,
                rationale="Invalid STL safety_function configuration.",
            )

        signal = self._extract_signal(trace, safety_function)
        if not signal:
            return ValidatorVerdict(
                rule_id=rule.rule_id,
                passed=False,
                confidence=0.0,
                rationale="No STL signal samples available from trace for configured safety function.",
            )

        engine = str(rule.parameters.get("engine", "builtin")).lower()
        if engine == "rtamt":
            return self._evaluate_with_rtamt(rule=rule, signal=signal)
        if engine == "argus":
            return ValidatorVerdict(
                rule_id=rule.rule_id,
                passed=False,
                confidence=0.0,
                rationale="Argus backend is not integrated yet. Use engine='rtamt' or builtin STL monitoring.",
            )

        comparator = str(safety_function.get("comparator", "<="))
        threshold = float(safety_function.get("threshold", 0.0))
        try:
            predicate_values = [self._predicate(sample, comparator, threshold) for sample in signal]
        except ValueError as error:
            return ValidatorVerdict(
                rule_id=rule.rule_id,
                passed=False,
                confidence=0.0,
                rationale=str(error),
            )

        if operator == "always":
            passed = all(predicate_values)
        elif operator == "eventually":
            passed = any(predicate_values)
        else:
            return ValidatorVerdict(
                rule_id=rule.rule_id,
                passed=False,
                confidence=0.0,
                rationale=f"Unsupported STL temporal operator: {operator}",
            )

        rationale = (
            f"STL '{operator}' satisfied for {len(signal)} samples with predicate value {comparator} {threshold:.3f}."
            if passed
            else f"STL '{operator}' violated for predicate value {comparator} {threshold:.3f} over {len(signal)} samples."
        )
        return ValidatorVerdict(
            rule_id=rule.rule_id,
            passed=passed,
            confidence=1.0,
            rationale=rationale,
            metadata={
                "operator": operator,
                "samples": len(signal),
                "signal_source": safety_function.get("source", "observation"),
                "engine": "builtin",
            },
        )

    def _extract_signal(
        self, trace: ExecutionTrace, safety_function: dict[str, Any]
    ) -> list[float]:
        source = str(safety_function.get("source", "observation")).lower()
        key = str(safety_function.get("key", ""))

        if not key:
            return []

        values: list[float] = []
        if source == "observation":
            for observation in trace.observations:
                if key in observation.data:
                    values.append(float(observation.data[key]))
        elif source == "action":
            if key == "name":
                expected_name = safety_function.get("expected_name")
                for action in trace.actions:
                    if expected_name is None:
                        values.append(1.0 if action.name else 0.0)
                    else:
                        values.append(1.0 if action.name == str(expected_name) else 0.0)
            else:
                for action in trace.actions:
                    if key in action.parameters:
                        values.append(float(action.parameters[key]))
        return values

    def _predicate(self, value: float, comparator: str, threshold: float) -> bool:
        if comparator == "<":
            return value < threshold
        elif comparator == "<=":
            return value <= threshold
        elif comparator == ">":
            return value > threshold
        elif comparator == ">=":
            return value >= threshold
        elif comparator == "==":
            return value == threshold
        elif comparator == "!=":
            return value != threshold
        raise ValueError(f"Unsupported STL comparator: {comparator}")

    def _evaluate_with_rtamt(self, rule: FormalRule, signal: list[float]) -> ValidatorVerdict:
        try:
            import rtamt  # type: ignore
        except ImportError:
            return ValidatorVerdict(
                rule_id=rule.rule_id,
                passed=False,
                confidence=0.0,
                rationale="RTAMT package is not installed. Install it to enable engine='rtamt'.",
                metadata={"engine": "rtamt"},
            )

        safety_function = rule.parameters.get("safety_function", {})
        operator = str(
            rule.parameters.get("temporal_operator", rule.parameters.get("operator", "always"))
        ).lower()
        comparator = str(safety_function.get("comparator", "<="))
        threshold = float(safety_function.get("threshold", 0.0))
        variable = str(rule.parameters.get("rtamt_variable", "signal"))

        spec_text = str(rule.parameters.get("rtamt_spec", "")).strip()
        if not spec_text:
            predicate = f"{variable} {comparator} {threshold}"
            if operator == "always":
                spec_text = f"always({predicate})"
            elif operator == "eventually":
                spec_text = f"eventually({predicate})"
            else:
                return ValidatorVerdict(
                    rule_id=rule.rule_id,
                    passed=False,
                    confidence=0.0,
                    rationale=f"Unsupported STL temporal operator for RTAMT translation: {operator}",
                    metadata={"engine": "rtamt"},
                )

        try:
            specification = rtamt.StlDiscreteTimeSpecification()
            specification.name = rule.rule_id
            specification.declare_var(variable, "float")
            specification.spec = spec_text
            specification.parse()
            # RTAMT discrete-time offline format: {"time": [...], var_name: [...]}
            dataset = {"time": list(range(len(signal))), variable: signal}
            robustness = specification.evaluate(dataset)
        except Exception as error:
            return ValidatorVerdict(
                rule_id=rule.rule_id,
                passed=False,
                confidence=0.0,
                rationale=f"RTAMT evaluation failed: {error}",
                metadata={"engine": "rtamt", "spec": spec_text},
            )

        robustness_value = self._extract_rtamt_robustness(robustness)
        passed = robustness_value >= 0.0
        rationale = (
            f"RTAMT robustness {robustness_value:.3f} indicates STL satisfaction."
            if passed
            else f"RTAMT robustness {robustness_value:.3f} indicates STL violation."
        )
        return ValidatorVerdict(
            rule_id=rule.rule_id,
            passed=passed,
            confidence=1.0,
            rationale=rationale,
            metadata={
                "engine": "rtamt",
                "robustness": robustness_value,
                "spec": spec_text,
                "samples": len(signal),
            },
        )

    def _extract_rtamt_robustness(self, robustness: Any) -> float:
        if isinstance(robustness, (int, float)):
            return float(robustness)
        if isinstance(robustness, list) and robustness:
            last_item = robustness[-1]
            if isinstance(last_item, (list, tuple)) and len(last_item) >= 2:
                return float(last_item[1])
            if isinstance(last_item, (int, float)):
                return float(last_item)
        raise ValueError(f"Unsupported RTAMT robustness output: {robustness}")


def run_formal_monitors(trace: ExecutionTrace, rules: list[FormalRule]) -> list[ValidatorVerdict]:
    verdicts: list[ValidatorVerdict] = []
    for rule in rules:
        if rule.rule_type == "speed_limit":
            verdicts.append(SpeedLimitMonitor().evaluate(trace, rule))
            continue
        if rule.rule_type == "stl":
            verdicts.append(STLMonitor().evaluate(trace, rule))
            continue

        verdicts.append(
            ValidatorVerdict(
                rule_id=rule.rule_id,
                passed=False,
                confidence=0.0,
                rationale=f"Unsupported formal monitor type: {rule.rule_type}",
            )
        )
    return verdicts
