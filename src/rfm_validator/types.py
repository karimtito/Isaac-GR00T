from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Observation:
    timestamp: float
    data: dict[str, Any]


@dataclass
class Action:
    timestamp: float
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionTrace:
    trace_id: str
    observations: list[Observation] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FormalRule:
    rule_id: str
    description: str
    rule_type: str = "speed_limit"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class InformalRule:
    rule_id: str
    text: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class Rulebook:
    formal_rules: list[FormalRule] = field(default_factory=list)
    informal_rules: list[InformalRule] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidatorVerdict:
    rule_id: str
    passed: bool
    confidence: float
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    formal_verdicts: list[ValidatorVerdict]
    informal_verdicts: list[ValidatorVerdict]
    overall_passed: bool
    details: dict[str, Any] = field(default_factory=dict)
