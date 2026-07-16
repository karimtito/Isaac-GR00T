from __future__ import annotations

import json
from pathlib import Path

from .formal_monitor import run_formal_monitors
from .goal_blind import sanitize_trace_for_validation
from .rulebook import load_rulebook
from .types import Action, ExecutionTrace, Observation, Rulebook, ValidationResult
from .vlm.base import VLMJudge
from .vlm.mock import MockVLMJudge


def load_trace(path: str | Path) -> ExecutionTrace:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    observations = [Observation(**observation) for observation in payload.get("observations", [])]
    actions = [Action(**action) for action in payload.get("actions", [])]
    metadata = payload.get("metadata", {})

    return ExecutionTrace(
        trace_id=payload.get("trace_id", "unknown-trace"),
        observations=observations,
        actions=actions,
        metadata=metadata,
    )


def summarize_trace_for_vlm(trace: ExecutionTrace) -> str:
    """Build compact textual evidence summary for VLM judges."""

    obs_count = len(trace.observations)
    action_count = len(trace.actions)
    metadata = trace.metadata
    return (
        f"trace_id={trace.trace_id}; observations={obs_count}; actions={action_count}; "
        f"metadata_keys={sorted(metadata.keys())}"
    )


def run_validation(
    trace: ExecutionTrace,
    rulebook: Rulebook,
    informal_validator: VLMJudge | None = None,
) -> ValidationResult:
    goal_blind_trace = sanitize_trace_for_validation(trace)

    formal_verdicts = run_formal_monitors(goal_blind_trace, rulebook.formal_rules)
    validator = informal_validator or MockVLMJudge()
    trace_summary = summarize_trace_for_vlm(goal_blind_trace)

    informal_verdicts = [
        validator.judge(
            rule_id=rule.rule_id,
            rule_text=rule.text,
            trace_summary=trace_summary,
        )
        for rule in rulebook.informal_rules
    ]

    overall_passed = all(verdict.passed for verdict in [*formal_verdicts, *informal_verdicts])
    return ValidationResult(
        formal_verdicts=formal_verdicts,
        informal_verdicts=informal_verdicts,
        overall_passed=overall_passed,
        details={"goal_blind": True},
    )


def run_validation_from_paths(
    trace_path: str | Path,
    rulebook_path: str | Path,
    informal_validator: VLMJudge | None = None,
) -> ValidationResult:
    trace = load_trace(trace_path)
    rulebook = load_rulebook(rulebook_path)
    return run_validation(trace, rulebook, informal_validator=informal_validator)
