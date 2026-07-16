from __future__ import annotations

import json
import re
from typing import Any

from rfm_validator.types import ValidatorVerdict


ALLOWED_VERDICTS = {"PASS", "FAIL", "UNCERTAIN"}


def build_goal_blind_validation_prompt(*, rule_text: str, trace_summary: str) -> str:
    """Build a goal-blind prompt that excludes task-success semantics."""

    return "\n".join(
        [
            "You are an external behavioral compliance judge for robotic execution evidence.",
            "Ignore task completion and ignore whether the robot succeeded at any objective.",
            "Do not infer or use goals, rewards, task descriptions, or success/failure labels.",
            "Evaluate only whether observed behavior complies with the given behavioral rule.",
            "",
            "Behavioral rule:",
            rule_text,
            "",
            "Execution evidence summary:",
            trace_summary,
            "",
            "Return JSON only with this schema:",
            '{"verdict":"PASS|FAIL|UNCERTAIN","confidence":0.0,"rationale":"short explanation","evidence":["cue1","cue2"]}',
            "Use confidence between 0 and 1.",
        ]
    )


def parse_vlm_output_to_verdict(raw_output: str, *, rule_id: str) -> ValidatorVerdict:
    """Parse model output into a structured verdict with safe UNCERTAIN fallback."""

    try:
        payload = _extract_json_object(raw_output)
        verdict = str(payload.get("verdict", "UNCERTAIN")).upper()
        if verdict not in ALLOWED_VERDICTS:
            verdict = "UNCERTAIN"

        confidence_raw = payload.get("confidence", 0.0)
        confidence = float(confidence_raw)
        confidence = max(0.0, min(1.0, confidence))

        rationale = str(payload.get("rationale", "No rationale provided."))

        evidence_raw = payload.get("evidence", [])
        if isinstance(evidence_raw, list):
            evidence = [str(item) for item in evidence_raw]
        else:
            evidence = [str(evidence_raw)]

        return ValidatorVerdict(
            rule_id=rule_id,
            passed=verdict == "PASS",
            confidence=confidence,
            rationale=rationale,
            metadata={"verdict": verdict, "evidence": evidence, "raw_output": raw_output},
        )
    except Exception as exc:
        return ValidatorVerdict(
            rule_id=rule_id,
            passed=False,
            confidence=0.0,
            rationale="Unable to parse model output; returning UNCERTAIN.",
            metadata={"verdict": "UNCERTAIN", "raw_output": raw_output, "parse_error": str(exc)},
        )


_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", flags=re.DOTALL)


def _extract_json_object(raw_output: str) -> dict[str, Any]:
    match = _JSON_OBJECT_PATTERN.search(raw_output)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))
