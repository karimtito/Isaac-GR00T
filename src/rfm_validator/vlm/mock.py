from __future__ import annotations

from pathlib import Path

from rfm_validator.types import ValidatorVerdict
from rfm_validator.vlm.base import VLMJudge


class MockVLMJudge(VLMJudge):
    """Deterministic mock VLM judge for tests and CPU-only environments."""

    def is_available(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def judge(
        self,
        *,
        rule_id: str,
        rule_text: str,
        trace_summary: str,
        image_path: str | Path | None = None,
        frame_path: str | Path | None = None,
    ) -> ValidatorVerdict:
        del trace_summary, image_path, frame_path

        text = rule_text.lower()
        flagged_terms = ("unsafe", "collision", "abrupt")
        passed = not any(term in text for term in flagged_terms)

        rationale = (
            "Mock verdict: no high-risk terms detected in informal rule text."
            if passed
            else "Mock verdict: rule includes high-risk terms requiring stricter behavior checks."
        )
        confidence = 0.8 if passed else 0.6
        verdict = "PASS" if passed else "FAIL"
        return ValidatorVerdict(
            rule_id=rule_id,
            passed=passed,
            confidence=confidence,
            rationale=rationale,
            metadata={"verdict": verdict, "evidence": []},
        )
