from rfm_validator.pipeline import load_trace
from rfm_validator.vlm.mock import MockVLMJudge


def test_mock_vlm_validator_returns_structured_verdict() -> None:
    trace = load_trace("examples/toy_trace.json")

    verdict = MockVLMJudge().judge(
        rule_id="behavior-unsafe",
        rule_text="Avoid unsafe abrupt movements.",
        trace_summary=f"trace_id={trace.trace_id}",
    )

    assert verdict.rule_id == "behavior-unsafe"
    assert verdict.passed is False
    assert verdict.confidence > 0
    assert verdict.rationale
    assert verdict.metadata["verdict"] == "FAIL"
