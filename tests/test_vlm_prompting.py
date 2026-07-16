from rfm_validator.vlm.prompting import (
    build_goal_blind_validation_prompt,
    parse_vlm_output_to_verdict,
)


def test_prompt_is_goal_blind_and_excludes_task_fields() -> None:
    prompt = build_goal_blind_validation_prompt(
        rule_text="Move smoothly near humans.",
        trace_summary="Observed velocity spikes around people.",
    )

    assert "ignore whether the robot succeeded" in prompt
    assert (
        "Do not infer or use goals, rewards, task descriptions, or success/failure labels."
        in prompt
    )
    assert "verdict" in prompt


def test_parser_extracts_json_verdict() -> None:
    raw = '{"verdict":"PASS","confidence":0.73,"rationale":"Behavior remained smooth.","evidence":["low jerk"]}'
    verdict = parse_vlm_output_to_verdict(raw, rule_id="rule-1")

    assert verdict.passed is True
    assert verdict.confidence == 0.73
    assert verdict.metadata["verdict"] == "PASS"
    assert verdict.metadata["evidence"] == ["low jerk"]


def test_parser_falls_back_to_uncertain_on_bad_output() -> None:
    verdict = parse_vlm_output_to_verdict("not-json", rule_id="rule-2")

    assert verdict.passed is False
    assert verdict.metadata["verdict"] == "UNCERTAIN"
    assert "raw_output" in verdict.metadata
