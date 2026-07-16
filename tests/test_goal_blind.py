from rfm_validator.goal_blind import sanitize_trace_for_validation
from rfm_validator.pipeline import load_trace, summarize_trace_for_vlm


def test_goal_task_fields_removed_from_trace() -> None:
    trace = load_trace("examples/toy_trace.json")
    sanitized = sanitize_trace_for_validation(trace)

    assert "goal" not in sanitized.metadata
    assert "task" not in sanitized.metadata
    assert "success" not in sanitized.metadata
    assert "reward" not in sanitized.observations[0].data
    assert "task_description" not in sanitized.actions[0].parameters


def test_summary_does_not_leak_goal_or_success_values() -> None:
    trace = load_trace("examples/toy_trace.json")
    sanitized = sanitize_trace_for_validation(trace)
    summary = summarize_trace_for_vlm(sanitized)

    assert "pick up red cube" not in summary
    assert "place cube in bin" not in summary
    assert "True" not in summary
