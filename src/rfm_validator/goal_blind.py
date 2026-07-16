from __future__ import annotations

from dataclasses import replace
from typing import Any

from .types import ExecutionTrace


EXCLUDED_KEYS = {
    "goal",
    "goals",
    "task",
    "task_description",
    "reward",
    "success",
    "success_label",
}


def _strip_excluded_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key not in EXCLUDED_KEYS}


def sanitize_trace_for_validation(trace: ExecutionTrace) -> ExecutionTrace:
    """Create a goal-blind view of a trace by removing task/goal/reward/success fields."""

    sanitized_observations = [
        replace(observation, data=_strip_excluded_fields(observation.data))
        for observation in trace.observations
    ]
    sanitized_actions = [
        replace(action, parameters=_strip_excluded_fields(action.parameters))
        for action in trace.actions
    ]

    return replace(
        trace,
        observations=sanitized_observations,
        actions=sanitized_actions,
        metadata=_strip_excluded_fields(trace.metadata),
    )


def build_goal_blind_input(trace: ExecutionTrace) -> dict[str, Any]:
    sanitized_trace = sanitize_trace_for_validation(trace)
    return {
        "trace_id": sanitized_trace.trace_id,
        "metadata": sanitized_trace.metadata,
        "observations": [obs.data for obs in sanitized_trace.observations],
        "actions": [action.parameters for action in sanitized_trace.actions],
    }
