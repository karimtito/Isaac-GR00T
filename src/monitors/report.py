from __future__ import annotations

from .base import MonitorResult, Trace


def format_spec_report(spec_name: str, result: MonitorResult, trace: Trace) -> str:
    lines = [
        f"Spec: {spec_name}",
        f"Satisfied: {str(result.satisfied).lower()}",
        f"Robustness: {result.robustness}",
    ]

    if spec_name == "safe_distance" and "dist_to_human" in trace and trace["dist_to_human"]:
        minimum_distance = min(value for _, value in trace["dist_to_human"])
        if minimum_distance < 0.4:
            lines.append(
                f"Violation: minimum distance was {minimum_distance:.2f}m, threshold was 0.40m"
            )

    return "\n".join(lines)
