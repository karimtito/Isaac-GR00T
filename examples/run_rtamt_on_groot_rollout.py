from __future__ import annotations

from monitors import create_monitor
from monitors.report import format_spec_report
from monitors.specs import (
    EVENTUALLY_OBJECT_LIFTED,
    NO_COLLISION,
    SAFE_DISTANCE,
    SPEED_LIMIT_NEAR_HUMAN,
)
from monitors.trace import canonicalize_trace


def build_mock_trace() -> dict[str, list[tuple[float, float]]]:
    # dist_to_human intentionally drops below 0.4 to demonstrate a violation report.
    return {
        "ee_x": [(0.0, 0.12), (0.1, 0.14), (0.2, 0.16), (0.3, 0.18)],
        "ee_y": [(0.0, 0.55), (0.1, 0.57), (0.2, 0.58), (0.3, 0.59)],
        "ee_z": [(0.0, 0.91), (0.1, 0.89), (0.2, 0.90), (0.3, 0.92)],
        "ee_speed": [(0.0, 0.03), (0.1, 0.09), (0.2, 0.11), (0.3, 0.10)],
        "dist_to_human": [(0.0, 1.25), (0.1, 0.78), (0.2, 0.35), (0.3, 0.28)],
        "gripper_closed": [(0.0, 0.0), (0.1, 1.0), (0.2, 1.0), (0.3, 1.0)],
        "object_lifted": [(0.0, 0.0), (0.1, 0.0), (0.2, 1.0), (0.3, 1.0)],
        "collision": [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0)],
    }


def main() -> None:
    trace = canonicalize_trace(build_mock_trace())
    specs = [NO_COLLISION, SAFE_DISTANCE, SPEED_LIMIT_NEAR_HUMAN, EVENTUALLY_OBJECT_LIFTED]

    for spec in specs:
        monitor = create_monitor(spec["formula"], variables=spec["variables"], backend="rtamt")
        result = monitor.evaluate(trace)
        print(format_spec_report(spec["name"], result, trace))


if __name__ == "__main__":
    main()
