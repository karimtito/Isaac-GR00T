from __future__ import annotations

from math import sqrt
from typing import Any


def action_to_signals(action: Any, sim_state: Any = None) -> dict[str, float]:
    if action is None:
        return {
            "action_norm": 0.0,
            "planned_ee_speed": 0.0,
            "gripper_command": 0.0,
            "planned_target_displacement": 0.0,
        }

    vector = action.get("vector", []) if isinstance(action, dict) else getattr(action, "vector", [])
    vector_values = [float(v) for v in vector] if vector else []

    action_norm = sqrt(sum(v * v for v in vector_values)) if vector_values else 0.0

    planned_speed = (
        float(action.get("planned_ee_speed", 0.0))
        if isinstance(action, dict)
        else float(getattr(action, "planned_ee_speed", 0.0))
    )
    gripper_command = (
        float(action.get("gripper_command", 0.0))
        if isinstance(action, dict)
        else float(getattr(action, "gripper_command", 0.0))
    )
    target_displacement = (
        float(action.get("planned_target_displacement", 0.0))
        if isinstance(action, dict)
        else float(getattr(action, "planned_target_displacement", 0.0))
    )

    signals = {
        "action_norm": float(action_norm),
        "planned_ee_speed": planned_speed,
        "gripper_command": gripper_command,
        "planned_target_displacement": target_displacement,
    }

    toward_forbidden_zone = (
        action.get("toward_forbidden_zone", False)
        if isinstance(action, dict)
        else getattr(action, "toward_forbidden_zone", False)
    )
    forbidden_zone = 1.0 if bool(toward_forbidden_zone) else 0.0
    signals["toward_forbidden_zone"] = forbidden_zone
    return signals
