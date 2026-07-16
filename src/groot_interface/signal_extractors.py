from __future__ import annotations

from math import sqrt
from typing import Any


def _read(source: Any, *keys: str, default: Any = None) -> Any:
    current = source
    for key in keys:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
            continue
        if hasattr(current, key):
            current = getattr(current, key)
            continue
        return default
    return current


def extract_end_effector_pose(sim_state: Any) -> dict[str, float]:
    ee = _read(sim_state, "ee", default={})
    x = float(_read(ee, "x", default=0.0))
    y = float(_read(ee, "y", default=0.0))
    z = float(_read(ee, "z", default=0.0))
    return {"ee_x": x, "ee_y": y, "ee_z": z}


def extract_end_effector_speed(prev_state: Any, state: Any, dt: float) -> float:
    if dt <= 0.0:
        return 0.0
    prev_pose = extract_end_effector_pose(prev_state)
    pose = extract_end_effector_pose(state)
    dx = pose["ee_x"] - prev_pose["ee_x"]
    dy = pose["ee_y"] - prev_pose["ee_y"]
    dz = pose["ee_z"] - prev_pose["ee_z"]
    return float(sqrt(dx * dx + dy * dy + dz * dz) / dt)


def extract_distance_to_human(sim_state: Any) -> float:
    return float(_read(sim_state, "dist_to_human", default=0.0))


def extract_collision_flag(sim_state: Any) -> float:
    return 1.0 if bool(_read(sim_state, "collision", default=False)) else 0.0


def extract_object_lifted(sim_state: Any, object_name: str) -> float:
    object_height = _read(sim_state, "objects", object_name, "z", default=None)
    if object_height is None:
        return 0.0
    table_height = float(_read(sim_state, "table_height", default=0.0))
    return 1.0 if float(object_height) > table_height + 0.05 else 0.0
