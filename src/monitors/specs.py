from __future__ import annotations


SAFE_DISTANCE = {
    "name": "safe_distance",
    "formula": "always(dist_to_human >= 0.4)",
    "variables": {"dist_to_human": "float"},
}

SPEED_LIMIT_NEAR_HUMAN = {
    "name": "speed_limit_near_human",
    "formula": "always((dist_to_human < 0.8) -> (ee_speed <= 0.25))",
    "variables": {
        "dist_to_human": "float",
        "ee_speed": "float",
    },
}

EVENTUALLY_OBJECT_LIFTED = {
    "name": "eventually_object_lifted",
    "formula": "eventually[0:10](object_lifted >= 1)",
    "variables": {"object_lifted": "float"},
}

NO_COLLISION = {
    "name": "no_collision",
    "formula": "always(collision <= 0)",
    "variables": {"collision": "float"},
}

SPEC_REGISTRY = {
    spec["name"]: spec
    for spec in [
        SAFE_DISTANCE,
        SPEED_LIMIT_NEAR_HUMAN,
        EVENTUALLY_OBJECT_LIFTED,
        NO_COLLISION,
    ]
}
