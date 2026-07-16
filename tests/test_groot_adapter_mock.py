from groot_interface.action_plan_adapter import action_to_signals
from groot_interface.rollout_logger import RolloutTraceLogger
from groot_interface.signal_extractors import (
    extract_collision_flag,
    extract_distance_to_human,
    extract_end_effector_pose,
    extract_object_lifted,
)


def test_rollout_logger_with_mock_state_and_action() -> None:
    states = [
        {
            "ee": {"x": 0.1, "y": 0.2, "z": 0.3},
            "dist_to_human": 1.1,
            "collision": False,
            "objects": {"cube": {"z": 0.01}},
            "table_height": 0.0,
        },
        {
            "ee": {"x": 0.2, "y": 0.2, "z": 0.4},
            "dist_to_human": 0.9,
            "collision": False,
            "objects": {"cube": {"z": 0.12}},
            "table_height": 0.0,
        },
    ]
    actions = [
        {"vector": [0.1, 0.2, 0.3], "planned_ee_speed": 0.2, "gripper_command": 1.0},
        {"vector": [0.0, 0.1, 0.1], "planned_ee_speed": 0.1, "gripper_command": 0.0},
    ]

    extractors = {
        "ee_x": lambda sim_state, _: extract_end_effector_pose(sim_state)["ee_x"],
        "ee_y": lambda sim_state, _: extract_end_effector_pose(sim_state)["ee_y"],
        "ee_z": lambda sim_state, _: extract_end_effector_pose(sim_state)["ee_z"],
        "dist_to_human": lambda sim_state, _: extract_distance_to_human(sim_state),
        "collision": lambda sim_state, _: extract_collision_flag(sim_state),
        "object_lifted": lambda sim_state, _: extract_object_lifted(sim_state, "cube"),
    }

    logger = RolloutTraceLogger(extractors)
    for i, (state, action) in enumerate(zip(states, actions)):
        t = i * 0.1
        logger.log_step(t=t, sim_state=state, groot_action=action)
        logger.merge_signals(t=t, extra_signals=action_to_signals(action, state))

    trace = logger.get_trace()
    expected_signals = {
        "ee_x",
        "ee_y",
        "ee_z",
        "dist_to_human",
        "collision",
        "object_lifted",
        "action_norm",
        "planned_ee_speed",
        "gripper_command",
        "planned_target_displacement",
        "toward_forbidden_zone",
    }
    assert expected_signals.issubset(set(trace.keys()))
