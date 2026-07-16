from __future__ import annotations

from groot_interface.policy_wrapper import LoggingPolicyWrapper
from groot_interface.rollout_logger import RolloutTraceLogger
import numpy as np
import pytest


DT = 0.05


class _FakePolicy:
    def __init__(self, chunk) -> None:
        self._chunk = chunk
        self.calls = 0
        self.was_reset = False
        self.model_name = "fake-groot"

    def get_action(self, observations):
        self.calls += 1
        return self._chunk, {"latency_ms": 1.0}

    def reset(self) -> None:
        self.was_reset = True


def _make_wrapper(horizon: int = 4):
    # action chunk shaped (n_envs=1, horizon), values 10, 11, 12, ...
    chunk = {"x": np.arange(10.0, 10.0 + horizon)[None, :]}
    policy = _FakePolicy(chunk)
    logger = RolloutTraceLogger(signal_extractors={"cmd_x": lambda sim_state, act: float(act["x"])})
    return LoggingPolicyWrapper(policy, logger, dt=DT), policy


def test_get_action_returns_inner_result_untouched() -> None:
    wrapper, policy = _make_wrapper()
    result = wrapper.get_action({"video": "frame"})
    actions, info = result  # unpacks exactly like the eval loop's line 334
    assert actions is policy._chunk
    assert info == {"latency_ms": 1.0}
    assert policy.calls == 1


def test_attribute_forwarding() -> None:
    wrapper, policy = _make_wrapper()
    assert wrapper.model_name == "fake-groot"
    wrapper.reset()
    assert policy.was_reset


def test_time_axis_is_continuous_across_chunks() -> None:
    wrapper, _ = _make_wrapper(horizon=4)
    wrapper.start_episode()
    wrapper.get_action({})
    wrapper.flush_steps(4)
    wrapper.get_action({})
    wrapper.flush_steps(4)
    trace = wrapper.finish_episode()
    times = [t for t, _ in trace["cmd_x"]]
    assert times == pytest.approx([j * DT for j in range(8)])


def test_actions_logged_per_timestep() -> None:
    wrapper, _ = _make_wrapper(horizon=4)
    wrapper.start_episode()
    wrapper.get_action({})
    wrapper.flush_steps(4)
    trace = wrapper.finish_episode()
    values = [v for _, v in trace["cmd_x"]]
    assert values == pytest.approx([10.0, 11.0, 12.0, 13.0])


def test_early_termination_mid_chunk() -> None:
    wrapper, _ = _make_wrapper(horizon=4)
    wrapper.start_episode()
    wrapper.get_action({})
    wrapper.flush_steps(2)  # `done` fired mid-chunk
    assert len(wrapper.finish_episode()["cmd_x"]) == 2
    wrapper.get_action({})
    wrapper.flush_steps(1)  # clock must resume, not restart
    times = [t for t, _ in wrapper.finish_episode()["cmd_x"]]
    assert times == pytest.approx([0.0, DT, 2 * DT])


def test_extra_per_step_signals_merged() -> None:
    wrapper, _ = _make_wrapper(horizon=4)
    wrapper.start_episode()
    wrapper.get_action({})
    wrapper.flush_steps(4, extra_per_step_signals={"success": [0.0, 0.0, 0.0, 1.0]})
    trace = wrapper.finish_episode()
    assert [v for _, v in trace["success"]] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert [t for t, _ in trace["success"]] == pytest.approx([j * DT for j in range(4)])


def test_extra_signals_shorter_than_steps_are_guarded() -> None:
    wrapper, _ = _make_wrapper(horizon=4)
    wrapper.start_episode()
    wrapper.get_action({})
    wrapper.flush_steps(4, extra_per_step_signals={"success": [0.0, 1.0]})
    trace = wrapper.finish_episode()
    assert len(trace["success"]) == 2
    assert len(trace["cmd_x"]) == 4


def test_flush_without_staged_chunk_raises() -> None:
    wrapper, _ = _make_wrapper()
    wrapper.start_episode()
    with pytest.raises(RuntimeError):
        wrapper.flush_steps(4)


def test_start_episode_resets_trace_and_clock() -> None:
    wrapper, _ = _make_wrapper(horizon=4)
    wrapper.start_episode()
    wrapper.get_action({})
    wrapper.flush_steps(4)
    wrapper.start_episode()  # new episode: fresh trace, t back to zero
    wrapper.get_action({})
    wrapper.flush_steps(2)
    trace = wrapper.finish_episode()
    assert len(trace["cmd_x"]) == 2
    assert trace["cmd_x"][0][0] == pytest.approx(0.0)
