from __future__ import annotations

from typing import Any

from monitors.base import Trace
import numpy as np

from groot_interface.rollout_logger import RolloutTraceLogger


class LoggingPolicyWrapper:
    """Transparent policy wrapper that records (observation, action) pairs.

    Sits between a rollout loop and the GR00T policy. Each ``get_action``
    call stages the observation it saw and the action chunk it returned.
    The rollout runner then reports how many inner env-steps actually
    executed (``info["n_env_steps"]``) via :meth:`flush_steps`, and the
    wrapper logs exactly that many per-timestep entries into the
    :class:`RolloutTraceLogger`.

    Notes:
        * Observations are only available at chunk boundaries (the
          MultiStepWrapper consumes intermediate frames), so ``sim_state``
          is held constant across the inner steps of one chunk.
        * Actions are logged at full per-timestep resolution from the chunk.
        * Batched interfaces are assumed; ``env_index`` selects which env
          to log (v1 targets single-env rollouts).
    """

    def __init__(
        self,
        policy: Any,
        logger: RolloutTraceLogger,
        dt: float,
        env_index: int = 0,
    ) -> None:
        self._policy = policy
        self._logger = logger
        self._dt = float(dt)
        self._env_index = int(env_index)
        self._global_step = 0
        self._pending_obs: Any = None
        self._pending_chunk: Any = None

    # ------- policy interface (what the rollout loop sees) -------

    def get_action(self, observations: Any) -> Any:
        result = self._policy.get_action(observations)
        actions = result[0] if isinstance(result, tuple) else result
        self._pending_obs = observations
        self._pending_chunk = actions
        return result

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    # ------- logging interface (what the rollout runner drives) -------

    def start_episode(self) -> None:
        self._logger.reset()
        self._global_step = 0
        self._pending_obs = None
        self._pending_chunk = None

    def flush_steps(
        self,
        n_steps: int,
        extra_per_step_signals: dict[str, list[float]] | None = None,
    ) -> None:
        """Log the first ``n_steps`` timesteps of the staged action chunk.

        ``n_steps`` may be smaller than the chunk length if the episode
        terminated mid-chunk. ``extra_per_step_signals`` maps a signal
        name to a list of per-inner-step values harvested from env info.
        """
        if self._pending_chunk is None:
            raise RuntimeError("flush_steps() called with no staged action chunk")
        for j in range(int(n_steps)):
            t = (self._global_step + j) * self._dt
            self._logger.log_step(
                t,
                sim_state=self._pending_obs,
                groot_action=self._slice_action(self._pending_chunk, j),
            )
            if extra_per_step_signals:
                self._logger.merge_signals(
                    t,
                    {
                        name: float(values[j])
                        for name, values in extra_per_step_signals.items()
                        if j < len(values)
                    },
                )
        self._global_step += int(n_steps)
        self._pending_obs = None
        self._pending_chunk = None

    def finish_episode(self) -> Trace:
        return self._logger.get_trace()

    # ------- helpers -------

    def _slice_action(self, chunk: Any, j: int) -> Any:
        if isinstance(chunk, dict):
            return {k: np.asarray(v)[self._env_index, j] for k, v in chunk.items()}
        return np.asarray(chunk)[self._env_index, j]
