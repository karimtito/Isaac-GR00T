from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from rfm_validator.types import ValidatorVerdict


class VLMJudge(ABC):
    """Interface for goal-blind VLM judging of informal behavioral rules."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether this judge can run in the current environment."""

    @abstractmethod
    def load(self) -> None:
        """Load model resources if needed."""

    @abstractmethod
    def judge(
        self,
        *,
        rule_id: str,
        rule_text: str,
        trace_summary: str,
        image_path: str | Path | None = None,
        frame_path: str | Path | None = None,
    ) -> ValidatorVerdict:
        """Return a structured verdict for one rule."""
