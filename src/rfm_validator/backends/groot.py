from __future__ import annotations

from pathlib import Path


class GrootBackend:
    """Adapter entry point for NVIDIA GR00T submodule-backed integration."""

    def __init__(self, repository_root: str | Path) -> None:
        self.repository_root = Path(repository_root)
        self.submodule_path = self.repository_root / "external" / "Isaac-GR00T"

    def is_available(self) -> bool:
        return self.submodule_path.exists()

    def require_available(self) -> None:
        if self.is_available():
            return
        raise FileNotFoundError(
            "GR00T submodule not found at external/Isaac-GR00T. "
            "Run: git submodule add https://github.com/Nvidia/Isaac-GR00T external/Isaac-GR00T "
            "and git submodule update --init --recursive"
        )

    def load(self) -> None:
        self.require_available()
        # TODO: load GR00T checkpoints from configured artifacts.
        # TODO: initialize observation preprocessing for multimodal robot inputs.

    def infer_action(self) -> None:
        self.require_available()
        # TODO: run GR00T action inference through the submodule's official API entrypoint.
