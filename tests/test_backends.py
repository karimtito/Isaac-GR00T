from pathlib import Path

import pytest
from rfm_validator.backends.groot import GrootBackend


def test_groot_backend_detects_missing_submodule(tmp_path: Path) -> None:
    backend = GrootBackend(repository_root=tmp_path)

    assert backend.is_available() is False
    with pytest.raises(FileNotFoundError):
        backend.require_available()


def test_groot_backend_detects_present_submodule(tmp_path: Path) -> None:
    (tmp_path / "external" / "Isaac-GR00T").mkdir(parents=True)
    backend = GrootBackend(repository_root=tmp_path)

    assert backend.is_available() is True
