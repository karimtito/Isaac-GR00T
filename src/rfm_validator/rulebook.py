from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .types import FormalRule, InformalRule, Rulebook


def _as_rulebook(payload: dict[str, Any]) -> Rulebook:
    formal_rules = [FormalRule(**rule) for rule in payload.get("formal_rules", [])]
    informal_rules = [InformalRule(**rule) for rule in payload.get("informal_rules", [])]
    metadata = payload.get("metadata", {})
    return Rulebook(formal_rules=formal_rules, informal_rules=informal_rules, metadata=metadata)


def load_rulebook(path: str | Path) -> Rulebook:
    rulebook_path = Path(path)
    suffix = rulebook_path.suffix.lower()

    if suffix == ".json":
        payload = json.loads(rulebook_path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(rulebook_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported rulebook format: {suffix}")

    if payload is None:
        payload = {}

    return _as_rulebook(payload)
