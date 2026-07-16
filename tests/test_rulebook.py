from pathlib import Path

from rfm_validator.rulebook import load_rulebook


def test_load_rulebook_from_yaml() -> None:
    rulebook = load_rulebook(Path("examples/toy_rulebook.yaml"))

    assert len(rulebook.formal_rules) == 1
    assert rulebook.formal_rules[0].rule_id == "speed-001"
    assert len(rulebook.informal_rules) == 1
