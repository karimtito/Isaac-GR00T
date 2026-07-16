from pathlib import Path

from rfm_validator.pipeline import run_validation_from_paths


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = run_validation_from_paths(
        trace_path=root / "toy_trace.json",
        rulebook_path=root / "toy_rulebook.yaml",
    )

    print(f"Overall passed: {result.overall_passed}")
    for verdict in result.formal_verdicts + result.informal_verdicts:
        print(
            f"- {verdict.rule_id}: passed={verdict.passed}, confidence={verdict.confidence}, rationale={verdict.rationale}"
        )
