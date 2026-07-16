from __future__ import annotations

import argparse
from pathlib import Path

from rfm_validator.pipeline import load_trace
from rfm_validator.rulebook import load_rulebook
from rfm_validator.vlm.hf_vlm import HuggingFaceVLMJudge


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one informal rule check with a Hugging Face VLM judge."
    )
    parser.add_argument(
        "--model-id", required=True, help="Hugging Face model id (e.g., google/gemma-3-4b-it)"
    )
    parser.add_argument("--image-path", required=True, help="Path to a representative frame image")
    parser.add_argument(
        "--rulebook-path",
        default=str(Path(__file__).resolve().parent / "toy_rulebook.yaml"),
        help="Path to a YAML/JSON rulebook",
    )
    parser.add_argument(
        "--trace-path",
        default=str(Path(__file__).resolve().parent / "toy_trace.json"),
        help="Path to a JSON execution trace",
    )
    args = parser.parse_args()

    rulebook = load_rulebook(args.rulebook_path)
    trace = load_trace(args.trace_path)

    if not rulebook.informal_rules:
        raise ValueError("Rulebook has no informal rules.")

    first_rule = rulebook.informal_rules[0]
    trace_summary = f"trace_id={trace.trace_id}; observations={len(trace.observations)}; actions={len(trace.actions)}"

    judge = HuggingFaceVLMJudge(model_id=args.model_id)
    verdict = judge.judge(
        rule_id=first_rule.rule_id,
        rule_text=first_rule.text,
        trace_summary=trace_summary,
        image_path=args.image_path,
    )

    print(verdict)


if __name__ == "__main__":
    main()
