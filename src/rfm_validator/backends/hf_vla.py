from __future__ import annotations


class HuggingFaceVLABackend:
    """Skeleton for future Hugging Face-hosted VLA policy backends."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def load(self) -> None:
        # TODO: load VLA policy model weights and tokenizer/processor.
        # TODO: support Qwen-style VLA model families and variants.
        return None

    def infer_action(self) -> None:
        # TODO: preprocess observations (vision, proprioception, language context).
        # TODO: build embodiment-specific prompting/input formatting.
        # TODO: decode actions from model outputs.
        # TODO: normalize actions to a backend-agnostic representation.
        # TODO: integrate with simulator and/or robot-control runtime.
        return None
