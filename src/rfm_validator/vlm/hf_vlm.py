from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from rfm_validator.types import ValidatorVerdict
from rfm_validator.vlm.base import VLMJudge
from rfm_validator.vlm.prompting import (
    build_goal_blind_validation_prompt,
    parse_vlm_output_to_verdict,
)


class HuggingFaceVLMJudge(VLMJudge):
    """Generic Hugging Face VLM judge for text+image behavioral validation."""

    def __init__(
        self,
        *,
        model_id: str,
        device: str | None = None,
        torch_dtype: str | Any | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        torch_module = _optional_import("torch")
        self.model_id = model_id
        self.device = device or (
            "cuda" if torch_module and torch_module.cuda.is_available() else "cpu"
        )
        self.torch_dtype = self._resolve_dtype(torch_dtype)
        self.max_new_tokens = max_new_tokens
        self._processor: Any | None = None
        self._model: Any | None = None

    def is_available(self) -> bool:
        return _optional_import("transformers") is not None

    def load(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        transformers_module = _require_import("transformers", "transformers")
        auto_processor = getattr(transformers_module, "AutoProcessor")
        model_class = self._resolve_model_class()
        self._processor = auto_processor.from_pretrained(self.model_id)
        model_kwargs: dict[str, Any] = {}
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype

        self._model = model_class.from_pretrained(self.model_id, **model_kwargs)
        if hasattr(self._model, "to"):
            self._model = self._model.to(self.device)

    def judge(
        self,
        *,
        rule_id: str,
        rule_text: str,
        trace_summary: str,
        image_path: str | Path | None = None,
        frame_path: str | Path | None = None,
    ) -> ValidatorVerdict:
        del frame_path  # TODO: add video/frame sequence handling.

        if self._processor is None or self._model is None:
            self.load()

        assert self._processor is not None
        assert self._model is not None

        prompt = build_goal_blind_validation_prompt(
            rule_text=rule_text, trace_summary=trace_summary
        )
        image = self._load_image(image_path) if image_path else None

        inputs = self._prepare_inputs(prompt=prompt, image=image)
        generated_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
        output_text = decoded[0] if decoded else ""

        return parse_vlm_output_to_verdict(output_text, rule_id=rule_id)

    def _prepare_inputs(self, *, prompt: str, image: Any | None) -> dict[str, Any]:
        assert self._processor is not None

        if image is not None and hasattr(self._processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                }
            ]
            prompt_text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = prompt

        if image is not None:
            model_inputs = self._processor(text=prompt_text, images=image, return_tensors="pt")
        else:
            model_inputs = self._processor(text=prompt_text, return_tensors="pt")

        prepared: dict[str, Any] = {}
        for key, value in model_inputs.items():
            prepared[key] = value.to(self.device) if hasattr(value, "to") else value
        return prepared

    @staticmethod
    def _resolve_dtype(torch_dtype: str | Any | None) -> Any | None:
        torch_module = _optional_import("torch")
        if torch_dtype is None:
            return torch_dtype
        if torch_module and isinstance(torch_dtype, torch_module.dtype):
            return torch_dtype
        if not torch_module:
            return None
        if not isinstance(torch_dtype, str):
            return None
        dtype_map: dict[str, Any] = {
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "bfloat16": torch_module.bfloat16,
            "bf16": torch_module.bfloat16,
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
        }
        return dtype_map.get(torch_dtype.lower())

    @staticmethod
    def _resolve_model_class() -> Any:
        # Prefer multimodal auto classes commonly used by Gemma/Qwen-style VLMs.
        transformers_module = _require_import("transformers", "transformers")
        preferred = getattr(transformers_module, "AutoModelForImageTextToText", None)
        fallback = getattr(transformers_module, "AutoModelForVision2Seq", None)
        if preferred is not None:
            return preferred
        if fallback is not None:
            return fallback
        raise ImportError("No multimodal auto model class found in installed transformers version.")

    @staticmethod
    def _load_image(image_path: str | Path) -> Any:
        image_module = _require_import("PIL.Image", "pillow")
        return image_module.open(image_path).convert("RGB")


def _optional_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _require_import(module_name: str, package_name: str) -> Any:
    module = _optional_import(module_name)
    if module is None:
        raise ImportError(
            f"{package_name} is required for HuggingFaceVLMJudge but is not installed."
        )
    return module
