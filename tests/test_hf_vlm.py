from pathlib import Path

from rfm_validator.vlm.hf_vlm import HuggingFaceVLMJudge


class _FakeTensor:
    def to(self, _device: str):
        return self


class _FakeProcessor:
    def apply_chat_template(self, *_args, **_kwargs):
        return "formatted-prompt"

    def __call__(self, **_kwargs):
        return {"input_ids": _FakeTensor(), "attention_mask": _FakeTensor()}

    def batch_decode(self, *_args, **_kwargs):
        return [
            '{"verdict":"FAIL","confidence":0.4,"rationale":"Abrupt motion observed.","evidence":["high jerk"]}'
        ]


class _FakeModel:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def to(self, _device: str):
        return self

    def generate(self, **_kwargs):
        return [[1, 2, 3]]


class _FakeAutoProcessor:
    @staticmethod
    def from_pretrained(*_args, **_kwargs):
        return _FakeProcessor()


class _FakeTransformersModule:
    AutoProcessor = _FakeAutoProcessor


def test_hf_vlm_judge_uses_parser_without_model_download(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_text("unused")

    monkeypatch.setattr(
        "rfm_validator.vlm.hf_vlm._require_import",
        lambda module_name, _package_name: (
            _FakeTransformersModule if module_name == "transformers" else object()
        ),
    )
    monkeypatch.setattr(
        "rfm_validator.vlm.hf_vlm.HuggingFaceVLMJudge._resolve_model_class",
        lambda *_args, **_kwargs: _FakeModel,
    )
    monkeypatch.setattr(
        "rfm_validator.vlm.hf_vlm.HuggingFaceVLMJudge._load_image",
        lambda *_args, **_kwargs: object(),
    )

    judge = HuggingFaceVLMJudge(model_id="fake/model", device="cpu")
    verdict = judge.judge(
        rule_id="rule-1",
        rule_text="Avoid abrupt movements.",
        trace_summary="Velocity spike near bystander.",
        image_path=image_path,
    )

    assert verdict.rule_id == "rule-1"
    assert verdict.passed is False
    assert verdict.metadata["verdict"] == "FAIL"
