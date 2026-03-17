from __future__ import annotations

from types import SimpleNamespace

import pytest

from galsenai_llm.config import GenerationConfig
from galsenai_llm.generation import (
    generate_first_assistant,
    resolve_context_window,
)
from galsenai_llm.schemas import Message


class FakePipe:
    def __init__(self) -> None:
        self.model = SimpleNamespace(generation_config=SimpleNamespace())
        self.tokenizer = SimpleNamespace(truncation_side="right", model_max_length=4096)
        self.payload = None
        self.kwargs = None
        self.truncation_side_during_call = None

    def __call__(self, payload, **kwargs):
        self.payload = payload
        self.kwargs = kwargs
        self.truncation_side_during_call = self.tokenizer.truncation_side
        return [{"generated_text": "Mangi fi rek."}]


def test_generate_first_assistant_applies_left_truncation_for_context_window() -> None:
    pipe = FakePipe()

    assistant = generate_first_assistant(
        pipe=pipe,
        messages=[Message(role="user", content="Nanga def?")],
        generation=GenerationConfig(max_new_tokens=32),
        context_window=2048,
    )

    assert assistant.content == "Mangi fi rek."
    assert pipe.kwargs["truncation"] is True
    assert pipe.kwargs["tokenizer_encode_kwargs"] == {"max_length": 2048}
    assert pipe.truncation_side_during_call == "left"
    assert pipe.tokenizer.truncation_side == "right"


def test_generate_first_assistant_skips_truncation_when_context_window_is_unset() -> None:
    pipe = FakePipe()

    generate_first_assistant(
        pipe=pipe,
        messages=[Message(role="user", content="Nanga def?")],
        generation=GenerationConfig(max_new_tokens=32),
    )

    assert "truncation" not in pipe.kwargs
    assert "tokenizer_encode_kwargs" not in pipe.kwargs
    assert pipe.truncation_side_during_call == "right"


def test_resolve_context_window_clamps_to_known_tokenizer_limit() -> None:
    tokenizer = SimpleNamespace(model_max_length=4096)

    assert resolve_context_window(tokenizer, 8192) == 4096


def test_resolve_context_window_rejects_non_positive_values() -> None:
    tokenizer = SimpleNamespace(model_max_length=4096)

    with pytest.raises(ValueError, match="positive integer"):
        resolve_context_window(tokenizer, 0)
