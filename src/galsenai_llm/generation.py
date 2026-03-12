from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .config import GenerationConfig
from .schemas import Message, ToolCall, ToolFunctionCall
from .tool_registry import get_tool_functions


def resolve_torch_dtype(dtype_name: str | None) -> Any:
    if dtype_name in {None, "", "auto", "none"}:
        return None
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def build_generation_pipeline(
    *,
    model_path: str,
    adapter_path: Path | None,
    base_model_name: str | None,
    device_map: str,
    dtype: str | None,
) -> Any:
    model_dtype = resolve_torch_dtype(dtype)
    tokenizer_ref = model_path

    if adapter_path is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            dtype=model_dtype,
            trust_remote_code=True,
        )
    else:
        from peft import PeftModel

        base_ref = base_model_name or model_path
        tokenizer_ref = str(adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_ref,
            device_map=device_map,
            dtype=model_dtype,
            trust_remote_code=True,
        )
        if len(tokenizer) != base_model.get_input_embeddings().weight.shape[0]:
            base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, str(adapter_path))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def message_to_chat_dict(message: Message) -> dict[str, Any]:
    return message.model_dump(mode="json", exclude_none=True)


def _tool_calls_from_json_array(payload: list[dict[str, Any]]) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    for index, tool_payload in enumerate(payload, start=1):
        tool_calls.append(
            ToolCall(
                id=f"call_{index}",
                function=ToolFunctionCall(
                    name=tool_payload["name"],
                    arguments=tool_payload.get("arguments", {}),
                ),
            )
        )
    return tool_calls


def extract_assistant_message(generated_output: Any) -> Message:
    if isinstance(generated_output, list) and generated_output:
        last_item = generated_output[-1]
        if isinstance(last_item, dict):
            payload = dict(last_item)
            payload.setdefault("role", "assistant")
            return Message.model_validate(payload)

    if isinstance(generated_output, dict):
        payload = dict(generated_output)
        payload.setdefault("role", "assistant")
        return Message.model_validate(payload)

    if isinstance(generated_output, str):
        stripped = generated_output.strip()
        if not stripped:
            return Message(role="assistant", content="")

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return Message(role="assistant", content=stripped)

        if isinstance(parsed, dict):
            if parsed.get("role") == "assistant" or "tool_calls" in parsed:
                parsed.setdefault("role", "assistant")
                return Message.model_validate(parsed)
            return Message(role="assistant", content=parsed)

        if (
            isinstance(parsed, list)
            and parsed
            and all(isinstance(item, dict) and "name" in item for item in parsed)
        ):
            return Message(role="assistant", tool_calls=_tool_calls_from_json_array(parsed))

        return Message(role="assistant", content=parsed)

    return Message(role="assistant", content=str(generated_output))


def generate_first_assistant(
    *,
    pipe: Any,
    messages: Sequence[Message],
    generation: GenerationConfig,
    tool_names: Sequence[str] | None = None,
    tool_registry: str = "none",
) -> Message:
    payload = [message_to_chat_dict(message) for message in messages]
    kwargs: dict[str, Any] = {
        "max_new_tokens": generation.max_new_tokens,
        "do_sample": generation.do_sample,
    }
    if generation.do_sample:
        kwargs["temperature"] = generation.temperature
        kwargs["top_p"] = generation.top_p

    if tool_registry == "sample" and tool_names:
        kwargs["tools"] = get_tool_functions(tool_names)

    response = pipe(payload, **kwargs)
    generated_output = response[0]["generated_text"] if isinstance(response, list) else response
    return extract_assistant_message(generated_output)
