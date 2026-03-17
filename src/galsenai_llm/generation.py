from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import logging as transformers_logging

from .config import GenerationConfig
from .schemas import Message, ToolCall, ToolFunctionCall
from .tool_registry import get_tool_functions

_KNOWN_TOKENIZER_MAX_LENGTH_CEILING = 1_000_000_000


def resolve_torch_dtype(dtype_name: str | None) -> Any:
    if dtype_name in {None, "", "auto", "none"}:
        return None
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def resolve_context_window(tokenizer: Any, context_window: int | None) -> int | None:
    if context_window is None:
        return None
    if context_window <= 0:
        raise ValueError("Context window must be a positive integer.")

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if (
        isinstance(tokenizer_limit, int)
        and 0 < tokenizer_limit < _KNOWN_TOKENIZER_MAX_LENGTH_CEILING
    ):
        return min(context_window, tokenizer_limit)
    return context_window


def _detect_peft_base_model(reference: str | Path) -> str | None:
    try:
        from peft import PeftConfig
    except ImportError:
        return None

    try:
        config = PeftConfig.from_pretrained(str(reference))
    except Exception:
        return None

    base_model_name = getattr(config, "base_model_name_or_path", None)
    if isinstance(base_model_name, str) and base_model_name.strip():
        return base_model_name.strip()
    return None


def _requires_trust_remote_code(error: Exception) -> bool:
    message = str(error)
    markers = (
        "trust_remote_code=True",
        "requires you to execute the configuration file",
        "requires you to execute the modeling file",
        "Please pass the argument `trust_remote_code=True`",
    )
    return any(marker in message for marker in markers)


def _load_tokenizer(reference: str) -> Any:
    try:
        return AutoTokenizer.from_pretrained(reference, trust_remote_code=False)
    except ValueError as error:
        if not _requires_trust_remote_code(error):
            raise
    return AutoTokenizer.from_pretrained(reference, trust_remote_code=True)


def _load_causal_lm(
    reference: str,
    *,
    device_map: str,
    dtype: Any,
) -> Any:
    common_kwargs = {
        "device_map": device_map,
        "dtype": dtype,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(
            reference,
            trust_remote_code=False,
            **common_kwargs,
        )
    except ValueError as error:
        if not _requires_trust_remote_code(error):
            raise
    return AutoModelForCausalLM.from_pretrained(
        reference,
        trust_remote_code=True,
        **common_kwargs,
    )


@contextmanager
def _quiet_transformers_output() -> Any:
    verbosity = transformers_logging.get_verbosity()
    progress_bar_enabled = transformers_logging.is_progress_bar_enabled()
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Passing `generation_config` together with generation-related arguments=.*",
        )
        yield
    transformers_logging.set_verbosity(verbosity)
    if progress_bar_enabled:
        transformers_logging.enable_progress_bar()


@contextmanager
def _temporary_tokenizer_truncation_side(tokenizer: Any, truncation_side: str) -> Any:
    if tokenizer is None or not hasattr(tokenizer, "truncation_side"):
        yield
        return

    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = truncation_side
    try:
        yield
    finally:
        tokenizer.truncation_side = original_truncation_side


def _build_generation_config(model: Any, generation: GenerationConfig) -> Any:
    base_generation_config = getattr(model, "generation_config", None)
    if base_generation_config is None:
        return None

    request_config = copy.deepcopy(base_generation_config)
    request_config.max_new_tokens = generation.max_new_tokens
    request_config.max_length = None
    request_config.do_sample = generation.do_sample

    if generation.do_sample:
        request_config.temperature = generation.temperature
        request_config.top_p = generation.top_p
    else:
        for attribute in ("temperature", "top_p", "top_k", "typical_p"):
            if hasattr(request_config, attribute):
                setattr(request_config, attribute, None)

    return request_config


def build_generation_pipeline(
    *,
    model_path: str,
    adapter_path: Path | None,
    base_model_name: str | None,
    device_map: str,
    dtype: str | None,
    merge_adapter: bool = False,
    merge_dtype: str | None = None,
) -> Any:
    model_dtype = resolve_torch_dtype(dtype)
    merge_model_dtype = resolve_torch_dtype(merge_dtype) if merge_dtype else model_dtype
    tokenizer_ref = model_path
    resolved_base_model_name = base_model_name
    adapter_ref: str | None = None
    is_adapter_model = False
    merged_adapter = False

    with _quiet_transformers_output():
        if adapter_path is None:
            inferred_base_model_name = _detect_peft_base_model(model_path)
            if inferred_base_model_name is None:
                model = _load_causal_lm(
                    model_path,
                    device_map=device_map,
                    dtype=model_dtype,
                )
            else:
                from peft import PeftModel

                is_adapter_model = True
                adapter_ref = model_path
                tokenizer_ref = model_path
                resolved_base_model_name = (
                    resolved_base_model_name or inferred_base_model_name
                )
                if not resolved_base_model_name:
                    raise ValueError(
                        "Could not determine the base model for the adapter repository."
                    )
                tokenizer = _load_tokenizer(tokenizer_ref)
                base_model = _load_causal_lm(
                    resolved_base_model_name,
                    device_map=device_map,
                    dtype=merge_model_dtype if merge_adapter else model_dtype,
                )
                if len(tokenizer) != base_model.get_input_embeddings().weight.shape[0]:
                    base_model.resize_token_embeddings(len(tokenizer))
                model = PeftModel.from_pretrained(base_model, adapter_ref)
                if merge_adapter:
                    model = model.merge_and_unload()
                    merged_adapter = True
        else:
            from peft import PeftModel

            is_adapter_model = True
            adapter_ref = str(adapter_path)
            tokenizer_ref = adapter_ref
            inferred_base_model_name = _detect_peft_base_model(adapter_ref)
            resolved_base_model_name = resolved_base_model_name or inferred_base_model_name
            if not resolved_base_model_name:
                raise ValueError(
                    "Adapter loading requires --base-model-name or a valid adapter_config.json."
                )
            tokenizer = _load_tokenizer(tokenizer_ref)
            base_model = _load_causal_lm(
                resolved_base_model_name,
                device_map=device_map,
                dtype=merge_model_dtype if merge_adapter else model_dtype,
            )
            if len(tokenizer) != base_model.get_input_embeddings().weight.shape[0]:
                base_model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(base_model, adapter_ref)
            if merge_adapter:
                model = model.merge_and_unload()
                merged_adapter = True

        tokenizer = _load_tokenizer(tokenizer_ref)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    pipe._galsenai_model_info = {
        "model_path": model_path,
        "adapter_path": adapter_ref,
        "base_model_name": resolved_base_model_name,
        "is_adapter_model": is_adapter_model,
        "merged_adapter": merged_adapter,
        "merge_dtype": merge_dtype if merged_adapter else None,
        "tokenizer_ref": tokenizer_ref,
    }
    return pipe


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
    context_window: int | None = None,
) -> Message:
    payload = [message_to_chat_dict(message) for message in messages]
    kwargs: dict[str, Any] = {}
    generation_config = _build_generation_config(getattr(pipe, "model", None), generation)
    if generation_config is not None:
        kwargs["generation_config"] = generation_config
    else:
        kwargs["max_new_tokens"] = generation.max_new_tokens
        kwargs["do_sample"] = generation.do_sample
        if generation.do_sample:
            kwargs["temperature"] = generation.temperature
            kwargs["top_p"] = generation.top_p

    if tool_registry == "sample" and tool_names:
        kwargs["tools"] = get_tool_functions(tool_names)

    effective_context_window = resolve_context_window(
        getattr(pipe, "tokenizer", None),
        context_window,
    )
    if effective_context_window is not None:
        kwargs["truncation"] = True
        kwargs["tokenizer_encode_kwargs"] = {"max_length": effective_context_window}

    truncation_context = (
        _temporary_tokenizer_truncation_side(getattr(pipe, "tokenizer", None), "left")
        if effective_context_window is not None
        else nullcontext()
    )

    with truncation_context, _quiet_transformers_output():
        response = pipe(payload, **kwargs)
    generated_output = response[0]["generated_text"] if isinstance(response, list) else response
    return extract_assistant_message(generated_output)
