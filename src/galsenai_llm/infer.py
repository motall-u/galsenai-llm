from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import GenerationConfig, InferenceConfig
from .generation import build_generation_pipeline, generate_first_assistant
from .schemas import Message


def run_prompt_inference(
    *,
    model_path: str,
    prompt: str,
    adapter_path: str | None = None,
    base_model_name: str | None = None,
    device_map: str = "auto",
    dtype: str | None = None,
    system_prompt: str | None = None,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Inference requires a non-empty prompt.")

    messages: list[Message] = []
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))
    messages.append(Message(role="user", content=prompt))

    pipe = build_generation_pipeline(
        model_path=model_path,
        adapter_path=None if adapter_path is None else Path(adapter_path),
        base_model_name=base_model_name,
        device_map=device_map,
        dtype=dtype,
    )
    assistant = generate_first_assistant(
        pipe=pipe,
        messages=messages,
        generation=GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        ),
        tool_names=[],
        tool_registry="none",
    )
    return {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "prompt": prompt,
        "assistant": assistant.model_dump(mode="json", exclude_none=True),
    }


def run_inference(config: InferenceConfig, prompt_override: str | None = None) -> dict[str, Any]:
    prompt = prompt_override or config.prompt
    if not prompt:
        raise ValueError("Inference requires a prompt in the config or via --prompt.")

    return run_prompt_inference(
        model_path=config.model_path,
        prompt=prompt,
        adapter_path=None if config.adapter_path is None else str(config.adapter_path),
        base_model_name=config.base_model_name,
        device_map=config.device_map,
        dtype=config.dtype,
        system_prompt=config.system_prompt,
        max_new_tokens=config.generation.max_new_tokens,
        do_sample=config.generation.do_sample,
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
    )
