from __future__ import annotations

from typing import Any

from .config import InferenceConfig
from .generation import build_generation_pipeline, generate_first_assistant
from .schemas import Message


def run_inference(config: InferenceConfig, prompt_override: str | None = None) -> dict[str, Any]:
    prompt = prompt_override or config.prompt
    if not prompt:
        raise ValueError("Inference requires a prompt in the config or via --prompt.")

    messages: list[Message] = []
    if config.system_prompt:
        messages.append(Message(role="system", content=config.system_prompt))
    messages.append(Message(role="user", content=prompt))

    pipe = build_generation_pipeline(
        model_path=config.model_path,
        adapter_path=config.adapter_path,
        base_model_name=config.base_model_name,
        device_map=config.device_map,
        dtype=config.dtype,
    )
    assistant = generate_first_assistant(
        pipe=pipe,
        messages=messages,
        generation=config.generation,
        tool_names=config.tools,
        tool_registry=config.tool_registry,
    )
    return {
        "prompt": prompt,
        "assistant": assistant.model_dump(mode="json", exclude_none=True),
    }
