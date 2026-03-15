from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from .config import GenerationConfig, InferenceConfig
from .generation import build_generation_pipeline, generate_first_assistant
from .schemas import Message


def _assistant_display_text(message: Message) -> str:
    if message.content is not None:
        if isinstance(message.content, str):
            return message.content.strip()
        return json.dumps(message.content, ensure_ascii=False, indent=2)
    if message.tool_calls:
        payload = [
            tool_call.model_dump(mode="json", exclude_none=True)
            for tool_call in message.tool_calls
        ]
        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    return ""


def run_prompt_inference(
    *,
    model_path: str,
    prompt: str,
    adapter_path: str | None = None,
    base_model_name: str | None = None,
    device_map: str = "auto",
    dtype: str | None = None,
    merge_adapter: bool = False,
    merge_dtype: str | None = None,
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
        merge_adapter=merge_adapter,
        merge_dtype=merge_dtype,
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
    model_info = getattr(pipe, "_galsenai_model_info", {})
    return {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "resolved_model": model_info,
        "prompt": prompt,
        "merge_adapter": merge_adapter,
        "merge_dtype": merge_dtype if merge_adapter else None,
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


def run_chat_session(
    *,
    model_path: str,
    adapter_path: str | None = None,
    base_model_name: str | None = None,
    device_map: str = "auto",
    dtype: str | None = None,
    merge_adapter: bool = False,
    merge_dtype: str | None = None,
    system_prompt: str | None = None,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_turns: int = 0,
) -> dict[str, Any]:
    console = Console()
    pipe = build_generation_pipeline(
        model_path=model_path,
        adapter_path=None if adapter_path is None else Path(adapter_path),
        base_model_name=base_model_name,
        device_map=device_map,
        dtype=dtype,
        merge_adapter=merge_adapter,
        merge_dtype=merge_dtype,
    )

    history: list[Message] = []
    if system_prompt:
        history.append(Message(role="system", content=system_prompt))

    console.print("Interactive Wolof chat started. Use `/clear` to reset and `/exit` to quit.")
    turns = 0
    stop_reason = "user_exit"
    while True:
        user_text = console.input("[bold cyan]You[/bold cyan]> ").strip()
        if not user_text:
            continue

        lowered = user_text.lower()
        if lowered in {"/exit", "/quit", "exit", "quit"}:
            break
        if lowered == "/clear":
            history = [history[0]] if history and history[0].role == "system" else []
            console.print("Conversation history cleared.")
            continue

        history.append(Message(role="user", content=user_text))
        assistant = generate_first_assistant(
            pipe=pipe,
            messages=history,
            generation=GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            ),
            tool_names=[],
            tool_registry="none",
        )
        history.append(assistant)
        console.print(f"[bold green]Assistant[/bold green]> {_assistant_display_text(assistant)}")

        turns += 1
        if max_turns > 0 and turns >= max_turns:
            stop_reason = "max_turns"
            break

    return {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "resolved_model": getattr(pipe, "_galsenai_model_info", {}),
        "merge_adapter": merge_adapter,
        "merge_dtype": merge_dtype if merge_adapter else None,
        "turns": turns,
        "stop_reason": stop_reason,
    }
