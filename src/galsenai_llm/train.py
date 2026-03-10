from __future__ import annotations

import json
import logging
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from .config import TrainProjectConfig
from .dataset import load_examples
from .io import write_json
from .schemas import DatasetExample

logger = logging.getLogger(__name__)


def _examples_to_records(examples: list[DatasetExample]) -> list[dict[str, Any]]:
    return [example.model_dump(mode="json", exclude_none=True) for example in examples]


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _fallback_render_example(example: DatasetExample) -> str:
    sections: list[str] = []
    if example.tools:
        tool_payload = [tool.model_dump(mode="json", exclude_none=True) for tool in example.tools]
        sections.append("AVAILABLE_TOOLS")
        sections.append(json.dumps(tool_payload, ensure_ascii=True, indent=2, sort_keys=True))

    for message in example.messages:
        role_header = message.role.upper()
        if message.role == "tool":
            role_header = f"TOOL name={message.name} call_id={message.tool_call_id or ''}".strip()
        sections.append(role_header)

        if message.tool_calls:
            tool_calls = [
                call.model_dump(mode="json", exclude_none=True)
                for call in message.tool_calls
            ]
            sections.append(json.dumps(tool_calls, ensure_ascii=True, indent=2, sort_keys=True))

        if message.content is not None:
            sections.append(_stringify_content(message.content))

    return "\n\n".join(section for section in sections if section)


def _render_example(example: DatasetExample, tokenizer: Any) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            messages_payload = [
                message.model_dump(mode="json", exclude_none=True)
                for message in example.messages
            ]
            tools_payload = [
                tool.model_dump(mode="json", exclude_none=True)
                for tool in example.tools
            ] or None
            return tokenizer.apply_chat_template(
                messages_payload,
                tools=tools_payload,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            logger.warning(
                "apply_chat_template failed for example %s, using fallback renderer",
                example.id,
                exc_info=True,
            )

    return _fallback_render_example(example)


def _rendered_records(examples: list[DatasetExample], tokenizer: Any) -> list[dict[str, str]]:
    return [{"text": _render_example(example, tokenizer)} for example in examples]


def _load_model_and_tokenizer(config: TrainProjectConfig):
    if config.model.backend == "unsloth":
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model.model_name,
            max_seq_length=config.model.max_seq_length,
            dtype=config.model.dtype,
            load_in_4bit=config.model.load_in_4bit,
        )

        if not config.model.full_finetuning:
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.lora.r,
                lora_alpha=config.lora.lora_alpha,
                lora_dropout=config.lora.lora_dropout,
                bias=config.lora.bias,
                target_modules=config.lora.target_modules,
                use_gradient_checkpointing=config.training.gradient_checkpointing,
                random_state=config.training.seed,
                use_rslora=config.lora.use_rslora,
                loftq_config=config.lora.loftq_config,
            )
        return model, tokenizer

    torch_dtype = None if config.model.dtype is None else getattr(torch, config.model.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        device_map=config.model.device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=config.model.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        trust_remote_code=config.model.trust_remote_code,
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if not config.model.full_finetuning:
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            target_modules=config.lora.target_modules,
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer


def run_train(config: TrainProjectConfig) -> dict[str, Any]:
    if config.data.train_file is None:
        raise ValueError("Training requires data.train_file in the YAML config.")

    if config.model.backend == "unsloth":
        import unsloth  # noqa: F401 – ensures unsloth patches are applied

    train_examples = load_examples(config.data.train_file)
    eval_examples = load_examples(config.data.eval_file) if config.data.eval_file else None

    set_seed(config.training.seed)
    model, tokenizer = _load_model_and_tokenizer(config)
    train_dataset = Dataset.from_list(_rendered_records(train_examples, tokenizer))
    eval_dataset = (
        Dataset.from_list(_rendered_records(eval_examples, tokenizer))
        if eval_examples
        else None
    )

    sft_kwargs: dict[str, Any] = {
        "output_dir": str(config.training.output_dir),
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "per_device_eval_batch_size": config.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "num_train_epochs": config.training.num_train_epochs,
        "learning_rate": config.training.learning_rate,
        "warmup_steps": config.training.warmup_steps,
        "logging_steps": config.training.logging_steps,
        "save_steps": config.training.save_steps,
        "weight_decay": config.training.weight_decay,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "optim": config.training.optim,
        "packing": config.training.packing,
        "assistant_only_loss": config.training.assistant_only_loss,
        "gradient_checkpointing": bool(config.training.gradient_checkpointing),
        "max_length": config.training.max_length or config.model.max_seq_length,
        "dataset_text_field": "text",
        "report_to": config.training.report_to,
    }
    if getattr(tokenizer, "eos_token", None):
        sft_kwargs["eos_token"] = tokenizer.eos_token
    if config.training.max_steps is not None:
        sft_kwargs["max_steps"] = config.training.max_steps

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(**sft_kwargs),
    )

    train_result = trainer.train()
    trainer.save_model(str(config.training.output_dir))
    tokenizer.save_pretrained(str(config.training.output_dir))

    summary: dict[str, Any] = {
        "project_name": config.project_name,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples) if eval_examples else 0,
        "output_dir": str(config.training.output_dir),
        "train_metrics": train_result.metrics,
    }
    if eval_dataset is not None:
        summary["eval_metrics"] = trainer.evaluate()

    write_json(config.training.output_dir / "train_summary.json", summary)
    return summary
