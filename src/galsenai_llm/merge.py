from __future__ import annotations

from typing import Any

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MergeConfig
from .generation import resolve_torch_dtype
from .io import write_json


def run_merge(config: MergeConfig) -> dict[str, Any]:
    if config.backend == "unsloth":
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(config.adapter_path),
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            load_in_4bit=config.load_in_4bit,
        )
        model.save_pretrained_merged(
            str(config.output_dir),
            tokenizer,
            save_method=config.save_method,
        )
    else:
        base_model_name = config.base_model_name
        if base_model_name is None:
            peft_config = PeftConfig.from_pretrained(str(config.adapter_path))
            base_model_name = peft_config.base_model_name_or_path

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=config.device_map,
            torch_dtype=resolve_torch_dtype(config.dtype),
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(config.adapter_path)).merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        model.save_pretrained(str(config.output_dir))
        tokenizer.save_pretrained(str(config.output_dir))

    summary: dict[str, Any] = {
        "backend": config.backend,
        "adapter_path": str(config.adapter_path),
        "output_dir": str(config.output_dir),
        "save_method": config.save_method,
    }
    write_json(config.output_dir / "merge_summary.json", summary)
    return summary
