from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class YamlConfig(BaseModel):
    @classmethod
    def from_yaml(cls, path: Path) -> YamlConfig:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.model_validate(raw)


class ModelConfig(BaseModel):
    backend: Literal["unsloth", "transformers"] = "unsloth"
    model_name: str
    max_seq_length: int = 4096
    dtype: str | None = None
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    load_in_16bit: bool = False
    full_finetuning: bool = False
    trust_remote_code: bool = True
    attn_implementation: str | None = None
    device_map: str = "auto"


class LoRAConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_rslora: bool = False
    loftq_config: dict[str, Any] | None = None


class DataConfig(BaseModel):
    train_file: Path | None = None
    eval_file: Path | None = None
    benchmark_file: Path | None = None


class TrainingConfig(BaseModel):
    output_dir: Path
    seed: int = 3407
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 1.0
    max_steps: int | None = None
    learning_rate: float = 1e-4
    warmup_steps: int = 0
    logging_steps: int = 1
    save_steps: int = 50
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    packing: bool = False
    assistant_only_loss: bool = True
    gradient_checkpointing: bool | str = "unsloth"
    max_length: int | None = None
    report_to: list[str] = Field(default_factory=list)


class TrainProjectConfig(YamlConfig):
    project_name: str = "galsenai-tool-calling"
    model: ModelConfig
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    data: DataConfig
    training: TrainingConfig


class MergeConfig(YamlConfig):
    backend: Literal["unsloth", "peft"] = "unsloth"
    adapter_path: Path
    output_dir: Path
    base_model_name: str | None = None
    max_seq_length: int = 4096
    dtype: str | None = None
    load_in_4bit: bool = False
    save_method: Literal["merged_16bit", "merged_4bit", "lora"] = "merged_16bit"
    device_map: str = "auto"


class GenerationConfig(BaseModel):
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


class InferenceConfig(YamlConfig):
    model_path: str
    adapter_path: Path | None = None
    base_model_name: str | None = None
    device_map: str = "auto"
    dtype: str | None = None
    prompt: str | None = None
    system_prompt: str | None = None
    tool_registry: Literal["none", "sample"] = "none"
    tools: list[str] = Field(default_factory=list)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


class EvaluationConfig(YamlConfig):
    dataset_file: Path
    predictions_file: Path
    report_file: Path


class BenchmarkConfig(YamlConfig):
    mode: Literal["oracle", "predictions", "transformers"] = "predictions"
    dataset_file: Path
    predictions_file: Path | None = None
    report_file: Path = Path("outputs/benchmark/report.json")
    model_path: str | None = None
    tool_registry: Literal["none", "sample"] = "none"
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
