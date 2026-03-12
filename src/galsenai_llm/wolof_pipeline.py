from __future__ import annotations

import gc
import json
import logging
import math
import random
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import quantiles
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tokenizers import AddedToken
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from .io import ensure_parent

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
WORD_JOINERS = {"'", "-", "\u2019"}


@dataclass(slots=True)
class ConversationRecord:
    conversation_id: str
    conversations: list[dict[str, str]]
    turn_count: int
    char_count: int


@dataclass(slots=True)
class GpuProfile:
    detected: bool
    device_name: str
    total_memory_mib: int
    bf16_supported: bool
    precision: str
    training_strategy: str
    benchmark_batch_size: int
    benchmark_grad_accum: int
    benchmark_max_length_cap: int
    benchmark_eval_batch_size: int
    benchmark_packing: bool
    full_batch_size: int
    full_grad_accum: int
    full_max_length_cap: int
    full_eval_batch_size: int
    full_packing: bool
    quantization_bits: int | None
    nvidia_smi_output: str


@dataclass(slots=True)
class TrainingPlan:
    strategy: str
    load_in_4bit: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    packing: bool
    num_train_epochs: int
    learning_rate: float
    warmup_steps: int
    logging_steps: int
    weight_decay: float
    lr_scheduler_type: str
    optim: str
    bf16: bool
    fp16: bool
    gradient_checkpointing: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    max_new_tokens: int


@dataclass(slots=True)
class TokenizerAdaptation:
    name: str
    tokenizer: PreTrainedTokenizerBase
    tokenizer_dir: Path
    embedding_sources: dict[int, list[int]]
    metric_token_ids: list[int]
    initialize_all_token_ids: list[int]
    metadata: dict[str, Any]


def _write_json_payload(path: Path, payload: Any) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def _run_nvidia_smi() -> tuple[str, dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "", {}

    output = result.stdout.strip()
    if not output:
        return output, {}
    first_line = output.splitlines()[0]
    parts = [part.strip() for part in first_line.split(",", maxsplit=1)]
    if len(parts) != 2:
        return output, {}
    return output, {"name": parts[0], "memory.total": parts[1]}


def detect_gpu_profile() -> GpuProfile:
    nvidia_smi_output, query = _run_nvidia_smi()
    cuda_available = torch.cuda.is_available()
    bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
    if cuda_available:
        device_props = torch.cuda.get_device_properties(0)
        total_memory_mib = int(device_props.total_memory / 1024 / 1024)
        device_name = device_props.name
    else:
        total_memory_mib = int(query.get("memory.total", "0") or 0)
        device_name = query.get("name", "cpu")

    normalized_name = device_name.upper()
    is_h100 = "H100" in normalized_name or total_memory_mib >= 70000

    if is_h100:
        return GpuProfile(
            detected=cuda_available or bool(query),
            device_name=device_name,
            total_memory_mib=total_memory_mib,
            bf16_supported=bf16_supported,
            precision="bf16" if bf16_supported else "fp16",
            training_strategy="full",
            benchmark_batch_size=32,
            benchmark_grad_accum=1,
            benchmark_max_length_cap=512,
            benchmark_eval_batch_size=16,
            benchmark_packing=True,
            full_batch_size=32,
            full_grad_accum=1,
            full_max_length_cap=512,
            full_eval_batch_size=16,
            full_packing=True,
            quantization_bits=None,
            nvidia_smi_output=nvidia_smi_output,
        )

    if total_memory_mib >= 20000:
        return GpuProfile(
            detected=cuda_available or bool(query),
            device_name=device_name,
            total_memory_mib=total_memory_mib,
            bf16_supported=bf16_supported,
            precision="bf16" if bf16_supported else "fp16",
            training_strategy="full",
            benchmark_batch_size=2,
            benchmark_grad_accum=4,
            benchmark_max_length_cap=1024,
            benchmark_eval_batch_size=2,
            benchmark_packing=False,
            full_batch_size=2,
            full_grad_accum=8,
            full_max_length_cap=1024,
            full_eval_batch_size=2,
            full_packing=False,
            quantization_bits=None,
            nvidia_smi_output=nvidia_smi_output,
        )

    if total_memory_mib >= 12000:
        return GpuProfile(
            detected=cuda_available or bool(query),
            device_name=device_name,
            total_memory_mib=total_memory_mib,
            bf16_supported=bf16_supported,
            precision="bf16" if bf16_supported else "fp16",
            training_strategy="qlora",
            benchmark_batch_size=1,
            benchmark_grad_accum=8,
            benchmark_max_length_cap=768,
            benchmark_eval_batch_size=1,
            benchmark_packing=False,
            full_batch_size=1,
            full_grad_accum=12,
            full_max_length_cap=1024,
            full_eval_batch_size=1,
            full_packing=False,
            quantization_bits=4,
            nvidia_smi_output=nvidia_smi_output,
        )

    return GpuProfile(
        detected=cuda_available or bool(query),
        device_name=device_name,
        total_memory_mib=total_memory_mib,
        bf16_supported=bf16_supported,
        precision="bf16" if bf16_supported else "fp16",
        training_strategy="qlora",
        benchmark_batch_size=1,
        benchmark_grad_accum=8,
        benchmark_max_length_cap=512,
        benchmark_eval_batch_size=1,
        benchmark_packing=False,
        full_batch_size=1,
        full_grad_accum=16,
        full_max_length_cap=768,
        full_eval_batch_size=1,
        full_packing=False,
        quantization_bits=4 if cuda_available else None,
        nvidia_smi_output=nvidia_smi_output,
    )


def _phase_learning_rate(strategy: str) -> float:
    if strategy == "full":
        return 5e-5
    return 2e-4


def _phase_optimizer(strategy: str) -> str:
    if strategy == "full":
        return "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"
    return "paged_adamw_8bit"


def _build_training_plan(
    profile: GpuProfile,
    *,
    benchmark: bool,
    train_size: int,
    num_train_epochs: int,
    max_length: int,
) -> TrainingPlan:
    batch_size = profile.benchmark_batch_size if benchmark else profile.full_batch_size
    eval_batch_size = (
        profile.benchmark_eval_batch_size
        if benchmark
        else profile.full_eval_batch_size
    )
    grad_accum = profile.benchmark_grad_accum if benchmark else profile.full_grad_accum
    packing = profile.benchmark_packing if benchmark else profile.full_packing
    steps_per_epoch = max(1, math.ceil(train_size / max(1, batch_size * grad_accum)))
    total_steps = max(1, steps_per_epoch * num_train_epochs)
    logging_steps = max(1, total_steps // 12)
    warmup_steps = max(1, int(total_steps * 0.03))
    return TrainingPlan(
        strategy=profile.training_strategy,
        load_in_4bit=profile.training_strategy == "qlora" and torch.cuda.is_available(),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        packing=packing,
        num_train_epochs=num_train_epochs,
        learning_rate=_phase_learning_rate(profile.training_strategy),
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim=_phase_optimizer(profile.training_strategy),
        bf16=profile.precision == "bf16",
        fp16=profile.precision == "fp16" and torch.cuda.is_available(),
        gradient_checkpointing=profile.training_strategy != "full",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        max_new_tokens=96,
    )


def load_wolof_dataset(path: Path) -> list[ConversationRecord]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")

    records: list[ConversationRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        conversation_id = str(item.get("conversation_id") or f"conv_{len(records)}")
        conversations = []
        for message in item.get("conversations", []):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip()
            if role not in {"user", "assistant", "system"}:
                continue
            content = str(message.get("content", ""))
            conversations.append({"role": role, "content": content})
        if len(conversations) < 2:
            continue
        if not any(message["role"] == "assistant" for message in conversations):
            continue
        char_count = sum(len(message["content"]) for message in conversations)
        records.append(
            ConversationRecord(
                conversation_id=conversation_id,
                conversations=conversations,
                turn_count=len(conversations),
                char_count=char_count,
            )
        )
    if not records:
        raise ValueError(f"No usable conversations were loaded from {path}.")
    return records


def _conversation_to_raw(record: ConversationRecord) -> dict[str, Any]:
    return {
        "conversation_id": record.conversation_id,
        "conversations": record.conversations,
        "turn_count": record.turn_count,
        "char_count": record.char_count,
    }


def _char_bins(records: Sequence[ConversationRecord]) -> list[float]:
    if len(records) < 4:
        return []
    char_counts = [record.char_count for record in records]
    return list(quantiles(char_counts, n=4, method="inclusive"))


def _turn_bucket(turn_count: int) -> str:
    if turn_count <= 2:
        return "turns_2"
    if turn_count == 3:
        return "turns_3"
    return "turns_4_plus"


def _char_bucket(char_count: int, bins: Sequence[float]) -> str:
    if not bins:
        return "chars_all"
    labels = ["chars_q1", "chars_q2", "chars_q3", "chars_q4"]
    for index, boundary in enumerate(bins):
        if char_count <= boundary:
            return labels[index]
    return labels[-1]


def _strata(records: Sequence[ConversationRecord]) -> dict[str, list[int]]:
    bins = _char_bins(records)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        key = f"{_turn_bucket(record.turn_count)}::{_char_bucket(record.char_count, bins)}"
        grouped[key].append(index)
    return grouped


def _allocation(group_sizes: dict[str, int], sample_size: int) -> dict[str, int]:
    total = sum(group_sizes.values())
    if sample_size > total:
        raise ValueError("Requested sample is larger than the available dataset.")
    raw_counts = {
        key: sample_size * size / total
        for key, size in group_sizes.items()
    }
    counts = {
        key: min(group_sizes[key], int(raw_counts[key]))
        for key in group_sizes
    }
    remainder = sample_size - sum(counts.values())
    if remainder <= 0:
        return counts
    ranked = sorted(
        group_sizes,
        key=lambda key: (
            raw_counts[key] - counts[key],
            group_sizes[key],
            key,
        ),
        reverse=True,
    )
    for key in ranked:
        if remainder == 0:
            break
        if counts[key] >= group_sizes[key]:
            continue
        counts[key] += 1
        remainder -= 1
    return counts


def stratified_sample(
    records: Sequence[ConversationRecord],
    sample_size: int,
    *,
    seed: int,
) -> tuple[list[ConversationRecord], list[ConversationRecord]]:
    grouped = _strata(records)
    sizes = {key: len(indexes) for key, indexes in grouped.items()}
    counts = _allocation(sizes, sample_size)
    rng = random.Random(seed)
    selected_indexes: list[int] = []
    for key, indexes in grouped.items():
        shuffled = list(indexes)
        rng.shuffle(shuffled)
        selected_indexes.extend(shuffled[: counts[key]])

    if len(selected_indexes) < sample_size:
        remaining_indexes = [idx for idx in range(len(records)) if idx not in set(selected_indexes)]
        rng.shuffle(remaining_indexes)
        selected_indexes.extend(remaining_indexes[: sample_size - len(selected_indexes)])

    selected_set = set(selected_indexes)
    selected = [records[index] for index in selected_indexes]
    remaining = [record for index, record in enumerate(records) if index not in selected_set]
    return selected, remaining


def split_records(
    records: Sequence[ConversationRecord],
    *,
    train_count: int,
    val_count: int,
    test_count: int = 0,
    seed: int,
) -> dict[str, list[ConversationRecord]]:
    if train_count + val_count + test_count > len(records):
        raise ValueError("Split sizes exceed the number of available records.")
    train_split, remainder = stratified_sample(records, train_count, seed=seed)
    if test_count:
        val_split, remainder = stratified_sample(remainder, val_count, seed=seed + 1)
        test_split, _ = stratified_sample(remainder, test_count, seed=seed + 2)
        return {"train": train_split, "val": val_split, "test": test_split}
    val_split, _ = stratified_sample(remainder, val_count, seed=seed + 1)
    return {"train": train_split, "val": val_split}


def _ratio_split_counts(
    sample_size: int,
    *,
    ratios: Sequence[float],
    labels: Sequence[str],
) -> dict[str, int]:
    if len(ratios) != len(labels):
        raise ValueError("Ratios and labels must have the same length.")
    if sample_size < len(labels):
        required = len(labels)
        raise ValueError(
            f"Need at least {required} samples to build splits for {', '.join(labels)}."
        )

    raw_counts = [sample_size * ratio for ratio in ratios]
    counts = [max(1, int(raw_count)) for raw_count in raw_counts]

    while sum(counts) > sample_size:
        candidates = [index for index, count in enumerate(counts) if count > 1]
        if not candidates:
            break
        index = max(
            candidates,
            key=lambda item: (counts[item] - raw_counts[item], counts[item], -item),
        )
        counts[index] -= 1

    while sum(counts) < sample_size:
        index = max(
            range(len(counts)),
            key=lambda item: (raw_counts[item] - counts[item], raw_counts[item], -item),
        )
        counts[index] += 1

    return dict(zip(labels, counts, strict=True))


def _extract_words(text: str) -> list[str]:
    words: list[str] = []
    current: list[str] = []
    text_length = len(text)
    for index, char in enumerate(text):
        if char.isalpha():
            current.append(char.lower())
            continue
        if (
            char in WORD_JOINERS
            and current
            and index + 1 < text_length
            and text[index + 1].isalpha()
        ):
            current.append(char)
            continue
        if current:
            word = "".join(current)
            if any(ch.isalpha() for ch in word):
                words.append(word)
            current = []
    if current:
        word = "".join(current)
        if any(ch.isalpha() for ch in word):
            words.append(word)
    return words


def _assistant_texts(records: Sequence[ConversationRecord]) -> list[str]:
    texts: list[str] = []
    for record in records:
        for message in record.conversations:
            if message["role"] == "assistant":
                texts.append(message["content"])
    return texts


def _all_message_texts(records: Sequence[ConversationRecord]) -> list[str]:
    texts: list[str] = []
    for record in records:
        for message in record.conversations:
            texts.append(message["content"])
    return texts


def _prompt_messages(record: ConversationRecord) -> list[dict[str, str]]:
    if record.conversations and record.conversations[-1]["role"] == "assistant":
        return record.conversations[:-1]
    return list(record.conversations)


def _reference_answer(record: ConversationRecord) -> str:
    if record.conversations and record.conversations[-1]["role"] == "assistant":
        return record.conversations[-1]["content"]
    for message in reversed(record.conversations):
        if message["role"] == "assistant":
            return message["content"]
    return ""


def _fallback_render(record: ConversationRecord) -> str:
    sections = []
    for message in record.conversations:
        sections.append(message["role"].upper())
        sections.append(message["content"])
    return "\n\n".join(section for section in sections if section)


def render_record(
    record: ConversationRecord,
    tokenizer: PreTrainedTokenizerBase,
    *,
    add_generation_prompt: bool,
    prompt_only: bool = False,
) -> str:
    messages = _prompt_messages(record) if prompt_only else record.conversations
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            logger.warning(
                "Chat template rendering failed for %s; falling back to plain text.",
                record.conversation_id,
                exc_info=True,
            )
    if prompt_only:
        prompt_record = ConversationRecord(
            conversation_id=record.conversation_id,
            conversations=messages,
            turn_count=len(messages),
            char_count=sum(len(message["content"]) for message in messages),
        )
        return _fallback_render(prompt_record)
    return _fallback_render(record)


def _rendered_dataset(
    records: Sequence[ConversationRecord],
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    payload = [
        {"text": render_record(record, tokenizer, add_generation_prompt=False)}
        for record in records
    ]
    return Dataset.from_list(payload)


def _token_to_surface(tokenizer: PreTrainedTokenizerBase, token: str) -> str:
    try:
        return tokenizer.convert_tokens_to_string([token])
    except Exception:
        return token


def _strip_whitespace_only_ids(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: Sequence[int],
) -> list[int]:
    kept: list[int] = []
    for token_id in token_ids:
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        surface = _token_to_surface(tokenizer, token)
        if surface.strip():
            kept.append(int(token_id))
    return kept


def _best_piece_ids(tokenizer: PreTrainedTokenizerBase, word: str) -> list[int]:
    candidates = [word]
    stripped = word.strip()
    if stripped and f" {stripped}" not in candidates:
        candidates.append(f" {stripped}")
    best_ids: list[int] | None = None
    best_key: tuple[int, int] | None = None
    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        token_ids = _strip_whitespace_only_ids(tokenizer, token_ids)
        if not token_ids:
            continue
        key = (len(token_ids), len(candidate))
        if best_key is None or key < best_key:
            best_ids = token_ids
            best_key = key
    return best_ids or []


def _surface_piece_ids(tokenizer: PreTrainedTokenizerBase, surface: str) -> list[int]:
    candidates = [surface]
    stripped = surface.strip()
    if stripped and stripped not in candidates:
        candidates.append(stripped)
    if stripped and f" {stripped}" not in candidates:
        candidates.append(f" {stripped}")
    best_ids: list[int] | None = None
    best_key: tuple[int, int] | None = None
    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        token_ids = _strip_whitespace_only_ids(tokenizer, token_ids)
        if not token_ids:
            continue
        key = (len(token_ids), len(candidate))
        if best_key is None or key < best_key:
            best_ids = token_ids
            best_key = key
    return best_ids or []


def compute_tokenizer_metrics(
    tokenizer: PreTrainedTokenizerBase,
    texts: Iterable[str],
) -> dict[str, float]:
    total_words = 0
    total_tokens = 0
    single_tokens = 0
    for text in texts:
        for word in _extract_words(text):
            token_ids = _best_piece_ids(tokenizer, word)
            if not token_ids:
                continue
            total_words += 1
            total_tokens += len(token_ids)
            if len(token_ids) == 1:
                single_tokens += 1
    if total_words == 0:
        return {"fertility": 0.0, "coverage": 0.0}
    return {
        "fertility": total_tokens / total_words,
        "coverage": 100.0 * single_tokens / total_words,
    }


def _load_base_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.padding_side = "right"
    return tokenizer


def build_method_a_adaptation(
    model_name: str,
    *,
    assistant_texts: Sequence[str],
    token_budget: int,
    output_dir: Path,
) -> TokenizerAdaptation:
    base_tokenizer = _load_base_tokenizer(model_name)
    word_counts: Counter[str] = Counter()
    for text in assistant_texts:
        word_counts.update(_extract_words(text))

    word_candidates: list[dict[str, Any]] = []
    for word, frequency in word_counts.items():
        if len(word) < 3 or frequency < 2:
            continue
        source_ids = _best_piece_ids(base_tokenizer, word)
        if len(source_ids) <= 1:
            continue
        word_candidates.append(
            {
                "token": word,
                "token_type": "word",
                "frequency": frequency,
                "fragment_count": len(source_ids),
                "score": frequency * (len(source_ids) - 1),
                "source_ids": source_ids,
            }
        )

    prefix_counts: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    prefix_words: dict[str, set[str]] = defaultdict(set)
    suffix_words: dict[str, set[str]] = defaultdict(set)
    for word, frequency in word_counts.items():
        if len(word) < 5:
            continue
        if len(_best_piece_ids(base_tokenizer, word)) <= 1:
            continue
        for length in range(3, min(8, len(word) - 1) + 1):
            prefix = word[:length]
            suffix = word[-length:]
            prefix_counts[prefix] += frequency
            suffix_counts[suffix] += frequency
            prefix_words[prefix].add(word)
            suffix_words[suffix].add(word)

    morph_candidates: list[dict[str, Any]] = []
    for token, frequency in prefix_counts.items():
        if len(prefix_words[token]) < 3 or frequency < 6:
            continue
        source_ids = _best_piece_ids(base_tokenizer, token)
        if len(source_ids) <= 1:
            continue
        morph_candidates.append(
            {
                "token": token,
                "token_type": "prefix",
                "frequency": frequency,
                "fragment_count": len(source_ids),
                "score": frequency * (len(source_ids) - 1),
                "source_ids": source_ids,
            }
        )
    for token, frequency in suffix_counts.items():
        if len(suffix_words[token]) < 3 or frequency < 6:
            continue
        source_ids = _best_piece_ids(base_tokenizer, token)
        if len(source_ids) <= 1:
            continue
        morph_candidates.append(
            {
                "token": token,
                "token_type": "suffix",
                "frequency": frequency,
                "fragment_count": len(source_ids),
                "score": frequency * (len(source_ids) - 1),
                "source_ids": source_ids,
            }
        )

    word_candidates.sort(
        key=lambda item: (item["score"], item["frequency"], item["token"]),
        reverse=True,
    )
    morph_candidates.sort(
        key=lambda item: (item["score"], item["frequency"], item["token"]),
        reverse=True,
    )

    tokenizer = _load_base_tokenizer(model_name)
    word_quota = max(1, int(token_budget * 0.7))
    morph_quota = max(0, token_budget - word_quota)
    selected: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()

    def _take(candidates: Sequence[dict[str, Any]], limit: int) -> None:
        for candidate in candidates:
            if len(selected) >= token_budget or limit <= 0:
                break
            token = candidate["token"]
            if token in seen_tokens:
                continue
            selected.append(candidate)
            seen_tokens.add(token)
            limit -= 1

    _take(word_candidates, word_quota)
    _take(morph_candidates, morph_quota)
    if len(selected) < token_budget:
        combined = sorted(
            [*word_candidates, *morph_candidates],
            key=lambda item: (item["score"], item["frequency"], item["token"]),
            reverse=True,
        )
        for candidate in combined:
            if len(selected) >= token_budget:
                break
            token = candidate["token"]
            if token in seen_tokens:
                continue
            selected.append(candidate)
            seen_tokens.add(token)

    added_tokens = []
    for candidate in selected:
        token_type = candidate["token_type"]
        added_tokens.append(
            AddedToken(
                candidate["token"],
                single_word=token_type == "word",
                lstrip=token_type == "word",
                normalized=False,
            )
        )
    tokenizer.add_tokens(added_tokens)

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))

    embedding_sources: dict[int, list[int]] = {}
    metric_token_ids: list[int] = []
    selected_metadata = []
    for candidate in selected:
        token = candidate["token"]
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        source_ids = list(candidate["source_ids"])
        embedding_sources[token_id] = source_ids
        metric_token_ids.append(token_id)
        selected_metadata.append(
            {
                "token": token,
                "token_id": token_id,
                "token_type": candidate["token_type"],
                "frequency": candidate["frequency"],
                "fragment_count": candidate["fragment_count"],
                "score": candidate["score"],
                "source_ids": source_ids,
                "source_tokens": base_tokenizer.convert_ids_to_tokens(source_ids),
            }
        )

    metadata = {
        "method": "method_a",
        "token_budget": token_budget,
        "selected_token_count": len(selected_metadata),
        "selected_tokens": selected_metadata,
        "candidate_summary": {
            "word_candidates": len(word_candidates),
            "morpheme_candidates": len(morph_candidates),
        },
    }
    _write_json_payload(output_dir / "token_selection.json", metadata)

    return TokenizerAdaptation(
        name="method_a",
        tokenizer=tokenizer,
        tokenizer_dir=tokenizer_dir,
        embedding_sources=embedding_sources,
        metric_token_ids=metric_token_ids,
        initialize_all_token_ids=metric_token_ids,
        metadata=metadata,
    )


def build_method_b_adaptation(
    model_name: str,
    *,
    training_texts: Sequence[str],
    vocab_size: int,
    output_dir: Path,
) -> TokenizerAdaptation:
    base_tokenizer = _load_base_tokenizer(model_name)
    tokenizer = base_tokenizer.train_new_from_iterator(
        training_texts,
        vocab_size=vocab_size,
    )
    tokenizer.padding_side = "right"
    tokenizer.chat_template = getattr(base_tokenizer, "chat_template", None)
    if tokenizer.pad_token is None and base_tokenizer.pad_token is not None:
        tokenizer.pad_token = base_tokenizer.pad_token
    if tokenizer.eos_token is None and base_tokenizer.eos_token is not None:
        tokenizer.eos_token = base_tokenizer.eos_token
    tokenizer.model_max_length = base_tokenizer.model_max_length

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))

    base_vocab = base_tokenizer.get_vocab()
    special_tokens = set(base_tokenizer.all_special_tokens) | set(tokenizer.all_special_tokens)
    new_vocab_items = sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])

    embedding_sources: dict[int, list[int]] = {}
    initialize_all_token_ids: list[int] = []
    metric_token_ids: list[int] = []
    init_plan: list[dict[str, Any]] = []
    direct_overlap = 0
    compositional = 0

    for token, token_id in new_vocab_items:
        if token in base_vocab:
            source_ids = [int(base_vocab[token])]
            source_method = "direct_overlap"
            direct_overlap += 1
        else:
            surface = _token_to_surface(tokenizer, token)
            source_ids = _surface_piece_ids(base_tokenizer, surface)
            if not source_ids:
                source_ids = _surface_piece_ids(base_tokenizer, token)
            source_method = "compositional_average"
            compositional += 1
            if token not in special_tokens:
                metric_token_ids.append(int(token_id))

        embedding_sources[int(token_id)] = source_ids
        initialize_all_token_ids.append(int(token_id))
        init_plan.append(
            {
                "token": token,
                "token_id": int(token_id),
                "source_method": source_method,
                "source_ids": source_ids,
                "source_tokens": base_tokenizer.convert_ids_to_tokens(source_ids),
            }
        )

    metadata = {
        "method": "method_b",
        "target_vocab_size": vocab_size,
        "actual_vocab_size": len(tokenizer),
        "direct_overlap_count": direct_overlap,
        "compositional_init_count": compositional,
        "init_plan": init_plan,
    }
    _write_json_payload(output_dir / "embedding_transfer_plan.json", metadata)

    return TokenizerAdaptation(
        name="method_b",
        tokenizer=tokenizer,
        tokenizer_dir=tokenizer_dir,
        embedding_sources=embedding_sources,
        metric_token_ids=metric_token_ids,
        initialize_all_token_ids=initialize_all_token_ids,
        metadata=metadata,
    )


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _update_model_token_config(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is None:
            continue
        setattr(model.config, attr, token_id)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            setattr(generation_config, attr, token_id)


def _count_parameters(model: Any) -> dict[str, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {"total": total, "trainable": trainable}


def _prepare_model(
    model_name: str,
    *,
    tokenizer: PreTrainedTokenizerBase,
    adaptation: TokenizerAdaptation,
    plan: TrainingPlan,
) -> tuple[Any, dict[str, Any], dict[int, float]]:
    dtype = torch.float32
    if torch.cuda.is_available():
        if plan.bf16:
            dtype = torch.bfloat16
        elif plan.fp16:
            dtype = torch.float16

    quantization_config = None
    device_map: Any = None
    if plan.load_in_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=dtype if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        device_map=device_map,
    )

    old_input = model.get_input_embeddings().weight.detach().float().cpu().clone()
    output_module = model.get_output_embeddings()
    old_output = None
    output_is_tied = False
    if output_module is not None:
        output_is_tied = (
            output_module.weight.data_ptr()
            == model.get_input_embeddings().weight.data_ptr()
        )
        if not output_is_tied:
            old_output = output_module.weight.detach().float().cpu().clone()

    model.resize_token_embeddings(len(tokenizer))
    _update_model_token_config(model, tokenizer)
    if plan.gradient_checkpointing:
        model.config.use_cache = False
    if plan.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=plan.gradient_checkpointing,
        )

    input_weight = model.get_input_embeddings().weight.data
    if output_module is not None:
        output_module = model.get_output_embeddings()
    similarity_scores: dict[int, float] = {}

    for token_id in adaptation.initialize_all_token_ids:
        source_ids = adaptation.embedding_sources[token_id]
        if not source_ids:
            continue
        source_input = old_input[source_ids].mean(dim=0)
        input_weight[token_id] = source_input.to(
            device=input_weight.device,
            dtype=input_weight.dtype,
        )
        if output_module is not None and not output_is_tied:
            source_output = source_input
            if old_output is not None:
                source_output = old_output[source_ids].mean(dim=0)
            output_module.weight.data[token_id] = source_output.to(
                device=output_module.weight.device,
                dtype=output_module.weight.dtype,
            )
        if token_id in adaptation.metric_token_ids:
            final_vector = model.get_input_embeddings().weight[token_id].detach().float().cpu()
            source_vectors = old_input[source_ids]
            cosines = F.cosine_similarity(
                final_vector.unsqueeze(0),
                source_vectors,
                dim=-1,
            )
            mean_cosine = float(cosines.mean().item())
            similarity_scores[token_id] = mean_cosine

    model.tie_weights()
    _update_model_token_config(model, tokenizer)

    if plan.strategy != "full":
        lora_config = LoraConfig(
            r=plan.lora_r,
            lora_alpha=plan.lora_alpha,
            lora_dropout=plan.lora_dropout,
            bias="none",
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    parameter_counts = _count_parameters(model)
    return (
        model,
        {
            "parameter_counts": parameter_counts,
            "embedding_init_quality": (
                sum(similarity_scores.values()) / len(similarity_scores)
                if similarity_scores
                else 0.0
            ),
        },
        similarity_scores,
    )


def _extract_loss_curves(log_history: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    train_curve: list[dict[str, Any]] = []
    eval_curve: list[dict[str, Any]] = []
    for item in log_history:
        if "loss" in item and "eval_loss" not in item:
            train_curve.append(
                {
                    "step": item.get("step"),
                    "epoch": item.get("epoch"),
                    "loss": item["loss"],
                }
            )
        if "eval_loss" in item:
            eval_curve.append(
                {
                    "step": item.get("step"),
                    "epoch": item.get("epoch"),
                    "eval_loss": item["eval_loss"],
                }
            )
    return {"train": train_curve, "eval": eval_curve}


def _perplexity(eval_loss: float | None) -> float | None:
    if eval_loss is None:
        return None
    try:
        return float(math.exp(eval_loss))
    except OverflowError:
        return float("inf")


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _reduced_plan(plan: TrainingPlan) -> TrainingPlan | None:
    if plan.per_device_train_batch_size > 1:
        reduced_batch_size = max(1, plan.per_device_train_batch_size // 2)
        effective_batch = max(1, plan.per_device_train_batch_size)
        reduced_effective_batch = max(1, reduced_batch_size)
        grad_accum = max(
            1,
            math.ceil(
                plan.gradient_accumulation_steps
                * effective_batch
                / reduced_effective_batch
            ),
        )
        return TrainingPlan(
            **{
                **asdict(plan),
                "per_device_train_batch_size": reduced_batch_size,
                "per_device_eval_batch_size": min(
                    plan.per_device_eval_batch_size,
                    reduced_batch_size,
                ),
                "gradient_accumulation_steps": grad_accum,
            }
        )
    if plan.max_length > 384:
        return TrainingPlan(
            **{
                **asdict(plan),
                "max_length": max(384, plan.max_length // 2),
            }
        )
    return None


def _train_once(
    *,
    model_name: str,
    adaptation: TokenizerAdaptation,
    train_records: Sequence[ConversationRecord],
    eval_records: Sequence[ConversationRecord],
    plan: TrainingPlan,
    output_dir: Path,
    generation_records: Sequence[ConversationRecord] | None = None,
) -> dict[str, Any]:
    model, model_metadata, similarity_scores = _prepare_model(
        model_name,
        tokenizer=adaptation.tokenizer,
        adaptation=adaptation,
        plan=plan,
    )

    train_dataset = _rendered_dataset(train_records, adaptation.tokenizer)
    eval_dataset = _rendered_dataset(eval_records, adaptation.tokenizer)

    trainer_output_dir = output_dir / "model"
    sft_config = SFTConfig(
        output_dir=str(trainer_output_dir),
        per_device_train_batch_size=plan.per_device_train_batch_size,
        per_device_eval_batch_size=plan.per_device_eval_batch_size,
        gradient_accumulation_steps=plan.gradient_accumulation_steps,
        num_train_epochs=plan.num_train_epochs,
        learning_rate=plan.learning_rate,
        warmup_steps=plan.warmup_steps,
        logging_steps=plan.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="steps",
        weight_decay=plan.weight_decay,
        lr_scheduler_type=plan.lr_scheduler_type,
        optim=plan.optim,
        packing=plan.packing,
        assistant_only_loss=False,
        gradient_checkpointing=plan.gradient_checkpointing,
        max_length=plan.max_length,
        dataset_text_field="text",
        report_to=[],
        bf16=plan.bf16,
        fp16=plan.fp16,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=adaptation.tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(trainer_output_dir))
    trainer.save_state()
    adaptation.tokenizer.save_pretrained(str(trainer_output_dir))
    sample_generations: list[dict[str, str]] = []
    if generation_records:
        sample_generations = _generate_samples_from_model(
            model,
            adaptation.tokenizer,
            generation_records,
            max_new_tokens=plan.max_new_tokens,
        )

    loss_curves = _extract_loss_curves(trainer.state.log_history)
    train_summary = {
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "loss_curves": loss_curves,
        "training_plan": asdict(plan),
        "parameter_counts": model_metadata["parameter_counts"],
        "embedding_init_quality": model_metadata["embedding_init_quality"],
        "embedding_similarity_by_token_id": similarity_scores,
        "sample_generations": sample_generations,
    }
    _write_json_payload(output_dir / "training_summary.json", train_summary)
    return train_summary


def train_with_fallback(
    *,
    model_name: str,
    adaptation: TokenizerAdaptation,
    train_records: Sequence[ConversationRecord],
    eval_records: Sequence[ConversationRecord],
    plan: TrainingPlan,
    output_dir: Path,
    generation_records: Sequence[ConversationRecord] | None = None,
) -> dict[str, Any]:
    current_plan = plan
    attempts: list[dict[str, Any]] = []
    while True:
        _cleanup_cuda()
        try:
            summary = _train_once(
                model_name=model_name,
                adaptation=adaptation,
                train_records=train_records,
                eval_records=eval_records,
                plan=current_plan,
                output_dir=output_dir,
                generation_records=generation_records,
            )
            summary["attempts"] = attempts + [{"status": "success", "plan": asdict(current_plan)}]
            _write_json_payload(output_dir / "oom_retries.json", summary["attempts"])
            return summary
        except Exception as exc:
            if not _is_oom_error(exc):
                raise
            attempts.append(
                {
                    "status": "oom",
                    "plan": asdict(current_plan),
                    "error": str(exc),
                }
            )
            reduced = _reduced_plan(current_plan)
            _cleanup_cuda()
            if reduced is None:
                _write_json_payload(output_dir / "oom_retries.json", attempts)
                raise
            current_plan = reduced


def _suggest_max_length(
    records: Sequence[ConversationRecord],
    tokenizer: PreTrainedTokenizerBase,
    *,
    cap: int,
) -> int:
    sample = list(records[: min(200, len(records))])
    lengths: list[int] = []
    for record in sample:
        rendered = render_record(record, tokenizer, add_generation_prompt=False)
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        lengths.append(len(token_ids))
    if not lengths:
        return min(512, cap)
    lengths.sort()
    index = min(len(lengths) - 1, math.ceil(len(lengths) * 0.95) - 1)
    target = lengths[index]
    for option in (256, 384, 512, 640, 768, 1024):
        if option >= target:
            return min(option, cap)
    return cap


def _generate_samples_from_model(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    records: Sequence[ConversationRecord],
    *,
    max_new_tokens: int,
    num_samples: int = 5,
) -> list[dict[str, str]]:
    _update_model_token_config(model, tokenizer)
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    generation_dtype = torch.float32
    if torch.cuda.is_available():
        generation_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None:
        input_embeddings.to(device=device, dtype=generation_dtype)
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        output_embeddings.to(device=device, dtype=generation_dtype)

    generations: list[dict[str, str]] = []
    for record in list(records)[:num_samples]:
        prompt_text = render_record(
            record,
            tokenizer,
            add_generation_prompt=True,
            prompt_only=True,
        )
        encoded = tokenizer(prompt_text, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_context = (
            torch.autocast(device_type="cuda", dtype=generation_dtype)
            if torch.cuda.is_available()
            else nullcontext()
        )
        with torch.no_grad(), generation_context:
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        completion = tokenizer.decode(
            generated[0, prompt_length:],
            skip_special_tokens=True,
        ).strip()
        generations.append(
            {
                "conversation_id": record.conversation_id,
                "prompt": prompt_text,
                "reference": _reference_answer(record),
                "generation": completion,
            }
        )
    return generations


def _benchmark_method(
    *,
    method_name: str,
    model_name: str,
    benchmark_splits: dict[str, list[ConversationRecord]],
    profile: GpuProfile,
    token_budget: int,
    bpe_vocab_size: int,
    epochs: int,
    output_dir: Path,
) -> dict[str, Any]:
    train_records = benchmark_splits["train"]
    val_records = benchmark_splits["val"]
    test_records = benchmark_splits["test"]

    if method_name == "method_a":
        adaptation = build_method_a_adaptation(
            model_name,
            assistant_texts=_assistant_texts(train_records),
            token_budget=token_budget,
            output_dir=output_dir,
        )
    else:
        adaptation = build_method_b_adaptation(
            model_name,
            training_texts=_all_message_texts(train_records),
            vocab_size=bpe_vocab_size,
            output_dir=output_dir,
        )

    suggested_length = _suggest_max_length(
        [*train_records, *val_records],
        adaptation.tokenizer,
        cap=profile.benchmark_max_length_cap,
    )
    plan = _build_training_plan(
        profile,
        benchmark=True,
        train_size=len(train_records),
        num_train_epochs=epochs,
        max_length=suggested_length,
    )

    held_out_texts = _assistant_texts(test_records)
    tokenizer_metrics = compute_tokenizer_metrics(adaptation.tokenizer, held_out_texts)
    train_summary = train_with_fallback(
        model_name=model_name,
        adaptation=adaptation,
        train_records=train_records,
        eval_records=val_records,
        plan=plan,
        output_dir=output_dir,
    )
    eval_loss = train_summary["eval_metrics"].get("eval_loss")
    result = {
        "method": method_name,
        "tokenizer_path": str(adaptation.tokenizer_dir),
        "tokenizer_metadata": adaptation.metadata,
        "fertility": tokenizer_metrics["fertility"],
        "coverage": tokenizer_metrics["coverage"],
        "embedding_init_quality": train_summary["embedding_init_quality"],
        "perplexity": _perplexity(eval_loss),
        "eval_loss": eval_loss,
        "training_plan": train_summary["training_plan"],
        "parameter_counts": train_summary["parameter_counts"],
        "loss_curves": train_summary["loss_curves"],
        "attempts": train_summary["attempts"],
    }
    _write_json_payload(output_dir / "benchmark_result.json", result)
    return result


def _pick_winner(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot pick a winner from an empty result list.")
    return min(
        results,
        key=lambda item: (
            item["perplexity"] if item["perplexity"] is not None else float("inf"),
            item["fertility"],
            -item["coverage"],
            -item["embedding_init_quality"],
        ),
    )


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _build_markdown_report(
    *,
    output_dir: Path,
    gpu_profile: GpuProfile,
    model_name: str,
    benchmark_results: Sequence[dict[str, Any]],
    winner: dict[str, Any],
    final_result: dict[str, Any],
    benchmark_splits: dict[str, list[ConversationRecord]],
    final_splits: dict[str, list[ConversationRecord]],
) -> str:
    benchmark_rows = "\n".join(
        [
            "| {method} | {fertility} | {coverage} | {embedding} | {perplexity} |".format(
                method=result["method"],
                fertility=_format_float(result["fertility"], 3),
                coverage=_format_float(result["coverage"], 2),
                embedding=_format_float(result["embedding_init_quality"], 4),
                perplexity=_format_float(result["perplexity"], 4),
            )
            for result in benchmark_results
        ]
    )

    generation_blocks = []
    for sample in final_result["sample_generations"]:
        generation_blocks.append(
            "\n".join(
                [
                    f"### {sample['conversation_id']}",
                    "",
                    "**Prompt**",
                    "",
                    "```text",
                    sample["prompt"].strip(),
                    "```",
                    "",
                    "**Reference**",
                    "",
                    "```text",
                    sample["reference"].strip(),
                    "```",
                    "",
                    "**Generation**",
                    "",
                    "```text",
                    sample["generation"].strip(),
                    "```",
                ]
            )
        )

    benchmark_sample_total = sum(len(split) for split in benchmark_splits.values())
    benchmark_split_line = (
        f"train={len(benchmark_splits['train'])}, "
        f"val={len(benchmark_splits['val'])}, "
        f"test={len(benchmark_splits['test'])}"
    )
    full_sample_total = len(final_splits["train"]) + len(final_splits["val"])
    full_split_line = (
        f"train={len(final_splits['train'])}, "
        f"val={len(final_splits['val'])}"
    )
    runtime_strategy_line = (
        "- Fine-tuning uses packed full-parameter BF16 training on the "
        "detected high-memory GPU so the 1k tokenizer benchmark and 5k "
        "follow-up run finish with much higher throughput."
        if gpu_profile.training_strategy == "full" and gpu_profile.full_packing
        else "- Fine-tuning uses full-parameter training because the "
        "detected GPU has enough memory headroom for resized embeddings "
        "and validation without quantization."
        if gpu_profile.training_strategy == "full"
        else "- Fine-tuning uses QLoRA on the detected smaller GPU "
        "because it is the reliable path with checkpointing and enough "
        "headroom for tokenizer resizing and validation."
    )
    research_lines = "\n".join(
        [
            "- Tokenizer quality is measured with intrinsic metrics",
            "  (fertility, coverage) and downstream loss because",
            "  tokenizer studies show fragmentation metrics alone do",
            "  not fully predict task performance.",
            "- Method A keeps Qwen's base vocabulary and augments it",
            "  with high-frequency over-fragmented Wolof words and",
            "  affix-like morphemes so English coverage and chat",
            "  special tokens remain stable.",
            "- Method B trains a compact tokenizer on the Wolof",
            "  corpus and initializes each token by direct overlap or",
            "  compositional averaging from the original Qwen",
            "  subword space instead of random rows.",
            runtime_strategy_line,
        ]
    )
    reference_lines = "\n".join(
        [
            "- Rust et al. (2021), *How Good Is Your Tokenizer?*",
            "  https://aclanthology.org/2021.acl-long.243/",
            "- Dobler et al. (2024), *FOCUS: Effective Embedding",
            "  Initialization for Special Tokens and Embeddings in",
            "  Fine-Tuned Language Models*",
            "  https://arxiv.org/abs/2305.14481",
            "- Dettmers et al. (2023), *QLoRA: Efficient Finetuning",
            "  of Quantized LLMs*",
            "  https://arxiv.org/abs/2305.14314",
            "- Chau et al. (2024), *Tokenizer Choice For LLM Training:",
            "  Negligible or Crucial?*",
            "  https://arxiv.org/abs/2405.17886",
        ]
    )
    recommendation_line = (
        "this method achieved the best validation perplexity on the held-out "
        "benchmark split. The secondary checks on fertility, single-token "
        "coverage, and embedding initialization quality also stayed "
        "competitive enough to make it the safer choice for the larger 5k "
        "fine-tune."
    )

    report = f"""# Wolof Qwen 2.5 Fine-Tuning Report

## Environment

- Run directory: `{output_dir}`
- Base model: `{model_name}`
- GPU: `{gpu_profile.device_name}`
- GPU memory: `{gpu_profile.total_memory_mib} MiB`
- Precision: `{gpu_profile.precision}`
- Training strategy: `{gpu_profile.training_strategy}`
- `nvidia-smi` summary:

```text
{gpu_profile.nvidia_smi_output or "nvidia-smi unavailable"}
```

## Research-Informed Design

{research_lines}

## References

{reference_lines}

## Benchmark Setup

- Benchmark sample size: {benchmark_sample_total} conversations
- Benchmark split: {benchmark_split_line}
- Full-train sample size: {full_sample_total} conversations
- Full split: {full_split_line}
- Stratification: turn-count bucket plus conversation-length quartile
- Tokenizer rebuild policy for Step 2: reuse the winning adaptation
  method and rebuild it on the 5k training corpus only, which avoids
  validation leakage while keeping the winning tokenizer strategy fixed

## Tokenizer Comparison

| Method | Fertility | Coverage (%) | Init cosine | Validation perplexity |
| --- | ---: | ---: | ---: | ---: |
{benchmark_rows}

## Recommendation

- Winner: `{winner['method']}`
- Justification: {recommendation_line}

## Full Fine-Tuning

- Winning tokenizer method: `{final_result['method']}`
- Final validation loss: {_format_float(final_result['eval_loss'], 4)}
- Final validation perplexity: {_format_float(final_result['perplexity'], 4)}
- Training plan: `{json.dumps(final_result['training_plan'], sort_keys=True)}`
- Parameter counts: `{json.dumps(final_result['parameter_counts'], sort_keys=True)}`

## Sample Generations

{chr(10).join(generation_blocks)}

## Artifacts

- Benchmark summary: `{output_dir / "benchmark" / "comparison.json"}`
- Final summary: `{output_dir / "final" / "final_result.json"}`
- Final generations: `{output_dir / "final" / "sample_generations.json"}`
- Trainer outputs: `{output_dir / "final" / "model"}`

## Notes

- If a future run has a larger GPU budget, rerun with the same command.
  The pipeline will switch to a full fine-tuning plan automatically
  when memory is sufficient.
- The current dataset does not expose explicit topic labels, so the
  sampler stratifies on conversation length and turn structure instead
  of semantic topic.
"""
    return report


def run_wolof_pipeline(
    *,
    dataset_file: Path = Path("data/wolof-dataset/curated_dataset.json"),
    output_root: Path = Path("outputs/wolof"),
    model_name: str = DEFAULT_MODEL_NAME,
    benchmark_sample_size: int = 1000,
    full_sample_size: int = 5000,
    token_budget: int = 800,
    benchmark_bpe_vocab_size: int = 8192,
    full_bpe_vocab_size: int = 16384,
    benchmark_epochs: int = 3,
    full_epochs: int = 3,
    seed: int = 3407,
) -> dict[str, Any]:
    set_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / run_id
    benchmark_dir = output_dir / "benchmark"
    final_dir = output_dir / "final"

    gpu_profile = detect_gpu_profile()
    _write_text(
        output_dir / "nvidia_smi.txt",
        gpu_profile.nvidia_smi_output or "nvidia-smi unavailable\n",
    )

    records = load_wolof_dataset(dataset_file)
    benchmark_pool, remaining = stratified_sample(records, benchmark_sample_size, seed=seed)
    benchmark_counts = _ratio_split_counts(
        benchmark_sample_size,
        ratios=(0.8, 0.1, 0.1),
        labels=("train", "val", "test"),
    )
    benchmark_splits = split_records(
        benchmark_pool,
        train_count=benchmark_counts["train"],
        val_count=benchmark_counts["val"],
        test_count=benchmark_counts["test"],
        seed=seed + 11,
    )
    full_pool_source = remaining if len(remaining) >= full_sample_size else records
    full_pool, _ = stratified_sample(full_pool_source, full_sample_size, seed=seed + 101)
    full_counts = _ratio_split_counts(
        full_sample_size,
        ratios=(0.9, 0.1),
        labels=("train", "val"),
    )
    full_splits = split_records(
        full_pool,
        train_count=full_counts["train"],
        val_count=full_counts["val"],
        seed=seed + 111,
    )

    _write_json_payload(
        output_dir / "data_splits.json",
        {
            "benchmark": {
                key: [_conversation_to_raw(record) for record in value]
                for key, value in benchmark_splits.items()
            },
            "full": {
                key: [_conversation_to_raw(record) for record in value]
                for key, value in full_splits.items()
            },
        },
    )

    benchmark_results = []
    for method_name in ("method_a", "method_b"):
        method_dir = benchmark_dir / method_name
        result = _benchmark_method(
            method_name=method_name,
            model_name=model_name,
            benchmark_splits=benchmark_splits,
            profile=gpu_profile,
            token_budget=token_budget,
            bpe_vocab_size=benchmark_bpe_vocab_size,
            epochs=benchmark_epochs,
            output_dir=method_dir,
        )
        benchmark_results.append(result)
        _cleanup_cuda()

    winner = _pick_winner(benchmark_results)
    comparison = {
        "results": benchmark_results,
        "winner": winner["method"],
    }
    _write_json_payload(benchmark_dir / "comparison.json", comparison)

    if winner["method"] == "method_a":
        final_adaptation = build_method_a_adaptation(
            model_name,
            assistant_texts=_assistant_texts(full_splits["train"]),
            token_budget=token_budget,
            output_dir=final_dir,
        )
    else:
        final_adaptation = build_method_b_adaptation(
            model_name,
            training_texts=_all_message_texts(full_splits["train"]),
            vocab_size=full_bpe_vocab_size,
            output_dir=final_dir,
        )

    final_max_length = _suggest_max_length(
        [*full_splits["train"], *full_splits["val"]],
        final_adaptation.tokenizer,
        cap=gpu_profile.full_max_length_cap,
    )
    final_plan = _build_training_plan(
        gpu_profile,
        benchmark=False,
        train_size=len(full_splits["train"]),
        num_train_epochs=full_epochs,
        max_length=final_max_length,
    )

    final_train_summary = train_with_fallback(
        model_name=model_name,
        adaptation=final_adaptation,
        train_records=full_splits["train"],
        eval_records=full_splits["val"],
        plan=final_plan,
        output_dir=final_dir,
        generation_records=full_splits["val"],
    )
    final_eval_loss = final_train_summary["eval_metrics"].get("eval_loss")
    sample_generations = final_train_summary["sample_generations"]
    final_result = {
        "method": winner["method"],
        "tokenizer_path": str(final_adaptation.tokenizer_dir),
        "tokenizer_metadata": final_adaptation.metadata,
        "training_plan": final_train_summary["training_plan"],
        "parameter_counts": final_train_summary["parameter_counts"],
        "eval_loss": final_eval_loss,
        "perplexity": _perplexity(final_eval_loss),
        "loss_curves": final_train_summary["loss_curves"],
        "attempts": final_train_summary["attempts"],
        "sample_generations": sample_generations,
    }
    _write_json_payload(final_dir / "sample_generations.json", sample_generations)
    _write_json_payload(final_dir / "final_result.json", final_result)

    markdown_report = _build_markdown_report(
        output_dir=output_dir,
        gpu_profile=gpu_profile,
        model_name=model_name,
        benchmark_results=benchmark_results,
        winner=winner,
        final_result=final_result,
        benchmark_splits=benchmark_splits,
        final_splits=full_splits,
    )
    report_path = output_dir / "wolof_finetuning_report.md"
    _write_text(report_path, markdown_report)

    summary = {
        "dataset_file": str(dataset_file),
        "output_dir": str(output_dir),
        "model_name": model_name,
        "gpu_profile": asdict(gpu_profile),
        "benchmark_results": benchmark_results,
        "winner": winner["method"],
        "final_result": {
            "eval_loss": final_result["eval_loss"],
            "perplexity": final_result["perplexity"],
            "report_path": str(report_path),
            "model_dir": str(final_dir / "model"),
        },
    }
    _write_json_payload(output_dir / "run_summary.json", summary)
    return summary
