#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DATASET = REPO_ROOT / "data/wolof-dataset/curated_dataset.json"
DEFAULT_EXTENSION_DATASET = (
    REPO_ROOT / "data/wolof-data/gen-from-dict/dictionary_vocab_dataset.jsonl"
)
DEFAULT_OUTPUT_DATASET = REPO_ROOT / "data/wolof-dataset/curated_dataset_extended.json"
ALLOWED_ROLES = {"system", "user", "assistant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the Wolof curated conversation dataset with the dictionary-generated "
            "vocabulary dataset and save the result in the curated conversation format."
        )
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=DEFAULT_BASE_DATASET,
        help="Path to the existing curated Wolof JSON dataset.",
    )
    parser.add_argument(
        "--extension-dataset",
        type=Path,
        default=DEFAULT_EXTENSION_DATASET,
        help="Path to the dictionary-generated JSONL dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DATASET,
        help="Path where the merged curated dataset should be written.",
    )
    return parser.parse_args()


def _normalize_message(message: Any) -> dict[str, str] | None:
    if not isinstance(message, dict):
        return None
    role = str(message.get("role", "")).strip().lower()
    if role not in ALLOWED_ROLES:
        return None
    content = str(message.get("content", "")).strip()
    if not content:
        return None
    return {"role": role, "content": content}


def _canonical_signature(conversations: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{message['role']}::{message['content'].strip()}"
        for message in conversations
        if message.get("content", "").strip()
    )


def _validate_curated_record(item: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    conversation_id = str(item.get("conversation_id") or f"conv_{index}").strip()
    if not conversation_id:
        conversation_id = f"conv_{index}"
    conversations = []
    for raw_message in item.get("conversations", []):
        normalized = _normalize_message(raw_message)
        if normalized is not None:
            conversations.append(normalized)
    if len(conversations) < 2:
        return None
    if not any(message["role"] == "assistant" for message in conversations):
        return None
    return {
        "conversation_id": conversation_id,
        "conversations": conversations,
    }


def load_base_dataset(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        normalized = _validate_curated_record(item, index)
        if normalized is not None:
            records.append(normalized)
    if not records:
        raise ValueError(f"No usable curated conversations were found in {path}.")
    return records


def _convert_dictionary_example(item: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    conversation_id = str(item.get("id") or f"dict-vocab-{index:05d}").strip()
    if not conversation_id:
        conversation_id = f"dict-vocab-{index:05d}"

    conversations: list[dict[str, str]] = []
    for raw_message in item.get("messages", []):
        normalized = _normalize_message(raw_message)
        if normalized is None:
            continue
        if normalized["role"] == "system":
            continue
        conversations.append(normalized)

    if len(conversations) < 2:
        return None
    if conversations[0]["role"] != "user":
        return None
    if conversations[-1]["role"] != "assistant":
        return None
    if not any(message["role"] == "assistant" for message in conversations):
        return None

    return {
        "conversation_id": conversation_id,
        "conversations": conversations,
    }


def load_dictionary_dataset(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            normalized = _convert_dictionary_example(item, index)
            if normalized is not None:
                records.append(normalized)
    if not records:
        raise ValueError(f"No usable dictionary conversations were found in {path}.")
    return records


def merge_datasets(
    base_records: list[dict[str, Any]],
    extension_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    merged = list(base_records)
    seen_ids = {record["conversation_id"] for record in merged}
    seen_signatures = {_canonical_signature(record["conversations"]) for record in merged}

    added = 0
    skipped_duplicate_ids = 0
    skipped_duplicate_content = 0

    for record in extension_records:
        conversation_id = record["conversation_id"]
        signature = _canonical_signature(record["conversations"])
        if conversation_id in seen_ids:
            skipped_duplicate_ids += 1
            continue
        if signature in seen_signatures:
            skipped_duplicate_content += 1
            continue
        merged.append(record)
        seen_ids.add(conversation_id)
        seen_signatures.add(signature)
        added += 1

    stats = {
        "base_records": len(base_records),
        "extension_records": len(extension_records),
        "added_records": added,
        "skipped_duplicate_ids": skipped_duplicate_ids,
        "skipped_duplicate_content": skipped_duplicate_content,
        "merged_records": len(merged),
    }
    return merged, stats


def write_output(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    base_records = load_base_dataset(args.base_dataset)
    extension_records = load_dictionary_dataset(args.extension_dataset)
    merged_records, stats = merge_datasets(base_records, extension_records)
    write_output(args.output, merged_records)

    print(f"[done] base dataset: {args.base_dataset}")
    print(f"[done] extension dataset: {args.extension_dataset}")
    print(f"[done] output dataset: {args.output}")
    for key, value in stats.items():
        print(f"[done] {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
