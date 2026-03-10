from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .io import load_jsonl, write_jsonl
from .schemas import DatasetExample


def load_examples(path: Path) -> list[DatasetExample]:
    return [DatasetExample.model_validate(record) for record in load_jsonl(path)]


def dump_examples(path: Path, examples: list[DatasetExample]) -> None:
    write_jsonl(
        path,
        [example.model_dump(mode="json", exclude_none=True) for example in examples],
    )


def summarize_examples(examples: list[DatasetExample]) -> dict[str, Any]:
    category_counts = Counter(example.category for example in examples)
    tool_counts = Counter(
        tool.function.name for example in examples for tool in example.tools
    )
    return {
        "num_examples": len(examples),
        "categories": dict(category_counts),
        "tools": dict(tool_counts),
    }


def _check_example_semantics(
    example: DatasetExample,
    line: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check semantic rules that go beyond Pydantic structural validation."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def _err(message: str) -> None:
        errors.append({"id": example.id, "line": line, "message": message})

    def _warn(message: str) -> None:
        warnings.append({"id": example.id, "line": line, "message": message})

    available_tool_names = {tool.function.name for tool in example.tools}
    has_tool_calls = any(
        msg.tool_calls for msg in example.messages if msg.role == "assistant"
    )

    # Category vs tools consistency
    if example.category == "no_tool":
        if example.tools:
            _warn("no_tool example defines tools (they will be ignored).")
        if has_tool_calls:
            _err("no_tool example contains assistant tool_calls.")
    else:
        if not example.tools:
            _err(
                f"{example.category} example has no tool definitions in 'tools'."
            )
        if not has_tool_calls:
            _err(
                f"{example.category} example has no assistant tool_calls."
            )

    # Tool call names must reference defined tools
    for msg in example.messages:
        if msg.role != "assistant" or not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            if tc.function.name not in available_tool_names:
                _err(
                    f"Tool call '{tc.function.name}' not found in "
                    f"tools: {sorted(available_tool_names)}."
                )

    # Tool messages should reference a preceding tool call id
    issued_call_ids: set[str] = set()
    for msg in example.messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.id:
                    issued_call_ids.add(tc.id)
        if msg.role == "tool":
            if msg.tool_call_id and msg.tool_call_id not in issued_call_ids:
                _warn(
                    f"Tool message tool_call_id '{msg.tool_call_id}' "
                    f"does not match any preceding assistant tool call id."
                )
            if msg.name and msg.name not in available_tool_names:
                _warn(
                    f"Tool message name '{msg.name}' not in defined tools."
                )

    # First message should be system or user
    first_role = example.messages[0].role
    if first_role not in ("system", "user"):
        _warn(f"First message has role '{first_role}', expected 'system' or 'user'.")

    return errors, warnings


def validate_dataset(path: Path) -> dict[str, Any]:
    """Validate dataset structure and semantics, returning a detailed report."""
    raw_records = load_jsonl(path)
    all_errors: list[dict[str, Any]] = []
    all_warnings: list[dict[str, Any]] = []
    valid_examples: list[DatasetExample] = []
    seen_ids: dict[str, int] = {}

    for line_number, record in enumerate(raw_records, start=1):
        # Structural validation via Pydantic
        try:
            example = DatasetExample.model_validate(record)
        except ValidationError as exc:
            for err in exc.errors():
                loc = " -> ".join(str(part) for part in err["loc"])
                all_errors.append({
                    "id": record.get("id", "<unknown>"),
                    "line": line_number,
                    "message": f"{loc}: {err['msg']}",
                })
            continue

        # Duplicate ID check
        if example.id in seen_ids:
            all_errors.append({
                "id": example.id,
                "line": line_number,
                "message": (
                    f"Duplicate id '{example.id}' "
                    f"(first seen on line {seen_ids[example.id]})."
                ),
            })
        seen_ids[example.id] = line_number

        # Semantic checks
        errors, warnings = _check_example_semantics(example, line_number)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        valid_examples.append(example)

    summary = summarize_examples(valid_examples)
    return {
        "dataset_file": str(path),
        "valid": len(all_errors) == 0,
        "num_lines": len(raw_records),
        "num_valid": len(valid_examples),
        **summary,
        "errors": all_errors,
        "warnings": all_warnings,
    }
