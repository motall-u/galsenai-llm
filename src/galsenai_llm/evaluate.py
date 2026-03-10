from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import EvaluationConfig
from .dataset import load_examples
from .io import load_jsonl, write_json
from .schemas import Message, PredictionRecord, ToolCall, first_assistant_message


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, sort_keys=True, ensure_ascii=True)
    return " ".join(text.lower().split())


def _tool_signature(tool_call: ToolCall) -> tuple[str, str]:
    arguments = json.dumps(
        tool_call.function.arguments,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return tool_call.function.name, arguments


def compare_assistant_messages(expected: Message, predicted: Message) -> dict[str, Any]:
    expected_tool_calls = expected.tool_calls or []
    predicted_tool_calls = predicted.tool_calls or []

    decision_match = bool(expected_tool_calls) == bool(predicted_tool_calls)
    tool_names_match = [call.function.name for call in expected_tool_calls] == [
        call.function.name for call in predicted_tool_calls
    ]
    tool_arguments_match = [_tool_signature(call) for call in expected_tool_calls] == [
        _tool_signature(call) for call in predicted_tool_calls
    ]
    content_match = _normalize_text(expected.content) == _normalize_text(predicted.content)
    overall_match = tool_arguments_match if expected_tool_calls else content_match

    return {
        "decision_match": decision_match,
        "tool_names_match": tool_names_match,
        "tool_arguments_match": tool_arguments_match,
        "content_match": content_match,
        "overall_match": overall_match,
    }


def _load_predictions(path: Path) -> dict[str, Message]:
    predictions: dict[str, Message] = {}
    for record in load_jsonl(path):
        parsed = PredictionRecord.model_validate(record)
        predictions[parsed.id] = parsed.assistant
    return predictions


def score_predictions(dataset_file: Path, predictions_file: Path) -> dict[str, Any]:
    examples = load_examples(dataset_file)
    predictions = _load_predictions(predictions_file)

    rows: list[dict[str, Any]] = []
    by_category: dict[str, list[bool]] = defaultdict(list)

    for example in examples:
        expected = first_assistant_message(example)
        predicted = predictions.get(example.id)
        if predicted is None:
            row = {
                "id": example.id,
                "category": example.category,
                "missing_prediction": True,
                "overall_match": False,
            }
            rows.append(row)
            by_category[example.category].append(False)
            continue

        comparison = compare_assistant_messages(expected, predicted)
        row = {
            "id": example.id,
            "category": example.category,
            **comparison,
        }
        rows.append(row)
        by_category[example.category].append(comparison["overall_match"])

    total = len(rows)
    matches = sum(1 for row in rows if row["overall_match"])

    return {
        "dataset_file": str(dataset_file),
        "predictions_file": str(predictions_file),
        "total_examples": total,
        "overall_accuracy": 0.0 if total == 0 else matches / total,
        "by_category": {
            category: {
                "count": len(results),
                "accuracy": 0.0 if not results else sum(results) / len(results),
            }
            for category, results in sorted(by_category.items())
        },
        "rows": rows,
    }


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    report = score_predictions(config.dataset_file, config.predictions_file)
    write_json(config.report_file, report)
    return report
