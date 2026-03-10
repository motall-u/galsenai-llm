from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .dataset import load_examples
from .evaluate import score_predictions
from .generation import build_generation_pipeline, generate_first_assistant
from .io import write_json, write_jsonl
from .schemas import PredictionRecord, first_assistant_message, prompt_messages


def _oracle_predictions(dataset_file: Path) -> list[PredictionRecord]:
    examples = load_examples(dataset_file)
    return [
        PredictionRecord(id=example.id, assistant=first_assistant_message(example))
        for example in examples
    ]


def _transformers_predictions(config: BenchmarkConfig) -> list[PredictionRecord]:
    if not config.model_path:
        raise ValueError("Benchmark mode=transformers requires model_path in the config.")

    examples = load_examples(config.dataset_file)
    pipe = build_generation_pipeline(
        model_path=config.model_path,
        adapter_path=None,
        base_model_name=None,
        device_map="auto",
        dtype=None,
    )

    predictions: list[PredictionRecord] = []
    for example in examples:
        tool_names = [tool.function.name for tool in example.tools]
        assistant = generate_first_assistant(
            pipe=pipe,
            messages=prompt_messages(example),
            generation=config.generation,
            tool_names=tool_names,
            tool_registry=config.tool_registry,
        )
        predictions.append(PredictionRecord(id=example.id, assistant=assistant))
    return predictions


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    if config.mode == "predictions":
        if config.predictions_file is None:
            raise ValueError("Benchmark mode=predictions requires predictions_file.")
        report = score_predictions(config.dataset_file, config.predictions_file)
        write_json(config.report_file, report)
        return report

    if config.mode == "oracle":
        predictions = _oracle_predictions(config.dataset_file)
    else:
        predictions = _transformers_predictions(config)

    temp_predictions_path = config.report_file.parent / "benchmark_predictions.jsonl"
    write_jsonl(
        temp_predictions_path,
        [prediction.model_dump(mode="json", exclude_none=True) for prediction in predictions],
    )
    report = score_predictions(config.dataset_file, temp_predictions_path)
    report["benchmark_mode"] = config.mode
    report["generated_predictions_file"] = str(temp_predictions_path)
    write_json(config.report_file, report)
    return report
