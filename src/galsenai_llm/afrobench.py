from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from .io import ensure_parent, write_json

AFROBENCH_WOLOF_TASK_TEMPLATES: dict[str, str] = {
    "afrimmlu": "afrimmlu_wol_prompt_{variant}",
    "afrixnli": "afrixnli_wol_prompt_{variant}",
    "belebele": "belebele_wol_prompt_{variant}",
    "afriqa": "afriqa_wol_prompt_{variant}",
    "sib": "sib_wol_prompt_{variant}",
    "afrimgsm_direct": "afrimgsm_direct_wol",
    "afrimgsm_translate": "afrimgsm_translate_wol",
    "afrimgsm_en_cot": "afrimgsm_en_cot_wol",
    "afrimmlu_math": "afrimmlu_math_wol",
}

DEFAULT_AFROBENCH_WOLOF_TASKS: tuple[str, ...] = (
    "afrimmlu",
    "afrixnli",
    "belebele",
)

PRIMARY_METRIC_PRIORITY: tuple[str, ...] = (
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
    "f1,none",
    "bleu,none",
    "chrf,none",
    "ter,none",
)

PROMPT_SUFFIX_PATTERN = re.compile(r"_prompt_(\d+)$")


@dataclass(slots=True)
class AfroBenchModelSpec:
    pretrained_model: str
    tokenizer_path: str | None
    adapter_path: str | None
    base_model_name: str | None
    model_kind: str
    source_reference: str


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")
    return payload


def _detect_base_model_name(reference: Path | str) -> str | None:
    try:
        from peft import PeftConfig
    except ImportError:
        return None

    try:
        config = PeftConfig.from_pretrained(str(reference))
    except Exception:
        return None

    base_model_name = getattr(config, "base_model_name_or_path", None)
    if isinstance(base_model_name, str) and base_model_name.strip():
        return base_model_name.strip()
    return None


def _parse_csv_items(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _parse_prompt_variants(raw_value: str | None) -> list[int]:
    if raw_value is None:
        return [1, 2, 3, 4, 5]
    variants = []
    for item in _parse_csv_items(raw_value):
        variants.append(int(item))
    if not variants:
        raise ValueError("At least one prompt variant is required.")
    invalid = [variant for variant in variants if variant < 1 or variant > 5]
    if invalid:
        raise ValueError(f"Prompt variants must be between 1 and 5. Invalid values: {invalid}")
    return variants


def _expand_afrobench_tasks(task_names: list[str], prompt_variants: list[int]) -> list[str]:
    expanded_tasks: list[str] = []
    for task_name in task_names:
        template = AFROBENCH_WOLOF_TASK_TEMPLATES.get(task_name)
        if template is None:
            valid_names = ", ".join(sorted(AFROBENCH_WOLOF_TASK_TEMPLATES))
            raise ValueError(
                f"Unsupported AfroBench task '{task_name}'. Valid values: {valid_names}"
            )
        if "{variant}" in template:
            expanded_tasks.extend(
                template.format(variant=variant)
                for variant in prompt_variants
            )
        else:
            expanded_tasks.append(template)
    return expanded_tasks


def _pick_primary_metric(metrics: dict[str, Any]) -> tuple[str | None, float | None]:
    for metric_name in PRIMARY_METRIC_PRIORITY:
        metric_value = metrics.get(metric_name)
        if isinstance(metric_value, (int, float)):
            return metric_name, float(metric_value)

    for metric_name, metric_value in metrics.items():
        if metric_name == "alias" or metric_name.endswith("_stderr"):
            continue
        if isinstance(metric_value, (int, float)):
            return metric_name, float(metric_value)

    return None, None


def _infer_suite_task_name(task_name: str) -> str:
    for suite_task, template in AFROBENCH_WOLOF_TASK_TEMPLATES.items():
        if "{variant}" in template:
            prefix = template.split("{variant}", 1)[0]
            if task_name.startswith(prefix):
                return suite_task
        elif task_name == template:
            return suite_task
    return task_name


def _infer_prompt_variant(task_name: str) -> int | None:
    match = PROMPT_SUFFIX_PATTERN.search(task_name)
    if match is None:
        return None
    return int(match.group(1))


def _normalize_reference_path(value: str | Path) -> str:
    path = value if isinstance(value, Path) else Path(value).expanduser()
    return str(path.resolve())


def _normalize_model_reference(value: str) -> str:
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    return value


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    if match is None:
        return 10**12
    return int(match.group(1))


def _resolve_epoch_checkpoint_dir(run_dir: Path, checkpoint: dict[str, Any]) -> Path | None:
    checkpoint_name = checkpoint.get("checkpoint_name")
    if isinstance(checkpoint_name, str) and checkpoint_name.strip():
        candidate = (run_dir / "final" / "model" / checkpoint_name.strip()).resolve()
        if candidate.exists():
            return candidate

    checkpoint_dir = checkpoint.get("checkpoint_dir")
    if isinstance(checkpoint_dir, str) and checkpoint_dir.strip():
        candidate = Path(checkpoint_dir).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    return None


def _load_epoch_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    epoch_summary_path = run_dir / "final" / "epoch_checkpoints.json"
    checkpoints: list[dict[str, Any]] = []
    if epoch_summary_path.exists():
        payload = _load_json_object(epoch_summary_path)
        raw_checkpoints = payload.get("checkpoints")
        if isinstance(raw_checkpoints, list):
            for item in raw_checkpoints:
                if isinstance(item, dict):
                    resolved_dir = _resolve_epoch_checkpoint_dir(run_dir, item)
                    if resolved_dir is None:
                        continue
                    epoch_raw = item.get("epoch")
                    epoch = (
                        int(epoch_raw)
                        if isinstance(epoch_raw, (int, float))
                        else len(checkpoints) + 1
                    )
                    checkpoint = {
                        **item,
                        "epoch": epoch,
                        "checkpoint_dir": str(resolved_dir),
                    }
                    checkpoints.append(checkpoint)

    if checkpoints:
        checkpoints.sort(
            key=lambda item: (
                int(item.get("epoch", 0)),
                _checkpoint_step(Path(item["checkpoint_dir"])),
            )
        )
        return checkpoints

    checkpoint_dirs = sorted(
        (
            path
            for path in (run_dir / "final" / "model").iterdir()
            if path.is_dir() and path.name.startswith("checkpoint-")
        ),
        key=_checkpoint_step,
    )
    for index, checkpoint_dir in enumerate(checkpoint_dirs, start=1):
        checkpoints.append(
            {
                "epoch": index,
                "checkpoint_name": checkpoint_dir.name,
                "checkpoint_dir": str(checkpoint_dir.resolve()),
                "tokenizer_path": str(checkpoint_dir.resolve()),
            }
        )
    return checkpoints


def _resolve_model_spec(
    *,
    model_path: str | None,
    run_dir: Path | None,
    adapter_path: Path | None,
    tokenizer_path: Path | None,
    base_model_name: str | None,
) -> AfroBenchModelSpec:
    resolved_run_dir = run_dir.resolve() if run_dir is not None else None
    resolved_adapter_path = (
        Path(_normalize_reference_path(adapter_path)) if adapter_path is not None else None
    )
    resolved_tokenizer_path = (
        Path(_normalize_reference_path(tokenizer_path)) if tokenizer_path is not None else None
    )

    if resolved_run_dir is not None:
        final_model_dir = resolved_run_dir / "final" / "model"
        final_tokenizer_dir = resolved_run_dir / "final" / "tokenizer"
        if not final_model_dir.exists():
            raise ValueError(f"Could not find a final model directory at {final_model_dir}.")

        inferred_base_model_name = _detect_base_model_name(final_model_dir)
        if inferred_base_model_name is not None:
            tokenizer_ref = (
                resolved_tokenizer_path
                or final_model_dir
                or (final_tokenizer_dir if final_tokenizer_dir.exists() else None)
            )
            resolved_base_model_name = base_model_name or inferred_base_model_name
            if resolved_base_model_name is None:
                raise ValueError(
                    "Could not determine the base model from the run directory "
                    "adapter configuration."
                )
            return AfroBenchModelSpec(
                pretrained_model=resolved_base_model_name,
                tokenizer_path=None if tokenizer_ref is None else str(tokenizer_ref),
                adapter_path=str(final_model_dir),
                base_model_name=resolved_base_model_name,
                model_kind="adapter",
                source_reference=str(resolved_run_dir),
            )

        tokenizer_ref = (
            resolved_tokenizer_path
            or (final_model_dir if (final_model_dir / "tokenizer.json").exists() else None)
            or (final_tokenizer_dir if final_tokenizer_dir.exists() else None)
        )
        return AfroBenchModelSpec(
            pretrained_model=str(final_model_dir),
            tokenizer_path=None if tokenizer_ref is None else str(tokenizer_ref),
            adapter_path=None,
            base_model_name=base_model_name,
            model_kind="full_model",
            source_reference=str(resolved_run_dir),
        )

    if model_path is None:
        raise ValueError("Provide either --run-dir or --model-path.")

    normalized_model_path = _normalize_model_reference(model_path)
    inferred_base_model_name = _detect_base_model_name(normalized_model_path)
    if resolved_adapter_path is not None:
        tokenizer_ref = resolved_tokenizer_path or resolved_adapter_path
        resolved_base_model_name = base_model_name or _detect_base_model_name(resolved_adapter_path)
        if resolved_base_model_name is None:
            raise ValueError(
                "Adapter evaluation requires --base-model-name or an adapter_config.json with "
                "base_model_name_or_path."
            )
        return AfroBenchModelSpec(
            pretrained_model=resolved_base_model_name,
            tokenizer_path=str(tokenizer_ref),
            adapter_path=str(resolved_adapter_path),
            base_model_name=resolved_base_model_name,
            model_kind="adapter",
            source_reference=str(resolved_adapter_path),
        )

    if inferred_base_model_name is not None:
        tokenizer_ref = (
            str(resolved_tokenizer_path)
            if resolved_tokenizer_path is not None
            else normalized_model_path
        )
        resolved_base_model_name = base_model_name or inferred_base_model_name
        return AfroBenchModelSpec(
            pretrained_model=resolved_base_model_name,
            tokenizer_path=tokenizer_ref,
            adapter_path=normalized_model_path,
            base_model_name=resolved_base_model_name,
            model_kind="adapter",
            source_reference=normalized_model_path,
        )

    return AfroBenchModelSpec(
        pretrained_model=normalized_model_path,
        tokenizer_path=None if resolved_tokenizer_path is None else str(resolved_tokenizer_path),
        adapter_path=None,
        base_model_name=base_model_name,
        model_kind="full_model",
        source_reference=normalized_model_path,
    )


def _build_model_args(
    *,
    model_spec: AfroBenchModelSpec,
    dtype: str | None,
    load_in_4bit: bool,
    trust_remote_code: bool,
    use_fast_tokenizer: bool,
    add_bos_token: bool,
) -> str:
    model_args = [f"pretrained={model_spec.pretrained_model}"]

    if model_spec.tokenizer_path:
        model_args.append(f"tokenizer={model_spec.tokenizer_path}")
    if model_spec.adapter_path:
        model_args.append(f"peft={model_spec.adapter_path}")
    if dtype and dtype not in {"", "auto", "none"}:
        model_args.append(f"dtype={dtype}")
    if load_in_4bit:
        model_args.append("load_in_4bit=True")
    if trust_remote_code:
        model_args.append("trust_remote_code=True")
    if not use_fast_tokenizer:
        model_args.append("use_fast_tokenizer=False")
    if add_bos_token:
        model_args.append("add_bos_token=True")

    return ",".join(model_args)


def _write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")


def _patch_lm_eval_transformers_compatibility() -> bool:
    import transformers

    if hasattr(transformers, "AutoModelForVision2Seq"):
        return False
    if not hasattr(transformers, "AutoModelForImageTextToText"):
        return False

    spec = importlib.util.find_spec("lm_eval")
    if spec is None or spec.origin is None:
        return False

    target_path = Path(spec.origin).resolve().parent / "models" / "hf_vlms.py"
    if not target_path.exists():
        return False

    original_line = "    AUTO_MODEL_CLASS = transformers.AutoModelForVision2Seq\n"
    patched_line = (
        "    AUTO_MODEL_CLASS = getattr(\n"
        '        transformers, "AutoModelForVision2Seq", transformers.AutoModelForImageTextToText\n'
        "    )\n"
    )

    source_text = target_path.read_text(encoding="utf-8")
    if patched_line in source_text:
        return False
    if original_line not in source_text:
        return False

    target_path.write_text(source_text.replace(original_line, patched_line), encoding="utf-8")
    return True


def _summarize_task_results(
    *,
    tasks: list[str],
    raw_results: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result_rows: list[dict[str, Any]] = []
    suite_groups: dict[str, list[float]] = {}
    suite_metrics: dict[str, str] = {}

    task_results = raw_results.get("results", {})
    for task_name in tasks:
        metrics = task_results.get(task_name, {})
        primary_metric_name, primary_metric_value = _pick_primary_metric(metrics)
        suite_task = _infer_suite_task_name(task_name)
        prompt_variant = _infer_prompt_variant(task_name)
        row = {
            "task": task_name,
            "suite_task": suite_task,
            "prompt_variant": prompt_variant,
            "primary_metric": primary_metric_name,
            "primary_score": primary_metric_value,
            "metrics": metrics,
        }
        result_rows.append(row)
        if primary_metric_name is None or primary_metric_value is None:
            continue
        suite_groups.setdefault(suite_task, []).append(primary_metric_value)
        suite_metrics.setdefault(suite_task, primary_metric_name)

    aggregate_rows: list[dict[str, Any]] = []
    for suite_task in sorted(suite_groups):
        aggregate_rows.append(
            {
                "suite_task": suite_task,
                "primary_metric": suite_metrics[suite_task],
                "mean_primary_score": mean(suite_groups[suite_task]),
                "num_task_variants": len(suite_groups[suite_task]),
            }
        )

    return result_rows, aggregate_rows


def _render_report_markdown(
    *,
    model_spec: AfroBenchModelSpec,
    task_names: list[str],
    prompt_variants: list[int],
    batch_size: str,
    num_fewshot: int,
    limit: float | None,
    result_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    raw_results_path: Path,
) -> str:
    task_lines = "\n".join(f"- `{task_name}`" for task_name in task_names)
    per_task_lines = "\n".join(
        (
            f"| `{row['task']}` | `{row['suite_task']}` | "
            f"{row['prompt_variant'] if row['prompt_variant'] is not None else '-'} | "
            f"`{row['primary_metric'] or 'n/a'}` | "
            f"{'n/a' if row['primary_score'] is None else format(row['primary_score'], '.4f')} |"
        )
        for row in result_rows
    )
    aggregate_lines = "\n".join(
        (
            f"| `{row['suite_task']}` | `{row['primary_metric']}` | "
            f"{row['mean_primary_score']:.4f} | {row['num_task_variants']} |"
        )
        for row in aggregate_rows
    )

    limit_text = "full task datasets" if limit is None else str(limit)
    return f"""# Wolof AfroBench Benchmark

## Run Setup

- Source reference: `{model_spec.source_reference}`
- Model kind: `{model_spec.model_kind}`
- Base model: `{model_spec.base_model_name or model_spec.pretrained_model}`
- Pretrained model reference passed to `lm_eval`: `{model_spec.pretrained_model}`
- Tokenizer path: `{model_spec.tokenizer_path or model_spec.pretrained_model}`
- Adapter path: `{model_spec.adapter_path or 'none'}`
- Prompt variants: `{", ".join(str(variant) for variant in prompt_variants)}`
- Few-shot examples: `{num_fewshot}`
- Batch size: `{batch_size}`
- Limit per task: `{limit_text}`

## Evaluated Tasks

{task_lines}

## Per-Task Results

| Task | Suite Task | Prompt | Primary Metric | Score |
| --- | --- | --- | --- | --- |
{per_task_lines if per_task_lines else '| n/a | n/a | n/a | n/a | n/a |'}

## Aggregated Results

| Suite Task | Primary Metric | Mean Score | Variants |
| --- | --- | --- | --- |
{aggregate_lines if aggregate_lines else '| n/a | n/a | n/a | n/a |'}

## Raw Output

- Raw `lm_eval` output: `{raw_results_path}`
"""


def _render_epoch_report_markdown(
    *,
    run_dir: Path,
    tasks: list[str],
    prompt_variants: list[int],
    limit: float | None,
    batch_size: str,
    epoch_results: list[dict[str, Any]],
    summary_path: Path,
) -> str:
    task_lines = "\n".join(f"- `{task_name}`" for task_name in tasks)
    rows = []
    for item in epoch_results:
        score = item.get("overall_mean_primary_score")
        score_text = "n/a" if score is None else format(score, ".4f")
        rows.append(
            "| {epoch} | `{checkpoint}` | {score} | `{summary}` | `{report}` |".format(
                epoch=item["epoch"],
                checkpoint=Path(item["checkpoint_dir"]).name,
                score=score_text,
                summary=item["summary_file"],
                report=item["report_file"],
            )
        )

    limit_text = "full task datasets" if limit is None else str(limit)
    return f"""# Wolof AfroBench Epoch Comparison

## Run Setup

- Run directory: `{run_dir}`
- Prompt variants: `{", ".join(str(variant) for variant in prompt_variants)}`
- Batch size: `{batch_size}`
- Limit per task: `{limit_text}`

## Evaluated Tasks

{task_lines}

## Epoch Results

| Epoch | Checkpoint | Mean Aggregate Score | Summary | Report |
| --- | --- | ---: | --- | --- |
{chr(10).join(rows) if rows else '| n/a | n/a | n/a | n/a | n/a |'}

## Raw Summary

- Aggregated JSON summary: `{summary_path}`
"""


def run_wolof_afrobench_benchmark(
    *,
    model_path: str | None = None,
    run_dir: Path | None = None,
    adapter_path: Path | None = None,
    tokenizer_path: Path | None = None,
    base_model_name: str | None = None,
    tasks: str | None = None,
    prompt_variants: str | None = None,
    output_dir: Path | None = None,
    limit: float | None = None,
    batch_size: str = "auto",
    max_batch_size: int | None = None,
    num_fewshot: int = 0,
    device: str | None = "cuda",
    dtype: str | None = "auto",
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    use_fast_tokenizer: bool = True,
    add_bos_token: bool = False,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    log_samples: bool = False,
    write_out: bool = False,
    bootstrap_iters: int = 0,
) -> dict[str, Any]:
    compatibility_patch_applied = _patch_lm_eval_transformers_compatibility()

    try:
        from lm_eval import evaluator
        from lm_eval.tasks import TaskManager
    except ImportError as exc:
        raise RuntimeError(
            "AfroBench benchmarking requires the TRC lm-evaluation-harness fork. "
            "Run `uv sync --extra train --extra dev` to install the project dependencies."
        ) from exc

    selected_tasks = _parse_csv_items(tasks) or list(DEFAULT_AFROBENCH_WOLOF_TASKS)
    selected_prompt_variants = _parse_prompt_variants(prompt_variants)
    expanded_tasks = _expand_afrobench_tasks(selected_tasks, selected_prompt_variants)

    model_spec = _resolve_model_spec(
        model_path=model_path,
        run_dir=run_dir,
        adapter_path=adapter_path,
        tokenizer_path=tokenizer_path,
        base_model_name=base_model_name,
    )
    model_args = _build_model_args(
        model_spec=model_spec,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        trust_remote_code=trust_remote_code,
        use_fast_tokenizer=use_fast_tokenizer,
        add_bos_token=add_bos_token,
    )

    if output_dir is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        if run_dir is not None:
            resolved_output_dir = run_dir.resolve() / "afrobench" / timestamp
        else:
            resolved_output_dir = Path("outputs/afrobench-wolof") / timestamp
    else:
        resolved_output_dir = output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    task_manager = TaskManager("ERROR")
    missing_tasks = [
        task_name for task_name in expanded_tasks if task_name not in task_manager.all_tasks
    ]
    if missing_tasks:
        raise ValueError(
            "The installed lm-evaluation-harness build does not expose these AfroBench tasks: "
            + ", ".join(missing_tasks)
        )

    raw_results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=expanded_tasks,
        task_manager=task_manager,
        num_fewshot=num_fewshot if num_fewshot > 0 else None,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        device=device,
        limit=limit,
        log_samples=log_samples,
        write_out=write_out,
        bootstrap_iters=bootstrap_iters,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
    )

    raw_results_path = resolved_output_dir / "raw_results.json"
    _write_json_payload(raw_results_path, raw_results)

    result_rows, aggregate_rows = _summarize_task_results(
        tasks=expanded_tasks,
        raw_results=raw_results,
    )

    summary = {
        "source_reference": model_spec.source_reference,
        "model_kind": model_spec.model_kind,
        "pretrained_model": model_spec.pretrained_model,
        "base_model_name": model_spec.base_model_name,
        "tokenizer_path": model_spec.tokenizer_path,
        "adapter_path": model_spec.adapter_path,
        "tasks": selected_tasks,
        "expanded_tasks": expanded_tasks,
        "prompt_variants": selected_prompt_variants,
        "batch_size": batch_size,
        "max_batch_size": max_batch_size,
        "num_fewshot": num_fewshot,
        "device": device,
        "dtype": dtype,
        "load_in_4bit": load_in_4bit,
        "trust_remote_code": trust_remote_code,
        "apply_chat_template": apply_chat_template,
        "fewshot_as_multiturn": fewshot_as_multiturn,
        "limit": limit,
        "bootstrap_iters": bootstrap_iters,
        "compatibility_patch_applied": compatibility_patch_applied,
        "task_results": result_rows,
        "aggregate_results": aggregate_rows,
        "raw_results_path": str(raw_results_path),
        "output_dir": str(resolved_output_dir),
        "model_args": model_args,
    }
    summary_path = resolved_output_dir / "summary.json"
    write_json(summary_path, summary)

    report_markdown = _render_report_markdown(
        model_spec=model_spec,
        task_names=expanded_tasks,
        prompt_variants=selected_prompt_variants,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        limit=limit,
        result_rows=result_rows,
        aggregate_rows=aggregate_rows,
        raw_results_path=raw_results_path,
    )
    report_path = resolved_output_dir / "report.md"
    ensure_parent(report_path)
    report_path.write_text(report_markdown, encoding="utf-8")

    return {
        "source_reference": model_spec.source_reference,
        "model_kind": model_spec.model_kind,
        "pretrained_model": model_spec.pretrained_model,
        "base_model_name": model_spec.base_model_name,
        "tokenizer_path": model_spec.tokenizer_path,
        "adapter_path": model_spec.adapter_path,
        "tasks": selected_tasks,
        "expanded_tasks": expanded_tasks,
        "prompt_variants": selected_prompt_variants,
        "batch_size": batch_size,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "device": device,
        "dtype": dtype,
        "load_in_4bit": load_in_4bit,
        "compatibility_patch_applied": compatibility_patch_applied,
        "output_dir": str(resolved_output_dir),
        "summary_file": str(summary_path),
        "report_file": str(report_path),
        "aggregate_results": aggregate_rows,
    }


def run_wolof_afrobench_epoch_benchmarks(
    *,
    run_dir: Path,
    tasks: str | None = None,
    prompt_variants: str | None = None,
    output_dir: Path | None = None,
    limit: float | None = None,
    batch_size: str = "auto",
    max_batch_size: int | None = None,
    num_fewshot: int = 0,
    device: str | None = "cuda",
    dtype: str | None = "auto",
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    use_fast_tokenizer: bool = True,
    add_bos_token: bool = False,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    log_samples: bool = False,
    write_out: bool = False,
    bootstrap_iters: int = 0,
) -> dict[str, Any]:
    resolved_run_dir = run_dir.resolve()
    checkpoints = _load_epoch_checkpoints(resolved_run_dir)
    if not checkpoints:
        raise ValueError(
            f"No epoch checkpoints were found under {resolved_run_dir / 'final' / 'model'}."
        )

    selected_tasks = _parse_csv_items(tasks) or list(DEFAULT_AFROBENCH_WOLOF_TASKS)
    selected_prompt_variants = _parse_prompt_variants(prompt_variants)
    resolved_output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else resolved_run_dir / "afrobench" / "epoch-checkpoints"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    epoch_results: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        epoch = int(checkpoint["epoch"])
        checkpoint_dir = Path(checkpoint["checkpoint_dir"]).resolve()
        checkpoint_output_dir = resolved_output_dir / f"epoch-{epoch}"
        result = run_wolof_afrobench_benchmark(
            model_path=str(checkpoint_dir),
            run_dir=None,
            adapter_path=None,
            tokenizer_path=checkpoint_dir,
            base_model_name=None,
            tasks=tasks,
            prompt_variants=prompt_variants,
            output_dir=checkpoint_output_dir,
            limit=limit,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            num_fewshot=num_fewshot,
            device=device,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
            add_bos_token=add_bos_token,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            log_samples=log_samples,
            write_out=write_out,
            bootstrap_iters=bootstrap_iters,
        )
        aggregate_results = result.get("aggregate_results", [])
        aggregate_scores = [
            float(item["mean_primary_score"])
            for item in aggregate_results
            if isinstance(item, dict) and isinstance(item.get("mean_primary_score"), (int, float))
        ]
        overall_mean_primary_score = mean(aggregate_scores) if aggregate_scores else None
        epoch_results.append(
            {
                "epoch": epoch,
                "checkpoint_dir": str(checkpoint_dir),
                "summary_file": result["summary_file"],
                "report_file": result["report_file"],
                "aggregate_results": aggregate_results,
                "overall_mean_primary_score": overall_mean_primary_score,
                "eval_loss": checkpoint.get("eval_loss"),
                "perplexity": checkpoint.get("perplexity"),
            }
        )

    best_epoch = None
    best_epoch_selection = (
        "highest overall_mean_primary_score, then lowest perplexity, "
        "then lowest eval_loss, then latest epoch"
    )
    ranked_epochs = [
        item
        for item in epoch_results
        if isinstance(item.get("overall_mean_primary_score"), (int, float))
    ]
    if ranked_epochs:
        def _epoch_rank_key(item: dict[str, Any]) -> tuple[float, float, float, int]:
            perplexity = item.get("perplexity")
            eval_loss = item.get("eval_loss")
            return (
                float(item["overall_mean_primary_score"]),
                -float(perplexity)
                if isinstance(perplexity, (int, float))
                else float("-inf"),
                -float(eval_loss)
                if isinstance(eval_loss, (int, float))
                else float("-inf"),
                int(item["epoch"]),
            )

        best_epoch = max(
            ranked_epochs,
            key=_epoch_rank_key,
        )["epoch"]

    summary = {
        "run_dir": str(resolved_run_dir),
        "tasks": selected_tasks,
        "prompt_variants": selected_prompt_variants,
        "batch_size": batch_size,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "device": device,
        "dtype": dtype,
        "load_in_4bit": load_in_4bit,
        "epoch_results": epoch_results,
        "best_epoch": best_epoch,
        "best_epoch_selection": best_epoch_selection,
        "output_dir": str(resolved_output_dir),
    }
    summary_path = resolved_output_dir / "summary.json"
    write_json(summary_path, summary)

    report_markdown = _render_epoch_report_markdown(
        run_dir=resolved_run_dir,
        tasks=selected_tasks,
        prompt_variants=selected_prompt_variants,
        limit=limit,
        batch_size=batch_size,
        epoch_results=epoch_results,
        summary_path=summary_path,
    )
    report_path = resolved_output_dir / "report.md"
    ensure_parent(report_path)
    report_path.write_text(report_markdown, encoding="utf-8")

    return {
        "run_dir": str(resolved_run_dir),
        "epoch_checkpoint_count": len(checkpoints),
        "best_epoch": best_epoch,
        "best_epoch_selection": best_epoch_selection,
        "output_dir": str(resolved_output_dir),
        "summary_file": str(summary_path),
        "report_file": str(report_path),
        "epoch_results": epoch_results,
    }
