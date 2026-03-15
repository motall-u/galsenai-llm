from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from .io import ensure_parent, write_json


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return
    ensure_parent(dst)
    shutil.copy2(src, dst)


def _relative_path(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _detect_model_kind(model_dir: Path) -> str:
    if (model_dir / "adapter_config.json").exists():
        return "adapter"
    return "full_model"


def _resolve_base_model_name(run_summary: dict[str, Any], model_dir: Path) -> str | None:
    adapter_config_path = model_dir / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_config = _load_json(adapter_config_path)
        base_model_name = adapter_config.get("base_model_name_or_path")
        if isinstance(base_model_name, str) and base_model_name.strip():
            return base_model_name.strip()
    model_name = run_summary.get("model_name")
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    return None


def _build_model_card(
    *,
    repo_id: str,
    run_summary: dict[str, Any],
    model_kind: str,
    uploaded_model_source: str,
    include_benchmark: bool,
    include_report: bool,
    include_sample_generations: bool,
) -> str:
    repo_name = repo_id.split("/")[-1]
    base_model_name = run_summary.get("model_name", "unknown")
    winner = run_summary.get("winner", "unknown")
    gpu_profile = run_summary.get("gpu_profile", {})
    training_strategy = gpu_profile.get("training_strategy", "unknown")
    final_result = run_summary.get("final_result", {})
    final_perplexity = final_result.get("perplexity")
    final_eval_loss = final_result.get("eval_loss")
    library_name = "peft" if model_kind == "adapter" else "transformers"
    strategy_tag = "peft" if model_kind == "adapter" else "full-finetuning"
    repo_slug = repo_id.replace("/", "--")
    final_perplexity_text = "n/a" if final_perplexity is None else f"{final_perplexity:.4f}"
    final_eval_loss_text = "n/a" if final_eval_loss is None else f"{final_eval_loss:.4f}"

    included_sections = [
        "- model files at the repository root",
        "- `artifacts/final_result.json`",
        "- `artifacts/run_summary.json`",
    ]
    if include_benchmark:
        included_sections.append("- `benchmark/` tokenizer benchmark results")
        included_sections.append("- `afrobench/` downstream Wolof benchmark results when available")
    if include_report:
        included_sections.append("- `artifacts/wolof_finetuning_report.md`")
    if include_sample_generations:
        included_sections.append("- `artifacts/sample_generations.json`")

    return f"""---
base_model: {base_model_name}
library_name: {library_name}
pipeline_tag: text-generation
language:
- wo
tags:
- wolof
- qwen2.5
- text-generation
- {strategy_tag}
---

# {repo_name}

This repository contains a Wolof fine-tune of `{base_model_name}` produced with `galsenai-llm`.

## Training Summary

- Base model: `{base_model_name}`
- Final model type: `{model_kind}`
- Uploaded model source: `{uploaded_model_source}`
- Training strategy: `{training_strategy}`
- Winning tokenizer method: `{winner}`
- Final validation loss: `{final_eval_loss_text}`
- Final validation perplexity: `{final_perplexity_text}`

## Included Artifacts

{chr(10).join(included_sections)}

## CLI Usage

Run inference directly from the Hub:

```bash
uv run galsenai wolof infer --model-path {repo_id} --prompt "Nanga def?"
```

Start an interactive chat session:

```bash
uv run galsenai wolof chat --model-path {repo_id}
```

Download the repository locally:

```bash
uv run galsenai wolof download --repo-id {repo_id} --local-dir models/{repo_slug}
```

Run inference from the downloaded local directory:

```bash
uv run galsenai wolof infer --model-path models/{repo_slug} --prompt "Nanga def?"
```

## Notes

- If this repository contains a PEFT adapter, the CLI loader will read the
  adapter configuration and load the correct base model automatically.
- Benchmark comparison results are stored separately from the model weights
  so you can inspect tokenizer quality without opening the training run
  directory.
"""


def _load_epoch_checkpoint_index(run_dir: Path) -> list[dict[str, Any]]:
    epoch_index_path = run_dir / "final" / "epoch_checkpoints.json"
    if not epoch_index_path.exists():
        return []
    payload = _load_json(epoch_index_path)
    checkpoints = payload.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        return []
    return [item for item in checkpoints if isinstance(item, dict)]


def _resolve_upload_model_dir(
    *,
    run_dir: Path,
    checkpoint_name: str | None,
    checkpoint_epoch: int | None,
) -> tuple[Path, str]:
    if checkpoint_name and checkpoint_epoch is not None:
        raise ValueError("Pass only one of checkpoint_name or checkpoint_epoch.")

    default_model_dir = run_dir / "final" / "model"
    if checkpoint_name is None and checkpoint_epoch is None:
        return default_model_dir, "final/model"

    if checkpoint_name is not None:
        candidate = default_model_dir / checkpoint_name
        if not candidate.exists():
            raise ValueError(f"Could not find checkpoint directory at {candidate}.")
        return candidate, checkpoint_name

    checkpoints = _load_epoch_checkpoint_index(run_dir)
    for checkpoint in checkpoints:
        if int(checkpoint.get("epoch", -1)) != int(checkpoint_epoch):
            continue
        checkpoint_dir = Path(str(checkpoint["checkpoint_dir"]))
        if checkpoint_dir.exists():
            return checkpoint_dir, checkpoint_dir.name
    raise ValueError(
        f"Could not resolve checkpoint for epoch {checkpoint_epoch} under "
        f"{run_dir / 'final' / 'epoch_checkpoints.json'}."
    )


def build_wolof_hub_bundle(
    *,
    run_dir: Path,
    target_dir: Path,
    repo_id: str,
    source_model_dir: Path | None = None,
    uploaded_model_source: str | None = None,
    include_benchmark: bool = True,
    include_report: bool = False,
    include_sample_generations: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    model_dir = (source_model_dir or (run_dir / "final" / "model")).resolve()
    run_summary_path = run_dir / "run_summary.json"
    final_result_path = run_dir / "final" / "final_result.json"

    if not model_dir.exists():
        raise ValueError(f"Could not find a final model directory at {model_dir}.")
    if not run_summary_path.exists():
        raise ValueError(f"Could not find run_summary.json in {run_dir}.")
    if not final_result_path.exists():
        raise ValueError(f"Could not find final/final_result.json in {run_dir}.")

    run_summary = _load_json(run_summary_path)
    model_kind = _detect_model_kind(model_dir)
    base_model_name = _resolve_base_model_name(run_summary, model_dir)
    model_source_label = uploaded_model_source or _relative_path(model_dir, run_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    included_paths: list[str] = []

    skipped_model_files = {
        "README.md",
        "trainer_state.json",
        "training_args.bin",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "epoch_summary.json",
    }
    for source_path in sorted(model_dir.iterdir()):
        if source_path.name in skipped_model_files:
            continue
        if source_path.name.startswith("checkpoint-"):
            continue
        destination = target_dir / source_path.name
        _copy_path(source_path, destination)
        included_paths.append(_relative_path(destination, target_dir))

    artifacts_dir = target_dir / "artifacts"
    for source_path, relative_destination in (
        (final_result_path, Path("artifacts/final_result.json")),
        (run_summary_path, Path("artifacts/run_summary.json")),
        (run_dir / "english_replay_dataset.json", Path("artifacts/english_replay_dataset.json")),
        (run_dir / "math_replay_dataset.json", Path("artifacts/math_replay_dataset.json")),
        (run_dir / "train_mix.json", Path("artifacts/train_mix.json")),
    ):
        destination = target_dir / relative_destination
        if not source_path.exists():
            continue
        _copy_path(source_path, destination)
        included_paths.append(_relative_path(destination, target_dir))

    if include_benchmark:
        benchmark_dir = run_dir / "benchmark"
        if benchmark_dir.exists():
            benchmark_files = [
                benchmark_dir / "comparison.json",
                benchmark_dir / "method_a" / "benchmark_result.json",
                benchmark_dir / "method_a" / "training_summary.json",
                benchmark_dir / "method_a" / "oom_retries.json",
                benchmark_dir / "method_a" / "token_selection.json",
                benchmark_dir / "method_b" / "benchmark_result.json",
                benchmark_dir / "method_b" / "training_summary.json",
                benchmark_dir / "method_b" / "oom_retries.json",
                benchmark_dir / "method_b" / "embedding_transfer_plan.json",
            ]
            for source_path in benchmark_files:
                if not source_path.exists():
                    continue
                destination = target_dir / _relative_path(source_path, run_dir)
                _copy_path(source_path, destination)
                included_paths.append(_relative_path(destination, target_dir))

        afrobench_dir = run_dir / "afrobench"
        if afrobench_dir.exists():
            for source_path in sorted(afrobench_dir.rglob("*")):
                if source_path.is_dir():
                    continue
                destination = target_dir / _relative_path(source_path, run_dir)
                _copy_path(source_path, destination)
                included_paths.append(_relative_path(destination, target_dir))

    if include_report:
        report_path = run_dir / "wolof_finetuning_report.md"
        if report_path.exists():
            destination = artifacts_dir / "wolof_finetuning_report.md"
            _copy_path(report_path, destination)
            included_paths.append(_relative_path(destination, target_dir))

    if include_sample_generations:
        sample_generations_path = run_dir / "final" / "sample_generations.json"
        if sample_generations_path.exists():
            destination = artifacts_dir / "sample_generations.json"
            _copy_path(sample_generations_path, destination)
            included_paths.append(_relative_path(destination, target_dir))

    metadata = {
        "repo_id": repo_id,
        "base_model_name": base_model_name,
        "model_kind": model_kind,
        "training_strategy": run_summary.get("gpu_profile", {}).get("training_strategy"),
        "winner": run_summary.get("winner"),
        "final_perplexity": run_summary.get("final_result", {}).get("perplexity"),
        "uploaded_model_source": model_source_label,
        "included_benchmark": include_benchmark,
        "included_report": include_report,
        "included_sample_generations": include_sample_generations,
        "source_run_dir": str(run_dir),
        "source_model_dir": str(model_dir),
        "included_paths": included_paths,
    }
    write_json(target_dir / "galsenai_wolof_metadata.json", metadata)

    model_card = _build_model_card(
        repo_id=repo_id,
        run_summary=run_summary,
        model_kind=model_kind,
        uploaded_model_source=model_source_label,
        include_benchmark=include_benchmark,
        include_report=include_report,
        include_sample_generations=include_sample_generations,
    )
    ensure_parent(target_dir / "README.md")
    (target_dir / "README.md").write_text(model_card, encoding="utf-8")

    return metadata


def _resolve_hf_token(token: str | None) -> str | None:
    return token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def upload_wolof_run_to_hub(
    *,
    run_dir: Path,
    repo_id: str,
    checkpoint_name: str | None = None,
    checkpoint_epoch: int | None = None,
    token: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
    include_benchmark: bool = True,
    include_report: bool = False,
    include_sample_generations: bool = False,
) -> dict[str, Any]:
    resolved_run_dir = run_dir.resolve()
    source_model_dir, uploaded_model_source = _resolve_upload_model_dir(
        run_dir=resolved_run_dir,
        checkpoint_name=checkpoint_name,
        checkpoint_epoch=checkpoint_epoch,
    )
    resolved_token = _resolve_hf_token(token)
    api = HfApi(token=resolved_token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="galsenai-hf-upload-") as temporary_directory:
        staging_dir = Path(temporary_directory) / "bundle"
        metadata = build_wolof_hub_bundle(
            run_dir=resolved_run_dir,
            target_dir=staging_dir,
            repo_id=repo_id,
            source_model_dir=source_model_dir,
            uploaded_model_source=uploaded_model_source,
            include_benchmark=include_benchmark,
            include_report=include_report,
            include_sample_generations=include_sample_generations,
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(staging_dir),
            commit_message=commit_message
            or f"Upload Wolof fine-tuning artifacts ({uploaded_model_source})",
            token=resolved_token,
        )

    return {
        "repo_id": repo_id,
        "repo_type": "model",
        "private": private,
        "repo_url": f"https://huggingface.co/{repo_id}",
        "run_dir": str(resolved_run_dir),
        "uploaded_model_source": uploaded_model_source,
        "source_model_dir": str(source_model_dir),
        "model_kind": metadata["model_kind"],
        "base_model_name": metadata["base_model_name"],
        "included_benchmark": include_benchmark,
        "included_report": include_report,
        "included_sample_generations": include_sample_generations,
    }


def download_wolof_repo_from_hub(
    *,
    repo_id: str,
    local_dir: Path | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    resolved_token = _resolve_hf_token(token)
    resolved_local_dir = (
        local_dir
        if local_dir is not None
        else Path("models") / repo_id.replace("/", "--")
    )
    resolved_local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        token=resolved_token,
        local_dir=str(resolved_local_dir),
    )
    return {
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": revision or "main",
        "local_dir": str(resolved_local_dir.resolve()),
        "snapshot_path": snapshot_path,
    }
