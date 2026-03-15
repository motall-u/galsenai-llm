from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .config import (
    BenchmarkConfig,
    EvaluationConfig,
    InferenceConfig,
    MergeConfig,
    TrainProjectConfig,
)
from .dataset import validate_dataset
from .evaluate import run_evaluation

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
benchmark_app = typer.Typer(no_args_is_help=True)
wolof_app = typer.Typer(no_args_is_help=True)
console = Console()

app.add_typer(data_app, name="data")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(wolof_app, name="wolof")

DEFAULT_WOLOF_SYSTEM_PROMPT = (
    "You are a friendly, helpful multilingual assistant. "
    "Respond in the language requested by the user, or match the user's language "
    "when not specified. Follow the user's instructions closely and answer clearly "
    "and directly."
)


def _print_report(report: dict) -> None:
    console.print_json(data=report)


@data_app.command("validate")
def data_validate(dataset_file: Path) -> None:
    report = validate_dataset(dataset_file)

    # Print errors and warnings with context
    for error in report.get("errors", []):
        console.print(
            f"  [red]ERROR[/red] line {error['line']} "
            f"(id={error['id']}): {error['message']}"
        )
    for warning in report.get("warnings", []):
        console.print(
            f"  [yellow]WARN[/yellow]  line {warning['line']} "
            f"(id={warning['id']}): {warning['message']}"
        )

    # Print summary
    console.print()
    _print_report({
        "dataset_file": report["dataset_file"],
        "valid": report["valid"],
        "num_lines": report["num_lines"],
        "num_valid": report["num_valid"],
        "num_examples": report["num_examples"],
        "categories": report["categories"],
        "tools": report["tools"],
        "num_errors": len(report["errors"]),
        "num_warnings": len(report["warnings"]),
    })

    if not report["valid"]:
        raise typer.Exit(code=1)


@app.command()
def train(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    from .train import run_train

    _print_report(run_train(TrainProjectConfig.from_yaml(config)))


@app.command()
def merge(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    from .merge import run_merge

    _print_report(run_merge(MergeConfig.from_yaml(config)))


@app.command()
def evaluate(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    _print_report(run_evaluation(EvaluationConfig.from_yaml(config)))


@app.command()
def infer(
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    prompt: str | None = typer.Option(None, "--prompt"),
) -> None:
    from .infer import run_inference

    _print_report(run_inference(InferenceConfig.from_yaml(config), prompt_override=prompt))


@benchmark_app.command("run")
def benchmark_run(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    from .benchmark import run_benchmark

    _print_report(run_benchmark(BenchmarkConfig.from_yaml(config)))


@wolof_app.command("infer")
def wolof_infer(
    prompt: str = typer.Option(..., "--prompt"),
    model_path: str = typer.Option(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--model-path",
        help="Local model directory or Hugging Face repo id.",
    ),
    adapter_path: Path | None = typer.Option(None, "--adapter-path"),
    base_model_name: str | None = typer.Option(None, "--base-model-name"),
    system_prompt: str | None = typer.Option(
        DEFAULT_WOLOF_SYSTEM_PROMPT,
        "--system-prompt",
    ),
    device_map: str = typer.Option("auto", "--device-map"),
    dtype: str | None = typer.Option("bfloat16", "--dtype"),
    max_new_tokens: int = typer.Option(128, "--max-new-tokens"),
    do_sample: bool = typer.Option(False, "--do-sample"),
    temperature: float = typer.Option(0.0, "--temperature"),
    top_p: float = typer.Option(1.0, "--top-p"),
) -> None:
    from .infer import run_prompt_inference

    _print_report(
        run_prompt_inference(
            model_path=model_path,
            prompt=prompt,
            adapter_path=None if adapter_path is None else str(adapter_path),
            base_model_name=base_model_name,
            device_map=device_map,
            dtype=dtype,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
    )


@wolof_app.command("chat")
def wolof_chat(
    model_path: str = typer.Option(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--model-path",
        help="Local model directory or Hugging Face repo id.",
    ),
    adapter_path: Path | None = typer.Option(None, "--adapter-path"),
    base_model_name: str | None = typer.Option(None, "--base-model-name"),
    system_prompt: str | None = typer.Option(
        DEFAULT_WOLOF_SYSTEM_PROMPT,
        "--system-prompt",
    ),
    device_map: str = typer.Option("auto", "--device-map"),
    dtype: str | None = typer.Option("bfloat16", "--dtype"),
    max_new_tokens: int = typer.Option(128, "--max-new-tokens"),
    do_sample: bool = typer.Option(False, "--do-sample"),
    temperature: float = typer.Option(0.0, "--temperature"),
    top_p: float = typer.Option(1.0, "--top-p"),
    max_turns: int = typer.Option(0, "--max-turns"),
) -> None:
    from .infer import run_chat_session

    run_chat_session(
        model_path=model_path,
        adapter_path=None if adapter_path is None else str(adapter_path),
        base_model_name=base_model_name,
        device_map=device_map,
        dtype=dtype,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_turns=max_turns,
    )


@wolof_app.command("run")
def wolof_run(
    dataset_file: Path = typer.Option(
        Path("data/wolof-dataset/curated_dataset.json"),
        "--dataset-file",
        exists=True,
        readable=True,
    ),
    method_a_text_corpus_file: Path | None = typer.Option(
        None,
        "--method-a-text-corpus-file",
        help=(
            "Separate text-only Wolof corpus used only by Method A to compute "
            "word frequencies and the BPE fragmentation threshold."
        ),
    ),
    english_replay_dataset: str | None = typer.Option(
        None,
        "--english-replay-dataset",
        help=(
            "Optional English replay dataset mixed into the final training set "
            "to reduce catastrophic forgetting. Accepts a Hugging Face dataset "
            "repo id or a local .json/.jsonl conversation dataset file."
        ),
    ),
    english_replay_config: str | None = typer.Option(
        None,
        "--english-replay-config",
        help="Optional Hugging Face dataset config name for the English replay dataset.",
    ),
    english_replay_split: str = typer.Option(
        "train_sft",
        "--english-replay-split",
        help="Split name used when loading the English replay dataset from Hugging Face.",
    ),
    english_replay_sample_ratio: float = typer.Option(
        0.2,
        "--english-replay-sample-ratio",
        help=(
            "Replay sample size as a fraction of --full-samples. "
            "For example, 0.2 adds English replay equal to 20% of the Wolof full sample size."
        ),
    ),
    english_replay_cache_dir: Path | None = typer.Option(
        None,
        "--english-replay-cache-dir",
        help="Optional cache directory for Hugging Face English replay downloads.",
    ),
    math_replay_dataset: str | None = typer.Option(
        None,
        "--math-replay-dataset",
        help=(
            "Optional math replay dataset mixed into the final training set. "
            "Accepts a Hugging Face dataset repo id or a local "
            ".json/.jsonl conversation dataset file."
        ),
    ),
    math_replay_config: str | None = typer.Option(
        None,
        "--math-replay-config",
        help="Optional Hugging Face dataset config name for the math replay dataset.",
    ),
    math_replay_split: str = typer.Option(
        "train",
        "--math-replay-split",
        help="Split name used when loading the math replay dataset from Hugging Face.",
    ),
    math_replay_sample_ratio: float = typer.Option(
        0.0,
        "--math-replay-sample-ratio",
        help=(
            "Math replay sample size as a fraction of --full-samples "
            "when exact mix ratios are not used."
        ),
    ),
    math_replay_cache_dir: Path | None = typer.Option(
        None,
        "--math-replay-cache-dir",
        help="Optional cache directory for Hugging Face math replay downloads.",
    ),
    wolof_mix_ratio: float | None = typer.Option(
        None,
        "--wolof-mix-ratio",
        help=(
            "Exact target ratio for Wolof in the final training mix. "
            "Use together with --english-mix-ratio and --math-mix-ratio."
        ),
    ),
    english_mix_ratio: float | None = typer.Option(
        None,
        "--english-mix-ratio",
        help=(
            "Exact target ratio for English replay in the final training mix. "
            "Use together with --wolof-mix-ratio and --math-mix-ratio."
        ),
    ),
    math_mix_ratio: float | None = typer.Option(
        None,
        "--math-mix-ratio",
        help=(
            "Exact target ratio for math replay in the final training mix. "
            "Use together with --wolof-mix-ratio and --english-mix-ratio."
        ),
    ),
    output_dir: Path = typer.Option(Path("outputs/wolof"), "--output-dir"),
    model_name: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", "--model-name"),
    benchmark_samples: int = typer.Option(1000, "--benchmark-samples"),
    full_samples: int = typer.Option(5000, "--full-samples"),
    token_budget: int = typer.Option(800, "--token-budget"),
    benchmark_bpe_vocab_size: int = typer.Option(8192, "--benchmark-bpe-vocab-size"),
    full_bpe_vocab_size: int = typer.Option(16384, "--full-bpe-vocab-size"),
    benchmark_epochs: int = typer.Option(3, "--benchmark-epochs"),
    full_epochs: int = typer.Option(
        3,
        "--full-epochs",
        help="Number of epochs for the final fine-tuning stage. Maximum: 3.",
    ),
    include_method_c: bool = typer.Option(False, "--include-method-c/--no-method-c"),
    seed: int = typer.Option(3407, "--seed"),
) -> None:
    from .wolof_pipeline import run_wolof_pipeline

    _print_report(
        run_wolof_pipeline(
            dataset_file=dataset_file,
            method_a_text_corpus_file=method_a_text_corpus_file,
            english_replay_dataset=english_replay_dataset,
            english_replay_config=english_replay_config,
            english_replay_split=english_replay_split,
            english_replay_sample_ratio=english_replay_sample_ratio,
            english_replay_cache_dir=english_replay_cache_dir,
            math_replay_dataset=math_replay_dataset,
            math_replay_config=math_replay_config,
            math_replay_split=math_replay_split,
            math_replay_sample_ratio=math_replay_sample_ratio,
            math_replay_cache_dir=math_replay_cache_dir,
            wolof_mix_ratio=wolof_mix_ratio,
            english_mix_ratio=english_mix_ratio,
            math_mix_ratio=math_mix_ratio,
            output_root=output_dir,
            model_name=model_name,
            benchmark_sample_size=benchmark_samples,
            full_sample_size=full_samples,
            token_budget=token_budget,
            benchmark_bpe_vocab_size=benchmark_bpe_vocab_size,
            full_bpe_vocab_size=full_bpe_vocab_size,
            benchmark_epochs=benchmark_epochs,
            full_epochs=full_epochs,
            include_method_c=include_method_c,
            seed=seed,
        )
    )


@wolof_app.command("benchmark")
def wolof_benchmark(
    run_dir: Path | None = typer.Option(
        None,
        "--run-dir",
        exists=True,
        file_okay=False,
        readable=True,
        help=(
            "Optional Wolof training run directory. When provided, the command "
            "auto-loads final/model and final/tokenizer from the run output."
        ),
    ),
    model_path: str | None = typer.Option(
        None,
        "--model-path",
        help=(
            "Local model directory, local adapter directory, or Hugging Face "
            "full-model / adapter repo id."
        ),
    ),
    adapter_path: Path | None = typer.Option(
        None,
        "--adapter-path",
        help="Optional PEFT adapter path when --model-path points to the base model.",
    ),
    tokenizer_path: Path | None = typer.Option(
        None,
        "--tokenizer-path",
        help="Optional tokenizer directory with the resized Wolof tokenizer.",
    ),
    base_model_name: str | None = typer.Option(
        None,
        "--base-model-name",
        help="Override the base model name when benchmarking a PEFT adapter.",
    ),
    tasks: str = typer.Option(
        "afrimmlu,afrixnli,belebele",
        "--tasks",
        help=(
            "Comma-separated Wolof AfroBench suites. Supported values: "
            "afrimmlu, afrixnli, belebele, afriqa, sib, afrimgsm_direct, "
            "afrimgsm_translate, afrimgsm_en_cot, afrimmlu_math."
        ),
    ),
    prompt_variants: str = typer.Option(
        "1,2,3,4,5",
        "--prompt-variants",
        help="Comma-separated AfroBench prompt template variants. Use `1` for a smoke run.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Optional directory for the AfroBench benchmark artifacts.",
    ),
    all_epochs: bool = typer.Option(
        False,
        "--all-epochs/--single-model",
        help=(
            "When used with --run-dir, benchmark every saved final fine-tuning epoch "
            "checkpoint instead of only the final model."
        ),
    ),
    limit: float | None = typer.Option(
        None,
        "--limit",
        help="Optional per-task example limit. Use a small value such as 1 or 2 for smoke tests.",
    ),
    batch_size: str = typer.Option("auto", "--batch-size"),
    max_batch_size: int | None = typer.Option(None, "--max-batch-size"),
    num_fewshot: int = typer.Option(0, "--num-fewshot"),
    device: str | None = typer.Option("cuda", "--device"),
    dtype: str | None = typer.Option("auto", "--dtype"),
    load_in_4bit: bool = typer.Option(False, "--load-in-4bit/--no-load-in-4bit"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code/--no-trust-remote-code"),
    use_fast_tokenizer: bool = typer.Option(True, "--use-fast-tokenizer/--no-fast-tokenizer"),
    add_bos_token: bool = typer.Option(False, "--add-bos-token/--no-add-bos-token"),
    apply_chat_template: bool = typer.Option(
        False,
        "--apply-chat-template/--no-chat-template",
    ),
    fewshot_as_multiturn: bool = typer.Option(
        False,
        "--fewshot-as-multiturn/--no-fewshot-as-multiturn",
    ),
    log_samples: bool = typer.Option(False, "--log-samples/--no-log-samples"),
    write_out: bool = typer.Option(False, "--write-out/--no-write-out"),
    bootstrap_iters: int = typer.Option(0, "--bootstrap-iters"),
) -> None:
    from .afrobench import (
        run_wolof_afrobench_benchmark,
        run_wolof_afrobench_epoch_benchmarks,
    )

    if all_epochs:
        if run_dir is None:
            raise typer.BadParameter("--all-epochs requires --run-dir.")
        _print_report(
            run_wolof_afrobench_epoch_benchmarks(
                run_dir=run_dir,
                tasks=tasks,
                prompt_variants=prompt_variants,
                output_dir=output_dir,
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
        )
        return

    _print_report(
        run_wolof_afrobench_benchmark(
            model_path=model_path,
            run_dir=run_dir,
            adapter_path=adapter_path,
            tokenizer_path=tokenizer_path,
            base_model_name=base_model_name,
            tasks=tasks,
            prompt_variants=prompt_variants,
            output_dir=output_dir,
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
    )


@wolof_app.command("upload")
def wolof_upload(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, readable=True),
    repo_id: str = typer.Option(..., "--repo-id"),
    checkpoint_name: str | None = typer.Option(
        None,
        "--checkpoint-name",
        help=(
            "Optional checkpoint directory name under final/model/, such as "
            "checkpoint-2389. When omitted, the upload uses final/model."
        ),
    ),
    checkpoint_epoch: int | None = typer.Option(
        None,
        "--checkpoint-epoch",
        help=(
            "Optional final fine-tuning epoch number to upload instead of the "
            "default final/model directory."
        ),
    ),
    token: str | None = typer.Option(None, "--token"),
    private: bool = typer.Option(False, "--private/--public"),
    commit_message: str | None = typer.Option(None, "--commit-message"),
    include_benchmark: bool = typer.Option(True, "--include-benchmark/--no-benchmark"),
    include_report: bool = typer.Option(False, "--include-report/--no-report"),
    include_sample_generations: bool = typer.Option(
        False,
        "--include-sample-generations/--no-sample-generations",
    ),
) -> None:
    from .wolof_hub import upload_wolof_run_to_hub

    _print_report(
        upload_wolof_run_to_hub(
            run_dir=run_dir,
            repo_id=repo_id,
            checkpoint_name=checkpoint_name,
            checkpoint_epoch=checkpoint_epoch,
            token=token,
            private=private,
            commit_message=commit_message,
            include_benchmark=include_benchmark,
            include_report=include_report,
            include_sample_generations=include_sample_generations,
        )
    )


@wolof_app.command("download")
def wolof_download(
    repo_id: str = typer.Option(..., "--repo-id"),
    local_dir: Path | None = typer.Option(None, "--local-dir"),
    revision: str | None = typer.Option(None, "--revision"),
    token: str | None = typer.Option(None, "--token"),
) -> None:
    from .wolof_hub import download_wolof_repo_from_hub

    _print_report(
        download_wolof_repo_from_hub(
            repo_id=repo_id,
            local_dir=local_dir,
            revision=revision,
            token=token,
        )
    )


if __name__ == "__main__":
    app()
