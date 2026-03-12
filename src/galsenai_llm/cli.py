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
    model_path: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", "--model-path"),
    adapter_path: Path | None = typer.Option(None, "--adapter-path"),
    base_model_name: str | None = typer.Option(None, "--base-model-name"),
    system_prompt: str | None = typer.Option(
        (
            "You are a helpful assistant for Wolof language tasks. "
            "Reply in Wolof unless the user asks otherwise."
        ),
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


@wolof_app.command("run")
def wolof_run(
    dataset_file: Path = typer.Option(
        Path("data/wolof-dataset/curated_dataset.json"),
        "--dataset-file",
        exists=True,
        readable=True,
    ),
    output_dir: Path = typer.Option(Path("outputs/wolof"), "--output-dir"),
    model_name: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", "--model-name"),
    benchmark_samples: int = typer.Option(1000, "--benchmark-samples"),
    full_samples: int = typer.Option(5000, "--full-samples"),
    token_budget: int = typer.Option(800, "--token-budget"),
    benchmark_bpe_vocab_size: int = typer.Option(8192, "--benchmark-bpe-vocab-size"),
    full_bpe_vocab_size: int = typer.Option(16384, "--full-bpe-vocab-size"),
    benchmark_epochs: int = typer.Option(3, "--benchmark-epochs"),
    full_epochs: int = typer.Option(3, "--full-epochs"),
    seed: int = typer.Option(3407, "--seed"),
) -> None:
    from .wolof_pipeline import run_wolof_pipeline

    _print_report(
        run_wolof_pipeline(
            dataset_file=dataset_file,
            output_root=output_dir,
            model_name=model_name,
            benchmark_sample_size=benchmark_samples,
            full_sample_size=full_samples,
            token_budget=token_budget,
            benchmark_bpe_vocab_size=benchmark_bpe_vocab_size,
            full_bpe_vocab_size=full_bpe_vocab_size,
            benchmark_epochs=benchmark_epochs,
            full_epochs=full_epochs,
            seed=seed,
        )
    )


if __name__ == "__main__":
    app()
