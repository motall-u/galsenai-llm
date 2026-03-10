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
console = Console()

app.add_typer(data_app, name="data")
app.add_typer(benchmark_app, name="benchmark")


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


if __name__ == "__main__":
    app()
