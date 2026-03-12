# Wolof Run Guide

This guide covers the full Wolof execution workflow for:

- local development on a smaller GPU such as an 8 GB laptop GPU
- production-style runs on an H100-class GPU
- full-dataset execution after the tokenizer benchmark

The pipeline is CLI-first and runs end to end:

1. detect the GPU with `nvidia-smi`
2. benchmark two tokenizer adaptation methods on a sampled benchmark split
3. pick the winning tokenizer
4. rebuild the winning tokenizer on the final training pool
5. fine-tune Qwen 2.5 0.5B on the selected training pool
6. save checkpoints, metrics, generations, and a Markdown report

## Dataset

The expected dataset path is:

```text
data/wolof-dataset/curated_dataset.json
```

The file must contain the full conversation list used by the Wolof pipeline.

## Important Flags

These are the two flags that matter most when you run the pipeline:

- `--benchmark-samples`: number of conversations used for the tokenizer benchmark
- `--full-samples`: size of the final fine-tuning pool after the benchmark

Important: `--full-samples` is not "use the whole dataset automatically". It is the exact size of the final training pool you want.

Defaults:

- benchmark sample size: `1000`
- full fine-tuning sample size: `5000`
- benchmark epochs: `3`
- full fine-tuning epochs: `3`

## Install

Run this once from the repo root:

```bash
uv sync --extra train --extra dev
```

The Wolof pipeline depends on the `train` extra. Do not rely on `uv pip install torch` alone.

If you want to confirm the project environment is complete, run:

```bash
uv run python -c "import torch, datasets, peft, transformers, trl; print(torch.__version__)"
```

## Pre-Flight Check

Confirm the GPU and driver:

```bash
nvidia-smi
```

Run a quick prompt against the base model before fine-tuning:

```bash
uv run galsenai wolof infer --prompt "Tekkil ci Wolof: Hello my friend."
```

## Local GPU Run

This path is intended for smaller GPUs such as an 8 GB RTX laptop GPU.

### What the pipeline will do automatically

- switch to QLoRA
- use mixed precision when supported
- keep conservative batch sizes
- disable packing by default on small-memory hardware
- fall back to smaller plans if an OOM happens

### Recommended local smoke test

Use this before launching the default 1k and 5k run:

```bash
uv run galsenai wolof run \
  --dataset-file data/wolof-dataset/curated_dataset.json \
  --benchmark-samples 32 \
  --full-samples 64 \
  --token-budget 64 \
  --benchmark-bpe-vocab-size 1024 \
  --full-bpe-vocab-size 2048 \
  --benchmark-epochs 1 \
  --full-epochs 1 \
  --output-dir outputs/wolof-local-smoke
```

### Default local end-to-end run

This runs the benchmark on `1000` samples and the final fine-tune on `5000` samples:

```bash
uv run galsenai wolof run \
  --dataset-file data/wolof-dataset/curated_dataset.json \
  --output-dir outputs/wolof-local
```

After training finishes, upload that run with:

```bash
uv run galsenai wolof upload \
  --run-dir outputs/wolof-local/<run-id> \
  --repo-id your-username/your-wolof-model
```

### Expected local profile

On a smaller GPU, the generated report should usually show something close to:

- training strategy: `qlora`
- precision: `bf16` when available, otherwise `fp16`
- benchmark batch size: `1`
- full batch size: `1`
- gradient accumulation larger than `1`

## H100 Run

This is the recommended path for the real experiment run.

### What the pipeline will do automatically on an H100

- detect the H100 from `nvidia-smi`
- switch to full fine-tuning
- use `bf16`
- enable packed batches for both benchmark and final training
- use larger per-device batch sizes
- use fused AdamW

The current H100 auto-plan is:

- benchmark train batch size: `32`
- benchmark eval batch size: `16`
- full train batch size: `32`
- full eval batch size: `16`
- gradient accumulation: `1`
- max length cap: `512`
- packing: `True`

### H100 default end-to-end run

This runs the benchmark on `1000` samples and the final fine-tune on `5000` samples:

```bash
uv run galsenai wolof run \
  --dataset-file data/wolof-dataset/curated_dataset.json \
  --output-dir outputs/wolof-h100
```

After training finishes, upload that run with:

```bash
uv run galsenai wolof upload \
  --run-dir outputs/wolof-h100/<run-id> \
  --repo-id your-username/your-wolof-model
```

### H100 larger subset run

If you want a larger final training pool but not the full dataset:

```bash
uv run galsenai wolof run \
  --dataset-file data/wolof-dataset/curated_dataset.json \
  --benchmark-samples 1000 \
  --full-samples 20000 \
  --benchmark-epochs 3 \
  --full-epochs 3 \
  --output-dir outputs/wolof-h100-large
```

After training finishes, upload that run with:

```bash
uv run galsenai wolof upload \
  --run-dir outputs/wolof-h100-large/<run-id> \
  --repo-id your-username/your-wolof-model
```

## Full-Dataset Run

The current dataset contains `133442` usable conversations when loaded by the pipeline.

If you want to verify that number on your machine, run:

```bash
uv run python - <<'PY'
from pathlib import Path
from galsenai_llm.wolof_pipeline import load_wolof_dataset

records = load_wolof_dataset(Path("data/wolof-dataset/curated_dataset.json"))
print(len(records))
PY
```

### Recommended full-dataset run without overlap

This is the recommended command. It uses:

- `1000` conversations for the tokenizer benchmark
- the remaining `132442` conversations for the final fine-tune

```bash
uv run galsenai wolof run \
  --dataset-file data/wolof-dataset/curated_dataset.json \
  --benchmark-samples 1000 \
  --full-samples 132442 \
  --benchmark-epochs 3 \
  --full-epochs 3 \
  --output-dir outputs/wolof-full-all
```

After training finishes, upload that run with:

```bash
uv run galsenai wolof upload \
  --run-dir outputs/wolof-full-all/<run-id> \
  --repo-id your-username/your-wolof-model \
  --include-benchmark \
  --include-report \
  --include-sample-generations
```

Why `132442` instead of `133442`:

- total usable conversations: `133442`
- benchmark sample: `1000`
- final no-overlap training pool: `133442 - 1000 = 132442`

### Full-dataset run with overlap

Use this only if you intentionally want the benchmark conversations to be eligible for the final training pool too:

```bash
uv run galsenai wolof run \
  --dataset-file data/wolof-dataset/curated_dataset.json \
  --benchmark-samples 1000 \
  --full-samples 133442 \
  --benchmark-epochs 3 \
  --full-epochs 3 \
  --output-dir outputs/wolof-full-all-overlap
```

After training finishes, upload that run with:

```bash
uv run galsenai wolof upload \
  --run-dir outputs/wolof-full-all-overlap/<run-id> \
  --repo-id your-username/your-wolof-model \
  --include-benchmark \
  --include-report \
  --include-sample-generations
```

This is usually not the preferred setup because the final training pool can include the benchmark conversations.

## Artifacts

Each run writes a timestamped directory under the output root you pass with `--output-dir`:

```text
<output-dir>/<run-id>/
```

Examples:

- `outputs/wolof-local/<run-id>/`
- `outputs/wolof-h100/<run-id>/`
- `outputs/wolof-full-all/<run-id>/`

Important files:

- `nvidia_smi.txt`: captured GPU detection output
- `data_splits.json`: sampled benchmark and training splits
- `benchmark/comparison.json`: tokenizer benchmark comparison
- `benchmark/method_a/` and `benchmark/method_b/`: tokenizer artifacts and benchmark training summaries
- `final/final_result.json`: final validation metrics and selected tokenizer
- `final/sample_generations.json`: generated Wolof samples
- `final/model/`: trainer checkpoint output
- `run_summary.json`: compact machine-readable run summary
- `wolof_finetuning_report.md`: human-readable summary of the whole run

## Publish To Hugging Face

After training finishes, upload the final model plus benchmark artifacts with:

```bash
uv run galsenai wolof upload \
  --run-dir outputs/wolof-full-all/<run-id> \
  --repo-id your-username/your-wolof-model
```

Useful optional flags:

- `--private` to create a private model repo
- `--include-benchmark` to upload the tokenizer benchmark results
- `--include-report` if you also want to upload the Markdown report
- `--include-sample-generations` if you want to upload generated validation samples

Default behavior:

- uploads the final model files to the repository root
- uploads benchmark artifacts under `benchmark/`
- uploads compact run metadata under `artifacts/`
- generates a cleaner `README.md` model card than the default trainer output

Authentication:

- pass `--token <hf_token>`
- or set `HF_TOKEN`

## Download From Hugging Face

Download a previously uploaded model repository locally:

```bash
uv run galsenai wolof download \
  --repo-id your-username/your-wolof-model \
  --local-dir models/your-wolof-model
```

If you omit `--local-dir`, the default is:

```text
models/<repo-id-with-slashes-replaced-by-->
```

## How To Read The Report

Open:

```text
<output-dir>/<run-id>/wolof_finetuning_report.md
```

Check these sections first:

- `Environment`: confirms GPU, precision, and training strategy
- `Tokenizer Comparison`: compares Method A and Method B
- `Recommendation`: shows which tokenizer won
- `Full Fine-Tuning`: contains final validation loss and perplexity
- `Sample Generations`: qualitative Wolof outputs

## Inference From Hugging Face Or A Downloaded Model

The inference commands accept either:

- a Hugging Face repo id such as `your-username/your-wolof-model`
- or a downloaded local directory such as `models/your-wolof-model`

### One-shot CLI inference

Directly from the Hub:

```bash
uv run galsenai wolof infer \
  --model-path your-username/your-wolof-model \
  --prompt "Nanga def?"
```

From a downloaded local directory:

```bash
uv run galsenai wolof infer \
  --model-path models/your-wolof-model \
  --prompt "Nanga def?"
```

### Interactive chat mode

Directly from the Hub:

```bash
uv run galsenai wolof chat --model-path your-username/your-wolof-model
```

From a downloaded local directory:

```bash
uv run galsenai wolof chat --model-path models/your-wolof-model
```

Notes:

- if the uploaded repository is a PEFT adapter repo, the loader will read `adapter_config.json` and load the base model automatically
- use `/clear` to reset the chat history
- use `/exit` or `/quit` to leave chat mode

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

This usually means `uv run` is not using an environment that has the `train` extra installed.

Fix:

```bash
uv sync --extra train --extra dev
```

Then verify:

```bash
uv run python -c "import torch, datasets, peft, transformers, trl; print(torch.__version__)"
```

Do not rely on `uv pip install torch` by itself.

### OOM or CUDA memory error

The pipeline already retries with a reduced training plan. If you still hit OOM:

- reduce `--full-samples` for the first test
- reduce epochs for the first validation run
- use the local smoke command before launching a large run

### Run restarts after failure

Each invocation creates a new timestamped output directory. The pipeline does not currently resume from a previous run directory automatically.

### Hugging Face authentication errors

If upload or download fails because of authentication:

- set `HF_TOKEN` in the shell
- or pass `--token` explicitly to `wolof upload` or `wolof download`

## Typical Workflows

### Local development

1. `uv sync --extra train --extra dev`
2. `nvidia-smi`
3. `uv run galsenai wolof infer --prompt "Nanga def?"`
4. run the local smoke test
5. run the default local command if the smoke test succeeds

### H100 default workflow

1. `uv sync --extra train --extra dev`
2. `nvidia-smi`
3. `uv run galsenai wolof infer --prompt "Nanga def?"`
4. run the H100 default end-to-end command
5. inspect `wolof_finetuning_report.md`
6. upload with `uv run galsenai wolof upload --run-dir ... --repo-id ...`

### H100 full-dataset workflow

1. `uv sync --extra train --extra dev`
2. `nvidia-smi`
3. optionally verify the usable conversation count
4. run the recommended full-dataset no-overlap command
5. inspect `run_summary.json` and `wolof_finetuning_report.md`
6. upload with `uv run galsenai wolof upload --run-dir ... --repo-id ...`
7. verify with `uv run galsenai wolof infer --model-path your-username/your-wolof-model --prompt "Nanga def?"`

## Notes

- The default workflow is not the entire 133k conversation dataset. It benchmarks on `1000` samples and fine-tunes on `5000` samples.
- The pipeline is designed to adapt itself to the detected GPU. You do not need separate code paths for local and H100.
- The research rationale for the tokenizer and low-resource choices is documented in `resources/wolof_low_resource_strategy.md`.
