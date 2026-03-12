# Wolof Run Guide

This guide covers the two supported execution paths for the Wolof pipeline:

- local development on a smaller GPU such as an 8 GB laptop GPU
- production-style execution on an H100-class GPU

The pipeline is CLI-first and runs end to end:

1. detect the GPU with `nvidia-smi`
2. benchmark two tokenizer adaptation methods on 1,000 sampled conversations
3. pick the winning tokenizer
4. fine-tune on 5,000 sampled conversations
5. save checkpoints, metrics, generations, and a Markdown report

## Dataset

The expected dataset path is:

```text
data/wolof-dataset/curated_dataset.json
```

The file must contain the full conversation list used by the Wolof pipeline.

## Install

Run this once on either local or H100:

```bash
uv sync --extra train --extra dev
```

## Sanity Check Before Training

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

```bash
uv run galsenai wolof run \
  --dataset-file data/wolof-dataset/curated_dataset.json
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

### H100 end-to-end run

```bash
uv run galsenai wolof run \
  --dataset-file data/wolof-dataset/curated_dataset.json \
  --output-dir outputs/wolof-h100
```

### Optional H100 larger run

If you want to go beyond the default 5,000-sample fine-tuning stage, increase the sample sizes explicitly:

```bash
uv run galsenai wolof run \
  --dataset-file data/wolof-dataset/curated_dataset.json \
  --benchmark-samples 1000 \
  --full-samples 20000 \
  --benchmark-epochs 3 \
  --full-epochs 3 \
  --output-dir outputs/wolof-h100-large
```

Use that only if you intentionally want a larger training subset than the default workflow.

## Artifacts

Each run writes a timestamped directory:

```text
outputs/wolof/<run-id>/
```

Important files:

- `nvidia_smi.txt`: captured GPU detection output
- `data_splits.json`: sampled benchmark and training splits
- `benchmark/comparison.json`: tokenizer benchmark comparison
- `benchmark/method_a/` and `benchmark/method_b/`: tokenizer artifacts and benchmark training summaries
- `final/final_result.json`: final validation metrics and selected tokenizer
- `final/sample_generations.json`: generated Wolof samples
- `final/model/`: trainer checkpoint output
- `wolof_finetuning_report.md`: human-readable summary of the whole run

## How To Read The Report

Open:

```text
outputs/wolof/<run-id>/wolof_finetuning_report.md
```

Check these sections first:

- `Environment`: confirms GPU, precision, and training strategy
- `Tokenizer Comparison`: compares Method A and Method B
- `Recommendation`: shows which tokenizer won
- `Full Fine-Tuning`: contains final validation loss and perplexity
- `Sample Generations`: qualitative Wolof outputs

## Typical Workflow

### Local development

1. `uv sync --extra train --extra dev`
2. `nvidia-smi`
3. `uv run galsenai wolof infer --prompt "Nanga def?"`
4. run the local smoke test
5. run the default local end-to-end command if the smoke test succeeds

### H100 execution

1. `uv sync --extra train --extra dev`
2. `nvidia-smi`
3. `uv run galsenai wolof infer --prompt "Nanga def?"`
4. run the H100 end-to-end command
5. inspect `wolof_finetuning_report.md`

## Notes

- The default workflow is not the entire 133k conversation dataset. It benchmarks on 1,000 samples and fine-tunes on 5,000 samples.
- The pipeline is designed to adapt itself to the detected GPU. You do not need separate code paths for local and H100.
- If a training run fails with OOM, the pipeline will automatically retry with a reduced plan.
- The research rationale for the tokenizer and low-resource choices is documented in `resources/wolof_low_resource_strategy.md`.
