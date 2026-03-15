# GalsenAI LLM

CLI-first finetuning project for no-tool, tool-calling, multi-tool, and reasoning-heavy tool-calling datasets.

The original notebook export is still in the repo as source material. The new project layout moves the useful parts into a package you can run from the command line with `uv`.

## Quickstart

```bash
uv sync --extra train --extra unsloth --extra dev
uv run galsenai data validate data/samples/finetune_sample.jsonl
uv run galsenai train --config configs/train/sample_qwen_tool_calling.yaml
uv run galsenai merge --config configs/merge/sample_merge.yaml
uv run galsenai benchmark run --config configs/benchmark/sample_benchmark.yaml
uv run galsenai wolof infer --prompt "Nanga def?"
uv run galsenai wolof run --dataset-file data/wolof-dataset/curated_dataset.json --method-a-text-corpus-file <path-to-text-only-wolof-corpus>
uv run galsenai wolof benchmark --run-dir outputs/wolof/<run-id> --prompt-variants 1 --limit 2
```

## Wolof End-to-End Run

For the Wolof pipeline, `uv sync` plus a single command is enough to run the
tokenizer benchmark and the follow-up fine-tuning job:

```bash
uv sync --extra train --extra dev
uv run galsenai wolof run \
  --dataset-file data/wolof-dataset/curated_dataset.json \
  --method-a-text-corpus-file <path-to-text-only-wolof-corpus>
```

After a run completes, publish it with:

```bash
uv run galsenai wolof upload --run-dir outputs/wolof/<run-id> --repo-id your-username/your-wolof-model
```

Run the Wolof downstream AfroBench smoke benchmark with:

```bash
uv run galsenai wolof benchmark \
  --run-dir outputs/wolof/<run-id> \
  --tasks afrimmlu,afrixnli,belebele \
  --prompt-variants 1 \
  --limit 2
```

Or benchmark a model directly from Hugging Face:

```bash
uv run galsenai wolof benchmark \
  --model-path your-username/your-wolof-model \
  --tasks afrimmlu,afrixnli,belebele \
  --prompt-variants 1
```

The command always runs both stages:

- Step 1 benchmark on 1,000 sampled conversations with Method A vs Method B.
- Method A now uses a separate text-only Wolof corpus to compute word frequencies and the BPE-based over-fragmentation threshold.
- Optional: add `--include-method-c` to benchmark a third Unigram-based tokenizer adaptation method.
- Optional: add `--english-replay-dataset <repo-or-file>` and `--math-replay-dataset <repo-or-file>` to mix English and math replay sets into the final fine-tuning stage.
- Math replay also accepts OpenReasoning-style rows with `problem` and `generated_solution`; the loader drops `<think>...</think>` and keeps only the final response.
- For `nvidia/OpenMathReasoning`, the loader automatically falls back from `train` to `cot`.
- Optional: add `--wolof-mix-ratio 0.7 --english-mix-ratio 0.2 --math-mix-ratio 0.1` to target a 70/20/10 final train mix while keeping validation Wolof-only.
- Step 2 fine-tuning on `--full-samples` Wolof conversations using the winning tokenizer, with optional English and math replay mixed into the final training split only.
- Final fine-tuning is capped at `3` epochs, and the trainer saves a reusable checkpoint after each epoch under `final/model/checkpoint-*`.
- `final/epoch_checkpoints.json` indexes every saved epoch checkpoint with its epoch number, `eval_loss`, and perplexity.
- Markdown logging, tokenizer artifacts, checkpoints, validation loss, and sample generations under `outputs/wolof/<run-id>/`.
- Upload, download, CLI inference, interactive chat, and full run guide: `resources/wolof_run_guide.md`

To benchmark every saved final-training epoch and compare them directly:

```bash
uv run galsenai wolof benchmark \
  --run-dir outputs/wolof/<run-id> \
  --all-epochs \
  --tasks afrimmlu,afrixnli,belebele \
  --prompt-variants 1,2,3,4,5
```

This writes a comparison summary under `outputs/wolof/<run-id>/afrobench/epoch-checkpoints/`.

### Inspect Method A Token Selection

After a Wolof run, Method A writes its token selection artifact to:

- `outputs/wolof/<run-id>/benchmark/method_a/token_selection.json`

If Method A wins the tokenizer benchmark, the same artifact is also copied to:

- `outputs/wolof/<run-id>/final/token_selection.json`

This file contains the values used to decide whether a Wolof word is a candidate
for tokenizer expansion:

- `fragmentation_threshold`: the weighted average number of fragments produced by a reference Wolof BPE tokenizer trained on the separate text-only corpus passed with `--method-a-text-corpus-file`
- `candidate_summary.min_frequency`: the hard minimum frequency filter applied before ranking candidates
- `selected_tokens[*].frequency`: corpus frequency of the selected word
- `selected_tokens[*].fragment_count`: number of pieces produced by the base Qwen tokenizer
- `selected_tokens[*].reference_fragment_count`: number of pieces produced by the reference Wolof BPE tokenizer
- `selected_tokens[*].fragmentation_gap`: `fragment_count - fragmentation_threshold`
- `selected_tokens[*].score`: ranking score used for selection, currently `frequency * fragmentation_gap`

The current Method A candidate rule is:

- keep only words with `len(word) >= 3`
- keep only words with `frequency >= 2`
- keep only words where `fragment_count > fragmentation_threshold`
- rank the remaining words by `frequency * fragmentation_gap`
- keep the top `token_budget` words

The file `outputs/wolof/<run-id>/method_a_text_corpus.json` records which
external text corpus was used to compute these frequencies and the reference
fragmentation threshold.

On an H100-class GPU the pipeline now auto-switches to:

- BF16 full fine-tuning.
- Packed sequences for the benchmark and final run.
- Much larger per-device batch sizes tuned for the short Wolof conversations in this dataset.

## Project layout

```text
configs/                 Sample YAML configs for train / merge / infer / benchmark
data/                    Dataset docs and small starter datasets
scripts/                 Small wrappers for common flows
src/galsenai_llm/        CLI, data validation, training, merge, inference, benchmark code
notebook58eab54a54.py    Original local notebook export that this project was derived from
```

## Code organization

The codebase is intentionally split by responsibility so maintainers can change one stage without rewriting the others.

- `src/galsenai_llm/cli.py`: Typer entrypoints. If you add a new user-facing command, wire it here.
- `src/galsenai_llm/config.py`: Typed YAML config models for train, merge, infer, evaluate, and benchmark.
- `src/galsenai_llm/schemas.py`: Canonical Pydantic schemas for dataset examples, messages, tool calls, and prediction records.
- `src/galsenai_llm/io.py`: Small shared JSON / JSONL read-write helpers.
- `src/galsenai_llm/dataset.py`: Dataset loading, validation, and summary utilities.
- `src/galsenai_llm/train.py`: Finetuning entrypoint. This is where models are loaded, examples are rendered into trainable text, and `SFTTrainer` is launched.
- `src/galsenai_llm/merge.py`: Adapter merge stage for LoRA outputs.
- `src/galsenai_llm/generation.py`: Shared local generation helpers for Transformers-style inference and benchmark prediction.
- `src/galsenai_llm/infer.py`: Single-prompt inference flow built on top of `generation.py`.
- `src/galsenai_llm/evaluate.py`: Prediction scoring logic against the benchmark dataset.
- `src/galsenai_llm/benchmark.py`: Benchmark orchestration. It can score an existing predictions file, run oracle mode, or generate first-turn predictions with a local model.
- `src/galsenai_llm/afrobench.py`: Wolof downstream benchmarking with the TRC `lm-evaluation-harness` AfroBench task suite.
- `src/galsenai_llm/wolof_pipeline.py`: End-to-end Wolof tokenizer benchmarking and low-resource fine-tuning pipeline for Qwen 2.5 0.5B.
- `src/galsenai_llm/tool_registry.py`: Deterministic local reference tools used by the sample datasets and benchmarks.

### Maintainer notes

- The current training pipeline renders each structured chat example into a single `text` field before passing it to TRL. If you later move to native conversational training, the main place to change is `src/galsenai_llm/train.py`.
- `configs/train/smoke_tiny_gpt2.yaml` is the smallest CPU smoke test for the training code path.
- `configs/train/smoke_unsloth_qwen_tool_calling.yaml` is the smallest GPU smoke test for the real Unsloth + Qwen stack.
- `configs/train/sample_qwen_tool_calling.yaml` is the main starter config for the actual project direction.
- `data/README.md` is the source of truth for dataset authoring. If you change the schema, update both `src/galsenai_llm/schemas.py` and `data/README.md`.

### Where to edit what

- Add a new tool:
  Update `src/galsenai_llm/tool_registry.py`, then add examples in `data/samples/`, and update `data/README.md` if the format changes.
- Add a new pipeline stage:
  Add a config model in `src/galsenai_llm/config.py`, implement the stage in a dedicated module, then expose it in `src/galsenai_llm/cli.py`.
- Change benchmark scoring:
  Edit `src/galsenai_llm/evaluate.py`, not the CLI layer.
- Change model-loading behavior:
  Edit `src/galsenai_llm/train.py`, `src/galsenai_llm/merge.py`, or `src/galsenai_llm/generation.py` depending on the stage.

## Data format

The project uses a chat-style **JSONL** format (one JSON object per line). Each example represents a complete conversation for supervised finetuning.

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Stable unique identifier. |
| `category` | `string` | One of `no_tool`, `single_tool`, `multi_tool`, `reasoning_tool`. |
| `messages` | `array` | Ordered chat transcript (at least 2 messages, including one `assistant`). |
| `tools` | `array` | OpenAI-style tool definitions available for this example. Use `[]` for `no_tool`. |

### Optional fields

| Field | Type | Description |
|-------|------|-------------|
| `metadata` | `object` | Free-form tags (split, difficulty, domain, source). |
| `expected` | `object` | Benchmark-only expectations (`first_assistant`, `tool_sequence`, `final_answer`). |

### Message roles

- **`system`** -- System prompt (optional, must be first if present).
- **`user`** -- User turn.
- **`assistant`** -- Model response. Must contain `content`, `tool_calls`, or both.
- **`tool`** -- Tool result. Must contain `role`, `name`, `tool_call_id`, and `content`.

### Tool call format

```json
{
  "id": "call_001",
  "type": "function",
  "function": {
    "name": "get_vector_sum",
    "arguments": { "a": [1, -1, 2], "b": [3, 0, -4] }
  }
}
```

### Tool definition format

```json
{
  "type": "function",
  "function": {
    "name": "get_vector_sum",
    "description": "Get the element-wise sum of two vectors.",
    "parameters": {
      "type": "object",
      "properties": {
        "a": { "type": "array", "items": { "type": "number" } },
        "b": { "type": "array", "items": { "type": "number" } }
      },
      "required": ["a", "b"]
    }
  }
}
```

### Category rules

| Category | Tools list | Assistant tool_calls |
|----------|-----------|---------------------|
| `no_tool` | Should be `[]` | Must **not** have tool calls |
| `single_tool` | At least one tool | At least one tool call |
| `multi_tool` | At least one tool | Multiple tool calls across turns |
| `reasoning_tool` | At least one tool | Tool calls with intermediate reasoning |

### Validation

Run the built-in validator to check both structure (schema) and semantics (category consistency, duplicate IDs, tool reference integrity):

```bash
uv run galsenai data validate data/samples/finetune_sample.jsonl
```

The validator checks:
- JSON structure matches the Pydantic schema.
- No duplicate `id` values.
- `no_tool` examples don't contain tool calls; other categories do.
- Tool call names reference tools defined in the `tools` list.
- Tool message `tool_call_id` values match preceding assistant tool call IDs.
- First message is `system` or `user`.

See [data/README.md](data/README.md) for full authoring guidance and a minimal example.

## What is included

- A typed dataset format for tool-calling finetuning.
- Sample data for `no_tool`, `single_tool`, `multi_tool`, and `reasoning_tool`.
- A CLI with separate stages for training, merge, evaluation, inference, and benchmarking.
- A deterministic local sample tool registry based on the math/inventory functions from the notebook.
- A benchmark runner that can either score a predictions file or generate first-turn predictions from a local Transformers model.

## Notes

- Training and local generation dependencies are kept behind `uv` extras so the repo can stay lightweight by default.
- The sample benchmark is deterministic and uses fixed local tool outputs instead of live external APIs.
- The CLI assumes your finetuning dataset follows the format documented in [data/README.md](/home/motall-u/my_projects/galsenai-llm/data/README.md).

## Timeline

### 2026-03-10 -- Project initialization

- Initialized the repository and set up the `uv`-managed Python package (`pyproject.toml`, extras for train/unsloth/dev).
- Built the full CLI-first architecture with Typer: `train`, `merge`, `evaluate`, `infer`, `benchmark`, and `data validate` commands.
- Implemented the SFT training pipeline (`train.py`) with dual backend support (Unsloth and Transformers) and LoRA/QLoRA configuration.
- Created typed YAML config system (`config.py`) and Pydantic data schemas (`schemas.py`) for dataset validation.
- Added LoRA adapter merge stage (`merge.py`) supporting Unsloth and PEFT backends.
- Built local inference (`generation.py`, `infer.py`) and benchmark orchestration (`benchmark.py`) with oracle, predictions, and transformers modes.
- Implemented prediction scoring logic (`evaluate.py`) with per-category accuracy reporting (decision, tool names, arguments, content matching).
- Created a deterministic local tool registry (`tool_registry.py`) with math and inventory functions for reproducible benchmarks.
- Designed the chat-style JSONL data format with four categories: `no_tool`, `single_tool`, `multi_tool`, `reasoning_tool`.
- Authored sample training data (`finetune_sample.jsonl` -- 6 examples) and benchmark data (`benchmark_sample.jsonl` -- 4 examples).
- Wrote dataset validation with structural (Pydantic) and semantic checks (duplicate IDs, category-tool consistency, tool reference integrity).
- Ran CPU smoke test with `tiny-gpt2` (full finetuning, 1 step) -- passed.
- Ran GPU smoke test with `Qwen2.5-Coder-1.5B-Instruct` via Unsloth (LoRA, 1 step) -- passed.
- Ran oracle benchmark -- 100% accuracy across all categories.

### 2026-03-11 -- Shared learning resources

- Added `resources/shared.md` -- a curated resource guide for the team covering fine-tuning fundamentals, core concepts (quantization, tool calling), a key terms glossary, training category explanations, and a recommended learning path.
