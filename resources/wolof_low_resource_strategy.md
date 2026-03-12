# Wolof Low-Resource Fine-Tuning Strategy

## Goal

Adapt `Qwen/Qwen2.5-0.5B-Instruct` to Wolof with a tokenizer benchmark first,
then a 5k-sample fine-tune using the winning tokenizer method.

## Why this design

1. Low-resource tokenization should be evaluated with both intrinsic and
   downstream metrics.
   Rust et al. (2021), "How Good Is Your Tokenizer?"
   https://aclanthology.org/2021.acl-long.243/

2. Fragmentation metrics alone are not enough.
   Chau et al. (2024), "Tokenizer Choice For LLM Training: Negligible or Crucial?"
   https://arxiv.org/abs/2405.17886

3. New tokens should not start from random embeddings if we can transfer from
   a stronger source tokenizer.
   Dobler et al. (2024), "FOCUS: Effective Embedding Initialization for Special
   Tokens and Embeddings in Fine-Tuned Language Models"
   https://arxiv.org/abs/2305.14481

4. On a single 8 GB GPU, QLoRA is the practical training path.
   Dettmers et al. (2023), "QLoRA: Efficient Finetuning of Quantized LLMs"
   https://arxiv.org/abs/2305.14314

5. LoRA remains the clean adapter baseline when full fine-tuning is too costly.
   Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models"
   https://arxiv.org/abs/2106.09685

## Implementation choices

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
  Reason: the data is conversational and Qwen's chat template is already
  aligned with supervised chat fine-tuning.

- Step 1 Method A:
  Keep Qwen's tokenizer, mine frequent Wolof words plus prefix/suffix-style
  morphemes that are currently split into multiple subwords, add them as
  regular tokens, then initialize each new row by averaging the source
  subword embeddings.

- Step 1 Method B:
  Train a compact BPE tokenizer from the Wolof benchmark corpus, preserve Qwen
  special tokens and chat template, and initialize each token by exact overlap
  or compositional averaging from Qwen's original subword space.

- Benchmark metrics:
  fertility, single-token coverage, embedding-initialization cosine,
  held-out perplexity, and loss curves.

- Step 2:
  Reuse the winning tokenizer method and rebuild it on the 5k train split
  only. This avoids validation leakage while keeping the tokenizer strategy
  fixed.

- Training mode on the current machine:
  4-bit QLoRA with BF16 compute, gradient checkpointing, and conservative
  batch sizing chosen from `nvidia-smi`.

## Repo entrypoint

```bash
uv run galsenai wolof run --dataset-file data/wolof-dataset/curated_dataset.json
```

The command writes timestamped artifacts under `outputs/wolof/`.
