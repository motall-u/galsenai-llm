# Wolof Tokenizer Research Report

## Scope

This report starts from the current project state:

- `method_b` is the strongest tokenizer baseline from your full Wolof run.
- `method_c` was added as a research branch and improved embedding
  initialization quality, but did not beat `method_b` on the local smoke
  benchmark.

The goal here is to identify **at least 10 additional tokenizer methods**
grounded in the literature, explain the concept of each one simply, define a
clear hypothesis for Wolof, and specify the experiment required to test it.

This report is a **research and experiment design report**. It does **not**
claim that all 10+ methods were implemented and run in this repository yet.

## Shared Evaluation Protocol

To make the methods comparable, use the same evaluation scaffold for every
candidate unless the method explicitly requires a different architecture.

### Stage 1: Intrinsic tokenizer benchmark

Build the tokenizer or tokenizer adaptation on the benchmark train split only.

Evaluate on held-out Wolof text:

- fertility: average tokens per word
- single-token coverage
- average tokens per character
- diacritic retention / reversibility
- optional fairness score versus French or English paraphrases

### Stage 2: Mini downstream benchmark

Use the current Wolof benchmark recipe:

- benchmark sample size: `1000`
- same train/val/test split policy
- same Qwen 2.5 0.5B base model
- same epochs and optimizer plan
- same LoRA or full-finetuning policy chosen by the hardware

Track:

- validation loss
- perplexity
- loss curve
- embedding initialization quality

### Stage 3: Promotion gate

Promote only methods that beat `method_b` on perplexity, or match it while
improving one of:

- fertility by at least 10%
- coverage by at least 5 points
- throughput or sequence compression
- robustness to Wolof diacritics and spelling variation

### Stage 4: H100 rerun

Run only the shortlisted methods on the H100 with the same full-data protocol:

- `--benchmark-samples 1000`
- `--full-samples 132442`
- `--benchmark-epochs 3`
- `--full-epochs 3`

## Ranking Heuristic

Use this priority scale:

- `P1`: high fit for current pipeline and high upside
- `P2`: interesting and plausible, but higher engineering cost or lower fit
- `P3`: architecture-level research track, not a drop-in tokenizer tweak

## Summary Table

| ID | Method | Core idea | Fit to current Qwen pipeline | Priority |
| --- | --- | --- | --- | --- |
| D | Unigram + Subword Regularization | Sample multiple segmentations from a Unigram tokenizer during training | Medium | P1 |
| E | BPE-Dropout | Stochastic BPE merges during training, deterministic at inference | High | P1 |
| F | WECHSEL | Semantic embedding transfer using multilingual static embeddings | Medium | P1 |
| G | Dictionary-based Vocabulary Transfer | Use bilingual dictionaries to initialize target subwords | Medium | P1 |
| H | OMP Tokenizer Transplantation | Sparse reconstruction of new tokens from shared anchors | Medium | P1 |
| I | Universal Multilingual Tokenizer | Train tokenizer on Wolof plus related support languages | High | P1 |
| J | VOLT | Learn vocabulary with optimal transport instead of plain BPE | Low-Medium | P2 |
| K | SelfSeg | Self-supervised neural subword segmentation | Low-Medium | P2 |
| L | BERTSeg | BERT-based unsupervised segmentation | Low-Medium | P2 |
| M | Morfessor Hybrid | Morphological segmentation alone or before subword learning | Medium | P2 |
| N | Morphology-aware Tokenizer | Explicit morphological boundary preservation | Low-Medium | P2 |
| O | Romanization / ASCII Wolof | Normalize script to improve overlap with base tokenizer | High | P1 |
| P | Trans-Tokenization / Hydra | Swap in target tokenizer embeddings or language heads | Low | P3 |
| Q | Vocabulary-free / Tokenizer-free | Remove fixed subword vocabulary entirely | Very Low | P3 |

## Method D: Unigram + Subword Regularization

### Concept

Instead of always choosing one fixed segmentation, a Unigram tokenizer defines a
distribution over possible segmentations and can sample multiple valid
segmentations during training.

### Why it may help Wolof

Wolof has productive morphology and orthographic variation. If the model sees
slightly different valid segmentations during training, it may become less
fragile on rare words and affixed forms.

### Hypothesis

Compared with plain `method_b`, this should improve robustness and slightly
reduce validation perplexity, especially when the benchmark split is small.

### Experiment

1. Train a SentencePiece Unigram tokenizer on the same Wolof benchmark text.
2. During SFT, enable sampled segmentations on the train split only.
3. Keep deterministic encoding for validation and generation.
4. Compare against:
   - `method_b` plain BPE
   - `method_c` plain Unigram without segmentation sampling

### Implementation notes

- The current fast-tokenizer path is not enough by itself; this likely needs a
  SentencePiece-backed tokenizer or a custom on-the-fly encode hook.
- Start with `alpha in {0.1, 0.2}` and `nbest_size = -1`.

### Source

- Subword Regularization: Improving Neural Network Translation Models with
  Multiple Subword Candidates
  https://aclanthology.org/P18-1007/
- SentencePiece: A simple and language independent subword tokenizer and
  detokenizer for Neural Text Processing
  https://arxiv.org/abs/1808.06226

## Method E: BPE-Dropout

### Concept

During training, randomly drop some BPE merges so the model sees alternative
segmentations. At inference, use the normal deterministic BPE tokenizer.

### Why it may help Wolof

This is the closest extension of your winning `method_b`, because it keeps the
same BPE family while regularizing against brittle segment boundaries.

### Hypothesis

This is one of the highest-probability wins. It may keep the strong perplexity
of `method_b` while improving robustness on rare and morphologically complex
forms.

### Experiment

1. Keep the current `method_b` tokenizer training procedure.
2. Add BPE-dropout only in train-time tokenization.
3. Test dropout rates `0.1`, `0.2`, and `0.3`.
4. Keep the rest of the benchmark identical.

### Implementation notes

- This is easier than most other candidates.
- If it works, it can likely be added as a train-time flag instead of a new
  tokenizer artifact family.

### Source

- BPE-Dropout: Simple and Effective Subword Regularization
  https://aclanthology.org/2020.acl-main.170/

## Method F: WECHSEL

### Concept

Replace the tokenizer with a target-language tokenizer, then initialize target
subword embeddings using multilingual static word embeddings so that the new
subwords land near semantically related source-language tokens.

### Why it may help Wolof

Your current transfer methods are mostly surface-based. WECHSEL introduces a
semantic transfer path, which could be more useful when the Wolof token surface
is very different from English or when direct overlap is weak.

### Hypothesis

WECHSEL-style initialization should beat simple subpiece averaging on embedding
initialization quality and may close the downstream gap between better
segmentation and better initialization.

### Experiment

1. Train the same Wolof tokenizer as in `method_b` or `method_c`.
2. Replace the current initialization rule with WECHSEL initialization.
3. Use aligned static embeddings for Wolof plus English or French.
4. Compare directly against:
   - `method_b`
   - `method_c`
   - OMP transplantation

### Implementation notes

- Needs cross-lingual static embeddings that cover Wolof.
- If Wolof coverage is weak, French can act as a pivot.

### Source

- WECHSEL: Effective initialization of subword embeddings for cross-lingual
  transfer of monolingual language models
  https://aclanthology.org/2022.naacl-main.293/

## Method G: Dictionary-based Cross-lingual Vocabulary Transfer

### Concept

Use a bilingual dictionary to estimate target-token embeddings instead of
relying only on string overlap. The 2025 paper specifically proposes iterative
subword embedding estimation using bilingual dictionaries.

### Why it may help Wolof

Wolof has much weaker overlap with Qwen's original tokenizer than English. A
dictionary-based bridge may recover meaning for tokens that surface heuristics
miss.

### Hypothesis

If you can get a decent Wolof-English or Wolof-French lexicon, this should
improve downstream transfer more than raw overlap-based initialization.

### Experiment

1. Build or obtain a bilingual lexicon for Wolof.
2. Train a Wolof tokenizer as in `method_b`.
3. Initialize new rows with the dictionary-based transfer method.
4. Compare against `method_b`, WECHSEL, and OMP.

### Implementation notes

- Strong candidate if lexicon coverage is good.
- Lower risk than fully new segmentation methods.

### Source

- Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for
  Low-Resource Languages Using Bilingual Dictionaries
  https://aclanthology.org/2025.findings-acl.1333.pdf

## Method H: OMP Tokenizer Transplantation

### Concept

Initialize unseen token embeddings as a sparse linear combination of shared
anchor tokens using Orthogonal Matching Pursuit instead of a simple average.

### Why it may help Wolof

This is a very strong fit for your current pipeline. It is effectively a better
initialization rule for transplanted tokenizers without needing model
retraining from scratch.

### Hypothesis

This has a realistic chance to improve `method_b` directly, because `method_b`
already wins on segmentation and may still be bottlenecked by imperfect
initialization.

### Experiment

1. Keep the current `method_b` tokenizer training recipe.
2. Replace embedding initialization with OMP reconstruction.
3. Sweep sparsity `k in {4, 8, 16}`.
4. Compare perplexity and initialization cosine.

### Implementation notes

- This is one of the best P1 candidates.
- It changes only initialization, not tokenization itself.

### Source

- Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit
  https://arxiv.org/abs/2506.06607

## Method I: Universal Multilingual Tokenizer

### Concept

Instead of training a tokenizer on Wolof only, train it on a broader
multilingual mixture that includes Wolof plus support languages such as French,
English, and possibly related African languages.

### Why it may help Wolof

A Wolof-only tokenizer may overfit a small corpus and lose compatibility with
the base model. A broader tokenizer may preserve transfer while still allocating
more useful tokens to Wolof.

### Hypothesis

A multilingual tokenizer mixture may outperform Wolof-only tokenizer training if
the Wolof corpus is too small to learn a stable vocabulary by itself.

### Experiment

1. Build a tokenizer-training corpus with mixtures such as:
   - `70% Wolof / 15% French / 15% English`
   - `50% Wolof / 25% French / 25% English`
2. Train BPE and Unigram variants.
3. Initialize via overlap, OMP, or WECHSEL.
4. Compare against Wolof-only `method_b`.

### Implementation notes

- Good balance between practicality and upside.
- Also helps retain French/English prompt handling if your downstream users mix
  languages.

### Source

- One Tokenizer To Rule Them All: Emergent Language Plasticity via Multilingual
  Tokenizers
  https://arxiv.org/abs/2506.10766
- Language Model Tokenizers Introduce Unfairness Between Languages
  https://arxiv.org/abs/2305.15425

## Method J: VOLT

### Concept

Vocabulary Learning via Optimal Transport chooses vocabulary units with an
optimal-transport objective rather than by simple merge-frequency heuristics.

### Why it may help Wolof

Wolof morphology may not align with the frequency-greedy decisions of standard
BPE. VOLT explicitly searches for a better vocabulary.

### Hypothesis

VOLT may reduce pathological over-fragmentation and outperform plain BPE on the
benchmark split, especially if the corpus is not extremely noisy.

### Experiment

1. Train a VOLT vocabulary on the benchmark train split.
2. Build a compatible tokenizer from that vocabulary.
3. Use the same initialization rule as `method_b`.
4. Compare against BPE and Unigram.

### Implementation notes

- Promising, but engineering is heavier than BPE-dropout or OMP.
- Best treated as a P2 segmentation research branch.

### Source

- Vocabulary Learning via Optimal Transport for Neural Machine Translation
  https://aclanthology.org/2021.acl-long.571/

## Method K: SelfSeg

### Concept

SelfSeg is a self-supervised neural segmentation method that uses monolingual
dictionaries rather than parallel data.

### Why it may help Wolof

It targets exactly the type of low-resource setting where hand-designed BPE
merges may be suboptimal.

### Hypothesis

SelfSeg could outperform BPE or Unigram when segmentation quality, not only
embedding initialization, is the dominant problem.

### Experiment

1. Build a Wolof lexicon from corpus frequency and any available dictionaries.
2. Train SelfSeg on the benchmark corpus.
3. Export the resulting segmentation to tokenizer training data.
4. Compare against VOLT, BERTSeg, and plain Unigram.

### Implementation notes

- Strong research idea, but clearly higher engineering cost.
- Better as a second-wave experiment after easier P1 methods.

### Source

- SelfSeg: A Self-supervised Sub-word Segmentation Method for Neural Machine
  Translation
  https://arxiv.org/abs/2307.16400

## Method L: BERTSeg

### Concept

BERTSeg uses a pretrained masked language model to infer subword segmentation
boundaries in an unsupervised way.

### Why it may help Wolof

If a multilingual encoder already captures useful contextual signals, it can be
used to score better boundaries than a purely count-based tokenizer.

### Hypothesis

BERTSeg may improve segmentation over BPE/VOLT in some settings, but its
success for Wolof will depend heavily on whether the underlying encoder already
handles Wolof reasonably well.

### Experiment

1. Choose a multilingual encoder with the best available Wolof coverage.
2. Train BERTSeg on the benchmark corpus.
3. Export segmentation and build the tokenizer.
4. Compare against SelfSeg and VOLT.

### Implementation notes

- Higher risk than SelfSeg because it inherits the weaknesses of the underlying
  encoder.
- Still worth benchmarking if a good multilingual encoder is available.

### Source

- BERTSeg: BERT Based Unsupervised Subword Segmentation for Neural Machine
  Translation
  https://aclanthology.org/2022.aacl-short.12/

## Method M: Morfessor Hybrid

### Concept

First segment words into morpheme-like units with Morfessor, then either use
those units directly or feed them into BPE/Unigram as a second stage.

### Why it may help Wolof

Wolof morphology is linguistically meaningful. A morphology-first tokenizer may
preserve stems and affixes more cleanly than a purely frequency-based subword
tokenizer.

### Hypothesis

Morfessor alone may over-segment, but a Morfessor+BPE hybrid could beat plain
subword tokenizers on coverage and robustness.

### Experiment

1. Train Morfessor on the Wolof training corpus.
2. Compare:
   - Morfessor-only tokenization
   - Morfessor -> BPE
   - Morfessor -> Unigram
3. Use the same Qwen embedding-transfer recipe as the baseline.

### Implementation notes

- Good P2 method because it is easy to reason about and linguistically
  interpretable.

### Source

- Morfessor 2.0: Python Implementation and Extensions for Morfessor Baseline
  https://aaltodoc.aalto.fi/items/8c912f1c-2b56-4c12-a60d-1c340aa24b01
- Effects of sub-word segmentation on performance of transformer language
  models
  https://aclanthology.org/2023.emnlp-main.459/

## Method N: Morphology-aware Tokenizer

### Concept

Use explicit morphological boundaries or morpheme resources to guide subword
construction, rather than relying only on corpus statistics.

### Why it may help Wolof

This is the linguistically strongest option if you can obtain even a lightweight
Wolof affix inventory or morphological analyzer.

### Hypothesis

For a language where grammatical information is often packed into affixes,
boundary-aware tokenization could improve both interpretability and generation.

### Experiment

1. Build a small Wolof morphological ruleset:
   - productive suffixes
   - common clitics
   - frequent derivational patterns
2. Train a morphology-aware tokenizer such as MorphPiece- or MorphBPE-style
   hybrid tokenization.
3. Compare against Morfessor hybrid and BPE.

### Implementation notes

- Best if you can work with a linguist or build a hand-crafted affix list.
- High upside, but higher manual effort.

### Source

- MorphPiece: A Linguistic Tokenizer for Large Language Models
  https://arxiv.org/abs/2307.07262
- MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for
  Efficient LLM Training Across Morphologies
  https://arxiv.org/abs/2502.00894

## Method O: Romanization / ASCII Wolof

### Concept

Normalize Wolof into a more overlap-friendly script form, for example by
reversible romanization or ASCII-friendly transliteration, then train or reuse
the tokenizer on that normalized representation.

### Why it may help Wolof

Wolof already uses Latin script, but characters like `e with diaeresis`,
`enye`, and mixed spelling conventions can still fragment heavily in
English-centric tokenizers.

### Hypothesis

This is a low-cost experiment with a realistic chance of improving overlap with
Qwen's base vocabulary, though it risks losing orthographic distinctions.

### Experiment

1. Define a reversible normalization such as:
   - `e with diaeresis -> e~`
   - `enye -> ny`
2. Train the same tokenizer families on normalized Wolof.
3. Evaluate both:
   - normalized training plus normalized inference
   - normalized training plus detokenized output restoration

### Implementation notes

- Good P1 experiment because it is cheap.
- Must be reversible; do not destroy distinctions without a detokenization map.

### Source

- RomanSetu: Efficiently unlocking multilingual capabilities of Large Language
  Models via Romanization
  https://aclanthology.org/2024.acl-long.833/
- Romanization-based Large-scale Adaptation of Multilingual Language Models
  https://arxiv.org/abs/2304.08865

## Method P: Trans-Tokenization / Hydra

### Concept

Trans-tokenization transfers a model to a new tokenizer more explicitly, and the
Hydra variant maintains multiple swappable embedding tables or heads.

### Why it may help Wolof

This is appealing if you want to preserve Qwen's original English behavior while
adding a stronger Wolof tokenizer without forcing one vocabulary to serve both
equally well.

### Hypothesis

This could be stronger than simple tokenizer replacement, but it is closer to a
model-architecture adaptation than a tokenizer tweak.

### Experiment

1. Keep Qwen body weights fixed or lightly adapted.
2. Learn a Wolof-specific embedding table and optionally a Wolof head.
3. Compare against plain tokenizer replacement under the same training budget.

### Implementation notes

- P3 research track.
- Not the next experiment to run unless you are ready to modify the model
  architecture.

### Source

- Trans-Tokenization and Cross-lingual Vocabulary Transfers: Language
  Adaptation of LLMs for Low-Resource NLP
  https://arxiv.org/abs/2408.04303

## Method Q: Vocabulary-free / Tokenizer-free Models

### Concept

Remove the fixed subword vocabulary entirely and operate on bytes, characters,
or learned sparse token-free representations.

### Why it may help Wolof

This completely avoids the problem of poor subword allocation for Wolof. It is
the cleanest answer in principle, but the farthest from your current Qwen
fine-tuning setup.

### Hypothesis

This is best viewed as a long-term research path, not a near-term replacement
for `method_b`.

### Experiment

1. Train or distill a byte-level or tokenizer-free student.
2. Compare compression, latency, and downstream loss against the subword model.
3. Only attempt after the subword search space is exhausted.

### Implementation notes

- P3 only.
- Requires a new model family, not just a new tokenizer artifact.

### Source

- A Vocabulary-Free Multilingual Neural Tokenizer for End-to-End Task Learning
  https://aclanthology.org/2022.repl4nlp-1.10/
- T-FREE: Subword Tokenizer-Free Generative LLMs via Sparse Representations for
  Memory-Efficient Embeddings
  https://aclanthology.org/2024.emnlp-main.1217/
- SpaceByte: Towards Deleting Tokenization from Large Language Modeling
  https://arxiv.org/abs/2404.14408

## Recommended Execution Order

If the objective is practical progress on the current Qwen/Wolof pipeline,
start here:

1. `Method E` BPE-dropout on top of `method_b`
2. `Method H` OMP initialization on top of `method_b`
3. `Method F` WECHSEL initialization
4. `Method I` multilingual tokenizer mixture
5. `Method O` reversible Wolof normalization / romanization

Second wave:

6. `Method G` dictionary-based vocabulary transfer
7. `Method M` Morfessor hybrid
8. `Method N` morphology-aware tokenizer
9. `Method J` VOLT
10. `Method K` SelfSeg
11. `Method L` BERTSeg

Long-term research:

12. `Method P` trans-tokenization / Hydra
13. `Method Q` tokenizer-free models

## Recommended H100 Experiment Matrix

If you want a serious next sweep, I recommend this matrix rather than jumping
straight to all methods at once:

### Wave 1: high-value low-risk

- `method_b + bpe_dropout`
- `method_b + omp_init`
- `method_b + wechsel_init`
- `method_b + multilingual_tokenizer_mix`
- `method_c + subword_regularization`

### Wave 2: linguistic segmentation

- `morfessor_bpe_hybrid`
- `morphology_aware_bpe`
- `volt_vocab`

### Wave 3: high-research-cost

- `selfseg`
- `bertseg`
- `dictionary_transfer`
- `trans_tokenization`

## Final Recommendation

Given the current evidence, **do not replace `method_b` as the default yet**.

The strongest next experiments are the ones that keep `method_b`'s segmentation
and improve only:

- train-time segmentation regularization
- embedding initialization
- tokenizer training data mix

In practice, the best first three candidates are:

1. `method_b + bpe_dropout`
2. `method_b + omp_init`
3. `method_b + wechsel_init`

Those three have the best trade-off between:

- literature support
- compatibility with the current repository
- likelihood of improving the current winner instead of restarting the search
  from scratch
