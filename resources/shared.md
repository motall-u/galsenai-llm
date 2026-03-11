# Shared Resources -- Fine-tuning LLMs for Tool Calling

## Context

This resource collection supports the **GalsenAI LLM** project. Our goal is to fine-tune medium-sized LLMs (< 14 billion parameters) to excel at **tool calling for task definition**. The model should learn to: recognize when a tool is needed, select the correct tool, construct valid arguments, interpret tool results, and reason through multi-step tool chains.

We target models like Qwen 2.5 (7B/14B) using parameter-efficient methods (LoRA/QLoRA via Unsloth) so training fits on consumer or single-GPU setups.

---

## 1. Fine-tuning Fundamentals

| # | Resource | Type | What you'll learn |
|---|----------|------|-------------------|
| 1 | [How to Fine-tune LLMs with Unsloth: Complete Guide](https://www.youtube.com/watch?v=Lt7KrFMcCis) | Video | End-to-end walkthrough of fine-tuning with Unsloth -- covers data prep, LoRA config, training loop, and model export. |

## 2. Core Concepts

| # | Resource | Type | What you'll learn |
|---|----------|------|-------------------|
| 2 | [What is Quantization](https://www.ibm.com/think/topics/quantization) | Article | How quantization (4-bit, 8-bit) reduces model size and memory usage while preserving accuracy -- critical for training large models on limited hardware. |
| 3 | [What is Tool Calling](https://www.ibm.com/think/topics/tool-calling) | Article | How LLMs invoke external tools/functions -- the exact capability we are fine-tuning for. Covers the request/response pattern, schema definitions, and use cases. |

## 3. Key Terms Glossary

| Term | Definition |
|------|------------|
| **Fine-tuning** | Continuing the training of a pre-trained model on a domain-specific dataset to specialize its behavior. |
| **LoRA (Low-Rank Adaptation)** | A parameter-efficient method that freezes the base model and trains small rank-decomposition matrices, drastically reducing memory and compute. |
| **QLoRA** | LoRA applied on top of a quantized (4-bit) base model, enabling fine-tuning of large models on a single GPU. |
| **Quantization** | Reducing the numerical precision of model weights (e.g., FP16 to 4-bit) to shrink memory footprint. |
| **Tool calling** | The ability of an LLM to emit structured function calls (name + arguments) instead of or alongside natural language, enabling it to interact with external APIs and tools. |
| **SFT (Supervised Fine-Tuning)** | Training a model on input-output pairs where the correct response is provided, as opposed to reinforcement learning approaches. |
| **Unsloth** | An optimized fine-tuning library that accelerates LoRA/QLoRA training with custom CUDA kernels and memory optimizations. |
| **Chat template** | A tokenizer-level format that structures multi-turn conversations (system, user, assistant, tool roles) into the token sequence the model expects. |

## 4. Our Training Categories

The dataset is organized into four progressive levels of tool-calling complexity:

| Category | Description | Example |
|----------|-------------|---------|
| `no_tool` | Conversational response with no tool use. Teaches the model when *not* to call a tool. | "What is 2+2?" -> direct answer |
| `single_tool` | One tool call per turn. The model learns tool selection and argument construction. | "Get the sum of vectors [1,2] and [3,4]" -> `get_vector_sum(a=[1,2], b=[3,4])` |
| `multi_tool` | Multiple tool calls across turns, building chains of tool interactions. | Query inventory, then compute restock cost based on the result. |
| `reasoning_tool` | Tool calls that require intermediate thinking/reasoning before selecting the tool and arguments. | A complex math word problem requiring decomposition before calling the right tools. |

## 5. Recommended Learning Path

1. **Start with the concepts** -- Read the articles on quantization (#2) and tool calling (#3) to build foundational understanding.
2. **Watch the tutorial** -- Follow the Unsloth fine-tuning video (#1) to see the full training pipeline in action.
3. **Explore the codebase** -- Read `data/README.md` for dataset format details, then look at sample data in `data/samples/`.
4. **Run a smoke test** -- Use `configs/train/smoke_tiny_gpt2.yaml` (CPU) or `configs/train/smoke_unsloth_qwen_tool_calling.yaml` (GPU) to verify the pipeline works.
5. **Train for real** -- Use `configs/train/sample_qwen_tool_calling.yaml` as the starting config for actual fine-tuning runs.

---

*Last updated: 2026-03-11*
