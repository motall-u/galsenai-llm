# Data Format

This project uses a chat-style JSONL format designed for supervised finetuning of models that must decide between:

- `no_tool`
- `single_tool`
- `multi_tool`
- `reasoning_tool`

Each line in a dataset file is one complete example.

## Required top-level fields

- `id`: Stable string identifier.
- `category`: One of `no_tool`, `single_tool`, `multi_tool`, or `reasoning_tool`.
- `messages`: Ordered chat transcript.
- `tools`: List of tool definitions available for that example. Use `[]` when no tools are available.

## Optional top-level fields

- `metadata`: Free-form tags such as split, difficulty, domain, source, or notes.
- `expected`: Benchmark-only expectations. The CLI uses `expected.first_assistant` when scoring first-turn benchmark predictions.

## Message format

Supported roles:

- `system`
- `user`
- `assistant`
- `tool`

Assistant messages can contain either direct natural-language `content`, structured `tool_calls`, or both.

Tool messages must contain:

- `role: "tool"`
- `name`
- `tool_call_id`
- `content`

## Tool call format

Tool calls use the common function-calling shape:

```json
{
  "id": "call_001",
  "type": "function",
  "function": {
    "name": "get_vector_sum",
    "arguments": {
      "a": [1, -1, 2],
      "b": [3, 0, -4]
    }
  }
}
```

## Tool definition format

Each item in `tools` is an OpenAI-style function schema:

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

## Category guidance

- `no_tool`: The assistant should answer directly and never call a tool.
- `single_tool`: One tool call is enough to solve the task.
- `multi_tool`: The task needs multiple tool calls across the trajectory.
- `reasoning_tool`: The task needs tool use plus intermediate composition or dependency handling. Do not store chain-of-thought. Store only the visible tool trajectory and final answer.

## Recommended authoring rules

- Keep tool outputs JSON-serializable.
- Keep tool arguments explicit and deterministic.
- Prefer full transcripts for training, not just the first tool call.
- Keep benchmark prompts separate from training prompts when you move beyond the starter sample.
- If a tool result is required before the next call, represent that dependency with sequential assistant/tool turns.

## Minimal example

```json
{
  "id": "single-tool-001",
  "category": "single_tool",
  "messages": [
    { "role": "system", "content": "You are a careful assistant. Use tools for calculations." },
    { "role": "user", "content": "Find the sum of a = [1, -1, 2] and b = [3, 0, -4]." },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_get_vector_sum_001",
          "type": "function",
          "function": {
            "name": "get_vector_sum",
            "arguments": { "a": [1, -1, 2], "b": [3, 0, -4] }
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_get_vector_sum_001",
      "name": "get_vector_sum",
      "content": [4, -1, -2]
    },
    { "role": "assistant", "content": "The vector sum is [4, -1, -2]." }
  ],
  "tools": [
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
  ],
  "metadata": {
    "split": "train",
    "difficulty": "easy"
  }
}
```

## Sample files in this repo

- [finetune_sample.jsonl](/home/motall-u/my_projects/galsenai-llm/data/samples/finetune_sample.jsonl)
- [benchmark_sample.jsonl](/home/motall-u/my_projects/galsenai-llm/data/samples/benchmark_sample.jsonl)

The local Python implementations for the sample tools live in [tool_registry.py](/home/motall-u/my_projects/galsenai-llm/src/galsenai_llm/tool_registry.py).
