from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ToolFunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    id: str | None = None
    type: Literal["function"] = "function"
    function: ToolFunctionCall


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

    @model_validator(mode="after")
    def validate_message(self) -> Message:
        if self.role == "assistant":
            if self.content is None and not self.tool_calls:
                raise ValueError("Assistant messages must include content or tool_calls.")
        else:
            if self.tool_calls:
                raise ValueError("Only assistant messages may include tool_calls.")

        if self.role == "tool":
            if not self.name:
                raise ValueError("Tool messages must include a tool name.")
            if self.content is None:
                raise ValueError("Tool messages must include a content payload.")

        return self


class ToolFunctionSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunctionSchema


class ExpectedOutcome(BaseModel):
    first_assistant: Message | None = None
    tool_sequence: list[str] = Field(default_factory=list)
    final_answer: str | None = None


class DatasetExample(BaseModel):
    id: str
    category: Literal["no_tool", "single_tool", "multi_tool", "reasoning_tool"]
    messages: list[Message]
    tools: list[ToolDefinition] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    expected: ExpectedOutcome | None = None

    @model_validator(mode="after")
    def validate_example(self) -> DatasetExample:
        if len(self.messages) < 2:
            raise ValueError("Each example must include at least two messages.")
        if not any(message.role == "assistant" for message in self.messages):
            raise ValueError("Each example must include at least one assistant message.")
        return self


class PredictionRecord(BaseModel):
    id: str
    assistant: Message


def first_assistant_message(example: DatasetExample) -> Message:
    if example.expected and example.expected.first_assistant is not None:
        return example.expected.first_assistant

    for message in example.messages:
        if message.role == "assistant":
            return message
    raise ValueError(f"Example {example.id} has no assistant message.")


def prompt_messages(example: DatasetExample) -> list[Message]:
    prompt: list[Message] = []
    for message in example.messages:
        if message.role == "assistant":
            break
        prompt.append(message)
    return prompt


def final_assistant_message(example: DatasetExample) -> Message:
    assistants = [message for message in example.messages if message.role == "assistant"]
    if not assistants:
        raise ValueError(f"Example {example.id} has no assistant message.")
    return assistants[-1]
