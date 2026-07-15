"""Exact scoring for structured output and fake tool-calling scenarios."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any]


class ToolScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    expected_calls: list[ToolCall]


class StructuredScore(BaseModel):
    parses: bool
    schema_valid: bool
    values_correct: bool


def score_structured_output(
    raw: str, expected: dict[str, Any], schema: type[BaseModel]
) -> StructuredScore:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return StructuredScore(parses=False, schema_valid=False, values_correct=False)
    try:
        validated = schema.model_validate(parsed)
    except ValidationError:
        return StructuredScore(parses=True, schema_valid=False, values_correct=False)
    return StructuredScore(
        parses=True, schema_valid=True, values_correct=validated.model_dump() == expected
    )


def score_tool_calls(scenario: ToolScenario, actual: list[ToolCall]) -> bool:
    return actual == scenario.expected_calls
