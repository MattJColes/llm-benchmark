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


def evaluate_structured_runs(
    responses: list[str], expected: dict[str, Any], schema: type[BaseModel]
) -> dict[str, float | bool]:
    scores = [score_structured_output(response, expected, schema) for response in responses]
    if not scores:
        raise ValueError("at least one response is required")
    total = len(scores)
    return {
        "parse_rate": sum(score.parses for score in scores) / total,
        "schema_rate": sum(score.schema_valid for score in scores) / total,
        "value_rate": sum(score.values_correct for score in scores) / total,
        "stable": len(set(responses)) == 1,
    }


def constrained_delta(
    freeform: dict[str, float | bool], constrained: dict[str, float | bool]
) -> float:
    return float(constrained["schema_rate"]) - float(freeform["schema_rate"])
