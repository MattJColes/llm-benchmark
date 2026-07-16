"""Exact scoring for structured output and fake tool-calling scenarios."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any]


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    arguments: dict[str, str] = Field(min_length=1)


class ToolScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    prompt: str
    expected_calls: list[ToolCall]
    tool_results: list[dict[str, Any]] = Field(default_factory=list)


class ToolManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools: list[ToolDefinition] = Field(min_length=1)
    scenarios: list[ToolScenario] = Field(min_length=1)

    @model_validator(mode="after")
    def validates_references_and_ids(self) -> ToolManifest:
        tool_names = [tool.name for tool in self.tools]
        scenario_ids = [scenario.id for scenario in self.scenarios]
        if len(tool_names) != len(set(tool_names)) or len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("tool and scenario IDs must be unique")
        unknown = {
            call.name
            for scenario in self.scenarios
            for call in scenario.expected_calls
            if call.name not in tool_names
        }
        if unknown:
            raise ValueError(f"unknown expected tools: {sorted(unknown)}")
        if any(
            len(scenario.tool_results) != len(scenario.expected_calls)
            for scenario in self.scenarios
        ):
            raise ValueError("each expected call requires a supplied tool result")
        return self


class StructuredScore(BaseModel):
    parses: bool
    schema_valid: bool
    values_correct: bool


def score_structured_output(
    raw: str, expected: dict[str, Any], schema: type[BaseModel]
) -> StructuredScore:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return StructuredScore(parses=False, schema_valid=False, values_correct=False)
    try:
        validated = schema.model_validate(parsed)
    except ValidationError:
        return StructuredScore(parses=True, schema_valid=False, values_correct=False)
    return StructuredScore(
        parses=True, schema_valid=True, values_correct=validated.model_dump() == expected
    )


def load_tool_manifest(path: Path) -> ToolManifest:
    return ToolManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def execute_tool_scenario(
    scenario: ToolScenario,
    tools: list[ToolDefinition],
    call_model: Callable[[list[dict[str, Any]], list[ToolDefinition]], ToolCall | None],
) -> list[ToolCall]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": scenario.prompt}]
    calls: list[ToolCall] = []
    for step in range(len(scenario.expected_calls) + 1):
        call = call_model(messages, tools)
        if call is None:
            break
        calls.append(call)
        messages.append({"role": "assistant", "tool_call": call.model_dump()})
        if step < len(scenario.tool_results):
            messages.append({"role": "tool", "content": scenario.tool_results[step]})
    return calls


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
    try:
        return float(constrained["schema_rate"]) - float(freeform["schema_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("schema rates must be numeric") from error
